"""From-scratch rotation-invariant candidate scorer: radial set terms + angular pair terms, listwise softmax over 4 candidates."""
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn, time, sys, argparse, os
sys.path.insert(0, 'dev'); import common, feats
ap = argparse.ArgumentParser()
ap.add_argument('--K', type=int, default=16); ap.add_argument('--hidden', type=int, default=96)
ap.add_argument('--epochs', type=int, default=80); ap.add_argument('--seeds', type=int, default=3)
ap.add_argument('--lr', type=float, default=2e-3); ap.add_argument('--wd', type=float, default=1e-3)
ap.add_argument('--drop', type=float, default=0.1); ap.add_argument('--bs', type=int, default=32)
ap.add_argument('--tab', type=int, default=0); ap.add_argument('--out', default='nn1')
ap.add_argument('--folds', type=str, default='0,1,2,3,4'); ap.add_argument('--nrbf', type=int, default=24)
ap.add_argument('--label_smooth', type=float, default=0.0); ap.add_argument('--head', default='plain'); ap.add_argument('--angK', type=int, default=0)
args = ap.parse_args()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
d = common.load(); Y = d['Y']; folds = d['folds']; N = len(Y)

def prep(env, zc, site):
    x = env[..., :3].astype(np.float32) * 8; r = env[..., 3].astype(np.float32) * 8
    zn = np.rint(env[..., 4] * 118).astype(np.int64)
    u = x / r[..., None]
    K = args.K
    cos = np.einsum('nckd,ncjd->nckj', u[:, :, :K], u[:, :, :K]).astype(np.float32)
    return dict(r=torch.tensor(r), zn=torch.tensor(zn), cos=torch.tensor(cos), zc=torch.tensor(zc.astype(np.int64)), site=torch.tensor(site.astype(np.int64)))
Tr = prep(d['env'], d['zc'], d['site']); Te = prep(d['env_t'], d['zc_t'], d['site_t'])

class RBF(nn.Module):
    def __init__(self, lo, hi, n):
        super().__init__(); self.register_buffer('mu', torch.linspace(lo, hi, n)); self.gamma = ((hi - lo) / n) ** -2
    def forward(self, x): return torch.exp(-self.gamma * (x[..., None] - self.mu) ** 2)

class Net(nn.Module):
    def __init__(self, H, K, ntab=0):
        super().__init__()
        self.K = K
        self.ez = nn.Embedding(120, 16); self.es = nn.Embedding(26, 4)
        self.rbf = RBF(0.7, 8.7, args.nrbf); self.arbf = RBF(-1, 1, 12)
        ec = 16 + 4 + 4  # center emb + continuous (EN, period/7, group/18, block/3)
        self.cmlp = nn.Sequential(nn.Linear(ec, H), nn.SiLU(), nn.Linear(H, H))
        self.rad = nn.Sequential(nn.Linear(args.nrbf + 16 + 1 + H, H), nn.SiLU(), nn.Linear(H, H), nn.SiLU())
        self.ang = nn.Sequential(nn.Linear(2 * args.nrbf + 12 + 16 + 16 + H, H), nn.SiLU(), nn.Linear(H, H), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(3 * H + ntab, H), nn.SiLU(), nn.Dropout(args.drop), nn.Linear(H, H // 2), nn.SiLU(), nn.Linear(H // 2, 1))
        self.exp_head = nn.Sequential(nn.Linear(H + 3, H // 2), nn.SiLU(), nn.Linear(H // 2, 1))
        self.scale_head = nn.Sequential(nn.Linear(H, H // 2), nn.SiLU(), nn.Linear(H // 2, 1))
        per, grp, blk = feats.period_group(np.arange(1, 120)); per, grp, blk = [np.concatenate([[0], a]) for a in (per, grp, blk)]
        en = np.concatenate([feats.EN, np.zeros(120 - len(feats.EN))])[:120]
        self.register_buffer('ccont', torch.tensor(np.stack([en, per / 7.0, grp / 18.0, blk / 3.0], -1), dtype=torch.float32))
    def forward(self, r, zn, cos, zc, site, tab=None):
        # r (B,4,48) zn (B,4,48) cos (B,4,K,K) zc (B,4) site (B,4)
        B = r.shape[0]
        c = torch.cat([self.ez(zc), self.es(site), self.ccont[zc]], -1); c = self.cmlp(c)  # (B,4,H)
        rb = self.rbf(r)  # (B,4,48,nrbf)
        zemb = self.ez(zn)
        hr = self.rad(torch.cat([rb, zemb, 1.0 / r[..., None] ** 3 * 10, c[:, :, None, :].expand(-1, -1, r.shape[2], -1)], -1))
        pool_r = torch.cat([hr.sum(2) / 48 ** 0.5, hr.max(2).values], -1)  # (B,4,2H)
        K = self.K
        iu = torch.triu_indices(K, K, 1, device=r.device)
        rbK = rb[:, :, :K]; zK = zemb[:, :, :K]
        pa = torch.cat([rbK[:, :, iu[0]], rbK[:, :, iu[1]], self.arbf(cos[:, :, iu[0], iu[1]]), zK[:, :, iu[0]] + zK[:, :, iu[1]], zK[:, :, iu[0]] * zK[:, :, iu[1]],
                        c[:, :, None, :].expand(-1, -1, iu.shape[1], -1)], -1)
        ha = self.ang(pa).sum(2) / iu.shape[1] ** 0.5
        h = torch.cat([pool_r, ha], -1)
        if tab is not None: h = torch.cat([h, tab], -1)
        if args.head == 'plain':
            return self.head(h).squeeze(-1)  # (B,4)
        # residual head: anomaly = |r_hat - expected(element, coarse env)| / scale(element)
        rel = r / r[..., :1]
        coarse = torch.stack([(rel <= 1.2).float().sum(-1) / 6 - 1, r[..., 0] / 2.3 - 1, self.ccont[zn][..., 0].mean(-1) - self.ccont[zc][..., 0]], -1)
        r_hat = self.head(h).squeeze(-1)
        expected = self.exp_head(torch.cat([c, coarse], -1)).squeeze(-1)
        scale = torch.exp(torch.clamp(self.scale_head(c).squeeze(-1), -4, 4))
        return torch.abs(r_hat - expected) / scale

def batches(idx, bs, shuffle, rng):
    idx = np.array(idx)
    if shuffle: idx = rng.permutation(idx)
    for i in range(0, len(idx), bs): yield idx[i:i + bs]
def get(T, idx, tab=None):
    out = {k: v[idx].to(dev) for k, v in T.items()}
    if tab is not None: out['tab'] = torch.tensor(tab[idx]).to(dev)
    return out

oof = np.zeros((N, 4), np.float32); test_logit = np.zeros((len(d['ids_t']), 4), np.float32); t0 = time.time()
use_folds = [int(f) for f in args.folds.split(',')]
for k in use_folds:
    trn = np.where(folds != k)[0]; val = np.where(folds == k)[0]
    tabX = tabXt = None; ntab = 0
    if args.tab:
        X, Xt, names = common.tab_features(d, trn)
        cols = [j for j, n in enumerate(names) if not (n.endswith('_rk'))]
        X = X[:, :, cols]; Xt = Xt[:, :, cols]
        mu = X[trn].reshape(-1, X.shape[-1]).mean(0); sd = X[trn].reshape(-1, X.shape[-1]).std(0) + 1e-6
        tabX = np.clip((X - mu) / sd, -5, 5).astype(np.float32); tabXt = np.clip((Xt - mu) / sd, -5, 5).astype(np.float32); ntab = X.shape[-1]
    for s in range(args.seeds):
        torch.manual_seed(1000 * k + s); rng = np.random.default_rng(1000 * k + s)
        net = Net(args.hidden, args.K, ntab).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
        steps = args.epochs * int(np.ceil(len(trn) / args.bs)); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.15)
        Yt = torch.tensor(Y)
        for ep in range(args.epochs):
            net.train()
            for b in batches(trn, args.bs, True, rng):
                inp = get(Tr, b, tabX); yb = Yt[b].to(dev)
                logit = net(**inp)
                loss = Fn.cross_entropy(logit, yb, label_smoothing=args.label_smooth)
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step(); sched.step()
        net.eval()
        with torch.no_grad():
            pv = torch.cat([net(**get(Tr, b, tabX)) for b in batches(val, 128, False, rng)]).cpu().numpy()
            pt = torch.cat([net(**get(Te, b, tabXt)) for b in batches(np.arange(len(d['ids_t'])), 128, False, rng)]).cpu().numpy()
        oof[val] += pv / args.seeds; test_logit += pt / (args.seeds * len(use_folds))
        print(f"fold {k} seed {s}: val acc {(pv.argmax(1) == Y[val]).mean():.4f}  t={time.time() - t0:.0f}s", flush=True)
    print(f"fold {k}: acc {(oof[val].argmax(1) == Y[val]).mean():.4f}", flush=True)
m = np.isin(folds, use_folds)
print(f"OOF acc ({args.out}): {(oof[m].argmax(1) == Y[m]).mean():.4f}")
os.makedirs('dev/out', exist_ok=True)
np.save(f'dev/out/{args.out}_oof.npy', oof); np.save(f'dev/out/{args.out}_test.npy', test_logit)
