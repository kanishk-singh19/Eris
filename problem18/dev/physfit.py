"""Parametric physics model: T_c = sum_k q[Z_k] (3xx^T - r^2 I)/r^5 ; r_c = log(1 + exp(a[Zc]) ||T_c||) ;
anomaly_c = |r_c - m[Zc] - beta[Zc]*cn_c| / exp(s[Zc]) ; listwise softmax. Grouped CV."""
import numpy as np, torch, sys, time, argparse
sys.path.insert(0, 'dev'); import common
ap = argparse.ArgumentParser(); ap.add_argument('--variant', default='base'); ap.add_argument('--epochs', type=int, default=300)
ap.add_argument('--lr', type=float, default=0.05); ap.add_argument('--init', default='en')
args = ap.parse_args()
d = common.load(); Y = d['Y']; folds = d['folds']; N = len(Y)
env = d['env']; zc = d['zc']; site = d['site']
x = torch.tensor(env[..., :3] * 8, dtype=torch.float64); r = torch.tensor(env[..., 3] * 8, dtype=torch.float64)
zn = torch.tensor(np.rint(env[..., 4] * 118).astype(np.int64)); Zc = torch.tensor(zc)
outer = x[..., :, None] * x[..., None, :]
B = (3 * outer - (r ** 2)[..., None, None] * torch.eye(3, dtype=torch.float64)) / (r ** 5)[..., None, None]  # (N,4,48,3,3)
rel = r / r[..., :1]; cn = (rel <= 1.2).double().sum(-1)  # (N,4)
d1 = r[..., 0]
from feats import EN
en = torch.tensor(EN.astype(np.float64))
# initial charges: EN-based sign guess relative to oxygen-ish: q0 = (2.5 - EN)*1.5 (cations positive, anions negative)
if args.init == 'en': q0 = (2.3 - en[:120]) * 1.5
else: q0 = torch.zeros(120, dtype=torch.float64) + 1.0
Yt = torch.tensor(Y)

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Parameter(q0.clone()); self.a = torch.nn.Parameter(torch.zeros(120, dtype=torch.float64))
        self.m = torch.nn.Parameter(torch.zeros(120, dtype=torch.float64)); self.s = torch.nn.Parameter(torch.zeros(120, dtype=torch.float64))
        self.beta = torch.nn.Parameter(torch.zeros(120, dtype=torch.float64)); self.gam = torch.nn.Parameter(torch.zeros(120, dtype=torch.float64))
        self.temp = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
    def forward(self, idx):
        q = self.q[zn[idx]]  # (b,4,48)
        T = (q[..., None, None] * B[idx]).sum(2)  # (b,4,3,3)
        fro = torch.sqrt((T ** 2).sum((-1, -2)) + 1e-12)
        z = Zc[idx]
        rr = torch.log1p(torch.exp(self.a[z]) * fro)
        if args.variant == 'base': expect = self.m[z]
        elif args.variant == 'cn': expect = self.m[z] + self.beta[z] * (cn[idx] - 6) / 3
        elif args.variant == 'cnd': expect = self.m[z] + self.beta[z] * (cn[idx] - 6) / 3 + self.gam[z] * (d1[idx] - 2.3)
        anom = torch.abs(rr - expect) / torch.exp(self.s[z])
        return anom * torch.exp(self.temp), rr, anom

oof = np.zeros((N, 4)); t0 = time.time()
for k in range(5):
    trn = torch.tensor(np.where(folds != k)[0]); val = torch.tensor(np.where(folds == k)[0])
    torch.manual_seed(k); m = M(); opt = torch.optim.Adam(m.parameters(), lr=args.lr)
    for ep in range(args.epochs):
        perm = trn[torch.randperm(len(trn))]
        for b in torch.split(perm, 128):
            logit, _, _ = m(b); loss = torch.nn.functional.cross_entropy(logit, Yt[b]) + 1e-3 * (m.q ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 100 == 99:
            with torch.no_grad():
                lt, _, _ = m(trn); lv, _, _ = m(val)
                print(f"fold {k} ep {ep}: train acc {(lt.argmax(1) == Yt[trn]).double().mean():.3f} val acc {(lv.argmax(1) == Yt[val]).double().mean():.3f} t={time.time() - t0:.0f}s", flush=True)
    with torch.no_grad(): oof[val.numpy()] = m(val)[0].numpy()
print(f"variant={args.variant} OOF acc: {(oof.argmax(1) == Y).mean():.4f}")
