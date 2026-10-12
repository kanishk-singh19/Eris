"""Shared fold-wise feature builder: base physics + symmetry + per-element standardisation + row-relative + nested TE + prototype votes + Wyckoff-letter row features."""
import numpy as np, collections, os, sys
sys.path.insert(0, os.path.dirname(__file__)); import feats, symm

def base_features(env, mask, zc, site, cache=None):
    F = feats.candidate_features(env, mask, zc, site)
    if cache is not None and os.path.exists(cache): S = dict(np.load(cache))
    else:
        S = symm.site_symmetry(env, mask)
        if cache is not None: np.savez(cache, **S)
    F.update(S)
    # Wyckoff-letter row features (no fitting)
    N = len(zc)
    same = np.array([[(site[i] == site[i, c]).sum() for c in range(4)] for i in range(N)], float)
    F['let_shared'] = same; F['let_unique'] = (same == 1).astype(float)
    F['let_rank'] = site.argsort(1).argsort(1).astype(float)
    F['let_is_min'] = (site == site.min(1, keepdims=True)).astype(float); F['let_is_max'] = (site == site.max(1, keepdims=True)).astype(float)
    F['row_let_ndistinct'] = np.repeat(np.array([len(set(site[i])) for i in range(N)], float)[:, None], 4, 1)
    F['row_let_min'] = np.repeat(site.min(1)[:, None], 4, 1).astype(float); F['row_let_max'] = np.repeat(site.max(1)[:, None], 4, 1).astype(float)
    F['let_minus_rowmin'] = (site - site.min(1, keepdims=True)).astype(float)
    r = env[..., 3] * 8
    for R in [3.0, 4.0, 5.0, 6.0]: F[f'nwithin{R}'] = (r <= R).sum(-1).astype(float)
    return F

def proto_parts(env, zc, site):
    r = env[..., 3] * 8; zn = np.rint(env[..., 4] * 118).astype(int); N = len(zc)
    per, grp, blk = feats.period_group(zc.ravel()); blk = blk.reshape(N, 4)
    P0 = []; P1 = []; P3 = []
    for i in range(N):
        a = []; b = []; c3 = []
        for c in range(4):
            sh = r[i, c] <= 1.2 * r[i, c, 0]
            comp = tuple(sorted(collections.Counter([int(np.where(zc[i] == z)[0][0]) if z in zc[i] else -1 for z in zn[i, c][sh]]).values()))
            p = (int(sh.sum()), comp); a.append(p); b.append(p + (int(blk[i, c]),)); c3.append(p + (int(blk[i, c]), int(site[i, c])))
        P0.append(a); P1.append(b); P3.append(c3)
    return [(P0, [tuple(sorted(p)) for p in P0]), (P1, [tuple(sorted(p)) for p in P1]), (P3, [tuple(sorted(p)) for p in P3])]

class FoldFeatures:
    """Everything fitted (element stats, target encodings, prototype votes) uses only rows in `trn`."""
    def __init__(self, F, Ft, zc, zc_t, site, site_t, Y, folds, protos, protos_t):
        self.F, self.Ft, self.zc, self.zc_t, self.site, self.site_t, self.Y, self.folds = F, Ft, zc, zc_t, site, site_t, Y, folds
        self.protos, self.protos_t = protos, protos_t
        self.N = len(Y); self.Nt = len(zc_t)
        self.Ycand = np.zeros((self.N, 4)); self.Ycand[np.arange(self.N), Y] = 1
        cnb = np.clip(F['cn120'], 1, 9).astype(int); cnb_t = np.clip(Ft['cn120'], 1, 9).astype(int)
        inv = F['pg_inv'].astype(int); inv_t = Ft['pg_inv'].astype(int); pg = np.minimum(F['pg_order'], 8).astype(int); pg_t = np.minimum(Ft['pg_order'], 8).astype(int)
        self.K = {'z': zc, 'zs': zc * 100 + site, 'zcn': zc * 100 + cnb, 'zinv': zc * 10 + inv, 'zpg': zc * 100 + pg}
        self.Kt = {'z': zc_t, 'zs': zc_t * 100 + site_t, 'zcn': zc_t * 100 + cnb_t, 'zinv': zc_t * 10 + inv_t, 'zpg': zc_t * 100 + pg_t}
    def te_fit(self, idx, key, m=10):
        k = key[idx].ravel(); a = self.Ycand[idx].ravel(); tot = {}; hit = {}
        for kk, aa in zip(k, a): tot[kk] = tot.get(kk, 0) + 1; hit[kk] = hit.get(kk, 0) + aa
        return {kk: (hit[kk] + 0.25 * m) / (tot[kk] + m) for kk in tot}
    @staticmethod
    def te_apply(mp, key): return np.vectorize(lambda v: mp.get(v, 0.25))(key)
    def te_features(self, trn, with_test):
        out = np.zeros((self.N, 4, len(self.K))); out_t = np.zeros((self.Nt, 4, len(self.K))); trf = self.folds[trn]
        for j, nm in enumerate(self.K):
            full = self.te_fit(trn, self.K[nm]); out[:, :, j] = self.te_apply(full, self.K[nm])
            if with_test: out_t[:, :, j] = self.te_apply(full, self.Kt[nm])
            for fo in np.unique(trf):
                itr = trn[trf != fo]; iva = trn[trf == fo]; out[iva, :, j] = self.te_apply(self.te_fit(itr, self.K[nm]), self.K[nm][iva])
        return out, out_t
    def proto_fit(self, idx, SIG, PP):
        votes = collections.defaultdict(collections.Counter); cnt = collections.Counter()
        for i in idx: votes[SIG[i]][PP[i][self.Y[i]]] += 1; cnt[SIG[i]] += 1
        return votes, cnt
    @staticmethod
    def proto_apply(model, idx, SIG, PP, out):
        votes, cnt = model
        for i in idx:
            v = votes.get(SIG[i]); n = cnt.get(SIG[i], 0)
            for c in range(4):
                ties = sum(1 for cc in range(4) if PP[i][cc] == PP[i][c])
                out[i, c, 0] = (v[PP[i][c]] / ties + 0.25) / (n + 1) if v is not None else 0.25
                out[i, c, 1] = np.log1p(n)
    def proto_features(self, trn, with_test):
        out = np.zeros((self.N, 4, 6)); out_t = np.zeros((self.Nt, 4, 6)); trf = self.folds[trn]
        for j, ((PP, SIG), (PPt, SIGt)) in enumerate(zip(self.protos, self.protos_t)):
            o = np.zeros((self.N, 4, 2)); model = self.proto_fit(trn, SIG, PP); self.proto_apply(model, np.arange(self.N), SIG, PP, o)
            if with_test:
                ot = np.zeros((self.Nt, 4, 2)); self.proto_apply(model, np.arange(self.Nt), SIGt, PPt, ot); out_t[:, :, 2 * j:2 * j + 2] = ot
            for fo in np.unique(trf):
                itr = trn[trf != fo]; iva = trn[trf == fo]; self.proto_apply(self.proto_fit(itr, SIG, PP), iva, SIG, PP, o)
            out[:, :, 2 * j:2 * j + 2] = o
        return out, out_t
    def build(self, trn, with_test=False):
        sub = {k: v[trn] for k, v in self.F.items() if k in feats.ELEM_KEYS}
        stats = feats.element_norm_fit(sub, self.zc[trn], feats.ELEM_KEYS)
        F = dict(self.F); F.update(feats.element_norm_apply(self.F, self.zc, stats)); F = feats.add_row_relative(F)
        names = sorted(F); X = np.stack([F[k] for k in names], -1).astype(np.float32)
        T, Tt = self.te_features(trn, with_test); Pp, Ppt = self.proto_features(trn, with_test)
        X = np.concatenate([X, T, Pp], -1).astype(np.float32)
        names = names + [f'te_{k}' for k in self.K] + [f'proto{j}_{w}' for j in range(3) for w in ('p', 'n')]
        Xt = None
        if with_test:
            Ft = dict(self.Ft); Ft.update(feats.element_norm_apply(self.Ft, self.zc_t, stats)); Ft = feats.add_row_relative(Ft)
            Xt = np.stack([Ft[k] for k in sorted(Ft)], -1).astype(np.float32)
            Xt = np.concatenate([Xt, Tt, Ppt], -1).astype(np.float32)
        return X, Xt, names
