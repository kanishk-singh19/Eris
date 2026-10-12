"""SOAP-like power-spectrum features and distortion indices (from-scratch, rotation invariant)."""
import numpy as np
from feats import EN, period_group, legendre

def legendre_all(c):
    P = {0: np.ones_like(c), 1: c, 2: 0.5 * (3 * c ** 2 - 1), 3: 0.5 * (5 * c ** 3 - 3 * c), 4: (35 * c ** 4 - 30 * c ** 2 + 3) / 8}
    return P

def neighbour_channels(zn, zc):
    """boolean channel masks (N,4,48,C)"""
    per, grp, blk = period_group(np.arange(1, 120)); per = np.r_[0, per]; grp = np.r_[0, grp]; blk = np.r_[0, blk]
    z = np.clip(zn, 0, 119)
    g = grp[z]; b = blk[z]
    ch = [z == 1, z == 8, z == 7, np.isin(z, [9, 17, 35, 53]), np.isin(z, [16, 34, 52]),
          (b == 1) & ~np.isin(z, [1, 7, 8, 9, 17, 35, 53, 16, 34, 52]),  # other p-block
          (b == 0) & (z > 1), b == 2, b == 3, zn == zc[..., None], np.ones_like(z, bool)]
    return np.stack(ch, -1).astype(np.float64)

def power_spectrum(env, mask, zc, lmax=4):
    x = env[..., :3].astype(np.float64) * 8; r = env[..., 3].astype(np.float64) * 8
    zn = np.rint(env[..., 4] * 118).astype(int)
    N = x.shape[0]
    u = x / r[..., None]
    C = np.einsum('ncki,ncji->nckj', u, u); C = np.clip(C, -1, 1)
    P = legendre_all(C)  # dict l -> (N,4,48,48)
    rel = r / r[..., :1]
    radial = {'g10': np.exp(-((rel - 1.0) / 0.12) ** 2), 'g13': np.exp(-((rel - 1.3) / 0.15) ** 2), 'g17': np.exp(-((rel - 1.7) / 0.2) ** 2),
              'g23': np.exp(-((rel - 2.3) / 0.3) ** 2), 'r3': 1 / r ** 3, 'r3a': np.exp(-(r - 2.0)) , 'one': np.ones_like(r) / 48}
    ch = neighbour_channels(zn, zc)
    chn = ['H', 'O', 'N', 'hal', 'chal', 'p', 's', 'd', 'f', 'same', 'all']
    F = {}
    for a, cname in enumerate(chn):
        for rn, wr in radial.items():
            if cname != 'all' and rn in ('r3a',): continue
            w = wr * ch[..., a] * mask
            # p_l = w^T P_l w
            for l in range(lmax + 1):
                pl = np.einsum('nck,nckj,ncj->nc', w, P[l], w)
                F[f'ps_{cname}_{rn}_l{l}'] = pl
            p0 = F[f'ps_{cname}_{rn}_l0']
            for l in range(1, lmax + 1):
                F[f'psn_{cname}_{rn}_l{l}'] = F[f'ps_{cname}_{rn}_l{l}'] / np.maximum(p0, 1e-12)
    return F

def distortion(env, mask):
    r = env[..., 3].astype(np.float64) * 8; x = env[..., :3].astype(np.float64) * 8
    rel = r / r[..., :1]
    sh = (rel <= 1.2); n = sh.sum(-1)
    F = {}
    dbar = (r * sh).sum(-1) / n
    F['dist_bondlen'] = (((r - dbar[..., None]) / dbar[..., None]) ** 2 * sh).sum(-1) / n  # bond length distortion
    # angle statistics within first shell
    u = x / r[..., None]
    C = np.clip(np.einsum('ncki,ncji->nckj', u, u), -1, 1)
    ang = np.degrees(np.arccos(C))
    pair = sh[..., :, None] & sh[..., None, :] & ~np.eye(48, dtype=bool)
    npair = np.maximum(pair.sum((-1, -2)), 1)
    amean = (ang * pair).sum((-1, -2)) / npair
    F['ang_mean'] = amean; F['ang_var'] = (((ang - amean[..., None, None]) ** 2) * pair).sum((-1, -2)) / npair
    F['ang_min'] = np.where(pair, ang, 999).min((-1, -2)); F['ang_max'] = np.where(pair, ang, -1).max((-1, -2))
    # deviation from ideal angles for CN 2(180),3(120),4(109.47),6(90/180)
    ideal = {2: [180], 3: [120], 4: [109.47], 5: [90, 120, 180], 6: [90, 180], 8: [70.5, 109.5, 180], 12: [60, 90, 120, 180]}
    dev = np.zeros(n.shape)
    for i in range(n.shape[0]):
        for c in range(4):
            cn = int(n[i, c]); ids = ideal.get(cn)
            if ids is None or cn < 2: dev[i, c] = -1; continue
            a = ang[i, c][pair[i, c]]
            dev[i, c] = np.sqrt(np.mean(np.min(np.abs(a[:, None] - np.array(ids)[None]), 1) ** 2))
    F['ang_ideal_dev'] = dev
    # second-shell relative gap and ratio
    F['shell2_ratio'] = np.where(sh, 0, r).min(-1) / r[..., 0] if False else np.where(~sh, r, 1e3).min(-1) / r[..., 0]
    return F

def build2(env, mask, zc):
    F = power_spectrum(env, mask, zc); F.update(distortion(env, mask)); return F
