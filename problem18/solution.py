"""FieldRelay (problem18) - from-scratch crystal-site EFG anomaly localisation.

Approach (v1): candidate-wise LightGBM ranker over rotation-invariant physics descriptors of each site's
periodic local environment, trained from scratch on the released training rows only.
 - point-charge EFG tensors under several charge models (unit, nuclear charge, electronegativity polarity, first shell, ...)
 - Steinhardt bond-orientational order parameters, coordination-number estimates, bond-valence sums, radial profile
 - local site point-group order from the neighbour cloud, inversion, orbit structure
 - per-element robust standardisation of key descriptors (fitted on training candidates only) -> two-sided |deviation|
 - within-row comparisons (candidate minus row mean / max, within-row rank)
 - smoothed per-element target encodings, nested leave-one-fold-out so training rows never see their own labels
 - 5-seed LightGBM lambdarank over the 4 candidates of each row; argmax per row
Fixed plan: no wall-clock gates, no test-set statistics are used anywhere (every fitted statistic uses train rows only).
Usage: python solution.py <public_dataset_dir> <submission_csv_path>
"""
"""Candidate-wise feature extraction.
All features are per-candidate functions of the released local environment plus
standard physical constants (periodic table position, Pauling EN, covalent radii).
Row-relative features compare the four candidates of the same crystal.
Per-element robust normalisation statistics are fitted on TRAIN candidates only."""
import numpy as np

# ---- standard physical constants (Z-indexed, index 0 unused) -----------------
PAULING_EN = [0, 2.20, 0, 0.98, 1.57, 2.04, 2.55, 3.04, 3.44, 3.98, 0, 0.93, 1.31, 1.61, 1.90, 2.19, 2.58,
    3.16, 0, 0.82, 1.00, 1.36, 1.54, 1.63, 1.66, 1.55, 1.83, 1.88, 1.91, 1.90, 1.65, 1.81, 2.01, 2.18, 2.55,
    2.96, 3.00, 0.82, 0.95, 1.22, 1.33, 1.60, 2.16, 1.90, 2.20, 2.28, 2.20, 1.93, 1.69, 1.78, 1.96, 2.05, 2.10,
    2.66, 2.60, 0.79, 0.89, 1.10, 1.12, 1.13, 1.14, 1.13, 1.17, 1.20, 1.20, 1.10, 1.22, 1.23, 1.24, 1.25, 1.10,
    1.27, 1.30, 1.50, 2.36, 1.90, 2.20, 2.20, 2.28, 2.54, 2.00, 1.62, 2.33, 2.02, 2.00, 2.20, 2.20, 0.70, 0.90,
    1.10, 1.30, 1.50, 1.38, 1.36, 1.28]
COV_RAD = [0, 0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58, 1.66, 1.41, 1.21, 1.11, 1.07, 1.05,
    1.02, 1.06, 2.03, 1.76, 1.70, 1.60, 1.53, 1.39, 1.39, 1.32, 1.26, 1.24, 1.32, 1.22, 1.22, 1.20, 1.19, 1.20,
    1.20, 1.16, 2.20, 1.95, 1.90, 1.75, 1.64, 1.54, 1.47, 1.46, 1.42, 1.39, 1.45, 1.44, 1.42, 1.39, 1.39, 1.38,
    1.39, 1.40, 2.44, 2.15, 2.07, 2.04, 2.03, 2.01, 1.99, 1.98, 1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87,
    1.87, 1.75, 1.70, 1.62, 1.51, 1.44, 1.41, 1.36, 1.36, 1.32, 1.45, 1.46, 1.48, 1.40, 1.50, 1.50, 2.60, 2.21,
    2.15, 2.06, 2.00, 1.96, 1.90, 1.87]
EN = np.array(PAULING_EN + [0.0] * (119 - len(PAULING_EN)), dtype=np.float32)
RC = np.array(COV_RAD + [1.5] * (119 - len(COV_RAD)), dtype=np.float32)
_PERIOD_START = [1, 3, 11, 19, 37, 55, 87, 119]

def period_group(z):
    z = np.asarray(z).astype(int)
    period = np.searchsorted(np.array(_PERIOD_START), z, side='right')
    start = np.array(_PERIOD_START)[period - 1]
    off = z - start  # 0-based offset in period
    group = np.zeros_like(z)
    blk = np.zeros_like(z)  # 0 s,1 p,2 d,3 f
    for i, (p, o, zz) in enumerate(zip(period, off, z)):
        if p == 1:
            g = 1 if o == 0 else 18; b = 0
        elif p in (2, 3):
            g = o + 1 if o < 2 else o + 11; b = 0 if o < 2 else 1
        elif p in (4, 5):
            g = o + 1; b = 0 if o < 2 else (2 if o < 12 else 1)
        else:
            if o < 2: g = o + 1; b = 0
            elif o < 17: g = 3; b = 3  # lanthanide/actinide
            else: g = o - 14 + 1; b = 2 if g <= 12 else 1
        group[i] = g; blk[i] = b
    return period, group, blk

def legendre(l, x):
    if l == 2: return 0.5 * (3 * x ** 2 - 1)
    if l == 4: return (35 * x ** 4 - 30 * x ** 2 + 3) / 8
    if l == 6: return (231 * x ** 6 - 315 * x ** 4 + 105 * x ** 2 - 5) / 16
    raise ValueError

def steinhardt(u, w, l):
    """u: (...,K,3) unit vectors, w: (...,K) weights (0/1 mask or soft). returns q_l (...)"""
    c = np.einsum('...ki,...ji->...kj', u, u)
    c = np.clip(c, -1, 1)
    P = legendre(l, c)
    ww = w[..., :, None] * w[..., None, :]
    s = (P * ww).sum((-1, -2))
    n = w.sum(-1)
    return np.sqrt(np.maximum(s, 0)) / np.maximum(n, 1e-9)

def efg_tensor(x, r, w):
    """point-charge EFG: V = sum_k w_k (3 x x^T - r^2 I)/r^5 ; x (...,K,3), r (...,K), w (...,K)"""
    r5 = r ** 5
    outer = x[..., :, None] * x[..., None, :]  # (...,K,3,3)
    I = np.eye(3, dtype=x.dtype)
    T = (3 * outer - (r ** 2)[..., None, None] * I) / r5[..., None, None]
    V = (T * w[..., None, None]).sum(-3)
    return V

def tensor_feats(V, prefix, out):
    ev = np.linalg.eigvalsh(V)  # ascending
    idx = np.argsort(np.abs(ev), axis=-1)
    evs = np.take_along_axis(ev, idx, axis=-1)  # by |.| ascending
    vzz = evs[..., 2]
    eta = (evs[..., 0] - evs[..., 1]) / np.where(np.abs(vzz) > 1e-12, vzz, 1e-12)
    fro = np.sqrt((V ** 2).sum((-1, -2)))
    out[prefix + '_fro'] = fro
    out[prefix + '_logfro'] = np.log1p(fro)
    out[prefix + '_vzz'] = vzz
    out[prefix + '_absvzz'] = np.abs(vzz)
    out[prefix + '_eta'] = np.clip(np.abs(eta), 0, 1)
    out[prefix + '_ev0'] = ev[..., 0]; out[prefix + '_ev2'] = ev[..., 2]
    return fro

def candidate_features(env, mask, zc, site):
    """env (N,4,48,5), mask (N,4,48), zc (N,4), site (N,4) -> dict of (N,4) arrays"""
    N = env.shape[0]
    x = env[..., :3].astype(np.float64) * 8.0  # Å
    r = env[..., 3].astype(np.float64) * 8.0
    zn = np.rint(env[..., 4].astype(np.float64) * 118).astype(int)
    m = mask.astype(np.float64)
    r_safe = np.where(m > 0, r, 1e3)
    zc = zc.astype(int)
    F = {}
    # --- element descriptors
    per, grp, blk = period_group(zc.ravel())
    F['zc'] = zc.astype(float); F['period'] = per.reshape(N, 4).astype(float)
    F['group'] = grp.reshape(N, 4).astype(float); F['block'] = blk.reshape(N, 4).astype(float)
    F['site'] = site.astype(float)
    en_c = EN[zc]; rc_c = RC[zc]
    F['en_c'] = en_c; F['rc_c'] = rc_c
    en_n = EN[np.clip(zn, 0, 118)]; rc_n = RC[np.clip(zn, 0, 118)]
    # --- radial distribution
    d1 = r_safe[..., 0]
    F['d1'] = d1
    for k in range(1, 16):
        F[f'd{k+1}'] = r_safe[..., k]
    for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 19, 23, 29, 35, 47]:
        F[f'rr{k+1}'] = r_safe[..., k] / d1
    F['dmean8'] = r_safe[..., :8].mean(-1); F['dmean48'] = r_safe.mean(-1)
    F['d48'] = r_safe[..., 47]
    # --- coordination number estimates
    rel = r_safe / d1[..., None]
    for t in (1.1, 1.2, 1.3, 1.45):
        F[f'cn{int(t*100)}'] = (rel <= t).sum(-1).astype(float)
    F['cn_abs05'] = (r_safe <= d1[..., None] + 0.5).sum(-1).astype(float)
    # gap based CN within first 12
    ratios = r_safe[..., 1:13] / r_safe[..., :12]
    gap_idx = ratios.argmax(-1)
    F['cn_gap'] = (gap_idx + 1).astype(float); F['gap_max'] = ratios.max(-1)
    F['gap1'] = ratios[..., 0]
    # smooth CN (O'Keeffe-like) with Gaussian in relative distance
    F['cn_soft'] = np.exp(-((rel - 1) / 0.12) ** 2).sum(-1)
    F['cn_soft2'] = (1 / (1 + np.exp((rel - 1.25) / 0.05))).sum(-1)
    # first shell mask (1.2*d1) and 12-nearest and 6-nearest
    sh = (rel <= 1.2).astype(float)
    sh_soft = 1 / (1 + np.exp((rel - 1.25) / 0.05))
    nsh = sh.sum(-1)
    # --- neighbour chemistry of first shell
    def wmean(v, w):
        return (v * w).sum(-1) / np.maximum(w.sum(-1), 1e-9)
    F['sh_zn_mean'] = wmean(zn, sh); F['sh_zn_max'] = np.where(sh > 0, zn, -1).max(-1)
    F['sh_zn_min'] = np.where(sh > 0, zn, 999).min(-1)
    F['sh_en_mean'] = wmean(en_n, sh); F['sh_en_diff'] = F['sh_en_mean'] - en_c
    F['sh_en_std'] = np.sqrt(np.maximum(wmean(en_n ** 2, sh) - F['sh_en_mean'] ** 2, 0))
    F['sh_same_el'] = wmean((zn == zc[..., None]).astype(float), sh)
    F['sh_ndistinct_z'] = np.array([[len(set(zn[i, c][sh[i, c] > 0])) for c in range(4)] for i in range(N)], float)
    F['n12_same_el'] = (zn[..., :12] == zc[..., None]).mean(-1)
    F['n48_same_el'] = (zn == zc[..., None]).mean(-1)
    F['n48_en_mean'] = en_n.mean(-1); F['n48_zn_mean'] = zn.mean(-1)
    F['n12_en_mean'] = en_n[..., :12].mean(-1)
    # anion-like / cation-like indicator: EN of center vs neighbours
    F['en_polar'] = wmean(np.sign(en_n - en_c[..., None]), sh)
    # bond valence sum proxy
    R0 = rc_c[..., None] + rc_n
    F['bvs'] = (np.exp((R0 - r_safe) / 0.37) * m).sum(-1)
    F['bvs_sh'] = (np.exp((R0 - r_safe) / 0.37) * sh).sum(-1)
    F['d1_over_R0'] = d1 / R0[..., 0]
    F['sh_d_over_R0_mean'] = wmean(r_safe / R0, sh)
    # --- geometry: unit vectors, Steinhardt on first shell / 6 / 12
    u = x / r_safe[..., None]
    for l in (2, 4, 6):
        F[f'q{l}_sh'] = steinhardt(u, sh, l)
        F[f'q{l}_soft'] = steinhardt(u, sh_soft, l)
        F[f'q{l}_n6'] = steinhardt(u[..., :6, :], np.ones_like(r_safe[..., :6]), l)
        F[f'q{l}_n12'] = steinhardt(u[..., :12, :], np.ones_like(r_safe[..., :12]), l)
    # centroid offset of first shell (polarity)
    cen = (u * sh[..., None]).sum(-2) / np.maximum(nsh, 1)[..., None]
    F['sh_centroid'] = np.linalg.norm(cen, axis=-1)
    cen_r = (x * sh[..., None]).sum(-2) / np.maximum(nsh, 1)[..., None]
    F['sh_centroid_A'] = np.linalg.norm(cen_r, axis=-1)
    F['dip_r2'] = np.linalg.norm((u / r_safe[..., None] ** 2 * m[..., None]).sum(-2), axis=-1)
    F['dip_r2_en'] = np.linalg.norm((u / r_safe[..., None] ** 2 * ((en_n - en_c[..., None]) * m)[..., None]).sum(-2), axis=-1)
    # inertia tensor of first shell (unit vectors): anisotropy
    Iu = np.einsum('...k,...ki,...kj->...ij', sh, u, u) / np.maximum(nsh, 1)[..., None, None]
    evI = np.linalg.eigvalsh(Iu)
    F['sh_aniso'] = evI[..., 2] - evI[..., 0]; F['sh_aniso2'] = evI[..., 1] - evI[..., 0]
    F['sh_ev0'] = evI[..., 0]; F['sh_ev2'] = evI[..., 2]
    # --- point-charge EFG tensors under several charge models
    models = {
        'u': m,                                   # unit charges, all 48
        'z': m * zn / 118.0,                      # nuclear charge
        'e': m * (en_n - en_c[..., None]),        # ionic polarity proxy
        'sh': sh,                                 # first shell only, unit
        'she': sh * (en_n - en_c[..., None]),
        'exp': m * np.exp(-(rel - 1) / 0.5),      # smooth emphasis on near shell
        'n12': m * (np.arange(48) < 12),
    }
    norm_u = (m / r_safe ** 3).sum(-1)
    for name, w in models.items():
        V = efg_tensor(x, r_safe, w)
        fro = tensor_feats(V, 'efg_' + name, F)
        den = (np.abs(w) / r_safe ** 3).sum(-1)
        F[f'efg_{name}_shape'] = fro / np.maximum(den, 1e-9)
    F['sum_r3'] = norm_u; F['sum_r3_sh'] = (sh / r_safe ** 3).sum(-1)
    F['sum_zr3'] = (m * zn / r_safe ** 3).sum(-1)
    # --- symmetry heuristics
    rr = np.round(r_safe[..., :12], 2)
    nd = np.zeros((N, 4)); mx = np.zeros((N, 4)); ndsh = np.zeros((N, 4))
    for i in range(N):
        for c in range(4):
            vals, cnt = np.unique(rr[i, c], return_counts=True)
            nd[i, c] = len(vals); mx[i, c] = cnt.max()
            ndsh[i, c] = len(np.unique(np.round(r_safe[i, c][sh[i, c] > 0], 2)))
    F['ndist12'] = nd; F['maxmult12'] = mx; F['ndist_sh'] = ndsh
    F['sh_dspread'] = (np.where(sh > 0, r_safe, 0).max(-1) - d1) / d1
    # inversion partner fraction (first 12 neighbours): exists neighbour at -x within 0.05 Å
    xs = x[..., :12, :]
    diff = xs[..., :, None, :] + x[..., None, :, :]  # (N,4,12,48,3)
    dd = np.linalg.norm(diff, axis=-1)
    has_inv = (dd < 0.05).any(-1).astype(float)
    F['inv_frac12'] = has_inv.mean(-1)
    F['inv_frac_sh'] = (has_inv * sh[..., :12]).sum(-1) / np.maximum(sh[..., :12].sum(-1), 1)
    # mirror partners w.r.t. the three crystal-frame axes (cheap proxy for mirror symmetry)
    for ax in range(3):
        xm = xs.copy(); xm[..., ax] *= -1
        dd = np.linalg.norm(xm[..., :, None, :] - x[..., None, :, :], axis=-1)
        F[f'mir{ax}_frac12'] = (dd < 0.05).any(-1).mean(-1)
    F['mir_frac12_mean'] = (F['mir0_frac12'] + F['mir1_frac12'] + F['mir2_frac12']) / 3
    # --- row-level context
    F['row_n_el'] = np.repeat(np.array([len(set(zc[i])) for i in range(N)], float)[:, None], 4, 1)
    F['row_same_z_count'] = np.array([[(zc[i] == zc[i, c]).sum() for c in range(4)] for i in range(N)], float)
    F['row_zc_rank'] = zc.argsort(1).argsort(1).astype(float)
    F['row_site_rank'] = site.argsort(1).argsort(1).astype(float)
    F['row_en_max'] = np.repeat(en_c.max(1)[:, None], 4, 1); F['row_en_min'] = np.repeat(en_c.min(1)[:, None], 4, 1)
    F['is_max_en'] = (en_c == en_c.max(1, keepdims=True)).astype(float)
    F['is_min_en'] = (en_c == en_c.min(1, keepdims=True)).astype(float)
    return F

ROW_REL_KEYS = None

def add_row_relative(F, keys=None):
    """add candidate - row_mean and within-row rank for selected keys"""
    out = dict(F)
    if keys is None:
        keys = [k for k in F if not k.startswith('row_') and k not in ('zc', 'period', 'group', 'block')]
    for k in keys:
        v = F[k]
        out[k + '_rd'] = v - v.mean(1, keepdims=True)
        out[k + '_rk'] = v.argsort(1).argsort(1).astype(float)
        out[k + '_rmax'] = v - v.max(1, keepdims=True)
    return out

def element_norm_fit(F, zc, keys, min_n=8):
    """robust per-element (median, MAD) stats fitted on train candidates only"""
    stats = {}
    for k in keys:
        v = F[k].ravel(); z = zc.ravel()
        gmed = np.median(v); gmad = np.median(np.abs(v - gmed)) + 1e-6
        d = {}
        for el in np.unique(z):
            vv = v[z == el]
            n = len(vv)
            med = np.median(vv); mad = np.median(np.abs(vv - med)) + 1e-6
            lam = n / (n + min_n)  # shrink toward global
            d[int(el)] = (lam * med + (1 - lam) * gmed, lam * mad + (1 - lam) * gmad)
        stats[k] = (d, (gmed, gmad))
    return stats

def element_norm_apply(F, zc, stats):
    out = {}
    for k, (d, (gmed, gmad)) in stats.items():
        v = F[k]; z = zc
        med = np.vectorize(lambda e: d.get(int(e), (gmed, gmad))[0])(z)
        mad = np.vectorize(lambda e: d.get(int(e), (gmed, gmad))[1])(z)
        out[k + '_ez'] = (v - med) / mad
        out[k + '_eabs'] = np.abs(v - med) / mad
    return out

ELEM_KEYS = ['efg_u_logfro', 'efg_z_logfro', 'efg_e_logfro', 'efg_sh_logfro', 'efg_exp_logfro', 'efg_n12_logfro',
             'efg_u_shape', 'efg_sh_shape', 'efg_e_shape', 'd1', 'cn120', 'cn_soft', 'q2_sh', 'q4_sh', 'q6_sh',
             'bvs', 'bvs_sh', 'sh_en_diff', 'sh_centroid', 'sum_r3', 'sh_aniso', 'd1_over_R0', 'site']

def build(env, mask, zc, site, stats=None, fit=False):
    F = candidate_features(env, mask, zc, site)
    if fit:
        stats = element_norm_fit(F, zc, ELEM_KEYS)
    if stats is not None:
        F.update(element_norm_apply(F, zc, stats))
    F = add_row_relative(F)
    names = sorted(F.keys())
    X = np.stack([F[k] for k in names], -1)  # (N,4,D)
    return X.astype(np.float32), names, stats


"""Local site symmetry from the neighbour cloud.
(A) frame-based: count candidate point operations (48 signed permutations + hexagonal ops about z) that map the environment to itself.
(B) frame-free: orbit structure of neighbours under local equivalence (same element, same centre distance, same sorted neighbour-neighbour distance profile)."""
import itertools

def op_set():
    ops = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            M = np.zeros((3, 3)); 
            for i, p in enumerate(perm): M[i, p] = signs[i]
            ops.append(M)
    for k in range(1, 6):  # hexagonal rotations about z (60 deg multiples), with and without z-mirror, plus vertical mirrors
        a = np.pi / 3 * k; c, s = np.cos(a), np.sin(a)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        for mz in (1, -1):
            Mz = np.diag([1, 1, mz]); ops.append(R @ Mz)
            ops.append(R @ np.diag([1, -1, 1]) @ Mz)  # vertical mirror composed with rotation
    ops.append(np.diag([1, -1, 1])); ops.append(np.diag([1, -1, -1])); ops.append(np.diag([1, 1, -1]) @ np.diag([1, -1, 1]))
    # dedupe
    uniq = []
    for M in ops:
        if not any(np.allclose(M, U, atol=1e-6) for U in uniq): uniq.append(M)
    return np.stack(uniq)
OPS = op_set()

def frame_ops(x, r, zn, tol=0.03):
    """x (48,3) Å, r (48,), zn (48,) -> number of ops in OPS mapping complete shells onto themselves, and flags"""
    rmax = r[-1] - 0.02
    m = r < rmax  # complete shells only
    X = x[m]; Z = zn[m]
    if len(X) < 2: return 1, 0, 1
    key = Z * 1000.0  # element offset for matching
    found = []
    for M in OPS:
        Xp = X @ M.T
        d = np.linalg.norm(Xp[:, None, :] - X[None, :, :], axis=-1) + np.abs(key[:, None] - key[None, :])
        if (d.min(1) < tol).all(): found.append(M)
    n = len(found)
    has_inv = any(np.allclose(M, -np.eye(3), atol=1e-6) for M in found)
    # max rotation order among proper rotations
    maxord = 1
    for M in found:
        if np.linalg.det(M) > 0:
            tr = np.trace(M); ang = np.arccos(np.clip((tr - 1) / 2, -1, 1))
            if ang > 1e-3:
                order = int(round(2 * np.pi / ang))
                maxord = max(maxord, order)
    return n, int(has_inv), maxord

def orbits(x, r, zn, rc=4.5, tol=0.02):
    """frame-free: group neighbours (within rc) into equivalence classes by (Z, r, sorted dist profile to other neighbours within rc)."""
    m = r < rc
    X = x[m]; Z = zn[m]; R = r[m]
    if len(X) == 0: return 1, 1, 1.0
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    keys = []
    for j in range(len(X)):
        prof = np.sort(np.round(D[j] / tol).astype(int) * 1000 + Z)
        keys.append((int(Z[j]), int(round(R[j] / tol)), tuple(prof)))
    uniq = {}
    for k in keys: uniq[k] = uniq.get(k, 0) + 1
    sizes = np.array(list(uniq.values()))
    return len(uniq), sizes.max(), len(X) / len(uniq)

def site_symmetry(env, mask):
    x = env[..., :3].astype(np.float64) * 8; r = env[..., 3].astype(np.float64) * 8
    zn = np.rint(env[..., 4] * 118).astype(int)
    N = x.shape[0]
    out = {k: np.zeros((N, 4)) for k in ['pg_order', 'pg_inv', 'pg_maxrot', 'orb_n', 'orb_max', 'orb_ratio']}
    for i in range(N):
        for c in range(4):
            n, inv, mo = frame_ops(x[i, c], r[i, c], zn[i, c])
            out['pg_order'][i, c] = n; out['pg_inv'][i, c] = inv; out['pg_maxrot'][i, c] = mo
            a, b, q = orbits(x[i, c], r[i, c], zn[i, c])
            out['orb_n'][i, c] = a; out['orb_max'][i, c] = b; out['orb_ratio'][i, c] = q
    return out


# =============================================================================================
# Main pipeline
# =============================================================================================
import sys, os, time
import pandas as pd
import lightgbm as lgb

def main():
    t0 = time.time()
    data_dir, out_path = sys.argv[1], sys.argv[2]
    P = lambda f: os.path.join(data_dir, f)
    tr = np.load(P('train_inputs.npz')); te = np.load(P('test_inputs.npz'))
    y = pd.read_csv(P('train.csv')); f = pd.read_csv(P('training_folds.csv')); t = pd.read_csv(P('test.csv'))
    folds = y.merge(f, on='id', how='left').fold.values; Y = y.target.values.astype(int); N = len(Y)
    env, mask, zc, site = tr['candidate_environment'], tr['candidate_mask'], tr['candidate_center_z'].astype(int), tr['candidate_site_code'].astype(int)
    env_t, mask_t, zc_t, site_t = te['candidate_environment'], te['candidate_mask'], te['candidate_center_z'].astype(int), te['candidate_site_code'].astype(int)
    Nt = len(t)
    print(f'train {N} rows, test {Nt} rows', flush=True)

    # ---- candidate-wise physics features (deterministic functions of each candidate's own environment)
    F0 = candidate_features(env, mask, zc, site); F0t = candidate_features(env_t, mask_t, zc_t, site_t)
    S = site_symmetry(env, mask); St = site_symmetry(env_t, mask_t)
    F0.update(S); F0t.update(St)
    print(f'features built: {len(F0)} base features, t={time.time()-t0:.0f}s', flush=True)
    cnb = np.clip(F0['cn120'], 1, 9).astype(int); cnb_t = np.clip(F0t['cn120'], 1, 9).astype(int)
    Ycand = np.zeros((N, 4)); Ycand[np.arange(N), Y] = 1

    # ---- smoothed per-element target encodings (fitted on training rows only; nested leave-one-fold-out for training rows)
    def te_keys(zc_, site_, cnb_, S_):
        return {'z': zc_, 'zs': zc_ * 100 + site_, 'zcn': zc_ * 100 + cnb_, 'zinv': zc_ * 10 + S_['pg_inv'].astype(int),
                'zpg': zc_ * 100 + np.minimum(S_['pg_order'], 8).astype(int)}
    K_tr = te_keys(zc, site, cnb, S); K_te = te_keys(zc_t, site_t, cnb_t, St)
    def te_fit(idx, key):
        k = key[idx].ravel(); a = Ycand[idx].ravel(); tot = {}; hit = {}
        for kk, aa in zip(k, a): tot[kk] = tot.get(kk, 0) + 1; hit[kk] = hit.get(kk, 0) + aa
        return {kk: (hit[kk] + 0.25 * 10) / (tot[kk] + 10) for kk in tot}
    def te_apply(mp, key): return np.vectorize(lambda v: mp.get(v, 0.25))(key)
    def te_features(trn, test_keys=None):
        """training rows: encoded with the other training folds only; validation/test rows: encoded with all training rows"""
        out = np.zeros((N, 4, len(K_tr))); out_t = None if test_keys is None else np.zeros((Nt, 4, len(K_tr)))
        trf = folds[trn]
        for j, nm in enumerate(K_tr):
            full = te_fit(trn, K_tr[nm]); out[:, :, j] = te_apply(full, K_tr[nm])
            if test_keys is not None: out_t[:, :, j] = te_apply(full, test_keys[nm])
            for fo in np.unique(trf):
                inner_tr = trn[trf != fo]; inner_va = trn[trf == fo]
                out[inner_va, :, j] = te_apply(te_fit(inner_tr, K_tr[nm]), K_tr[nm][inner_va])
        return out, out_t

    def build(trn, with_test):
        sub = {k: v[trn] for k, v in F0.items() if k in ELEM_KEYS}
        stats = element_norm_fit(sub, zc[trn], ELEM_KEYS)           # per-element robust stats from training rows only
        F = dict(F0); F.update(element_norm_apply(F0, zc, stats)); F = add_row_relative(F)
        names = sorted(F); X = np.stack([F[k] for k in names], -1).astype(np.float32)
        Xt = None
        if with_test:
            Ft = dict(F0t); Ft.update(element_norm_apply(F0t, zc_t, stats)); Ft = add_row_relative(Ft)
            Xt = np.stack([Ft[k] for k in names], -1).astype(np.float32)
        T, Tt = te_features(trn, K_te if with_test else None)
        X = np.concatenate([X, T], -1).astype(np.float32)
        if with_test: Xt = np.concatenate([Xt, Tt], -1).astype(np.float32)
        return X, Xt

    PARAMS = dict(objective='lambdarank', verbose=-1, learning_rate=0.03, num_leaves=15, min_data_in_leaf=20, feature_fraction=0.3,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=4, deterministic=True, force_row_wise=True)
    ROUNDS = 300; SEEDS = [0, 1, 2, 3, 4]
    def fit_predict(X, trn, Xq_list):
        outs = [np.zeros((len(q), 4)) for q in Xq_list]
        for s in SEEDS:
            ds = lgb.Dataset(X[trn].reshape(-1, X.shape[-1]), Ycand[trn].ravel(), group=[4] * len(trn))
            m = lgb.train(dict(PARAMS, seed=s), ds, ROUNDS)
            for o, q in zip(outs, Xq_list): o += m.predict(q.reshape(-1, q.shape[-1])).reshape(-1, 4) / len(SEEDS)
        return outs

    # ---- grouped cross-validation with the supplied chemical-system folds (reporting only; does not alter the plan)
    oof = np.zeros((N, 4))
    for k in range(5):
        trn = np.where(folds != k)[0]; val = np.where(folds == k)[0]
        X, _ = build(trn, False)
        oof[val] = fit_predict(X, trn, [X[val]])[0]
        print(f'fold {k}: acc {(oof[val].argmax(1) == Y[val]).mean():.4f}  t={time.time()-t0:.0f}s', flush=True)
    print(f'CV accuracy (grouped 5-fold): {(oof.argmax(1) == Y).mean():.4f}', flush=True)

    # ---- final model on all training rows -> test predictions (one deterministic inference pass)
    trn = np.arange(N)
    X, Xt = build(trn, True)
    pt = fit_predict(X, trn, [Xt])[0]
    pred = pt.argmax(1).astype(int)
    sub = pd.DataFrame({'id': t.id.values, 'prediction': pred})
    sub.to_csv(out_path, index=False)
    print('prediction distribution:', np.bincount(pred, minlength=4).tolist())
    print(f'wrote {out_path}  total time {time.time()-t0:.0f}s')

if __name__ == '__main__':
    main()
