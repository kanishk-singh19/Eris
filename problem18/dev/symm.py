"""Local site symmetry from the neighbour cloud.
(A) frame-based: count candidate point operations (48 signed permutations + hexagonal ops about z) that map the environment to itself.
(B) frame-free: orbit structure of neighbours under local equivalence (same element, same centre distance, same sorted neighbour-neighbour distance profile)."""
import numpy as np, itertools

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
