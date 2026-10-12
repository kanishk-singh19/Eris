import numpy as np, pandas as pd, sys, os
sys.path.insert(0, os.path.dirname(__file__)); import feats
D = os.path.join(os.path.dirname(__file__), '..', 'public (12)') + '/'
def load():
    tr = np.load(D + "train_inputs.npz"); te = np.load(D + "test_inputs.npz")
    y = pd.read_csv(D + "train.csv"); f = pd.read_csv(D + "training_folds.csv"); t = pd.read_csv(D + "test.csv")
    folds = y.merge(f, on='id').fold.values; Y = y.target.values
    return dict(env=tr['candidate_environment'], mask=tr['candidate_mask'], zc=tr['candidate_center_z'].astype(int), site=tr['candidate_site_code'].astype(int),
                Y=Y, folds=folds, ids=y.id.values,
                env_t=te['candidate_environment'], mask_t=te['candidate_mask'], zc_t=te['candidate_center_z'].astype(int), site_t=te['candidate_site_code'].astype(int), ids_t=t.id.values)
def tab_features(d, fit_idx, elem=True, rowrel=True):
    """build tabular features; element stats fitted on fit_idx train rows only; returns Xtr (N,4,D), Xte, names"""
    F0 = feats.candidate_features(d['env'], d['mask'], d['zc'], d['site'])
    F0t = feats.candidate_features(d['env_t'], d['mask_t'], d['zc_t'], d['site_t'])
    F = dict(F0); Ft = dict(F0t)
    if elem:
        sub = {k: v[fit_idx] for k, v in F0.items()}
        stats = feats.element_norm_fit(sub, d['zc'][fit_idx], feats.ELEM_KEYS)
        F.update(feats.element_norm_apply(F0, d['zc'], stats)); Ft.update(feats.element_norm_apply(F0t, d['zc_t'], stats))
    if rowrel:
        F = feats.add_row_relative(F); Ft = feats.add_row_relative(Ft)
    names = sorted(F.keys())
    X = np.stack([F[k] for k in names], -1).astype(np.float32); Xt = np.stack([Ft[k] for k in names], -1).astype(np.float32)
    return X, Xt, names
