"""Assemble a self-contained solution.py (v2 ensemble) from the dev modules. usage: python dev/make_solution.py <out_path> <models comma list> <seeds>"""
import sys, re
out_path, models, seeds = sys.argv[1], sys.argv[2], int(sys.argv[3])
feats = open('dev/feats.py').read()
symm = open('dev/symm.py').read().replace('import numpy as np, itertools', 'import itertools')
sf = open('dev/stack_feats.py').read()
sf = sf.split('\n', 1)[1]  # drop docstring line
sf = sf.replace("import numpy as np, collections, os, sys\nsys.path.insert(0, os.path.dirname(__file__)); import feats, symm\n", "import collections\n")
sf = sf.replace("feats.", "").replace("symm.", "")
sf = re.sub(r"    if cache is not None and os.path.exists\(cache\): S = dict\(np.load\(cache\)\)\n    else:\n        S = site_symmetry\(env, mask\)\n        if cache is not None: np.savez\(cache, \*\*S\)\n", "    S = site_symmetry(env, mask)\n", sf)
main = r'''

# =============================================================================================
# Models and main pipeline
# =============================================================================================
import sys, os, time
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

MODELS = [%s]
SEEDS = list(range(%d))
NTHREADS = 4

def lgb_fit(obj, nl=15, md=20, ff_=0.3, lr=0.03, rounds=300):
    def fit(Xtr, ytr, Xq, s):
        p = dict(objective=obj, verbose=-1, learning_rate=lr, num_leaves=nl, min_data_in_leaf=md, feature_fraction=ff_, bagging_fraction=0.8, bagging_freq=1,
                 lambda_l2=1.0, num_threads=NTHREADS, seed=s, deterministic=True, force_row_wise=True)
        ds = lgb.Dataset(Xtr, ytr, group=[4] * (len(ytr) // 4)) if obj == 'lambdarank' else lgb.Dataset(Xtr, ytr)
        m = lgb.train(p, ds, rounds); return [m.predict(q) for q in Xq]
    return fit
def xgb_fit(depth=6, eta=0.03, rounds=400, cs=0.3):
    def fit(Xtr, ytr, Xq, s):
        p = dict(objective='rank:pairwise', max_depth=depth, eta=eta, subsample=0.8, colsample_bytree=cs, min_child_weight=5, reg_lambda=1.0, nthread=NTHREADS, seed=s, verbosity=0)
        dtr = xgb.DMatrix(Xtr, ytr); dtr.set_group([4] * (len(ytr) // 4)); m = xgb.train(p, dtr, rounds); return [m.predict(xgb.DMatrix(q)) for q in Xq]
    return fit
def cat_fit(iters=600, lr=0.05, depth=6):
    def fit(Xtr, ytr, Xq, s):
        from catboost import CatBoostRanker, Pool
        m = CatBoostRanker(loss_function='YetiRank', iterations=iters, learning_rate=lr, depth=depth, verbose=0, random_seed=s, thread_count=NTHREADS)
        m.fit(Pool(Xtr, ytr, group_id=np.repeat(np.arange(len(ytr) // 4), 4)))
        return [m.predict(Pool(q, group_id=np.repeat(np.arange(len(q) // 4), 4))) for q in Xq]
    return fit
FITTERS = {'lgb_rank': lgb_fit('lambdarank'), 'lgb_bin': lgb_fit('binary'), 'lgb_rank_deep': lgb_fit('lambdarank', nl=31, md=10, rounds=250),
           'xgb6': xgb_fit(6, 0.03, 400, 0.3), 'xgb4': xgb_fit(4, 0.05, 300, 0.3), 'xgb8': xgb_fit(8, 0.02, 400, 0.3), 'cat': cat_fit()}

def zscore(a):
    a = a - a.mean(1, keepdims=True); return a / (a.std() + 1e-9)

def main():
    t0 = time.time()
    data_dir, out_path = sys.argv[1], sys.argv[2]
    P = lambda f: os.path.join(data_dir, f)
    tr = np.load(P('train_inputs.npz')); te = np.load(P('test_inputs.npz'))
    y = pd.read_csv(P('train.csv')); f = pd.read_csv(P('training_folds.csv')); t = pd.read_csv(P('test.csv'))
    folds = y.merge(f, on='id', how='left').fold.values; Y = y.target.values.astype(int); N = len(Y)
    env, mask, zc, site = tr['candidate_environment'], tr['candidate_mask'], tr['candidate_center_z'].astype(int), tr['candidate_site_code'].astype(int)
    env_t, mask_t, zc_t, site_t = te['candidate_environment'], te['candidate_mask'], te['candidate_center_z'].astype(int), te['candidate_site_code'].astype(int)
    print(f'train {N} rows, test {len(t)} rows', flush=True)
    F = base_features(env, mask, zc, site); Ft = base_features(env_t, mask_t, zc_t, site_t)
    ff = FoldFeatures(F, Ft, zc, zc_t, site, site_t, Y, folds, proto_parts(env, zc, site), proto_parts(env_t, zc_t, site_t))
    Ycand = ff.Ycand
    print(f'base features built: {len(F)}  t={time.time()-t0:.0f}s', flush=True)

    def run_models(X, trn, Xq_list):
        """returns per-model averaged-over-seeds predictions for each query set"""
        outs = {name: [np.zeros((len(q), 4)) for q in Xq_list] for name in MODELS}
        Xtr = X[trn].reshape(-1, X.shape[-1]); ytr = Ycand[trn].ravel(); Xq = [q.reshape(-1, q.shape[-1]) for q in Xq_list]
        for name in MODELS:
            for s in SEEDS:
                preds = FITTERS[name](Xtr, ytr, Xq, s)
                for o, p in zip(outs[name], preds): o += p.reshape(-1, 4) / len(SEEDS)
        return outs
    def blend(outs, qi):
        return sum(zscore(outs[name][qi]) for name in MODELS)

    # ---- grouped cross-validation on the supplied chemical-system folds (reporting only)
    oof = {name: np.zeros((N, 4)) for name in MODELS}; oof_blend = np.zeros((N, 4))
    for k in range(5):
        trn = np.where(folds != k)[0]; val = np.where(folds == k)[0]
        X, _, _ = ff.build(trn, False)
        outs = run_models(X, trn, [X[val]])
        for name in MODELS: oof[name][val] = outs[name][0]
        oof_blend[val] = blend(outs, 0)
        print(f'fold {k}: blend acc {(oof_blend[val].argmax(1) == Y[val]).mean():.4f}  ' + ' '.join(f'{n}={(oof[n][val].argmax(1) == Y[val]).mean():.3f}' for n in MODELS) + f'  t={time.time()-t0:.0f}s', flush=True)
    print('CV accuracy per model: ' + ' '.join(f'{n}={(oof[n].argmax(1) == Y).mean():.4f}' for n in MODELS))
    print(f'CV accuracy (grouped 5-fold, blend): {(oof_blend.argmax(1) == Y).mean():.4f}', flush=True)

    # ---- final fit on all training rows, one deterministic inference pass on test
    trn = np.arange(N)
    X, Xt, _ = ff.build(trn, True)
    outs = run_models(X, trn, [Xt])
    pt = blend(outs, 0); pred = pt.argmax(1).astype(int)
    pd.DataFrame({'id': t.id.values, 'prediction': pred}).to_csv(out_path, index=False)
    print('prediction distribution:', np.bincount(pred, minlength=4).tolist())
    print(f'wrote {out_path}  total time {time.time()-t0:.0f}s')

if __name__ == '__main__':
    main()
''' % (', '.join(repr(m) for m in models.split(',')), seeds)
header = '''"""FieldRelay (problem18) - from-scratch crystal-site EFG anomaly localisation.

Approach (v2): ensemble of candidate-wise gradient-boosted rankers over rotation-invariant physics descriptors
of each site's periodic local environment, trained from scratch on the released training rows only.
 - point-charge EFG tensors under several charge models (unit, nuclear charge, electronegativity polarity, first shell, ...)
 - Steinhardt bond-orientational order parameters, coordination-number estimates, bond-valence sums, radial profile
 - local site point-group order from the neighbour cloud, inversion, orbit structure; Wyckoff-letter row context
 - per-element robust standardisation of key descriptors (fitted on training candidates only) -> two-sided |deviation|
 - within-row comparisons (candidate minus row mean / max, within-row rank)
 - smoothed per-element target encodings and structural-prototype votes, nested leave-one-fold-out so training rows
   never see their own labels; test rows use statistics from all training rows
 - several boosted rankers (LightGBM lambdarank/binary, XGBoost rank:pairwise, CatBoost YetiRank), multiple seeds,
   z-scored and averaged; argmax over the 4 candidates of each row
Fixed plan: no wall-clock gates, no runtime switches, nothing fitted on test rows.
Usage: python solution.py <public_dataset_dir> <submission_csv_path>
"""
'''
open(out_path, 'w').write(header + feats.replace('"""Candidate-wise feature extraction for FieldRelay (problem18).', '"""Candidate-wise feature extraction.', 1) + '\n\n' + symm + '\n\n' + sf + main)
print('written', out_path)
