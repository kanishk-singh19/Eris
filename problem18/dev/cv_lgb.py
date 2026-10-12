import numpy as np, pandas as pd, time, sys, lightgbm as lgb
sys.path.insert(0,'dev'); import feats
D="public (12)/"
tr=np.load(D+"train_inputs.npz")
y=pd.read_csv(D+"train.csv"); f=pd.read_csv(D+"training_folds.csv")
folds=y.merge(f,on='id').fold.values; Y=y.target.values
env,mask,zc,site=tr['candidate_environment'],tr['candidate_mask'],tr['candidate_center_z'],tr['candidate_site_code']
N=len(Y)
F0=feats.candidate_features(env,mask,zc,site)
def make(F0, idx_fit):
    sub={k:v[idx_fit] for k,v in F0.items()}
    stats=feats.element_norm_fit(sub, zc[idx_fit], feats.ELEM_KEYS)
    F=dict(F0); F.update(feats.element_norm_apply(F0, zc, stats)); F=feats.add_row_relative(F)
    names=sorted(F.keys()); X=np.stack([F[k] for k in names],-1).astype(np.float32)
    return X,names
Ycand=np.zeros((N,4)); Ycand[np.arange(N),Y]=1
def run(params, obj='binary', rounds=600, use_elem=True, drop_prefix=None):
    oof=np.zeros((N,4)); accs=[]
    for k in range(5):
        trn=np.where(folds!=k)[0]; val=np.where(folds==k)[0]
        X,names=make(F0, trn)
        cols=[j for j,n in enumerate(names) if (use_elem or not (n.endswith('_ez') or n.endswith('_eabs') or '_ez_' in n or '_eabs_' in n)) and (drop_prefix is None or not any(n.startswith(p) for p in drop_prefix))]
        Xtr=X[trn][:,:,cols].reshape(-1,len(cols)); Xva=X[val][:,:,cols].reshape(-1,len(cols))
        if obj=='binary':
            ds=lgb.Dataset(Xtr,Ycand[trn].ravel())
            p=dict(objective='binary',verbose=-1,**params)
            m=lgb.train(p,ds,rounds)
        else:
            ds=lgb.Dataset(Xtr,Ycand[trn].ravel(),group=[4]*len(trn))
            p=dict(objective='lambdarank',verbose=-1,**params)
            m=lgb.train(p,ds,rounds)
        pr=m.predict(Xva).reshape(-1,4); oof[val]=pr
        accs.append((pr.argmax(1)==Y[val]).mean())
    acc=(oof.argmax(1)==Y).mean()
    return acc, accs, oof, m, names, cols
base=dict(learning_rate=0.03,num_leaves=15,min_data_in_leaf=20,feature_fraction=0.3,bagging_fraction=0.8,bagging_freq=1,lambda_l2=1.0,seed=0,num_threads=8)
t=time.time()
for tag,kw in [('bin',dict(obj='binary')),('bin_noelem',dict(obj='binary',use_elem=False)),('rank',dict(obj='rank'))]:
    acc,accs,oof,m,names,cols=run(base,**kw)
    print(f"{tag}: acc={acc:.4f} folds={np.round(accs,3)} nfeat={len(cols)} t={time.time()-t:.0f}s", flush=True)
# importance
imp=pd.Series(m.feature_importance('gain'),index=[names[c] for c in cols]).sort_values(ascending=False)
print(imp.head(40).to_string())
