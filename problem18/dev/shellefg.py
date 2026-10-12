# EFG features from complete shells only (drop the possibly truncated last shell) + GBM comparison
import numpy as np, sys, lightgbm as lgb
sys.path.insert(0,'dev'); import common, feats
d=common.load(); Y=d['Y']; folds=d['folds']; N=len(Y); zc=d['zc']; site=d['site']; env=d['env']
Ycand=np.zeros((N,4)); Ycand[np.arange(N),Y]=1
x=env[...,:3].astype(np.float64)*8; r=env[...,3].astype(np.float64)*8; zn=np.rint(env[...,4]*118).astype(int)
complete=(r < r[...,-1:]-0.02).astype(float)
en_n=feats.EN[np.clip(zn,0,118)]; en_c=feats.EN[zc]
G={}
for name,w in {'cu':complete,'cz':complete*zn/118,'ce':complete*(en_n-en_c[...,None]),'cr3':complete*np.exp(-(r/r[...,:1]-1)/0.5)}.items():
    V=feats.efg_tensor(x,r,w); fro=feats.tensor_feats(V,'efg_'+name,G); G[f'efg_{name}_shape']=fro/np.maximum((np.abs(w)/r**3).sum(-1),1e-9)
G['n_complete']=complete.sum(-1)
ans=np.zeros((N,4),bool); ans[np.arange(N),Y]=True
for k in ['efg_cu_shape','efg_cu_logfro','efg_ce_logfro','efg_cz_logfro']:
    print(k, "argmax", (G[k].argmax(1)==Y).mean().round(3), "argmin", (G[k].argmin(1)==Y).mean().round(3))
F=feats.candidate_features(env,d['mask'],zc,site); S=dict(np.load('dev/cache/symm_train.npz')); F.update(S)
def cv(F_, name, seeds=(0,1)):
    names=sorted(F_); X=np.stack([F_[k] for k in names],-1).astype(np.float32); oof=np.zeros((N,4))
    for k in range(5):
        trn=folds!=k; val=folds==k
        for s in seeds:
            p=dict(objective='lambdarank',verbose=-1,learning_rate=0.03,num_leaves=15,min_data_in_leaf=20,feature_fraction=0.3,bagging_fraction=0.8,bagging_freq=1,num_threads=6,seed=s)
            m=lgb.train(p,lgb.Dataset(X[trn].reshape(-1,X.shape[-1]),Ycand[trn].ravel(),group=[4]*trn.sum()),300)
            oof[val]+=m.predict(X[val].reshape(-1,X.shape[-1])).reshape(-1,4)
    print(f"{name}: {(oof.argmax(1)==Y).mean():.4f}", flush=True)
cv(F,'base+symm'); F2=dict(F); F2.update(G); cv(F2,'base+symm+complete-shell EFG')
