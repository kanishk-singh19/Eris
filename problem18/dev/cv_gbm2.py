import numpy as np, pandas as pd, time, sys, lightgbm as lgb
sys.path.insert(0,'dev'); import common, feats, feats2
d=common.load(); Y=d['Y']; folds=d['folds']; N=len(Y)
Ycand=np.zeros((N,4)); Ycand[np.arange(N),Y]=1
F1=feats.candidate_features(d['env'],d['mask'],d['zc'],d['site'])
F2=feats2.build2(d['env'],d['mask'],d['zc'])
def mat(F): names=sorted(F); return np.stack([F[k] for k in names],-1).astype(np.float32), names
def cv(X, name, rounds=300, seeds=(0,1,2)):
    oof=np.zeros((N,4))
    for k in range(5):
        trn=folds!=k; val=folds==k
        for s in seeds:
            p=dict(objective='lambdarank',verbose=-1,learning_rate=0.03,num_leaves=15,min_data_in_leaf=20,feature_fraction=0.3,bagging_fraction=0.8,bagging_freq=1,num_threads=6,seed=s)
            ds=lgb.Dataset(X[trn].reshape(-1,X.shape[-1]),Ycand[trn].ravel(),group=[4]*trn.sum())
            m=lgb.train(p,ds,rounds); oof[val]+=m.predict(X[val].reshape(-1,X.shape[-1])).reshape(-1,4)
    print(f"{name}: {(oof.argmax(1)==Y).mean():.4f} nfeat={X.shape[-1]}", flush=True); return oof
X1,n1=mat(F1); X2,n2=mat(F2)
cv(X1,'F1 base')
cv(X2,'F2 powerspec+distortion')
cv(np.concatenate([X1,X2],-1),'F1+F2')
Fr=feats.add_row_relative({**F1,**F2}); Xr,nr=mat(Fr)
cv(Xr,'F1+F2 + rowrel')
