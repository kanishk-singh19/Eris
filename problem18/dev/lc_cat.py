import numpy as np, sys, lightgbm as lgb, time
sys.path.insert(0,'dev'); import common, feats, symm
from catboost import CatBoostRanker, Pool
d=common.load(); Y=d['Y']; folds=d['folds']; N=len(Y); zc=d['zc']; site=d['site']
Ycand=np.zeros((N,4)); Ycand[np.arange(N),Y]=1
F=feats.candidate_features(d['env'],d['mask'],zc,site)
S=dict(np.load('dev/cache/symm_train.npz')); F.update(S)
names=sorted(F); X=np.stack([F[k] for k in names],-1).astype(np.float32)
def lgb_cv(X, frac=1.0, seeds=(0,1)):
    oof=np.zeros((N,4))
    for k in range(5):
        trn=np.where(folds!=k)[0]; val=np.where(folds==k)[0]
        for s in seeds:
            rng=np.random.default_rng(s); tr=rng.choice(trn, int(len(trn)*frac), replace=False)
            p=dict(objective='lambdarank',verbose=-1,learning_rate=0.03,num_leaves=15,min_data_in_leaf=20,feature_fraction=0.3,bagging_fraction=0.8,bagging_freq=1,num_threads=6,seed=s)
            m=lgb.train(p,lgb.Dataset(X[tr].reshape(-1,X.shape[-1]),Ycand[tr].ravel(),group=[4]*len(tr)),300)
            oof[val]+=m.predict(X[val].reshape(-1,X.shape[-1])).reshape(-1,4)
    return (oof.argmax(1)==Y).mean()
t=time.time()
for frac in [0.25,0.5,0.75,1.0]:
    print(f"learning curve frac={frac}: {lgb_cv(X,frac):.4f}  t={time.time()-t:.0f}s", flush=True)
# CatBoost ranker (YetiRank) with categorical zc, site
cat_idx=[names.index('zc'), names.index('site')]
oof=np.zeros((N,4))
for k in range(5):
    trn=np.where(folds!=k)[0]; val=np.where(folds==k)[0]
    Xtr=X[trn].reshape(-1,X.shape[-1]).astype(object); Xva=X[val].reshape(-1,X.shape[-1]).astype(object)
    for ci in cat_idx: Xtr[:,ci]=Xtr[:,ci].astype(int).astype(str); Xva[:,ci]=Xva[:,ci].astype(int).astype(str)
    gid=np.repeat(np.arange(len(trn)),4)
    m=CatBoostRanker(loss_function='YetiRank',iterations=600,learning_rate=0.05,depth=6,verbose=0,random_seed=0,thread_count=6)
    m.fit(Pool(Xtr,Ycand[trn].ravel(),group_id=gid,cat_features=cat_idx))
    oof[val]=m.predict(Pool(Xva,group_id=np.repeat(np.arange(len(val)),4),cat_features=cat_idx)).reshape(-1,4)
print(f"CatBoost YetiRank: {(oof.argmax(1)==Y).mean():.4f} t={time.time()-t:.0f}s")
np.save('dev/cache/oof_cat.npy',oof)
