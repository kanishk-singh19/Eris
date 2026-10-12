import numpy as np, sys, xgboost as xgb, time
sys.path.insert(0,'dev'); import common, feats
d=common.load(); Y=d['Y']; folds=d['folds']; N=len(Y); zc=d['zc']; site=d['site']; env=d['env']
Ycand=np.zeros((N,4)); Ycand[np.arange(N),Y]=1
F=feats.candidate_features(env,d['mask'],zc,site); S=dict(np.load('dev/cache/symm_train.npz')); F.update(S)
r=env[...,3]*8
for R in [2.5,3.0,3.5,4.0,5.0,6.0]: F[f'nwithin{R}']=(r<=R).sum(-1).astype(float)
names=sorted(F); X=np.stack([F[k] for k in names],-1).astype(np.float32)
cnb=np.clip(F['cn120'],1,9).astype(int); inv=S['pg_inv'].astype(int); pg=np.minimum(S['pg_order'],8).astype(int)
KEYS={'z':zc,'zs':zc*100+site,'zcn':zc*100+cnb,'zinv':zc*10+inv,'zpg':zc*100+pg}
def te_fit(idx, key, m=10):
    k=key[idx].ravel(); a=Ycand[idx].ravel(); tot={}; hit={}
    for kk,aa in zip(k,a): tot[kk]=tot.get(kk,0)+1; hit[kk]=hit.get(kk,0)+aa
    return {kk:(hit[kk]+0.25*m)/(tot[kk]+m) for kk in tot}
def te_apply(mp,key): return np.vectorize(lambda v: mp.get(v,0.25))(key)
def te_features(trn):
    out=np.zeros((N,4,len(KEYS))); trf=folds[trn]
    for j,nm in enumerate(KEYS):
        key=KEYS[nm]; out[:,:,j]=te_apply(te_fit(trn,key),key)
        for fo in np.unique(trf):
            itr=trn[trf!=fo]; iva=trn[trf==fo]; out[iva,:,j]=te_apply(te_fit(itr,key),key[iva])
    return out
t=time.time()
for obj,extra in [('rank:pairwise',{}),('rank:ndcg',{}),('binary:logistic',{})]:
    for depth,eta,rounds,cs in [(4,0.05,300,0.3),(6,0.03,400,0.3),(3,0.05,500,0.5)]:
        oof=np.zeros((N,4))
        for k in range(5):
            trn=np.where(folds!=k)[0]; val=np.where(folds==k)[0]
            Xk=np.concatenate([X, te_features(trn)],-1)
            for s in (0,1):
                p=dict(objective=obj,max_depth=depth,eta=eta,subsample=0.8,colsample_bytree=cs,min_child_weight=5,reg_lambda=1.0,nthread=6,seed=s,verbosity=0)
                dtr=xgb.DMatrix(Xk[trn].reshape(-1,Xk.shape[-1]),Ycand[trn].ravel())
                if obj.startswith('rank'): dtr.set_group([4]*len(trn))
                m=xgb.train(p,dtr,rounds); oof[val]+=m.predict(xgb.DMatrix(Xk[val].reshape(-1,Xk.shape[-1]))).reshape(-1,4)
        print(f"xgb {obj} depth={depth} eta={eta} r={rounds} cs={cs}: {(oof.argmax(1)==Y).mean():.4f} t={time.time()-t:.0f}s", flush=True)
