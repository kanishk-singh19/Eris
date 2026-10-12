import numpy as np, sys, lightgbm as lgb, time
sys.path.insert(0,'dev'); import common, feats
d=common.load(); Y=d['Y']; folds=d['folds']; N=len(Y); zc=d['zc']; site=d['site']
Ycand=np.zeros((N,4)); Ycand[np.arange(N),Y]=1
F=feats.candidate_features(d['env'],d['mask'],zc,site); S=dict(np.load('dev/cache/symm_train.npz')); F.update(S)
names=sorted(F); X=np.stack([F[k] for k in names],-1).astype(np.float32)
cn=F['cn120']; cnb=np.clip(cn,1,9).astype(int)
def te_fit(idx, key):
    """smoothed target encoding P(ans|key) from rows idx"""
    k=key[idx].ravel(); a=Ycand[idx].ravel()
    tot={}; hit={}
    for kk,aa in zip(k,a): tot[kk]=tot.get(kk,0)+1; hit[kk]=hit.get(kk,0)+aa
    return {kk:(hit[kk]+0.25*10)/(tot[kk]+10) for kk in tot}
def te_apply(mp,key): return np.vectorize(lambda v: mp.get(v,0.25))(key)
def te_features(trn, allidx):
    """nested: for training rows use leave-one-training-fold-out; for other rows use all training rows"""
    keys={'z':zc, 'zs':zc*100+site, 'zcn':zc*100+cnb, 'zinv':zc*10+S['pg_inv'].astype(int), 'zpg':zc*100+np.minimum(S['pg_order'],8).astype(int)}
    out=np.zeros((N,4,len(keys)))
    trn_set=set(trn.tolist()); trf=folds[trn]
    for j,(nm,key) in enumerate(keys.items()):
        full=te_fit(trn,key)
        out[:,:,j]=te_apply(full,key)
        for fo in np.unique(trf):
            inner_tr=trn[trf!=fo]; inner_va=trn[trf==fo]
            out[inner_va,:,j]=te_apply(te_fit(inner_tr,key),key[inner_va])
    return out
def cv(build, name, rounds=300, seeds=(0,1), obj='lambdarank'):
    oof=np.zeros((N,4))
    for k in range(5):
        trn=np.where(folds!=k)[0]; val=np.where(folds==k)[0]
        Xk=build(trn)
        for s in seeds:
            p=dict(objective=obj,verbose=-1,learning_rate=0.03,num_leaves=15,min_data_in_leaf=20,feature_fraction=0.3,bagging_fraction=0.8,bagging_freq=1,num_threads=6,seed=s)
            ds=lgb.Dataset(Xk[trn].reshape(-1,Xk.shape[-1]),Ycand[trn].ravel(),group=[4]*len(trn))
            m=lgb.train(p,ds,rounds); oof[val]+=m.predict(Xk[val].reshape(-1,Xk.shape[-1])).reshape(-1,4)
    print(f"{name}: {(oof.argmax(1)==Y).mean():.4f}", flush=True); return oof
t=time.time()
cv(lambda trn: X, 'base+symm')
oof_te=cv(lambda trn: np.concatenate([X, te_features(trn,None)],-1), 'base+symm+TE')
cv(lambda trn: np.concatenate([X, te_features(trn,None)],-1), 'base+symm+TE binary', obj='binary')
np.save('dev/cache/oof_te.npy',oof_te)
# pairwise difference model: for each ordered pair (a,b) in a row, features [xa, xb, xa-xb], label a is answer & b not
def pair_cv(Xb, name, rounds=300):
    oof=np.zeros((N,4)); Dn=Xb.shape[-1]
    for k in range(5):
        trn=np.where(folds!=k)[0]; val=np.where(folds==k)[0]
        A=[];L=[]
        for i in trn:
            a=Y[i]
            for b in range(4):
                if b==a: continue
                A.append(np.concatenate([Xb[i,a],Xb[i,b],Xb[i,a]-Xb[i,b]])); L.append(1)
                A.append(np.concatenate([Xb[i,b],Xb[i,a],Xb[i,b]-Xb[i,a]])); L.append(0)
        A=np.array(A,np.float32); L=np.array(L)
        p=dict(objective='binary',verbose=-1,learning_rate=0.03,num_leaves=15,min_data_in_leaf=20,feature_fraction=0.3,bagging_fraction=0.8,bagging_freq=1,num_threads=6,seed=0)
        m=lgb.train(p,lgb.Dataset(A,L),rounds)
        for i in val:
            sc=np.zeros(4)
            for a in range(4):
                for b in range(4):
                    if a==b: continue
                    sc[a]+=m.predict(np.concatenate([Xb[i,a],Xb[i,b],Xb[i,a]-Xb[i,b]])[None])[0]
            oof[i]=sc
    print(f"{name}: {(oof.argmax(1)==Y).mean():.4f} t={time.time()-t:.0f}s", flush=True); return oof
sel=[j for j,n in enumerate(names) if not (n.endswith('_rd') or n.endswith('_rk') or n.endswith('_rmax'))]
oof_pair=pair_cv(X[:,:,sel],'pairwise'); np.save('dev/cache/oof_pair.npy',oof_pair)
