import numpy as np, sys, lightgbm as lgb, time
sys.path.insert(0,'dev'); import common, feats
d=common.load(); Y=d['Y']; folds=d['folds']; N=len(Y); zc=d['zc']; site=d['site']
Ycand=np.zeros((N,4)); Ycand[np.arange(N),Y]=1
F=feats.candidate_features(d['env'],d['mask'],zc,site); S=dict(np.load('dev/cache/symm_train.npz')); F.update(S)
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
PK=['efg_u_logfro','efg_z_logfro','efg_e_logfro','efg_sh_logfro','efg_exp_logfro','efg_u_shape','efg_sh_shape','d1','cn_soft','q2_sh','q4_sh','q6_sh','bvs_sh','sh_en_diff','sh_centroid','sum_r3','sh_aniso','d1_over_R0','pg_order','orb_max']
def pct_features(trn):
    """within-element percentile of key proxies (fitted on training candidates), and two-sided extremeness |pct-0.5|"""
    out=np.zeros((N,4,2*len(PK)))
    for j,k in enumerate(PK):
        v=F[k]; 
        for el in np.unique(zc):
            m=zc==el; ref=v[trn][zc[trn]==el]
            if len(ref)<5: ref=v[trn].ravel()
            ref=np.sort(ref.ravel()); p=np.searchsorted(ref, v[m], side='left')/len(ref)
            out[m,j]=p; out[m,len(PK)+j]=np.abs(p-0.5)
    return out
def cv(build, name, rounds=300, seeds=(0,1), nl=15, md=20, ff=0.3, lr=0.03):
    oof=np.zeros((N,4))
    for k in range(5):
        trn=np.where(folds!=k)[0]; val=np.where(folds==k)[0]; Xk=build(trn)
        for s in seeds:
            p=dict(objective='lambdarank',verbose=-1,learning_rate=lr,num_leaves=nl,min_data_in_leaf=md,feature_fraction=ff,bagging_fraction=0.8,bagging_freq=1,num_threads=6,seed=s)
            m=lgb.train(p,lgb.Dataset(Xk[trn].reshape(-1,Xk.shape[-1]),Ycand[trn].ravel(),group=[4]*len(trn)),rounds)
            oof[val]+=m.predict(Xk[val].reshape(-1,Xk.shape[-1])).reshape(-1,4)
    acc=(oof.argmax(1)==Y).mean(); print(f"{name}: {acc:.4f}", flush=True); return acc, oof, m
acc,oof,m=cv(lambda trn: np.concatenate([X, te_features(trn)],-1), 'X+TE5 ref (seeds 0,1)')
cv(lambda trn: np.concatenate([X, te_features(trn), pct_features(trn)],-1), 'X+TE5+pct')
cv(lambda trn: np.concatenate([X, te_features(trn)],-1), 'X+TE5 seeds 0-4', seeds=(0,1,2,3,4))
# feature selection by in-fold importance: keep top-K
def build_sel(K):
    def b(trn):
        Xk=np.concatenate([X, te_features(trn)],-1)
        p=dict(objective='lambdarank',verbose=-1,learning_rate=0.03,num_leaves=15,min_data_in_leaf=20,feature_fraction=0.3,bagging_fraction=0.8,bagging_freq=1,num_threads=6,seed=0)
        mm=lgb.train(p,lgb.Dataset(Xk[trn].reshape(-1,Xk.shape[-1]),Ycand[trn].ravel(),group=[4]*len(trn)),300)
        imp=mm.feature_importance('gain'); keep=np.argsort(-imp)[:K]
        return Xk[:,:,keep]
    return b
for K in [30,60,100]:
    cv(build_sel(K), f'top-{K} in-fold selected')
