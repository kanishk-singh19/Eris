import numpy as np, sys, lightgbm as lgb, time
sys.path.insert(0,'dev'); import common, feats
d=common.load(); Y=d['Y']; folds=d['folds']; N=len(Y); zc=d['zc']; site=d['site']
Ycand=np.zeros((N,4)); Ycand[np.arange(N),Y]=1
F=feats.candidate_features(d['env'],d['mask'],zc,site); S=dict(np.load('dev/cache/symm_train.npz')); F.update(S)
names=sorted(F); X=np.stack([F[k] for k in names],-1).astype(np.float32)
per,grp,blk=feats.period_group(zc.ravel()); per=per.reshape(N,4); grp=grp.reshape(N,4); blk=blk.reshape(N,4)
cnb=np.clip(F['cn120'],1,9).astype(int)
q2b=np.digitize(F['q2_sh'],[0.05,0.2,0.5]); shapeb=np.digitize(F['efg_u_shape'],[0.03,0.08,0.2,0.4])
inv=S['pg_inv'].astype(int); pg=np.minimum(S['pg_order'],8).astype(int)
# first-shell dominant neighbour element class
zn=np.rint(d['env'][...,4]*118).astype(int); r=d['env'][...,3]*8
dom=np.zeros((N,4),int)
for i in range(N):
    for c in range(4):
        sh=r[i,c]<=1.2*r[i,c,0]; vals,cnt=np.unique(zn[i,c][sh],return_counts=True); dom[i,c]=vals[cnt.argmax()]
KEYS={'z':zc,'zs':zc*100+site,'zcn':zc*100+cnb,'zinv':zc*10+inv,'zpg':zc*100+pg,
      'zq2':zc*10+q2b,'zshape':zc*10+shapeb,'zdom':zc*1000+dom,'gcn':grp*100+cnb+10000*blk,'pcn':per*100+cnb,'zcninv':zc*1000+cnb*10+inv,'g':grp*10+blk,'zdomcn':zc*100000+dom*100+cnb}
def te_fit(idx, key, m=10):
    k=key[idx].ravel(); a=Ycand[idx].ravel(); tot={}; hit={}
    for kk,aa in zip(k,a): tot[kk]=tot.get(kk,0)+1; hit[kk]=hit.get(kk,0)+aa
    return {kk:(hit[kk]+0.25*m)/(tot[kk]+m) for kk in tot}
def te_apply(mp,key): return np.vectorize(lambda v: mp.get(v,0.25))(key)
def te_features(trn, keys, m=10):
    out=np.zeros((N,4,len(keys))); trf=folds[trn]
    for j,nm in enumerate(keys):
        key=KEYS[nm]; out[:,:,j]=te_apply(te_fit(trn,key,m),key)
        for fo in np.unique(trf):
            itr=trn[trf!=fo]; iva=trn[trf==fo]; out[iva,:,j]=te_apply(te_fit(itr,key,m),key[iva])
    return out
def cv(build, name, rounds=300, seeds=(0,1), nl=15, md=20, ff=0.3, lr=0.03):
    oof=np.zeros((N,4))
    for k in range(5):
        trn=np.where(folds!=k)[0]; val=np.where(folds==k)[0]; Xk=build(trn)
        for s in seeds:
            p=dict(objective='lambdarank',verbose=-1,learning_rate=lr,num_leaves=nl,min_data_in_leaf=md,feature_fraction=ff,bagging_fraction=0.8,bagging_freq=1,num_threads=6,seed=s)
            m=lgb.train(p,lgb.Dataset(Xk[trn].reshape(-1,Xk.shape[-1]),Ycand[trn].ravel(),group=[4]*len(trn)),rounds)
            oof[val]+=m.predict(Xk[val].reshape(-1,Xk.shape[-1])).reshape(-1,4)
    acc=(oof.argmax(1)==Y).mean(); print(f"{name}: {acc:.4f}", flush=True); return acc, oof
base5=['z','zs','zcn','zinv','zpg']
cv(lambda trn: np.concatenate([X, te_features(trn, base5)],-1), 'X + TE5 (ref)')
cv(lambda trn: te_features(trn, base5), 'TE5 only')
cv(lambda trn: np.concatenate([X, te_features(trn, list(KEYS))],-1), 'X + TE all keys')
cv(lambda trn: np.concatenate([X, te_features(trn, base5+['zq2','zshape','zdom'])],-1), 'X + TE5 + q2/shape/dom')
cv(lambda trn: np.concatenate([X, te_features(trn, base5+['gcn','pcn','g'])],-1), 'X + TE5 + group/period keys')
cv(lambda trn: np.concatenate([X, te_features(trn, base5, m=3)],-1), 'X + TE5 m=3')
cv(lambda trn: np.concatenate([X, te_features(trn, base5, m=30)],-1), 'X + TE5 m=30')
# small feature set + TE
small=[names.index(k) for k in ['zc','site','period','group','block','en_c','d1','cn120','cn_soft','q2_sh','q4_sh','q6_sh','efg_u_shape','efg_u_logfro','efg_e_logfro','efg_sh_shape','sh_en_diff','sh_centroid','sh_aniso','bvs_sh','inv_frac12','pg_order','pg_inv','orb_max','sh_same_el','rr2','rr4','rr8','d1_over_R0']]
cv(lambda trn: np.concatenate([X[:,:,small], te_features(trn, base5)],-1), 'small + TE5')
for nl,md,rounds,lr in [(7,30,400,0.03),(31,10,200,0.03),(15,20,600,0.015),(15,40,300,0.03),(4,20,800,0.03)]:
    cv(lambda trn: np.concatenate([X, te_features(trn, base5)],-1), f'X+TE5 nl={nl} md={md} r={rounds} lr={lr}', rounds=rounds, nl=nl, md=md, lr=lr)
