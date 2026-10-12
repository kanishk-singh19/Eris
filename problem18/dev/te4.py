import numpy as np, sys, lightgbm as lgb, collections, time
sys.path.insert(0,'dev'); import common, feats
d=common.load(); Y=d['Y']; folds=d['folds']; N=len(Y); zc=d['zc']; site=d['site']; env=d['env']
Ycand=np.zeros((N,4)); Ycand[np.arange(N),Y]=1
F=feats.candidate_features(env,d['mask'],zc,site); S=dict(np.load('dev/cache/symm_train.npz')); F.update(S)
names=sorted(F); X=np.stack([F[k] for k in names],-1).astype(np.float32)
per,grp,blk=feats.period_group(zc.ravel()); grp=grp.reshape(N,4); blk=blk.reshape(N,4)
cnb=np.clip(F['cn120'],1,9).astype(int); inv=S['pg_inv'].astype(int); pg=np.minimum(S['pg_order'],8).astype(int)
r=env[...,3]*8; zn=np.rint(env[...,4]*118).astype(int)
# prototype parts
P=[]; 
for i in range(N):
    pp=[]
    for c in range(4):
        sh=r[i,c]<=1.2*r[i,c,0]
        comp=tuple(sorted(collections.Counter([int(np.where(zc[i]==z)[0][0]) if z in zc[i] else -1 for z in zn[i,c][sh]]).values()))
        pp.append((int(sh.sum()),comp))
    P.append(pp)
SIG0=[tuple(sorted(p)) for p in P]
P3=[[P[i][c]+(int(blk[i,c]),int(site[i,c])) for c in range(4)] for i in range(N)]; SIG3=[tuple(sorted(p)) for p in P3]
P1=[[P[i][c]+(int(blk[i,c]),) for c in range(4)] for i in range(N)]; SIG1=[tuple(sorted(p)) for p in P1]
KEYS={'z':zc,'zs':zc*100+site,'zcn':zc*100+cnb,'zinv':zc*10+inv,'zpg':zc*100+pg,'scn':site*100+cnb,'zscn':zc*10000+site*100+cnb,'sinv':site*10+inv}
def te_fit(idx, key, m=10):
    k=key[idx].ravel(); a=Ycand[idx].ravel(); tot={}; hit={}
    for kk,aa in zip(k,a): tot[kk]=tot.get(kk,0)+1; hit[kk]=hit.get(kk,0)+aa
    return {kk:(hit[kk]+0.25*m)/(tot[kk]+m) for kk in tot}
def te_apply(mp,key): return np.vectorize(lambda v: mp.get(v,0.25))(key)
def te_features(trn, keys):
    out=np.zeros((N,4,len(keys))); trf=folds[trn]
    for j,nm in enumerate(keys):
        key=KEYS[nm]; out[:,:,j]=te_apply(te_fit(trn,key),key)
        for fo in np.unique(trf):
            itr=trn[trf!=fo]; iva=trn[trf==fo]; out[iva,:,j]=te_apply(te_fit(itr,key),key[iva])
    return out
def proto_fit(idx, SIG, PP):
    votes=collections.defaultdict(lambda: collections.Counter()); cnt=collections.Counter()
    for i in idx: votes[SIG[i]][PP[i][Y[i]]]+=1; cnt[SIG[i]]+=1
    return votes,cnt
def proto_apply(model, idx, SIG, PP, out):
    votes,cnt=model
    for i in idx:
        v=votes.get(SIG[i]); n=cnt.get(SIG[i],0)
        for c in range(4):
            ties=sum(1 for cc in range(4) if PP[i][cc]==PP[i][c])
            out[i,c,0]=(v[PP[i][c]]/ties+0.25)/(n+1) if v is not None else 0.25
            out[i,c,1]=np.log1p(n)
def proto_features(trn):
    out=np.zeros((N,4,6)); trf=folds[trn]
    for j,(SIG,PP) in enumerate([(SIG0,P),(SIG1,P1),(SIG3,P3)]):
        o=np.zeros((N,4,2)); proto_apply(proto_fit(trn,SIG,PP), np.arange(N), SIG, PP, o)
        for fo in np.unique(trf):
            itr=trn[trf!=fo]; iva=trn[trf==fo]; proto_apply(proto_fit(itr,SIG,PP), iva, SIG, PP, o)
        out[:,:,2*j:2*j+2]=o
    return out
# within-row outlierness: distance of candidate to mean of the other three in standardized key-feature space
keyf=['d1','cn_soft','q2_sh','q4_sh','q6_sh','efg_u_shape','efg_u_logfro','sh_en_diff','sh_centroid','sh_aniso','bvs_sh','rr2','rr4','rr8','en_c','pg_order']
Kf=np.stack([F[k] for k in keyf],-1); Kf=(Kf-Kf.reshape(-1,Kf.shape[-1]).mean(0))/(Kf.reshape(-1,Kf.shape[-1]).std(0)+1e-6)
outl=np.zeros((N,4,2))
for c in range(4):
    others=np.delete(Kf,c,axis=1).mean(1); outl[:,c,0]=np.linalg.norm(Kf[:,c]-others,axis=-1)
    outl[:,c,1]=np.min([np.linalg.norm(Kf[:,c]-Kf[:,o],axis=-1) for o in range(4) if o!=c],axis=0)
def cv(build, name, rounds=300, seeds=(0,1,2), nl=15, md=20, ff=0.3, lr=0.03):
    oof=np.zeros((N,4))
    for k in range(5):
        trn=np.where(folds!=k)[0]; val=np.where(folds==k)[0]; Xk=build(trn)
        for s in seeds:
            p=dict(objective='lambdarank',verbose=-1,learning_rate=lr,num_leaves=nl,min_data_in_leaf=md,feature_fraction=ff,bagging_fraction=0.8,bagging_freq=1,num_threads=6,seed=s)
            m=lgb.train(p,lgb.Dataset(Xk[trn].reshape(-1,Xk.shape[-1]),Ycand[trn].ravel(),group=[4]*len(trn)),rounds)
            oof[val]+=m.predict(Xk[val].reshape(-1,Xk.shape[-1])).reshape(-1,4)
    acc=(oof.argmax(1)==Y).mean(); print(f"{name}: {acc:.4f}", flush=True); return acc, oof
base5=['z','zs','zcn','zinv','zpg']
cv(lambda trn: np.concatenate([X, te_features(trn, base5)],-1), 'X+TE5 (3 seeds)')
cv(lambda trn: np.concatenate([X, te_features(trn, base5+['scn','zscn','sinv'])],-1), 'X+TE8 (site keys)')
cv(lambda trn: np.concatenate([X, te_features(trn, base5), proto_features(trn)],-1), 'X+TE5+proto')
cv(lambda trn: np.concatenate([X, te_features(trn, base5), outl],-1), 'X+TE5+outlier')
a,oof=cv(lambda trn: np.concatenate([X, te_features(trn, base5+['scn','zscn','sinv']), proto_features(trn), outl],-1), 'X+TE8+proto+outlier')
np.save('dev/cache/oof_te4.npy',oof)
