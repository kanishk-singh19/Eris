# Row-level multiclass GBM with candidate-permutation augmentation (sees all 4 candidates jointly)
import numpy as np, sys, lightgbm as lgb, itertools, time
sys.path.insert(0,'dev'); import common, feats
d=common.load(); Y=d['Y']; folds=d['folds']; N=len(Y); zc=d['zc']; site=d['site']
F=feats.candidate_features(d['env'],d['mask'],zc,site); S=dict(np.load('dev/cache/symm_train.npz')); F.update(S)
keep=['zc','site','period','group','block','en_c','d1','cn120','cn_soft','q2_sh','q4_sh','q6_sh','efg_u_shape','efg_u_logfro','efg_e_logfro','efg_z_logfro','efg_sh_shape','sh_en_diff','sh_centroid','sh_aniso','bvs_sh','inv_frac12','pg_order','pg_inv','orb_max','sh_same_el','rr2','rr4','rr8','d1_over_R0','sh_zn_mean','n12_en_mean','ndist12','gap_max']
X=np.stack([F[k] for k in keep],-1).astype(np.float32)  # (N,4,D)
perms=list(itertools.permutations(range(4)))
rng=np.random.default_rng(0)
def aug(idx, nperm):
    Xs=[];Ys=[]
    for i in idx:
        ps=[perms[j] for j in rng.choice(24,nperm,replace=False)] if nperm<24 else perms
        for p in ps:
            Xs.append(X[i][list(p)].ravel()); Ys.append(list(p).index(Y[i]))
    return np.array(Xs,np.float32), np.array(Ys)
oof=np.zeros((N,4)); t=time.time()
for k in range(5):
    trn=np.where(folds!=k)[0]; val=np.where(folds==k)[0]
    Xtr,Ytr=aug(trn,24)
    p=dict(objective='multiclass',num_class=4,verbose=-1,learning_rate=0.05,num_leaves=15,min_data_in_leaf=40,feature_fraction=0.3,bagging_fraction=0.7,bagging_freq=1,num_threads=6,seed=0)
    m=lgb.train(p,lgb.Dataset(Xtr,Ytr),300)
    # test-time: average over all 24 permutations, mapping back
    for i in val:
        acc=np.zeros(4)
        Xq=np.array([X[i][list(pp)].ravel() for pp in perms],np.float32); pr=m.predict(Xq)
        for pp,prob in zip(perms,pr):
            for pos,cand in enumerate(pp): acc[cand]+=prob[pos]
        oof[i]=acc/24
    print(f"fold {k}: {(oof[val].argmax(1)==Y[val]).mean():.4f} t={time.time()-t:.0f}s", flush=True)
print(f"row-multiclass permutation-aug OOF: {(oof.argmax(1)==Y).mean():.4f}")
np.save('dev/cache/oof_rowmc.npy',oof)
