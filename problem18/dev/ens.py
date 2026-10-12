import numpy as np, sys, time, itertools, lightgbm as lgb, xgboost as xgb
sys.path.insert(0,'dev'); import common, feats, stack_feats as sf
from catboost import CatBoostRanker, Pool
d=common.load(); Y=d['Y']; folds=d['folds']; N=len(Y); zc=d['zc']; site=d['site']
F=sf.base_features(d['env'],d['mask'],zc,site,cache='dev/cache/symm_train.npz'); Ft=sf.base_features(d['env_t'],d['mask_t'],d['zc_t'],d['site_t'],cache='dev/cache/symm_test.npz')
ff=sf.FoldFeatures(F,Ft,zc,d['zc_t'],site,d['site_t'],Y,folds,sf.proto_parts(d['env'],zc,site),sf.proto_parts(d['env_t'],d['zc_t'],d['site_t']))
Ycand=ff.Ycand; SEEDS=[0,1,2]; t0=time.time()
FX={k: ff.build(np.where(folds!=k)[0]) for k in range(5)}
names=FX[0][2]; print("features:", len(names), flush=True)
small=[j for j,n in enumerate(names) if n in ['zc','site','period','group','block','en_c','d1','cn120','cn_soft','q2_sh','q4_sh','q6_sh','efg_u_shape','efg_u_logfro','efg_e_logfro','efg_z_logfro','efg_sh_shape','sh_en_diff','sh_centroid','sh_aniso','bvs_sh','inv_frac12','pg_order','pg_inv','orb_max','sh_same_el','rr2','rr4','rr8','d1_over_R0','sh_zn_mean','n12_en_mean','ndist12','gap_max','let_unique','let_rank','te_z','te_zs','te_zcn','proto2_p']]
def lgb_model(obj, nl=15, md=20, ff_=0.3, lr=0.03, rounds=300):
    def fit(Xtr,ytr,Xva,s):
        p=dict(objective=obj,verbose=-1,learning_rate=lr,num_leaves=nl,min_data_in_leaf=md,feature_fraction=ff_,bagging_fraction=0.8,bagging_freq=1,lambda_l2=1.0,num_threads=6,seed=s)
        ds=lgb.Dataset(Xtr,ytr,group=[4]*(len(ytr)//4)) if obj=='lambdarank' else lgb.Dataset(Xtr,ytr)
        return lgb.train(p,ds,rounds).predict(Xva)
    return fit
def xgb_model(depth=6, eta=0.03, rounds=400, cs=0.3):
    def fit(Xtr,ytr,Xva,s):
        p=dict(objective='rank:pairwise',max_depth=depth,eta=eta,subsample=0.8,colsample_bytree=cs,min_child_weight=5,reg_lambda=1.0,nthread=6,seed=s,verbosity=0)
        dtr=xgb.DMatrix(Xtr,ytr); dtr.set_group([4]*(len(ytr)//4)); return xgb.train(p,dtr,rounds).predict(xgb.DMatrix(Xva))
    return fit
def cat_model(iters=600, lr=0.05, depth=6):
    def fit(Xtr,ytr,Xva,s):
        m=CatBoostRanker(loss_function='YetiRank',iterations=iters,learning_rate=lr,depth=depth,verbose=0,random_seed=s,thread_count=6)
        m.fit(Pool(Xtr,ytr,group_id=np.repeat(np.arange(len(ytr)//4),4))); return m.predict(Pool(Xva,group_id=np.repeat(np.arange(len(Xva)//4),4)))
    return fit
MODELS={'lgb_rank':(lgb_model('lambdarank'),None),'lgb_bin':(lgb_model('binary'),None),'lgb_rank_deep':(lgb_model('lambdarank',nl=31,md=10,rounds=250),None),
        'xgb6':(xgb_model(6,0.03,400,0.3),None),'xgb4':(xgb_model(4,0.05,300,0.3),None),'xgb8':(xgb_model(8,0.02,400,0.3),None),'cat':(cat_model(),None),'lgb_rank_small':(lgb_model('lambdarank'),small)}
OOF={}
for name,(fit,cols) in MODELS.items():
    oof=np.zeros((N,4))
    for k in range(5):
        trn=np.where(folds!=k)[0]; val=np.where(folds==k)[0]; X=FX[k][0]
        if cols is not None: X=X[:,:,cols]
        Xtr=X[trn].reshape(-1,X.shape[-1]); Xva=X[val].reshape(-1,X.shape[-1])
        for s in SEEDS: oof[val]+=fit(Xtr,Ycand[trn].ravel(),Xva,s).reshape(-1,4)/len(SEEDS)
    OOF[name]=oof; np.save(f'dev/cache/ens_oof_{name}.npy',oof)
    print(f"{name}: {(oof.argmax(1)==Y).mean():.4f}  t={time.time()-t0:.0f}s", flush=True)
def z(a): a=a-a.mean(1,keepdims=True); return a/(a.std()+1e-9)
print("== blends")
keys=list(OOF)
for r in range(2,len(keys)+1):
    best=None
    for combo in itertools.combinations(keys,r):
        acc=(sum(z(OOF[c]) for c in combo).argmax(1)==Y).mean()
        if best is None or acc>best[0]: best=(acc,combo)
    print(r, f"{best[0]:.4f}", best[1], flush=True)
print("all:", (sum(z(OOF[c]) for c in keys).argmax(1)==Y).mean().round(4))
