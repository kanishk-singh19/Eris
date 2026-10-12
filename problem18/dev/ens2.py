import numpy as np, sys, time, lightgbm as lgb, xgboost as xgb
sys.path.insert(0,'dev'); import common, feats, stack_feats as sf
d=common.load(); Y=d['Y']; folds=d['folds']; N=len(Y); zc=d['zc']; site=d['site']
F=sf.base_features(d['env'],d['mask'],zc,site,cache='dev/cache/symm_train.npz'); Ft=sf.base_features(d['env_t'],d['mask_t'],d['zc_t'],d['site_t'],cache='dev/cache/symm_test.npz')
ff=sf.FoldFeatures(F,Ft,zc,d['zc_t'],site,d['site_t'],Y,folds,sf.proto_parts(d['env'],zc,site),sf.proto_parts(d['env_t'],d['zc_t'],d['site_t']))
Ycand=ff.Ycand; SEEDS=[0,1,2]; t0=time.time()
FX={k: ff.build(np.where(folds!=k)[0])[0] for k in range(5)}
def lgb_model(obj='lambdarank', nl=15, md=20, ff_=0.3, lr=0.03, rounds=300, extra=None):
    def fit(Xtr,ytr,Xva,s):
        p=dict(objective=obj,verbose=-1,learning_rate=lr,num_leaves=nl,min_data_in_leaf=md,feature_fraction=ff_,bagging_fraction=0.8,bagging_freq=1,lambda_l2=1.0,num_threads=5,seed=s)
        if extra: p.update(extra)
        ds=lgb.Dataset(Xtr,ytr,group=[4]*(len(ytr)//4)) if obj=='lambdarank' else lgb.Dataset(Xtr,ytr)
        return lgb.train(p,ds,rounds).predict(Xva)
    return fit
def xgb_model(depth=6, eta=0.03, rounds=400, cs=0.3, mcw=5):
    def fit(Xtr,ytr,Xva,s):
        p=dict(objective='rank:pairwise',max_depth=depth,eta=eta,subsample=0.8,colsample_bytree=cs,min_child_weight=mcw,reg_lambda=1.0,nthread=5,seed=s,verbosity=0)
        dtr=xgb.DMatrix(Xtr,ytr); dtr.set_group([4]*(len(ytr)//4)); return xgb.train(p,dtr,rounds).predict(xgb.DMatrix(Xva))
    return fit
MODELS={'lgb63':lgb_model(nl=63,md=5,rounds=200),'lgb31_400':lgb_model(nl=31,md=10,lr=0.02,rounds=400),'lgb_et31':lgb_model(nl=31,md=10,rounds=400,extra=dict(extra_trees=True)),
        'lgb_dart':lgb_model(nl=31,md=10,rounds=300,lr=0.05,extra=dict(boosting='dart',drop_rate=0.1)),'xgb10':xgb_model(10,0.02,400,0.3),'xgb6_800':xgb_model(6,0.015,800,0.3),'xgb6_mcw2':xgb_model(6,0.03,400,0.3,2),'xgb6_cs5':xgb_model(6,0.03,400,0.5)}
for name,fit in MODELS.items():
    oof=np.zeros((N,4))
    for k in range(5):
        trn=np.where(folds!=k)[0]; val=np.where(folds==k)[0]; X=FX[k]
        Xtr=X[trn].reshape(-1,X.shape[-1]); Xva=X[val].reshape(-1,X.shape[-1])
        for s in SEEDS: oof[val]+=fit(Xtr,Ycand[trn].ravel(),Xva,s).reshape(-1,4)/len(SEEDS)
    np.save(f'dev/cache/ens_oof_{name}.npy',oof); print(f"{name}: {(oof.argmax(1)==Y).mean():.4f}  t={time.time()-t0:.0f}s", flush=True)
