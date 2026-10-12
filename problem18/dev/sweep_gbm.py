import numpy as np, pandas as pd, time, sys, lightgbm as lgb, itertools
sys.path.insert(0,'dev'); import common, feats
d=common.load(); Y=d['Y']; folds=d['folds']; N=len(Y)
Ycand=np.zeros((N,4)); Ycand[np.arange(N),Y]=1
# precompute per-fold features
FX={}
for k in range(5):
    trn=np.where(folds!=k)[0]
    X,Xt,names=common.tab_features(d,trn)
    FX[k]=(X,names)
def run(params, obj, rounds, colsel, seeds=(0,)):
    oof=np.zeros((N,4))
    for k in range(5):
        trn=np.where(folds!=k)[0]; val=np.where(folds==k)[0]
        X,names=FX[k]; cols=[j for j,n in enumerate(names) if colsel(n)]
        Xtr=X[trn][:,:,cols].reshape(-1,len(cols)); Xva=X[val][:,:,cols].reshape(-1,len(cols))
        for s in seeds:
            p=dict(objective=obj,verbose=-1,seed=s,num_threads=6,**params)
            ds=lgb.Dataset(Xtr,Ycand[trn].ravel(),group=[4]*len(trn)) if obj=='lambdarank' else lgb.Dataset(Xtr,Ycand[trn].ravel())
            m=lgb.train(p,ds,rounds)
            pr=m.predict(Xva).reshape(-1,4)
            if obj=='binary': pr=np.log(pr/(1-pr))
            oof[val]+=pr/len(seeds)
    return (oof.argmax(1)==Y).mean(), oof
sels={'all':lambda n:True,
      'noelem':lambda n: not ('_ez' in n or '_eabs' in n),
      'norowrel':lambda n: not (n.endswith('_rd') or n.endswith('_rk') or n.endswith('_rmax')),
      'base':lambda n: not ('_ez' in n or '_eabs' in n or n.endswith('_rd') or n.endswith('_rk') or n.endswith('_rmax')),
      'rd_only':lambda n: not (n.endswith('_rk') or n.endswith('_rmax'))}
grid=[]
for obj in ['lambdarank','binary']:
  for sel in ['all','noelem','norowrel','base','rd_only']:
    for nl,md,ff,rounds,lr in [(7,30,0.2,400,0.03),(15,20,0.3,300,0.03),(31,10,0.5,300,0.03),(7,50,0.1,800,0.02),(3,30,0.3,800,0.05)]:
        grid.append((obj,sel,nl,md,ff,rounds,lr))
t=time.time(); res=[]
for obj,sel,nl,md,ff,rounds,lr in grid:
    params=dict(learning_rate=lr,num_leaves=nl,min_data_in_leaf=md,feature_fraction=ff,bagging_fraction=0.8,bagging_freq=1,lambda_l2=1.0)
    acc,_=run(params,obj,rounds,sels[sel],seeds=(0,1))
    res.append((obj,sel,nl,md,ff,rounds,lr,acc))
    print(f"{obj:10s} {sel:9s} nl={nl:2d} md={md:2d} ff={ff} r={rounds} lr={lr} acc={acc:.4f} t={time.time()-t:.0f}s",flush=True)
df=pd.DataFrame(res,columns=['obj','sel','nl','md','ff','rounds','lr','acc']).sort_values('acc',ascending=False)
print(df.head(15).to_string())
