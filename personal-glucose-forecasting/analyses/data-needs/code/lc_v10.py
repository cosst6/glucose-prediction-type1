import numpy as np, pandas as pd, warnings, sys; warnings.filterwarnings("ignore")
from data import *
import glucoforecast8 as g8, glucoforecast9 as G
from glucoforecast9 import build_frame9, restrict_to_logged_days, Hybrid
L1,L2,W1,W2,cgm=all_data()
hs=[60,120]
X=pd.read_pickle("/home/user/work/lc/X10.pkl")
g8.feature_list4=G.feature_list9
days=np.array(sorted(X.index.normalize().unique())); print(len(days),"days",days[0],days[-1],flush=True)
rng=np.random.default_rng(0); out=[]
for s in range(30):
    te_days=rng.choice(days,4,replace=False); pool=np.array([d for d in days if d not in te_days]); rng.shuffle(pool)
    te=X[X.index.normalize().isin(te_days)]
    for frac in (0.25,0.5,0.75,1.0):
        k=max(2,int(round(frac*len(pool)))); trd=pool[:k]
        tr=X[X.index.normalize().isin(trd)]
        # drop training rows whose 2h target falls on a test day (no peeking)
        tgt=(tr.index+pd.Timedelta("2h")).normalize(); tr=tr[~tgt.isin(te_days)]
        hyb=Hybrid().fit(tr,"B",hs)
        for h in hs:
            cols=G.feature_list9("B",h); y=f"y{h}"; t2=te.dropna(subset=cols+[y])
            P=hyb.predict(t2,h); e=(t2.g0+t2[y])-P.pred
            out.append(dict(split=s,frac=frac,ndays=k,h=h,rmse=float(np.sqrt((e**2).mean())),persist=float(np.sqrt((t2[y]**2).mean()))))
        print(s,frac,k,out[-1]["rmse"],flush=True)
pd.DataFrame(out).to_csv("/home/user/work/lc/v10_curve.csv",index=False)
