import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from data import *; from ctx3d import *
import glucoforecast8 as g8, glucoforecast9 as G
from glucoforecast9 import build_frame9, restrict_to_logged_days, Hybrid
L1,L2,W1,W2,cgm=all_data()
L=pd.concat([L1,L2]).sort_values('time').reset_index(drop=True); W=pd.concat([W1,W2]).sort_index()
hs=[60,120]
X=pd.read_pickle("X12.pkl")

base=G.feature_list9
def run(sel):
    g8.feature_list4 = (lambda m,h,u=True: base(m,h,u)+sel)
    days=[d for d in sorted(X.index.normalize().unique()) if d>=pd.Timestamp("2026-08-30")]; out=[]
    for day in days:
        te=X[X.index.normalize()==day]; tr=X[X.index<day-pd.Timedelta(minutes=240)]
        hyb=Hybrid().fit(tr,"B",hs)
        for h in hs:
            cols=g8.feature_list4("B",h); y=f"y{h}"; t2=te.dropna(subset=cols+[y])
            if not len(t2): continue
            P=hyb.predict(t2,h); out.append(pd.DataFrame({"h":h,"true":t2.g0+t2[y],"pred":P.pred},index=t2.index))
    return pd.concat(out)
res={}
for name,sel in [("ctx_v12",SM2),("steps3h",["ctx_steps3h"])]:
    P=run(sel); P.to_csv(f"preds12_{name}.csv")
    P["day"]=P.index.normalize()
    r={h:np.sqrt(((d.true-d.pred)**2).mean()) for h,d in P.groupby("h")}
    res[name]=r; print(name,{h:round(v,2) for h,v in r.items()},flush=True)
