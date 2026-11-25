import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import glucoforecast8 as g8, glucoforecast9 as G
from glucoforecast9 import Hybrid
X=pd.read_pickle("/home/user/work/lc/X10.pkl"); g8.feature_list4=G.feature_list9; hs=[60,120]
X=X[~X.index.normalize().isin(pd.to_datetime(["2026-08-29","2026-08-30","2026-08-31","2026-09-01"]))]
X["rare"]=((X.y120>=50)|(X.y120<=-50)).astype(int)
day=X.index.normalize(); rate=X.groupby(day).rare.mean(); print(rate.round(2).to_dict()); days=rate.index.values
rng=np.random.default_rng(5); out=[]
for s in range(30):
    te=rng.choice(days,3,replace=False); pool=[d for d in days if d not in te]; rng.shuffle(pool)
    base=pool[:5]; rest=sorted(pool[6:],key=lambda d: rate[d])
    sets={"base 6":base,"+4 ordinary":base+rest[:4],"+4 varied":base+rest[-4:]}
    Te=X[day.isin(te)]; c=G.feature_list9("B",120); Te=Te.dropna(subset=c+["y120"])
    for nm,ds in sets.items():
        tr=X[day.isin(ds)]; tr=tr[~(tr.index+pd.Timedelta("2h")).normalize().isin(te)]
        P=Hybrid().fit(tr,"B",hs).predict(Te,120).pred; e=(Te.g0+Te.y120-P)
        out.append(dict(split=s,set=nm,all=np.sqrt((e**2).mean()),rare=np.sqrt((e[Te.rare==1]**2).mean()),ordinary=np.sqrt((e[Te.rare==0]**2).mean())))
    print(s,flush=True)
R=pd.DataFrame(out); R.to_csv("/home/user/work/lc/variety.csv",index=False)
print(R.groupby("set")[["all","rare","ordinary"]].mean().round(1))
d=R.pivot(index="split",columns="set",values="rare"); print("varied better on rare in",( d["+4 varied"]<d["+4 ordinary"]).mean())
d=R.pivot(index="split",columns="set",values="all"); print("varied better overall in",( d["+4 varied"]<d["+4 ordinary"]).mean())
