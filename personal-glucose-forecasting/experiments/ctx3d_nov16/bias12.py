import numpy as np, pandas as pd
from data import *
P=pd.read_csv("preds12_v10.csv",index_col=0,parse_dates=True)
L1=pd.read_csv("/home/user/work/D/log.csv",parse_dates=["time"]); st=set(L1.dropna(subset=["bg"]).time.dt.round("15min"))
rm=lambda v: np.sqrt(np.mean(np.square(v)))
for h in (60,120):
    d=P[P.h==h].sort_index(); d["res"]=d.true-d.pred; d["mat"]=d.index+pd.Timedelta(minutes=h)
    for mode in ("all15","sticks"):
        src=d if mode=="all15" else d[d.index.isin(st)]
        for K in (2,5,10):
            for w in (0.5,1.0):
                corr=[]
                for t in d.index:
                    r=src[(src.mat<=t)&(src.mat>t-pd.Timedelta("72h"))].res
                    corr.append(w*r.sum()/(len(r)+K))
                c=np.array(corr); print(h,mode,"K",K,"w",w,"old",round(rm(d.res),2),"new",round(rm(d.res-c),2))
