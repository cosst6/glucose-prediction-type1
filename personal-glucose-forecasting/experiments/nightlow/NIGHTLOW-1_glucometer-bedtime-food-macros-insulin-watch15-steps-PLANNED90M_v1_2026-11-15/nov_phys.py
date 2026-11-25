import sys, numpy as np, pandas as pd
from feats import *; from nightmodel import load, prep; from phys import *
S=sys.argv[1]; D=load(); M=fit(D)
L=prep_log(pd.read_csv("glucose-model/v11_and_low_alarm/LOW-1_glucometer-optional-watch15-steps-cal-food-insulin-PLANNED_v1_2026-11-14/log_nov6-12.csv"))
W=load_watch7(f"{S}/raw/nov/glucometer diabetes data/heart rate steps"); out=[]
for _,r in L[L.lantus_u.notna()].iterrows():
    t=r.time; b=L[(L.bg.notna())&(abs((L.time-t).dt.total_seconds())<=40*60)]
    if not len(b): continue
    f=pd.Series(bed_features(t,L,W,float(b.bg.iloc[0]))); f=prep(f.to_frame().T).iloc[0]
    p,P=predict(M,f,70)
    ev=L[(L.time>t+pd.Timedelta("90min"))&(L.food.notna()|L.rapid_u.notna())]; end=min(ev.time.min() if len(ev) else t+pd.Timedelta("10.5h"), t+pd.Timedelta("10.5h"))
    x=L[(L.time>t+pd.Timedelta("90min"))&(L.time<=end)&L.bg.notna()]
    for _,x in x.iterrows():
        h=(x.time-t).total_seconds()/3600; hi=int(min(max(round(h),1),9))
        out.append(dict(bed=t,bed_bg=f.g_bed,milk=f.milk_bed,iob=round(f.iob,1),at_h=round(h,1),actual=x.bg,pred=round(p[hi-1]),p_low=round(P.max(),2)))
O=pd.DataFrame(out); print(O.to_string()); e=O.pred-O.actual; print("RMSE all",np.sqrt((e**2).mean()).round(1),"| without 11-06",np.sqrt((e[O.bed.dt.day!=6]**2).mean()).round(1),"| no-change",np.sqrt(((O.bed_bg-O.actual)**2).mean()).round(1))
O.to_csv(f"{S}/w/nov_check.csv",index=False)
