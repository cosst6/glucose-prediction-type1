import os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore"); os.environ["WSET"]="new"
from glucoforecast8 import *
D="../D"; cgm=load_cgm(glob.glob(f"{D}/*glucose*.csv")[0]); cgm=cgm[cgm.index>="2026-08-01"]
log=load_log4(f"{D}/log.csv"); watch=load_watch(f"{D}/noise hr logger")
X=restrict_to_logged_days(build_frame4(cgm,log,watch,[60,120]),log)
bg=log.dropna(subset=["bg"])[["time","bg"]].sort_values("time").reset_index(drop=True); rows=[]; cache={}
for i in range(len(bg)-1):
    a,b=bg.iloc[i],bg.iloc[i+1]; gap=(b.time-a.time).total_seconds()/60
    if not (40<=gap<=150) or a.time<pd.Timestamp("2026-08-30"): continue
    day=a.time.normalize()
    if day not in cache: cache[day]=Hybrid().fit(X[X.index<day-pd.Timedelta("4h")],"B",[60,120])
    idx=pd.DatetimeIndex([a.time]); F=pd.DataFrame({"g0":[a.bg]},index=idx).join(event_features4(idx,log,[60,120])).join(time_features(idx)).join(watch_features(idx,watch))
    p={h:float(cache[day].predict(F,h).pred.iloc[0]) for h in (60,120)}
    pr=np.interp(gap,[0,60,120],[a.bg,p[60],p[120]])
    rows.append(dict(t=a.time,bg0=a.bg,gap=round(gap),true=b.bg,pred=round(pr),persist=a.bg))
R=pd.DataFrame(rows); print(R.to_string()); print(len(R),"MAE model",(R.pred-R.true).abs().mean().round(1),"persist",(R.persist-R.true).abs().mean().round(1))
R.to_csv("fingerstick_check8.csv",index=False)
