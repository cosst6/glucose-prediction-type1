import sys,pandas as pd,numpy as np
sys.path.insert(0,"glucose-model/v8_B_planned_watch15")
from glucoforecast import load_cgm
C=load_cgm("glucose-model/v3/cgm_export_29-10-2026.csv");C=C[C.index>="2026-08-27"]
cgm=pd.concat([s.g.rolling(3,center=True,min_periods=1).mean() for _,s in C.groupby("seg")]).dropna()
R=pd.read_csv("../meals_timing_libre.csv",parse_dates=["t"]);H=lambda m:pd.Timedelta(minutes=m)
def onset(r):
    w=cgm[r.t:r.t+H(r.win_min)]; x=w[w>=r.pre+20]
    return (x.index[0]-r.t).total_seconds()/60 if len(x) else np.nan
R["on20"]=R.apply(onset,axis=1); R["dip"]=R["drop"]>=20; R["low55"]=R.nadir<55
R.to_csv("../meals_timing_libre.csv",index=False)
W=R[R.cat=="dose WITH food (0-9 min)"]
print(W.groupby("fam").agg(n=("rise","size"),rise30=("rise",lambda x:(x>=30).sum()),on20=("on20","median"),dip=("dip","sum"),low55=("low55","sum")))
m=W[W.fam=="milk only"]; m["d"]=np.where(m.dose<=1,"1u","2u+")
print(m.groupby("d").agg(n=("rise","size"),rise=("rise","median"),pre=("pre","median"),peak=("peak_min","median")))
# correlation rise vs pre
print(W.groupby("fam").apply(lambda d:d.rise.corr(d.pre)).round(2))
print(R[R.cat=="no insulin"].groupby("fam").agg(n=("rise","size"),on20=("on20","median"),peak=("peak_min","median")))
