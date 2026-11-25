import numpy as np, pandas as pd
from scipy.optimize import curve_fit
P = pd.read_csv("wf_preds.csv", index_col=0, parse_dates=True); P["e"]=P.true-P.pred
L = pd.read_csv("/home/user/work/D/log.csv", parse_dates=["time"])
from parse_meal import parse_meal
L["carbs"]=[parse_meal(f)[1]["carbs"] for f in L.food]; L["rapid_u"]=pd.to_numeric(L.rapid_u,errors="coerce").fillna(0)
meals=L[L.carbs>=5].time.values; doses=L[L.rapid_u>0]
def near(t, arr, before, after):
    return ((arr >= np.datetime64(t - pd.Timedelta(minutes=before))) & (arr <= np.datetime64(t + pd.Timedelta(minutes=after)))).any()
for h in (60,120):
    d=P[P.h==h].copy(); tot=(d.e**2).sum()
    d["low"]=d.g0<80
    d["meal"]=[near(t,meals,180,h) for t in d.index]
    corr=doses[[not near(t,meals,60,60) for t in doses.time]].time.values  # doses not near a meal = corrections
    d["corr"]=[near(t,corr,180,h) for t in d.index]
    d["night"]=(d.index.hour<6)
    cat=np.select([d.low.values.astype(bool),d["corr"].values.astype(bool),d.meal.values.astype(bool),d.night.values.astype(bool)],["starts low (<80)","correction dose","meal in window","night, no food"],"other")
    d["cat"]=cat
    g=d.groupby("cat").apply(lambda x: pd.Series({"share_rows":len(x)/len(d),"share_err":(x.e**2).sum()/tot,"rmse":np.sqrt((x.e**2).mean()),"bias":x.e.mean()}))
    print(h, "RMSE", round(np.sqrt((d.e**2).mean()),1)); print(g.round(2))
    hi=d[d.true>200]; lo=d[d.true<80]
    print(" true>200 n",len(hi),"mean miss",round(hi.e.mean(),1)," true<80 n",len(lo),"mean miss",round(lo.e.mean(),1))
    # how far forecasts move vs real
    print(" slope pred-change vs true-change", round(np.polyfit(d.true-d.g0, d.pred-d.g0,1)[0],2))
lc=pd.read_csv("lc.csv")
f=lambda n,a,b: a+b/np.sqrt(n)
for h in (60,120):
    s=lc[lc.h==h]; (a,b),_=curve_fit(f,s.days,s.hyb); print(h,"floor",round(a,1),[ (n,round(f(n,a,b),1)) for n in (13,30,60,90,180)])
