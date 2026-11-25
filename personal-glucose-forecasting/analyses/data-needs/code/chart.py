import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import roc_auc_score
O="/home/user/work/lc/"
f=lambda n,a,b: a+b/np.sqrt(n)
def fitx(n,y,targets):
    (a,b),_=curve_fit(f,n,y); return a,b,{t:f(t,a,b) for t in targets}
V=pd.read_csv(O+"v10_curve.csv"); v=V[V.h==120].groupby("ndays").rmse.agg(["mean","std"]); v1=V[V.h==60].groupby("ndays").rmse.mean()
M=pd.read_csv(O+"meal_curve.csv"); m=M.groupby("ndays")[["spike_auc30","rise_auc","spike_band_acc","rise_acc","nmeals"]].mean()
R=pd.read_csv(O+"low_curve.csv"); l=R.groupby(["ndays","rep"]).apply(lambda g: roc_auc_score(g.y,g.p)).groupby("ndays").mean()
lowsN=R.groupby("ndays").nlows.mean()
N=pd.read_csv(O+"night_curve.csv"); n=N.groupby("ndays").wake_err.apply(lambda e: np.sqrt(np.nanmean(e**2)))
out={}
out["v10 2h"]=fitx(v.index.values,v["mean"].values,[30,60,90]); out["v10 1h"]=fitx(v1.index.values,v1.values,[30,60,90])
out["SPIKE-1 AUC"]=fitx(m.index.values,m.spike_auc30.values,[30,60,90]); out["RISE40 AUC"]=fitx(m.index.values,m.rise_auc.values,[30,60,90])
out["LOW-1 AUC"]=fitx(l.index.values,l.values,[14,30,60]); out["NIGHT wake"]=fitx(n.index.values[1:],n.values[1:],[30,60,90])
for k,(a,b,t) in out.items(): print(k,"floor",round(a,2),{x:round(y,3) for x,y in t.items()})
print(v.round(1)); print(v1.round(1)); print(m.round(3)); print(l.round(3), lowsN); print(n.round(1))
fig,ax=plt.subplots(2,2,figsize=(11,8))
def panel(a,x,y,lab,fitk,xmax,ylab,title,better):
    a.plot(x,y,"o-",color="#2a6fdb",lw=2,label="measured"); A,B,_=out[fitk] if fitk else (0,0,0); xx=np.linspace(x.min(),xmax,100)
    if fitk: a.plot(xx,f(xx,A,B),"--",color="#999",label="trend (extrapolated)")
    a.set_xlim(0,xmax); a.set_title(title,fontsize=11); a.set_xlabel("days of training data" if fitk else "nights of training data"); a.set_ylabel(ylab)
    a.axvline(x.max(),color="#ddd",lw=1); a.text(0.98,0.04,better,transform=a.transAxes,ha="right",fontsize=8,color="#555"); a.grid(alpha=.3)
panel(ax[0,0],v.index.values,v["mean"].values,"","v10 2h",90,"2h error (mg/dL, vs Libre)","v10 forecast, 2h ahead","lower = better")
ax[0,0].plot(v1.index.values,v1.values,"o-",color="#e8a33d",lw=2,label="1h ahead"); A,B,_=out["v10 1h"]; xx=np.linspace(4,90,100); ax[0,0].plot(xx,f(xx,A,B),"--",color="#e8c58d"); ax[0,0].legend(fontsize=8)
panel(ax[0,1],m.index.values,m.spike_auc30.values,"","SPIKE-1 AUC",90,"AUC (0.5 = coin flip, 1 = perfect)","Meal alarms: SPIKE-1 (+30) and RISE40","higher = better")
ax[0,1].plot(m.index.values,m.rise_auc.values,"o-",color="#e8a33d",lw=2,label="RISE40"); A,B,_=out["RISE40 AUC"]; xx=np.linspace(5,90,100); ax[0,1].plot(xx,f(xx,A,B),"--",color="#e8c58d")
ax[0,1].lines[0].set_label("SPIKE-1"); ax[0,1].legend(fontsize=8)
panel(ax[1,0],l.index.values,l.values,"","LOW-1 AUC",60,"AUC","LOW-1 low alarm (fingerstick days)","higher = better")
for x,y,k in zip(l.index.values,l.values,lowsN.values): ax[1,0].annotate(f"{k:.0f} lows",(x,y),textcoords="offset points",xytext=(0,8),fontsize=8,ha="center")
panel(ax[1,1],n.index.values[1:],n.values[1:],"",None,20,"wake-up error (mg/dL, blood scale)","NIGHTLOW-1 wake-up value","lower = better")
ax[1,1].axhline(77,color="#c44",ls=":",lw=1); ax[1,1].text(4,82,"just guessing 'no change' = 77",fontsize=8,color="#c44")
plt.tight_layout(); plt.savefig("data-needs/learning-curves.png",dpi=150)
