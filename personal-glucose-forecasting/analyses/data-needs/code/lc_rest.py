import json, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score as AUC
from data import *
OUT="/home/user/work/lc/"; FR=(0.25,0.5,0.75,1.0); rng=np.random.default_rng(1)
# ---------- meals: SPIKE-1 & RISE40
T=pd.DataFrame(json.load(open('glucose-model/spike_alarm/explorer/explorer_data.json')))
T["t"]=pd.to_datetime(T.t); T["day"]=T.t.dt.normalize()
T["y30"]=(T.rise>=30).astype(int); T["y60"]=(T.rise>=60).astype(int); T["y40"]=(T.rise>=40).astype(int)
T["prec"]=T.pre.clip(40,250); T["low"]=(T.pre<70).astype(float); T["hr15"]=T.hr15.fillna(T.hr15.median()); T["steps60"]=T.steps60.fillna(0)
T["band"]=T.y30+T.y60
S1=["uncov","iob","steps60","hr15","prec","low"]
LR=lambda: make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=3000))
days=np.array(sorted(T.day.unique())); rows=[]
for s in range(40):
    te=rng.choice(days,6,replace=False); pool=np.array([d for d in days if d not in te]); rng.shuffle(pool)
    Te=T[T.day.isin(te)]
    for f in FR:
        k=int(round(f*len(pool))); Tr=T[T.day.isin(pool[:k])]
        r=dict(split=s,frac=f,ndays=k,nmeals=len(Tr))
        if Tr.y30.nunique()<2 or Tr.y60.nunique()<2 or Tr.y40.nunique()<2: continue
        p30=LR().fit(Tr[S1],Tr.y30).predict_proba(Te[S1])[:,1]; p60=LR().fit(Tr[S1],Tr.y60).predict_proba(Te[S1])[:,1]
        p40=LR().fit(Tr[S1],Tr.y40).predict_proba(Te[S1])[:,1]
        band=(p30>.5).astype(int)+(p60>.5).astype(int)
        r.update(spike_band_acc=(band==Te.band.values).mean(), spike_auc30=AUC(Te.y30,p30) if Te.y30.nunique()>1 else np.nan,
                 rise_acc=((p40>.5)==(Te.y40==1)).mean(), rise_auc=AUC(Te.y40,p40) if Te.y40.nunique()>1 else np.nan,
                 base_band=(Te.band==Tr.band.mode()[0]).mean(), base_rise=(Te.y40==int(Tr.y40.mean()>.5)).mean())
        rows.append(r)
M=pd.DataFrame(rows); M.to_csv(OUT+"meal_curve.csv",index=False)
print(M.groupby("frac")[["ndays","nmeals","spike_band_acc","spike_auc30","base_band","rise_acc","rise_auc","base_rise"]].mean().round(3),flush=True)
# ---------- NIGHTLOW-1
import phys as P0, night12 as N
from scipy.stats import norm
D=N.D; fit,pred=N.make([]); rows=[]
for i in D.index:
    others=np.array([j for j in D.index if j!=i])
    for rep in range(8):
        rng.shuffle(others)
        for f in FR:
            k=max(3,int(round(f*len(others)))); M_=fit(D.loc[others[:k]]); te=D.loc[i]; p,sd=pred(M_,te)
            gk=np.array([te[f"g{h}"] for h in P0.H],float); ok=~np.isnan(gk)
            rows.append(dict(night=i,rep=rep,frac=f,ndays=k,hour_se=float(np.mean((p-gk)[ok]**2)) if ok.any() else np.nan,
                 wake_err=(p[7]+p[8])/2-te.wake if pd.notna(te.wake) else np.nan, nochange=te.g_bed-te.wake if pd.notna(te.wake) else np.nan))
        if f==1.0: pass
Nn=pd.DataFrame(rows); Nn.to_csv(OUT+"night_curve.csv",index=False)
print(Nn.groupby("frac").agg(ndays=("ndays","mean"),hourly=("hour_se",lambda v:np.sqrt(v.mean())),wake=("wake_err",lambda v:np.sqrt((v**2).mean())),nochange=("nochange",lambda v:np.sqrt((v**2).mean()))).round(1),flush=True)
