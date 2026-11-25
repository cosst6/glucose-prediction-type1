import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from lowfeat import *
from data import *; from ctx3d import *
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
L1,L2,W1,W2,cgm=all_data()
L=pd.concat([L1,L2]).sort_values('time').reset_index(drop=True); W=pd.concat([W1,W2]).sort_index()
HS=[30,60,90,120,180,240,300,360]
B=L2.dropna(subset=["bg"])[["time","bg"]].sort_values("time").reset_index(drop=True)
te=[]
for Lm in HS:
    for _,r in B.iterrows():
        t=r.time-pd.Timedelta(minutes=Lm); prev=B[B.time<=t]
        nog=feats(pd.DatetimeIndex([t]),L2,W2,Lm).iloc[0]
        row=dict(target=r.time,t=t,L=Lm,bg=r.bg,y=int(r.bg<70),**{f"n_{k}":nog[k] for k in NOG})
        if len(prev):
            a=prev.iloc[-1]; age=(t-a.time).total_seconds()/60
            one=feats(pd.DatetimeIndex([a.time]),L2,W2,int(age+Lm),g0=np.array([a.bg])).iloc[0]
            row.update({f"o_{k}":one[k] for k in NOG},o_g0=a.bg,o_age=age)
        te.append(row)
T=pd.DataFrame(te)
C=context(pd.DatetimeIndex(T.t),L,W).reset_index(drop=True); T=pd.concat([T,C],axis=1)
T["ins0"]=T.n_ins_nextH; T["carb0"]=T.n_carb_next; T["fpu0"]=T.n_fpu_nextH; T["iob0"]=T.n_ins_ob-T.n_ins_nextH
T["ins"]=T.o_ins_nextH; T["carb"]=T.o_carb_next; T["fpu"]=T.o_fpu_nextH; T["H"]=T.o_H
T["day"]=T.target.dt.normalize(); T.to_pickle("low12.pkl")
Wc=["n_hr_15","n_steps_60","n_cal_60"]
CN=["ins0","carb0","fpu0","iob0","L"]+Wc; CO=["o_g0","ins","carb","fpu","H","o_age"]+Wc
SM=SM2; LOWC=["ctx_lows24","ctx_h_since_low"]
LR=lambda: make_pipeline(StandardScaler(),LogisticRegression(C=0.3,class_weight="balanced",max_iter=2000))
def lodo(cols,D):
    p=pd.Series(index=D.index,dtype=float)
    for d in D.day.unique():
        tr,tt=D.day!=d,D.day==d
        if D.y[tr].sum()==0: continue
        p[tt]=LR().fit(D.loc[tr,cols],D.loc[tr,"y"]).predict_proba(D.loc[tt,cols])[:,1]
    return p
print("rows",len(T),"lows",T[T.y==1].target.nunique(),"ctx_h median",T.ctx_h.median())
D1=T[T.o_g0.notna()&(T.o_age<=480)].copy()
for nm,cn,co in [("old",CN,CO),("+ctx all",CN+CTX,CO+CTX),("+ctx small",CN+SM,CO+SM),("+lows ctx",CN+LOWC,CO+LOWC),("+steps3h",CN+["ctx_steps3h"],CO+["ctx_steps3h"])]:
    pn=lodo(cn,D1); po=lodo(co,D1); s=[]
    for Lm in (60,120,180,240):
        d=D1.L==Lm; p,n=(D1.y==1)&d,(D1.y==0)&d
        s.append(f"{Lm//60 if Lm>=60 else Lm}h noG {int(((pn>.5)&p).sum())}/{int(p.sum())},{100*((pn>.5)&n).sum()/n.sum():.0f}% | 1stick {int(((po>.5)&p).sum())}/{int(p.sum())},{100*((po>.5)&n).sum()/n.sum():.0f}%")
    print(nm,"AUC noG",round(roc_auc_score(D1.y,pn.fillna(.5)),2),"1stick",round(roc_auc_score(D1.y,po.fillna(.5)),2)); print("   "," || ".join(s))
