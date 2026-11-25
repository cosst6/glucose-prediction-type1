import json, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score as AUC
from data import *; from ctx3d import *
L1,L2,W1,W2,cgm=all_data()
L=pd.concat([L1,L2]).sort_values('time').reset_index(drop=True); W=pd.concat([W1,W2]).sort_index()
T=pd.DataFrame(json.load(open('glucose-model/spike_alarm/explorer/explorer_data.json')))
T["t"]=pd.to_datetime(T.t); T["day"]=T.t.dt.normalize()
C=context(T.t,L,W).reset_index(drop=True); T=pd.concat([T,C],axis=1)
T["y30"]=(T.rise>=30).astype(int); T["y60"]=(T.rise>=60).astype(int); T["y40"]=(T.rise>=40).astype(int)
lib=T.src=="Libre"; T["rise_raw"]=np.where(lib,T.rise*0.72,T.rise); T["y40raw"]=(T.rise_raw>=40).astype(int)
T["prec"]=T.pre.clip(40,250); T["low"]=(T.pre<70).astype(float); T["hr15"]=T.hr15.fillna(T.hr15.median()); T["steps60"]=T.steps60.fillna(0)
#T.to_pickle("meals12.pkl")
LR=lambda: make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=3000))
def lodo(cols,y):
    p=pd.Series(index=T.index,dtype=float)
    for d in T.day.unique():
        tr,te=T.day!=d,T.day==d; p[te]=LR().fit(T.loc[tr,cols],T.loc[tr,y]).predict_proba(T.loc[te,cols])[:,1]
    return p
def rep(p,y):
    o={}
    for s in ("Libre","Glucometer"):
        m=T.src==s; o[s]=f"AUC {AUC(T[y][m],p[m]):.2f} acc {((p[m]>.5)==(T[y][m]==1)).mean()*100:.0f}% caught {int(((p>.5)&(T[y]==1)&m).sum())}/{int((T[y][m]==1).sum())} false {int(((p>.5)&(T[y]==0)&m).sum())}/{int((T[y][m]==0).sum())}"
    return o
B=["uncov","iob","steps60","hr15"]; S1=B+["prec","low"]
SM=SM2
print("meals",len(T),"ctx_h median",T.ctx_h.median(), "ctx_h<24:",(T.ctx_h<24).sum())
for nm,cols in [("SPIKE-1 step1 (pre stick) old",S1),("+ctx all",S1+CTX),("+ctx small",S1+SM),("+steps3h",S1+["ctx_steps3h"]),
                ("SPIKE-1 step0 old",B),("step0 +ctx all",B+CTX),("step0 +ctx small",B+SM)]:
    print(nm); 
    for y in ("y30","y60"): print("  ",y,rep(lodo(cols,y),y))
print("RISE40 (scored on blood-scale label for glucometer, both labels for Libre)")
for nm,cols,y in [("old",S1,"y40"),("raw Libre label",S1,"y40raw"),("raw + steps3h",S1+["ctx_steps3h"],"y40raw"),("raw + steps3h + ctx all",S1+CTX,"y40raw"),("raw + ctx small",S1+SM,"y40raw"),("old label + ctx all",S1+CTX,"y40")]:
    p=lodo(cols,y); print("  ",nm,"| vs y40raw:",rep(p,"y40raw"))
