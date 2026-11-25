import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
T = pd.read_pickle("../low_blood.pkl")
T["ins0"]=T.n_ins_nextH; T["carb0"]=T.n_carb_next; T["fpu0"]=T.n_fpu_nextH; T["iob0"]=T.n_ins_ob - T.n_ins_nextH
W=["n_hr_15","n_steps_60","n_cal_60"]
CN=["ins0","carb0","fpu0","iob0","L"]+W; CO=["o_g0","ins","carb","fpu","H","o_age"]+W
LR=lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000))
for c in ("pN","pO"): T[c]=np.nan
for d in T.day.unique():
    tr,te=T[T.day!=d],T[T.day==d]
    T.loc[te.index,"pN"]=LR().fit(tr[CN],tr.y).predict_proba(te[CN])[:,1]
    T.loc[te.index,"pO"]=LR().fit(tr[CO],tr.y).predict_proba(te[CO])[:,1]
rows=[]
for L in (30,60,90,120,180,240,300,360):
    d=T[(T.L==L)&(T.o_age<=480)]; p=d.y==1; n=d.y==0
    for name,f,ask in [("no glucose",d.pN>0.5,0*d.pN),("1 stick (<=8h old)",d.pO>0.5,1+0*d.pN),("2-step: ask stick only if risk",(d.pN>0.5)&(d.pO>0.5),(d.pN>0.5).astype(float))]:
        rows.append(dict(warn_min=L,model=name,caught=f"{int((f&p).sum())}/{int(p.sum())}",catch_pct=round(100*(f&p).sum()/p.sum()),false_alarm_pct=round(100*(f&n).sum()/n.sum()),sticks_asked_pct=round(100*ask.mean())))
R=pd.DataFrame(rows); R.to_csv("../low_alarm_results.csv",index=False); print(R.to_string(index=False))
