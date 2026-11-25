import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from lowfeat import *
from data import *; from d15 import log15, watch15
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
L1,L2,W1,W2,cgm=all_data()
L2=pd.concat([L2,log15(blind=False)]).sort_values("time").reset_index(drop=True); W2=pd.concat([W2,watch15()]).sort_index()
B=L2.dropna(subset=["bg"])[["time","bg"]].sort_values("time").reset_index(drop=True)
print("sticks",len(B),"lows",(B.bg<70).sum(),flush=True)
te=[]
for Lm in [60,120,180,240]:
    for _,r in B.iterrows():
        t=r.time-pd.Timedelta(minutes=Lm)
        nog=feats(pd.DatetimeIndex([t]),L2,W2,Lm).iloc[0]
        te.append(dict(target=r.time,L=Lm,y=int(r.bg<70),**{f"n_{k}":nog[k] for k in NOG}))
T=pd.DataFrame(te)
T["ins0"]=T.n_ins_nextH; T["carb0"]=T.n_carb_next; T["fpu0"]=T.n_fpu_nextH; T["iob0"]=T.n_ins_ob-T.n_ins_nextH
T["day"]=(T.target-pd.Timedelta("6h")).dt.normalize()   # night belongs to the day before
T.to_pickle("/home/user/work/lc/low.pkl")
CN=["ins0","carb0","fpu0","iob0","L","n_hr_15","n_steps_60","n_cal_60"]
LR=lambda: make_pipeline(StandardScaler(),LogisticRegression(C=0.3,class_weight="balanced",max_iter=2000))
days=np.array(sorted(T.day.unique())); print("days",len(days),T[T.y==1].groupby("day").target.nunique().to_dict())
rng=np.random.default_rng(2); rows=[]
for rep in range(15):
  for d in days:
    others=np.array([x for x in days if x!=d]); rng.shuffle(others)
    for f in (0.25,0.5,0.75,1.0):
        k=max(1,int(round(f*len(others)))); Tr=T[T.day.isin(others[:k])]; Te=T[T.day==d]
        nl=Tr[Tr.y==1].target.nunique()
        p=LR().fit(Tr[CN],Tr.y).predict_proba(Te[CN])[:,1] if nl>0 else np.full(len(Te),0.5)
        rows.append(pd.DataFrame(dict(rep=rep,frac=f,ndays=k,nlows=nl,y=Te.y.values,L=Te.L.values,p=p)))
R=pd.concat(rows); R.to_csv("/home/user/work/lc/low_curve.csv",index=False)
for f,g in R.groupby("frac"):
    aucs=[roc_auc_score(x.y,x.p) for _,x in g.groupby("rep")]
    h=g[g.L==120]; c=((h.p>.5)&(h.y==1)).sum()/(h.y==1).sum(); fa=((h.p>.5)&(h.y==0)).sum()/(h.y==0).sum()
    print(f, "days",g.ndays.mean().round(1),"lows in train",g.nlows.mean().round(1),"AUC",np.mean(aucs).round(3),"+-",np.std(aucs).round(3),"2h caught",round(c,2),"false",round(fa,2))
