import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import phys as P0
from scipy.optimize import lsq_linear
from scipy.stats import norm
from data import *; from ctx3d import *
L1,L2,W1,W2,cgm=all_data()
L=pd.concat([L1,L2]).sort_values('time').reset_index(drop=True); W=pd.concat([W1,W2]).sort_index()
D=pd.read_csv("glucose-model/nightlow/NIGHTLOW-1_glucometer-bedtime-food-macros-insulin-watch15-steps-PLANNED90M_v1_2026-11-15/nights_libre.csv",parse_dates=["bed"])
D["fpu"]=(9*D.fat4h+4*D.prot4h)/100; D["hr15"]=D.hr15.fillna(70); D["steps3h"]=np.log1p(D.steps3h.fillna(0))
D=pd.concat([D,context(pd.DatetimeIndex(D.bed),L,W).reset_index(drop=True)],axis=1)
if __name__=="__main__": print(D[["night","ctx_h","ctx_ins24","ctx_steps24","ctx_bg_mean","ctx_steps3h"]].round(0).to_string())
def make(extra, lam=30.):
    def design(T):
        X=P0.design(T)
        for c in extra: X[c]=(T[c]-D[c].mean())/(D[c].std()+1e-9)
        return X
    def fit(tr):
        M={}; X=design(tr); cols=list(X.columns); LB=P0.LB+[-np.inf]*len(extra); UB=P0.UB+[np.inf]*len(extra)
        lamd=dict(P0.LAM, **{c:lam for c in extra})
        for h in P0.H:
            y=(tr[f"g{h}"]-tr.g_bed); ok=y.notna(); A=X[ok].values; b=y[ok].values
            R=np.zeros((len(lamd),len(cols)))
            for j,(c,l) in enumerate(lamd.items()): R[j,cols.index(c)]=np.sqrt(l)
            sol=lsq_linear(np.vstack([A,R]),np.r_[b,np.zeros(len(lamd))],bounds=(LB,UB)).x
            r=b-A@sol; M[h]=(sol,float(np.sqrt((r**2).sum()/max(ok.sum()-5,1))))
        return M
    def pred(M,te):
        x=design(te.to_frame().T).astype(float).values[0]; return np.array([te.g_bed+x@M[h][0] for h in P0.H]), np.array([M[h][1] for h in P0.H])
    return fit,pred
def score(extra,lam=30.):
    fit,pred=make(extra,lam); e=[];we=[];res={70:[0,0,0,0],97:[0,0,0,0]}
    for i in D.index:
        te=D.loc[i]; M=fit(D.drop(i)); p,s=pred(M,te)
        gk=np.array([te[f"g{h}"] for h in P0.H],float); k=~np.isnan(gk); e+=list((p-gk)[k])
        if pd.notna(te.wake): we.append((p[7]+p[8])/2-te.wake)
        for thr in (70,97):
            P=norm.cdf((thr-p)/np.maximum(s,20)); act=te["min"]<thr; pr=P[k].max()>0.4
            r=res[thr]; r[0]+=act&pr; r[1]+=act; r[2]+=(~act)&pr; r[3]+=~act
    rm=lambda v: np.sqrt(np.mean(np.square(v)))
    return f"hourly RMSE {rm(e):.0f}, wake RMSE {rm(we):.0f} | blood<70 caught {res[70][0]}/{res[70][1]} false {res[70][2]}/{res[70][3]} | near-low(<97) caught {res[97][0]}/{res[97][1]} false {res[97][2]}/{res[97][3]}"
SM=["ctx_h","ctx_ins24","ctx_steps24","ctx_bg_mean"]
if __name__=="__main__":
  for nm,ex in [("old",[]),("+ctx all",CTX),("+ctx small",SM),("+ctx_ins24",["ctx_ins24"]),("+ctx_steps24",["ctx_steps24"]),("+ctx_bg_mean",["ctx_bg_mean"])]:
    print(nm,score(ex))
