import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge
SC="/tmp/claude-0/-home-user/9e2be325-d188-5d57-bb7b-ba9795152036/scratchpad"
COLS=["g_bed","iob","carbs4h","fpu","milk_bed","hr15","steps3h"]; A=3; H=range(1,10)
def load():
    D=pd.read_csv(f"{SC}/w/nights_libre.csv",parse_dates=["bed"]); return prep(D)
def prep(D):
    D=D.copy(); D["fpu"]=(9*D.fat4h+4*D.prot4h)/100; D["hr15"]=D.hr15.fillna(70); D["steps3h"]=np.log1p(D.steps3h.fillna(0)); return D
def fit(tr):
    M={}
    for h in H:
        y=tr[f"g{h}"]-tr.g_bed; ok=y.notna(); X=tr.loc[ok,COLS]; mu,sd=X.mean(),X.std()+1e-9
        m=Ridge(alpha=A).fit((X-mu)/sd,y[ok]); r=y[ok]-m.predict((X-mu)/sd)
        M[h]=(m,mu,sd,float(np.sqrt((r**2).sum()/max(ok.sum()-4,1))))
    return M
def predict(M,te,thr):
    p=[];P=[]
    for h in H:
        m,mu,sd,s=M[h]; g=float(te["g_bed"]+m.predict(((te[COLS].astype(float)-mu)/sd).to_frame().T.values)[0]); p.append(g); P.append(norm.cdf((thr-g)/s))
    return np.array(p),np.array(P)
