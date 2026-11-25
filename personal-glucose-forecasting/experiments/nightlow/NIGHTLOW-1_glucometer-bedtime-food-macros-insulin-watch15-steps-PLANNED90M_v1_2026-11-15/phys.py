import numpy as np, pandas as pd
from scipy.optimize import lsq_linear
from scipy.stats import norm
H=range(1,10)
# signs: intercept free, g_bed-120 (<=0 mean reversion), iob (<=0), carbs4h+snack (>=0), milk_bed (>=0), fpu (>=0), hr15-70 free small, steps3h free
COLS=["one","gdev","iob","carbs","milk","fpu","hr","steps"]
LB=[-np.inf,-1,-np.inf,0,0,0,-np.inf,-np.inf]; UB=[np.inf,0,0,np.inf,np.inf,np.inf,np.inf,np.inf]
def design(D):
    X=pd.DataFrame({"one":1.0,"gdev":D.g_bed-120,"iob":D.iob.clip(upper=5.5),"carbs":(D.carbs4h).clip(upper=80),"milk":D.milk_bed.clip(upper=500)/250,
        "fpu":D.fpu.clip(upper=8),"hr":(D.hr15-70).clip(-15,25)/10,"steps":D.steps3h},index=D.index); return X
LAM=dict(hr=30.,steps=30.)  # shrink watch terms (ridge rows)
def fit(tr):
    M={}; X=design(tr)
    for h in H:
        y=(tr[f"g{h}"]-tr.g_bed); ok=y.notna(); A=X[ok].values; b=y[ok].values
        R=np.zeros((len(LAM),len(COLS)));
        for j,(c,l) in enumerate(LAM.items()): R[j,COLS.index(c)]=np.sqrt(l)
        sol=lsq_linear(np.vstack([A,R]),np.r_[b,np.zeros(len(LAM))],bounds=(LB,UB)).x
        r=b-A@sol; M[h]=(sol,float(np.sqrt((r**2).sum()/max(ok.sum()-5,1))))
    return M
def predict(M,te,thr):
    x=design(te.to_frame().T if isinstance(te,pd.Series) else te).astype(float).values[0]
    p=np.array([te["g_bed"]+x@M[h][0] for h in H]); s=np.array([M[h][1] for h in H]); return p, norm.cdf((thr-p)/s)
def low_range(P,cut=0.3,hi=0.7):
    """hours after bedtime: start = first hour P>cut; end = first hour P>hi (1-3h wide)."""
    a=np.where(P>cut)[0]
    if not len(a): return None
    s=a[0]+1; b=np.where(P>hi)[0]; e=(b[0]+1) if len(b) else s+2
    e=min(max(e,s+1),s+3,9); return (s-0.5, e)
