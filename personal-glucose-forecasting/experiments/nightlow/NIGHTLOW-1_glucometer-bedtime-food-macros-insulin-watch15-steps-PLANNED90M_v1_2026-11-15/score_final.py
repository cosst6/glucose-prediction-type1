import numpy as np, pandas as pd; from nightmodel import load; from phys import *
D=load()
def fit_loo_sd(tr):
    E={h:[] for h in H}
    for j in tr.index:
        t=tr.loc[j]; M=fit(tr.drop(j)); p,_=predict(M,t,70)
        for h in H:
            if pd.notna(t[f"g{h}"]): E[h].append(p[h-1]-t[f"g{h}"])
    M=fit(tr); return {h:(M[h][0],max(float(np.sqrt(np.mean(np.square(E[h])))),20.0)) for h in H}
Ms={i:fit_loo_sd(D.drop(i)) for i in D.index}
for thr in (70,97):
  for cut in (0.2,0.3,0.4):
    tp=fp=fn=tn=0; hit=[]; wid=[]; wt=[0,0,0,0]
    for i in D.index:
        te=D.loc[i]; p,P=predict(Ms[i],te,thr); gk=np.array([te[f"g{h}"] for h in H],float)
        act=te["min"]<thr; pr=P[~np.isnan(gk)].max()>cut
        tp+=act&pr; fn+=act&~pr; fp+=(~act)&pr; tn+=(~act)&~pr
        if act and pr:
            r=low_range(P,cut); st=te.low_start_h if thr==70 else ((np.argmax(gk<thr)+1) if (gk<thr).any() else te.min_h)
            hit.append(r[0]-0.5<=st<=r[1]+0.5); wid.append(r[1]-r[0])
        if pd.notna(te.wake):
            a=te.wake<thr; b=max(P[7],P[8])>cut; wt[0]+=a&b; wt[1]+=a&~b; wt[2]+=(~a)&b; wt[3]+=(~a)&~b
    print(f"thr {thr} cut {cut}: caught {tp}/{tp+fn}, false {fp}/{fp+tn}; range hit {sum(hit)}/{len(hit)} width {np.mean(wid) if wid else 0:.1f}h; wake caught {wt[0]}/{wt[0]+wt[1]} false {wt[2]}/{wt[2]+wt[3]}")
