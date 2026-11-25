import numpy as np, pandas as pd; from nightmodel import load; from phys import *
D=load()
for thr,lab in ((70,"blood<70 (Libre<50)"),(97,"Libre<70 (blood<97)")):
  print("==",lab)
  for cut in (0.2,0.3,0.5):
    tp=fn=fp=tn=0; wtp=wfp=wfn=wtn=0; rng=[]
    for i in D.index:
        te=D.loc[i]; M=fit(D.drop(i)); p,P=predict(M,te,thr)
        gk=np.array([te[f"g{h}"] for h in H],float); known=~np.isnan(gk)
        act=te["min"]<thr; pr=P[known].max()>cut
        tp+=act&pr; fn+=act&~pr; fp+=(~act)&pr; tn+=(~act)&~pr
        if act and pr:
            hrs=np.where(P>cut)[0]+1; st=te.low_start_h if thr==70 else (np.argmax(gk<thr)+1 if (gk<thr).any() else te.min_h)
            rng.append(hrs.min()-1<=st<=hrs.max()+0.5)
        if pd.notna(te.wake):
            wa=te.wake<thr; wp=max(P[7],P[8])>cut; wtp+=wa&wp; wfn+=wa&~wp; wfp+=(~wa)&wp; wtn+=(~wa)&~wp
    print(f"cut {cut}: night caught {tp}/{tp+fn}, false {fp}/{fp+tn}, range hit {sum(rng)}/{len(rng)} | wake caught {wtp}/{wtp+wfn}, false {wfp}/{wfp+wtn}")
# baseline rules
D["lowL"]=D["min"]<97
print("rule 'no milk at bed' ->", pd.crosstab(D.milk_bed==0,D.lowL).values.tolist())
print("rule 'iob>=2 & no milk' ->", pd.crosstab((D.milk_bed==0)&(D.iob>=2),D.lowL).values.tolist())
