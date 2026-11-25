import numpy as np, pandas as pd
R=pd.read_csv("ex_bouts.csv",parse_dates=["start"]); R["spm"]=R.steps/R["min"]; P=pd.read_csv("preds8_B.csv",index_col=0,parse_dates=True)
P["res"]=P.true-P.hybrid
print("corr of glucose change with intensity:"); print(R[["d40","d60","d90","hrup","spm","min","g0"]].corr().round(2)[["d40","d60","d90"]])
# residual of v8 by forecast origin offset relative to bout start
out=[]
for _,b in R.iterrows():
    for h in (60,120):
        d=P[P.h==h]
        for off in range(-60,95,15):
            t=b.start+pd.Timedelta(minutes=off)
            if t in d.index:
                x=d.loc[[t]].iloc[0]; out.append({"start":b.start,"h":h,"off":off,"res":x.res,"hrup":b.hrup})
O=pd.DataFrame(out)
print("\nv8 mean residual (actual - forecast) by forecast time vs exercise start; + = v8 too low")
print(O.pivot_table(index="off",columns="h",values="res",aggfunc=["mean","count"]).round(1))
allr=P.groupby("h").res.agg(["mean"]).round(1); print("\nall rows mean residual:",allr.T)
# evening baseline 17-21h on non-exercise rows
ev=P[(P.index.hour>=17)&(P.index.hour<21)]; print("evening all mean residual",ev.groupby("h").res.mean().round(1).to_dict())
hi=R[R.hrup>=40].start; lo=R[R.hrup<40].start
for nm,S in (("hard (HR +40 above rest)",hi),("easy",lo)):
    s=O[O.start.isin(S)&(O.off.between(-15,30))]; print(nm, len(S), s.groupby("h").res.mean().round(1).to_dict())
