import glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from glucoforecast8 import *
from ex_detect import bouts
D="../D"; cgm=load_cgm(glob.glob(f"{D}/*glucose*.csv")[0]); cgm=cgm[cgm.index>="2026-08-01"]
log=load_log4(f"{D}/log.csv"); w=load_watch(f"{D}/noise hr logger")
days=set(log.time.dt.normalize())
B=bouts(w); B=B[B.start.dt.normalize().isin(days)].reset_index(drop=True)
g=cgm.g
def gat(t):
    i=g.index.get_indexer([t],method="nearest",tolerance=pd.Timedelta("8min"))[0]; return np.nan if i<0 else g.iloc[i]
rows=[]
for _,b in B.iterrows():
    g0=gat(b.start); r={"start":b.start,"min":b["min"],"steps":b.steps,"hrup":round(b.hr_above_rest,1),"g0":g0}
    for k in (20,40,60,90,120): r[f"d{k}"]=gat(b.start+pd.Timedelta(minutes=k))-g0
    # meal / insulin near start
    near=log[(log.time>b.start-pd.Timedelta("2h"))&(log.time<b.start+pd.Timedelta("2h"))]
    r["meal_before2h"]=near[(near.time<=b.start)].avail.sum(); r["meal_after2h"]=near[near.time>b.start].avail.sum()
    r["ins_before2h"]=near[near.time<=b.start].rapid_u.sum(); r["ins_after2h"]=near[near.time>b.start].rapid_u.sum()
    r["food_before"]="; ".join(near[near.time<=b.start].food.fillna("").astype(str))[:60]
    rows.append(r)
R=pd.DataFrame(rows); pd.set_option("display.width",250); pd.set_option("display.max_columns",30)
print(R.drop(columns="food_before").round(1).to_string())
R.to_csv("ex_bouts.csv",index=False)
print("\nall bouts n=",len(R)); print(R[[f"d{k}" for k in (20,40,60,90,120)]].describe().round(1))
clean=R[(R.meal_before2h<5)&(R.ins_before2h==0)]
print("\nno carbs/insulin in 2h before n=",len(clean)); print(clean[[f"d{k}" for k in (20,40,60,90,120)]].describe().round(1))
# baseline: same hour on days, random non-exercise starts
