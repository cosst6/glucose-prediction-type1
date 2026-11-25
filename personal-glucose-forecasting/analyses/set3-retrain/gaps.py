import sys, pandas as pd, numpy as np
sys.path.insert(0, "/home/user/work/code")
from parse_meal import parse_item
import re
D="/home/user/work/D"
FAM={"milk":"milk","milk_cow":"milk","milk_toned":"milk","milk_buffalo":"milk","curd":"curd","paneer":"paneer","paneer_from_milk":"paneer","egg":"egg",
 "apple":"fruit","banana":"fruit","guava":"fruit","pomegranate":"fruit","grape":"fruit","rice":"rice","roti":"roti","puri":"puri",
 "dal":"dal","moong":"dal","chickpea":"chole/chana","rajma":"rajma","soy":"soy","peas":"peas","veg":"veg","raw_veg":"veg","cucumber":"veg","pumpkin":"veg",
 "biscuit_hideseek":"biscuit","biscuit_goodday":"biscuit","chocolate":"sweet","melody":"sweet","almond":"nuts","walnut":"nuts","bhujiya":"namkeen","besan":"besan","sauce":"veg","explicit_carbs":"other","fish_oil":"nuts","UNKNOWN":"unknown"}
ev=[]
for s,f in [(1,"log.csv"),(2,"log_nov.csv"),(3,"log_set3.csv")]:
    L=pd.read_csv(f"{D}/{f}"); L["time"]=pd.to_datetime(L.time); L["set"]=s; ev.append(L)
E=pd.concat(ev).sort_values("time").reset_index(drop=True)
food=E[E.food.notna()].copy(); dose=E[E.rapid_u>0]
# group food events within 30 min into meals
food["mid"]=(food.time.diff()>pd.Timedelta("15min")).cumsum()
rows=[]
for k,g in food.groupby("mid"):
    t0=g.time.iloc[0]; items=[parse_item(x) for x in ";".join(g.food).split(";")]; items=[i for i in items if i]
    fams=sorted(set(FAM.get(i["food"],"unknown") for i in items)); carbs=sum(i["carbs"] for i in items); fat=sum(i["fat"] for i in items)
    d=dose[(dose.time>=t0-pd.Timedelta("45min"))&(dose.time<=t0+pd.Timedelta("40min"))]
    if len(d)==0: tim="no insulin"
    else:
        gap=(t0-d.time.iloc[0]).total_seconds()/60
        tim="30 min before" if gap>=25 else "15 min before" if gap>=10 else "with food" if gap>=-10 else "after food"
    # low treatment flag: glucose <75 in 20 min before
    bg=E[(E.bg.notna())&(E.time>=t0-pd.Timedelta("20min"))&(E.time<=t0+pd.Timedelta("5min"))]
    low=(bg.bg<75).any() or (E[(E.time>=t0-pd.Timedelta("25min"))&(E.time<=t0)].get("low_flag",pd.Series()).fillna(0)>0).any()
    rows.append(dict(time=t0,set=g.set.iloc[0],fams="+".join(fams),n_fam=len(fams),carbs=round(carbs),fat=round(fat),timing=tim,low_treat=low,items="; ".join(g.food)))
M=pd.DataFrame(rows); M.to_csv("/home/user/work/gaps/meals_all_sets.csv",index=False)
print("meals",len(M), M.groupby("set").size().to_dict())
fams=sorted(set(x for f in M.fams for x in f.split("+")))
T=[]
for f in fams:
    has=M[M.fams.str.split("+").apply(lambda s:f in s)]
    alone=has[has.n_fam==1]
    T.append(dict(food=f,meals=len(has),alone=len(alone),alone_not_low=int((~alone.low_treat).sum()),
      sets=",".join(map(str,sorted(has.set.unique()))),**{t:int((has.timing==t).sum()) for t in ["with food","15 min before","30 min before","after food","no insulin"]}))
T=pd.DataFrame(T).sort_values("meals",ascending=False); print(T.to_string(index=False)); T.to_csv("/home/user/work/gaps/food_counts.csv",index=False)
# pairs
from itertools import combinations
C={}
for f in M.fams:
    s=f.split("+")
    for a,b in combinations(sorted(s),2): C[(a,b)]=C.get((a,b),0)+1
print("pair counts (starch/sweet with fat-only):")
for st in ["rice","roti","puri","biscuit","fruit","dal","rajma","chole/chana","milk","sweet","namkeen"]:
    print(st, {o:C.get(tuple(sorted((st,o))),0) for o in ["paneer","egg","nuts","veg","milk","curd","rice","roti","dal"] if o!=st})
print(M.timing.value_counts())
print(M[M.set==3][["time","fams","carbs","timing","low_treat"]].to_string())
