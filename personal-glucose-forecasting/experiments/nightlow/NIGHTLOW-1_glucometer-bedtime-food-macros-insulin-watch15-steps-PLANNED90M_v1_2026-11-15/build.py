import sys, numpy as np, pandas as pd; from feats import *
S=sys.argv[1]
cgm=load_cgm(f"{S}/raw/libre_29oct.csv").g/R   # blood-equivalent
L=prep_log(pd.read_csv("glucose-model/v4/log.csv"))
W=load_watch7(f"{S}/raw/fms/FMS/noise hr logger")
rows=[]
for _,r in L[L.lantus_u.notna()].iterrows():
    t=r.time; tr=t.round("15min")
    if tr not in cgm.index or cgm[t:t+pd.Timedelta("9h")].size<30: continue
    # window ends at first food/dose after bedtime snack period (treating low or breakfast) or +9h
    later=L[(L.time>t+pd.Timedelta(minutes=SNACK))&(L.time<=t+pd.Timedelta("9h"))&(L.food.notna()|L.rapid_u.notna())]
    end=min(later.time.min() if len(later) else t+pd.Timedelta("9h"), t+pd.Timedelta("9h"))
    g=cgm[t:end]; f=bed_features(t,L,W,float(cgm[tr])); f["night"]=t.strftime("%Y-%m-%d"); f["bed"]=t; f["end_h"]=(end-t).total_seconds()/3600
    for h in range(1,10):
        v=cgm.get((t+pd.Timedelta(hours=h)).round("15min"),np.nan); f[f"g{h}"]=v if t+pd.Timedelta(hours=h)<=end else np.nan
    f["min"]=float(g.min()); f["min_h"]=(g.idxmin()-t).total_seconds()/3600
    lw=g[g<70]; f["low"]=int(len(lw)>0); f["low_start_h"]=(lw.index[0]-t).total_seconds()/3600 if len(lw) else np.nan
    f["low_end_h"]=(lw.index[-1]-t).total_seconds()/3600 if len(lw) else np.nan
    wk=cgm[t+pd.Timedelta("8h"):t+pd.Timedelta("9h")]; wk=wk[wk.index<=end]
    f["wake"]=float(wk.mean()) if len(wk) else np.nan
    rows.append(f)
D=pd.DataFrame(rows); D.to_csv(f"{S}/w/nights_libre.csv",index=False)
pd.set_option("display.width",250); print(D.drop(columns=["bed"]+[f"g{h}" for h in range(1,10)]).round(1).to_string())
