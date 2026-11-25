import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from glucoforecast import load_cgm, insulin_cum
from watch7 import load_watch7
from parse_meal import parse_meal
R=0.72  # Libre ~= 0.72 x blood (Nov 14, 37 pairs)
SNACK=90  # min after bedtime: bedtime snack / dose counted as planned input
def prep_log(L):
    L=L.copy(); L["time"]=pd.to_datetime(L.time)
    for c in ["rapid_u","lantus_u","bg"]: L[c]=pd.to_numeric(L[c],errors="coerce")
    tot=[parse_meal(f)[1] if isinstance(f,str) else dict(carbs=0,protein=0,fat=0,fiber=0) for f in L.food]
    for k in ["carbs","protein","fat","fiber"]: L[k]=[t[k] for t in tot]
    L["net"]=(L.carbs-L.fiber).clip(lower=0); L["milk_ml"]=[_milk(f) for f in L.food]
    return L.sort_values("time").reset_index(drop=True)
def _milk(f):
    import re
    if not isinstance(f,str): return 0.
    m=re.findall(r"(\d+)\s*ml[^;]*milk",f); return float(sum(map(float,m)))
def bed_features(t, L, watch, g_bed, trend=np.nan):
    """t = bedtime (Lantus time). g_bed = blood-scale reading. Uses food/insulin up to t+SNACK (planned)."""
    m=(t-L.time).dt.total_seconds()/60  # minutes since event (neg = future)
    known=(m>=-SNACK)
    ins=L.rapid_u.fillna(0)
    # NovoRapid still to act after bedtime (on board + planned): 1 - fraction absorbed by t
    iob=float((ins*(1-insulin_cum(m.clip(lower=0).where(m>=0,0))) )[known&(m>-SNACK)&(m<300)].sum())
    win=known&(m<=240); snack=(m<0)&known
    f=dict(g_bed=g_bed, iob=iob,
           carbs4h=float(L.net[win].sum()), prot4h=float(L.protein[win].sum()), fat4h=float(L.fat[win].sum()),
           milk_bed=float(L.milk_ml[(m<=60)&known].sum()), snack_carbs=float(L.net[snack].sum()),
           hrs_since_meal=float((m[(L.food.notna())&(m>=0)].min() if ((L.food.notna())&(m>=0)).any() else 600)/60),
           bed_hour=(t.hour+t.minute/60+12)%24-12)  # 23:00 -> -1, 01:00 -> 1
    if watch is not None:
        w=watch[t-pd.Timedelta("3h"):t]; w15=watch[t-pd.Timedelta("15min"):t]
        f["hr15"]=float(w15.hr.mean()) if w15.hr.notna().any() else np.nan
        f["steps3h"]=float(w.steps.sum()) if len(w) else np.nan
        wd=watch[t-pd.Timedelta("16h"):t]; f["steps_day"]=float(wd.steps.sum()) if len(wd) else np.nan
    return f
