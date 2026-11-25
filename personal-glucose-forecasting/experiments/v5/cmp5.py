import sys; TAG = sys.argv[1] if len(sys.argv) > 1 else ""
import numpy as np, pandas as pd, os
from glucoforecast4 import load_log4
from parse_meal import parse_meal
L = load_log4("log.csv"); F = L[L.avail.fillna(0) + L.protein.fillna(0) > 3]
milk_t = [m.time.ceil("15min") for _, m in F.iterrows() if (s := {i["food"] for i in parse_meal(m.food)[0]}) and all(x.startswith("milk") for x in s)]
dal_t = [m.time.ceil("15min") for _, m in F.iterrows() if {i["food"] for i in parse_meal(m.food)[0]} & {"dal", "moong", "peas"}]
def rm(e): return np.sqrt(np.mean(e**2))
def cov(d, scale):
    mid = d.hybrid; lo = mid - scale * (mid - d.lo); hi = mid + scale * (d.hi - mid); return ((d.true >= lo) & (d.true <= hi)).mean() * 100
def roll_cov(d):
    # scale chosen only from earlier days -> honest 80% range
    days = sorted(d.index.normalize().unique()); hit = []
    for i, day in enumerate(days):
        prev = d[d.index.normalize() < day]; cur = d[d.index.normalize() == day]
        s = 1.6
        if len(prev) > 100:
            for s in np.arange(1.0, 4.01, 0.05):
                if cov(prev, s) >= 80: break
        mid = cur.hybrid; lo = mid - s * (mid - cur.lo); hi = mid + s * (cur.hi - mid); hit += list((cur.true >= lo) & (cur.true <= hi))
    return 100 * np.mean(hit), s
rows = []
for M in "ABC":
    v4 = pd.read_csv(f"preds4_{M}.csv", index_col=0, parse_dates=True); v5 = pd.read_csv(f"preds5{TAG}_{M}_p0.csv", index_col=0, parse_dates=True)
    for h in sorted(v5.h.unique()):
        a = v4[v4.h == h]; b = v5[v5.h == h]; ix = a.index.intersection(b.index); a, b = a.loc[ix], b.loc[ix]
        r = dict(model=M, h=h, n=len(ix), persist=rm(a.persist - a.true), v4=rm(a.hybrid - a.true), v5=rm(b.hybrid - b.true), v5_ridge=rm(b.ridge5 - b.true))
        hi = a.true > 200; lo = a.true < 80
        r.update(v4_hi_bias=(a.hybrid - a.true)[hi].mean(), v5_hi_bias=(b.hybrid - b.true)[hi].mean(), v4_lo_bias=(a.hybrid - a.true)[lo].mean(), v5_lo_bias=(b.hybrid - b.true)[lo].mean())
        mk = ix.isin(milk_t); dk = ix.isin(dal_t)
        r.update(v4_milk=rm((a.hybrid - a.true)[mk]), v5_milk=rm((b.hybrid - b.true)[mk]), v4_milk_bias=(a.hybrid - a.true)[mk].mean(), v5_milk_bias=(b.hybrid - b.true)[mk].mean(),
                 v4_dal=rm((a.hybrid - a.true)[dk]), v5_dal=rm((b.hybrid - b.true)[dk]), n_milk=mk.sum(), n_dal=dk.sum())
        night = (ix.hour < 6); r.update(v4_night=rm((a.hybrid - a.true)[night]), v5_night=rm((b.hybrid - b.true)[night]))
        r["v4_cov_raw"] = cov(a, 1); r["v5_cov_raw"] = cov(b, 1); r["v5_cov_cal"], r["scale"] = roll_cov(b)
        for P in (60, 120):
            f = f"preds5_{M}_p{P}.csv"
            if h == P and os.path.exists(f):
                p = pd.read_csv(f, index_col=0, parse_dates=True); p = p[p.h == h]; jx = p.index.intersection(ix)
                r["plan_meal"] = rm(p.loc[jx].hybrid - p.loc[jx].true); r["plan_vs_v5_same"] = rm(b.loc[jx].hybrid - b.loc[jx].true)
        rows.append(r)
R = pd.DataFrame(rows).round(1); pd.set_option("display.width", 300); print(R.to_string(index=False)); R.to_csv(f"results5{TAG}.csv", index=False)
