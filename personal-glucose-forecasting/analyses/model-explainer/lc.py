import os, glob, sys, numpy as np, pandas as pd, warnings, pickle; warnings.filterwarnings("ignore")
os.environ.update(WSET="new", FIB="split", EXF="1")
from glucoforecast9 import *
D = "/home/user/work/D"
cgm = load_cgm(glob.glob(f"{D}/*glucose*.csv")[0]); cgm = cgm[cgm.index >= "2026-08-01"]
log = load_log9(f"{D}/log.csv"); watch = load_watch(f"{D}/noise hr logger")
hs = [60, 120]; X = restrict_to_logged_days(build_frame9(cgm, log, watch, hs), log); X.to_pickle("X10.pkl")
days = sorted(X.index.normalize().unique()); print(len(days), [d.date() for d in days])
test = [d for d in days[-7:] if len(X[(X.index.normalize()==d)].dropna(subset=["y120"]))>20][-6:]; print([d.date() for d in test]); rows = []
for N in (2, 4, 6, 8, 10, 13):
    e = {h: {"hyb": [], "body": [], "per": []} for h in hs}
    for d in test:
        prior = [p for p in days if p < d][-N:]
        tr = X[X.index.normalize().isin(prior) & (X.index < d - pd.Timedelta(minutes=240))]
        hyb = Hybrid().fit(tr, "B", hs); te = X[X.index.normalize() == d]
        for h in hs:
            cols = feature_list4("B", h); t2 = te.dropna(subset=cols + [f"y{h}"])
            if not len(t2): continue
            P = hyb.predict(t2, h); tru = t2.g0 + t2[f"y{h}"]
            e[h]["hyb"].append(tru - P.pred); e[h]["body"].append(tru - P.body); e[h]["per"].append(t2[f"y{h}"])
    for h in hs:
        r = {k: float(np.sqrt((pd.concat(v) ** 2).mean())) for k, v in e[h].items()}
        rows.append({"days": N, "h": h, **r}); print(rows[-1], flush=True)
pd.DataFrame(rows).round(1).to_csv("lc.csv", index=False)
