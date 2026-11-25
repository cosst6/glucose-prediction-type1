import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from glucoforecast9 import *
D = sys.argv[1]; tag = sys.argv[2]
cgm = load_cgm(glob.glob(f"{D}/*glucose*.csv")[0]); cgm = cgm[cgm.index >= "2026-08-01"]
log = load_log9(f"{D}/log.csv"); watch = load_watch(f"{D}/noise hr logger")
hs = [60, 120]; Xm = restrict_to_logged_days(build_frame9(cgm, log, watch, hs), log); Xm.to_pickle(f"X_{tag}.pkl")
days = [d for d in sorted(Xm.index.normalize().unique()) if d >= pd.Timestamp("2026-08-30")]; out = []
for day in days:
    te = Xm[Xm.index.normalize() == day]; tr = Xm[Xm.index < day - pd.Timedelta(minutes=max(hs) + 120)]
    hyb = Hybrid().fit(tr, "B", hs)
    for h in hs:
        cols = feature_list4("B", h); y = f"y{h}"; t2 = te.dropna(subset=cols + [y])
        if not len(t2): continue
        P = hyb.predict(t2, h)
        out.append(pd.DataFrame({"h": h, "true": t2.g0 + t2[y], "persist": t2.g0, "body": P.body, "hybrid": P.pred, "lo": P.lo, "hi": P.hi}, index=t2.index))
pd.concat(out).to_csv(f"preds_{tag}.csv"); print(tag, "done")
