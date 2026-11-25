import os, sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import Ridge; from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler
from glucoforecast5 import *
D = sys.argv[1]; PLAN = int(sys.argv[2]) if len(sys.argv) > 2 else 0; ONLY = sys.argv[3] if len(sys.argv) > 3 else "ABC"
cgm = load_cgm(glob.glob(f"{D}/*glucose*.csv")[0]); cgm = cgm[cgm.index >= "2026-08-01"]
log = load_log5(f"{D}/log.csv"); watch = load_watch(f"{D}/noise hr logger")
H = sorted(set(HORIZONS_A) | set(HORIZONS_B))
X = build_frame5(cgm, log, watch, H, PLAN); X.to_pickle(f"X5{os.environ.get('TAG','')}_p{PLAN}.pkl")
logged = list(set(log.time.dt.normalize()))
SETS = {"A": (restrict_to_logged_days(X, log), HORIZONS_A), "B": (restrict_to_logged_days(X, log), HORIZONS_B), "C": (X[X.index.normalize().isin(logged)], HORIZONS_C)}
for model in ONLY:
    Xm, hs = SETS[model]
    if PLAN: hs = [h for h in hs if h in (60, 120)]
    days = [d for d in sorted(Xm.index.normalize().unique()) if d >= pd.Timestamp("2026-08-30")]; out = []
    for day in days:
        te = Xm[Xm.index.normalize() == day]; tr = Xm[Xm.index < day - pd.Timedelta(minutes=max(hs) + PLAN)]
        hyb = Hybrid5().fit(tr, model, hs, use_watch=(model != "C"))
        risk = pd.Series(np.nan, index=te.index); ok = te.dropna(subset=hyb.spike_cols[0]).index
        if len(ok): risk[ok] = hyb.spike_risk(te.loc[ok])
        for h in hs:
            cols = feature_list5(model, h, model != "C"); y = f"y{h}"
            t2 = te.dropna(subset=cols + [y]); r2 = tr.dropna(subset=cols + [y])
            if not len(t2): continue
            lin = make_pipeline(StandardScaler(), Ridge(alpha=30)).fit(r2[cols], r2[y]); P = hyb.predict(t2, h)
            out.append(pd.DataFrame({"model": model, "h": h, "true": t2.g0 + t2[y], "persist": t2.g0, "ridge5": t2.g0 + lin.predict(t2[cols]),
                                     "body": P.body, "hybrid": P.pred, "lo": P.lo, "hi": P.hi, "risk": risk[t2.index], "rise2h": t2.rise2h}, index=t2.index))
        print(model, day.date(), flush=True)
    pd.concat(out).to_csv(f"preds5{os.environ.get('TAG','')}_{model}_p{PLAN}.csv")
