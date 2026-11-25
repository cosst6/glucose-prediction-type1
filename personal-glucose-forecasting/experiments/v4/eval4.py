import sys, glob, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import Ridge; from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler
from glucoforecast4 import *
D = sys.argv[1]
cgm = load_cgm(glob.glob(f"{D}/*glucose*.csv")[0]); cgm = cgm[cgm.index >= "2026-08-01"]
log = load_log4(f"{D}/log.csv"); watch = load_watch(f"{D}/noise hr logger")
H = sorted(set(HORIZONS_A) | set(HORIZONS_B))
X = build_frame4(cgm, log, watch, H); X.to_pickle("X4.pkl")
logged = list(set(log.time.dt.normalize()))
SETS = {"A": (restrict_to_logged_days(X, log), HORIZONS_A), "B": (restrict_to_logged_days(X, log), HORIZONS_B),
        "C": (X[X.index.normalize().isin(logged)], HORIZONS_C)}
for model, (Xm, hs) in SETS.items():
    days = [d for d in sorted(Xm.index.normalize().unique()) if d >= pd.Timestamp("2026-08-30")]
    out = []
    for day in days:
        te = Xm[Xm.index.normalize() == day]; tr = Xm[Xm.index < day - pd.Timedelta(minutes=max(hs))]
        hyb = Hybrid().fit(tr, model, hs, use_watch=(model != "C"))
        risk = pd.Series(np.nan, index=te.index); ok = te.dropna(subset=hyb.spike_cols[0]).index
        if len(ok): risk[ok] = hyb.spike_risk(te.loc[ok])
        for h in hs:
            cols = feature_list4(model, h, model != "C"); y = f"y{h}"
            t2 = te.dropna(subset=cols + [y]); r2 = tr.dropna(subset=cols + [y])
            if not len(t2): continue
            lin = make_pipeline(StandardScaler(), Ridge(alpha=30)).fit(r2[cols], r2[y])
            P = hyb.predict(t2, h)
            out.append(pd.DataFrame({"model": model, "h": h, "true": t2.g0 + t2[y], "persist": t2.g0, "ridge4": t2.g0 + lin.predict(t2[cols]),
                                     "body": P.body, "hybrid": P.pred, "lo": P.lo, "hi": P.hi, "risk": risk[t2.index], "rise2h": t2.rise2h,
                                     "d15": t2.get("d15")}, index=t2.index))
        print(model, day.date(), flush=True)
    pd.concat(out).to_csv(f"preds4_{model}.csv")
