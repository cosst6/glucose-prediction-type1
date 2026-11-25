import sys, numpy as np, pandas as pd
from glucoforecast import *
D = sys.argv[1]
cgm = load_cgm(glob.glob(f"{D}/*glucose*.csv")[0]); cgm = cgm[cgm.index >= "2026-08-01"]
log = load_log(f"{D}/log.csv")
X = build_frame(cgm, log, None, HORIZONS_C)
X = X[X.index.normalize().isin(list(set(log.time.dt.normalize())))]  # food-logged days only; no watch needed
days = [d for d in sorted(X.index.normalize().unique()) if d >= pd.Timestamp("2026-08-30")]
rows = []
for h in HORIZONS_C:
    for variant, drop in (("C", []), ("C_no_lantus", ["lantus_ob", "h_since_lantus"])):
        cols = [c for c in feature_list("C", h) if c not in drop]; y = f"y{h}"; out = []
        for day in days:
            te = X[X.index.normalize() == day].dropna(subset=cols + [y]); tr = X[X.index < day - pd.Timedelta(minutes=h)].dropna(subset=cols + [y])
            if len(te) == 0: continue
            m = make_pipeline(StandardScaler(), Ridge(alpha=30)).fit(tr[cols], tr[y])
            out.append(pd.DataFrame({"t": te.g0 + te[y], "p": te.g0 + m.predict(te[cols]), "g0": te.g0}))
        R = pd.concat(out)
        rows.append({"h": h, "variant": variant, "n": len(R), "persistence": np.sqrt(((R.g0 - R.t) ** 2).mean()), "rmse": np.sqrt(((R.p - R.t) ** 2).mean()), "mae": (R.p - R.t).abs().mean()})
T = pd.DataFrame(rows).round(1); T["gain_%"] = (100 * (1 - T.rmse / T.persistence)).round(0); print(T.to_string()); T.to_csv("results_C.csv", index=False)
