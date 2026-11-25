import sys, numpy as np, pandas as pd
from glucoforecast import *

DATA = sys.argv[1]  # folder with glucose csv, 'noise hr logger/', and log.csv
cgm = load_cgm(glob.glob(f"{DATA}/*glucose*.csv")[0])
cgm = cgm[cgm.index >= "2026-08-01"]
log = load_log(f"{DATA}/log.csv")
watch = load_watch(f"{DATA}/noise hr logger")
H = sorted(set(HORIZONS_A) | set(HORIZONS_B))
X = restrict_to_logged_days(build_frame(cgm, log, watch, H), log)
print("training/eval points per day:"); print(X.groupby(X.index.normalize()).size().to_string())
days = [d for d in sorted(X.index.normalize().unique()) if d >= pd.Timestamp("2026-08-30")]

def walk(model, h, use_watch=True, use_events=True):
    out = []
    for day in days:
        te = X[X.index.normalize() == day]; tr = X[X.index < day - pd.Timedelta(minutes=h)]
        cols = feature_list(model, h, use_watch)
        if not use_events:
            cols = [c for c in cols if c in ("g0", "d15", "d30", "d60", "accel", "tod_s", "tod_c") or (use_watch and c in WATCH)]
        y = f"y{h}"; tr = tr.dropna(subset=cols + [y]); te = te.dropna(subset=cols + [y])
        if len(te) == 0: continue
        m = make_pipeline(StandardScaler(), Ridge(alpha=30)).fit(tr[cols], tr[y])
        out.append(pd.DataFrame({"true": te.g0 + te[y], "pred": te.g0 + m.predict(te[cols]), "persist": te.g0}, index=te.index))
    return pd.concat(out)

rmse = lambda a, b: float(np.sqrt(((a - b) ** 2).mean()))
rows = []
for model, hs in (("A", HORIZONS_A), ("B", HORIZONS_B)):
    for h in hs:
        full = walk(model, h)
        r = {"model": model, "h": h, "n": len(full), "persistence": rmse(full.persist, full.true), "full": rmse(full.pred, full.true)}
        r["no_watch"] = rmse(walk(model, h, use_watch=False).pred, full.true)
        r["no_food_insulin"] = rmse(walk(model, h, use_events=False).pred, full.true)
        good = full[full.index >= "2026-09-03"]
        r["full_sep3_6"] = rmse(good.pred, good.true); r["persist_sep3_6"] = rmse(good.persist, good.true)
        r["MAE_full"] = float((full.pred - full.true).abs().mean())
        rows.append(r); print(r, flush=True)
        full.to_csv(f"preds_{model}_{h}.csv")
R = pd.DataFrame(rows).round(1)
R["gain_%"] = (100 * (1 - R.full / R.persistence)).round(0)
R.to_csv("results.csv", index=False); print(R.to_string())
