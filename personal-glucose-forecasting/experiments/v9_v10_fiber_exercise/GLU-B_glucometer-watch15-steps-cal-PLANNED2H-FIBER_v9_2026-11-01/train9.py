import sys, json, os, joblib, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
name, ptag = sys.argv[1], sys.argv[2]
from glucoforecast9 import *
import glucoforecast9 as G
D = "../D"; cgm = load_cgm(glob.glob(f"{D}/*glucose*.csv")[0]); cgm = cgm[cgm.index >= "2026-08-01"]
log = load_log9(f"{D}/log.csv"); watch = load_watch(f"{D}/noise hr logger")
X = restrict_to_logged_days(build_frame9(cgm, log, watch, [60, 120]), log)
hyb = Hybrid().fit(X, "B", [60, 120])
P = pd.read_csv(f"preds_{ptag}.csv", index_col=0); sc = {}
for h in (60, 120):
    d = P[P.h == h]
    for s in np.arange(1, 4, .05):
        if (((d.true >= d.hybrid - s * (d.hybrid - d.lo)) & (d.true <= d.hybrid + s * (d.hi - d.hybrid))).mean()) >= .8: break
    sc[h] = round(float(s), 2)
hyb.range_scale = sc; hyb.plan_min = PLAN; hyb.fiber_mode = G.FIB; hyb.exercise = G.EXF
os.makedirs(f"out/{name}", exist_ok=True); joblib.dump(hyb, f"out/{name}/{name}.joblib", compress=3)
json.dump({"model_id": name, "trained": "2026-11-01", "base": "GLU-B_glucometer-watch15-steps-cal-PLANNED2H_v8_2026-11-01", "rows": int(len(X)),
           "horizons_min": [60, 120], "planned_meal_window_min": PLAN, "fiber_mode": G.FIB, "exercise_features": G.EXF, "range_scale_for_80pct": sc,
           "body_model_params": hyb.body.params(), "features_per_horizon": {h: feature_list4("B", h) for h in (60, 120)}},
          open(f"out/{name}/{name}_card.json", "w"), indent=1)
print(name, len(X), sc)
