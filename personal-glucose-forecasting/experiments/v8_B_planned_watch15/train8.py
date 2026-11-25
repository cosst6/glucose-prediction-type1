import json, os, joblib, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
os.environ["WSET"] = "new"
from glucoforecast8 import *
D = "../D"; name = "GLU-B_glucometer-watch15-steps-cal-PLANNED2H_v8_2026-11-01"
cgm = load_cgm(glob.glob(f"{D}/*glucose*.csv")[0]); cgm = cgm[cgm.index >= "2026-08-01"]
log = load_log4(f"{D}/log.csv"); watch = load_watch(f"{D}/noise hr logger")
X = restrict_to_logged_days(build_frame4(cgm, log, watch, [60, 120]), log)
hyb = Hybrid().fit(X, "B", [60, 120])
P = pd.read_csv("../v4p7/preds4_B.csv", index_col=0); sc = {}
for h in (60, 120):
    d = P[P.h == h]
    for s in np.arange(1, 4, .05):
        if (((d.true >= d.hybrid - s * (d.hybrid - d.lo)) & (d.true <= d.hybrid + s * (d.hi - d.hybrid))).mean()) >= .8: break
    sc[h] = round(float(s), 2)
hyb.range_scale = sc; hyb.plan_min = PLAN; os.makedirs("out", exist_ok=True); joblib.dump(hyb, f"out/{name}.joblib", compress=3)
card = {"model_id": name, "trained": "2026-11-01", "base": "GLU-B_glucometer-watch-food-GI_v4_2026-10-29", "rows": int(len(X)), "horizons_min": [60, 120],
        "input_glucose": "one glucometer (fingerstick) reading; no CGM trend", "planned_meal_window_min": PLAN, "watch_features": WATCH, "range_scale_for_80pct": sc,
        "body_model_params": dict(zip(hyb.body.names, [round(float(v), 3) for v in hyb.body.coef])) if hasattr(hyb.body, "names") else None,
        "test_rmse_mgdl_walkforward_30Aug_17Sep_planned2h": {
            "60": {"v8": 20.7, "v4_same_planned_setup_old_watch": 20.7, "v5_style_B_new_watch": 21.5, "v5_style_no_watch": 22.2, "no_change": 26.2, "v4_as_published_no_planned": 22.0},
            "120": {"v8": 28.6, "v4_same_planned_setup_old_watch": 28.4, "v5_style_B_new_watch": 28.9, "v5_style_no_watch": 31.0, "no_change": 38.3, "v4_as_published_no_planned": 32.0}},
        "features_per_horizon": {h: feature_list4("B", h) for h in (60, 120)}}
json.dump(card, open(f"out/{name}_card.json", "w"), indent=1); print(sc, WATCH, len(X), card["body_model_params"])
