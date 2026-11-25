import json, joblib, pandas as pd, numpy as np, warnings, os; warnings.filterwarnings("ignore")
from glucoforecast7 import *
log = load_log5("D/log.csv"); X = restrict_to_logged_days(pd.read_pickle("X7_new_p120.pkl"), log)
name = "GLU-A_cgm-watch15-steps-cal-PLANNED2H_v7_2026-10-31"
hyb = Hybrid5().fit(X, "A", [60, 120]); P = pd.read_csv("preds7_new_A_p120.csv", index_col=0, parse_dates=True)
sc = {}
for h in (60, 120):
    d = P[P.h == h]
    for s in np.arange(1, 4, .05):
        if (((d.true >= d.hybrid - s * (d.hybrid - d.lo)) & (d.true <= d.hybrid + s * (d.hi - d.hybrid))).mean()) >= .8: break
    sc[h] = round(float(s), 2)
hyb.range_scale = sc; os.makedirs("out", exist_ok=True); joblib.dump(hyb, f"out/{name}.joblib", compress=3)
card = {"model_id": name, "trained": "2026-10-31", "base": "GLU-A_cgm-watch-food-PLANNEDMEAL_v5_2026-10-30", "rows": int(len(X)), "horizons_min": [60, 120],
        "planned_meal_window_min": 120, "watch_features": WATCH, "range_scale_for_80pct": sc,
        "body_model_params": dict(zip(hyb.body.names, [round(float(v), 3) for v in hyb.body.coef])),
        "test_rmse_mgdl_walkforward_30Aug_17Sep_planned2h": {"60": {"v7": 21.2, "v5_same_setup": 21.3, "no_watch": 21.9, "v7_plus_6-24h_exercise": 21.8, "no_change": 26.1},
                                                              "120": {"v7": 28.8, "v5_same_setup": 29.0, "no_watch": 30.6, "v7_plus_6-24h_exercise": 30.1, "no_change": 38.2}},
        "features_per_horizon": {h: feature_list5("A", h) + ["body"] for h in (60, 120)}}
json.dump(card, open(f"out/{name}_card.json", "w"), indent=1); print(sc, WATCH, len(X))
