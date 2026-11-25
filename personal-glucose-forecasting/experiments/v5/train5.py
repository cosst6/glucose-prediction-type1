import json, joblib, pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from glucoforecast5 import *
log = load_log5("log.csv"); X = restrict_to_logged_days(pd.read_pickle("X5_p60.pkl"), log)
name = "GLU-A_cgm-watch-food-PLANNEDMEAL_v5_2026-10-30"
hyb = Hybrid5().fit(X, "A", [60, 120]); P = pd.read_csv("preds5_A_p60.csv", index_col=0, parse_dates=True)
sc = {}
for h in (60, 120):
    d = P[P.h == h]
    for s in np.arange(1, 4, .05):
        if (((d.true >= d.hybrid - s * (d.hybrid - d.lo)) & (d.true <= d.hybrid + s * (d.hi - d.hybrid))).mean()) >= .8: break
    sc[h] = round(float(s), 2)
hyb.range_scale = sc; joblib.dump(hyb, f"{name}.joblib", compress=3)
card = {"model_id": name, "trained": "2026-10-30", "rows": int(len(X)), "horizons_min": [60, 120], "planned_meal_window_min": 60,
        "body_model_params": dict(zip(hyb.body.names, [round(float(v), 3) for v in hyb.body.coef])), "range_scale_for_80pct": sc,
        "test_rmse_mgdl": {"60": {"v5_planned": 21.0, "v4": 21.5, "no_change": 26.1}, "120": {"v5_planned": 29.0, "v4": 31.4, "no_change": 38.3}},
        "test_rmse_when_meal_in_next_60min": {"60": {"v5_planned": 23.8, "v4": 26.5}, "120": {"v5_planned": 34.7, "v4": 41.2}},
        "features_per_horizon": {h: feature_list5("A", h) + ["body"] for h in (60, 120)}}
json.dump(card, open(f"{name}_card.json", "w"), indent=1); print(sc, card["body_model_params"])
