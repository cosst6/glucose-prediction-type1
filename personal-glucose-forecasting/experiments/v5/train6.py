import json, joblib, pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from glucoforecast5 import *
log = load_log5("log.csv"); X = restrict_to_logged_days(pd.read_pickle("X5H6_p360.pkl"), log); HS = [60, 120, 180, 240, 300, 360]
name = "GLU-A_cgm-watch-food-PLANNED6H_v5_2026-10-30"; P = pd.read_csv("preds5H6_A_p360.csv", index_col=0, parse_dates=True)
hyb = Hybrid5().fit(X, "A", HS); sc = {}
for h in HS:
    d = P[P.h == h]
    for s in np.arange(1, 5, .05):
        if (((d.true >= d.hybrid - s * (d.hybrid - d.lo)) & (d.true <= d.hybrid + s * (d.hi - d.hybrid))).mean()) >= .8: break
    sc[h] = round(float(s), 2)
hyb.range_scale = sc; joblib.dump(hyb, f"{name}.joblib", compress=3)
json.dump({"model_id": name, "trained": "2026-10-30", "rows": int(len(X)), "horizons_min": HS, "planned_window_min": 360, "range_scale_for_80pct": sc,
           "test_rmse_mgdl": {"60": [21.5, 22.9], "120": [29.4, 32.0], "180": [33.1, 36.7], "240": [37.2, 41.1], "300": [39.6, 44.5], "360": [41.3, 46.0]},
           "test_rmse_note": "[with next 6h food+insulin given, without]"}, open(f"{name}_card.json", "w"), indent=1)
print(sc)
