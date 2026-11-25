import json, joblib, pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from glucoforecast4 import *
X = pd.read_pickle("X4.pkl"); log = load_log4("log.csv"); K = json.load(open("range_scale4.json"))
logged = list(set(log.time.dt.normalize()))
names = {"A": "GLU-A_cgm-watch-food-GI_v4_2026-10-29", "B": "GLU-B_glucometer-watch-food-GI_v4_2026-10-29", "C": "GLU-C_glucometer-food-GI_v2_2026-10-29"}
sets = {"A": (restrict_to_logged_days(X, log), HORIZONS_A), "B": (restrict_to_logged_days(X, log), HORIZONS_B), "C": (X[X.index.normalize().isin(logged)], HORIZONS_C)}
R = pd.read_csv("results4.csv")
for M, (Xm, hs) in sets.items():
    hyb = Hybrid().fit(Xm, M, hs, use_watch=(M != "C")); hyb.range_scale = {int(h): v for h, v in K[M].items()}
    joblib.dump(hyb, f"{names[M]}.joblib", compress=3)
    card = {"model_id": names[M], "trained": "2026-10-29", "rows": int(len(Xm)), "horizons_min": hs,
            "body_model_params": hyb.body.params(), "range_scale_for_80pct": hyb.range_scale,
            "test_rmse_mgdl": {int(r.h): {"v4": r.hybrid, "previous": r.v3, "no_change": r.persist} for r in R[R.model == M].itertuples()},
            "features_per_horizon": {int(h): feature_list4(M, h, M != "C") + ["body"] for h in hs}}
    json.dump(card, open(f"{names[M]}_card.json", "w"), indent=1)
    print(M, hyb.body.params())
