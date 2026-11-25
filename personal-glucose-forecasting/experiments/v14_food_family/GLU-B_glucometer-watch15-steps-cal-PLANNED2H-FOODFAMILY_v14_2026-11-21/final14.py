import json, joblib, numpy as np, pandas as pd, warnings, sys; warnings.filterwarnings("ignore")
sys.argv = ["x", "lit"]
exec(open("exp14.py").read().split("v10 = joblib")[0])   # loads data + refam with lit shapes
F.set_shapes(SH["lit"])
Xo_ = refam(Xo, logs["logo"], SH["lit"]); X3_ = refam(X3, logs["l3"], SH["lit"])
S = pd.concat([refam(P, logs[l], SH["lit"]) for P, l in ((Po, "logo"), (P2, "l2"), (P3, "l3"))]).assign(w=5.0)
m = FamHybrid().fit(pd.concat([Xo_.assign(w=1.0), X3_.assign(w=1.0), S]), "B", [60, 120])
m.range_scale = {60: 1.0, 120: 1.0}; m.plan_min = 120; m.fiber_mode = "split"; m.exercise = True; m.family_shapes = SH["lit"]; m.family_map = F.FAMILY
N = "GLU-B_glucometer-watch15-steps-cal-PLANNED2H-FOODFAMILY_v14_2026-11-21"
import os; os.makedirs(f"../out/{N}", exist_ok=True)
joblib.dump(m, f"../out/{N}/{N}.joblib", compress=3)
json.dump({"model_id": N, "trained": "2026-11-21", "base": "GLU-B v10", "cgm_rows": int(len(Xo_) + len(X3_)), "fingerstick_pairs": int(len(S)), "fingerstick_weight": 5,
  "horizons_min": [60, 120], "planned_meal_window_min": 120, "families": F.FAMILY, "family_shapes_min_peakfrac": SH["lit"], "body_model_params": m.body.params()},
  open(f"../out/{N}/{N}_card.json", "w"), indent=1)
print(m.body.params())
