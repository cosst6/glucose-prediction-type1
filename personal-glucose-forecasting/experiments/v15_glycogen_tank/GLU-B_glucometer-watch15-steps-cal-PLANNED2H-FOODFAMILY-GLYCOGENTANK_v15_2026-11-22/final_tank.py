import json, os, joblib, numpy as np, pandas as pd, warnings, sys; warnings.filterwarnings("ignore")
sys.argv = ["x", "lit"]
exec(open("exp14.py").read().split("v10 = joblib")[0])
F.set_shapes(SH["lit"])
from tank import *
from watch7 import load_watch7 as LW
W = {"logo": LW("../D/noise hr logger"), "l2": LW("../D/nov_watch"), "l3": LW("../D/set3_watch")}
Xo_ = add_tank(refam(Xo, logs["logo"], SH["lit"]), logs["logo"], W["logo"]); X3_ = add_tank(refam(X3, logs["l3"], SH["lit"]), logs["l3"], W["l3"])
S = pd.concat([add_tank(refam(P, logs[l], SH["lit"]), logs[l], W[l]) for P, l in ((Po, "logo"), (P2, "l2"), (P3, "l3"))]).assign(w=5.0)
m = TankFamHybrid().fit(pd.concat([Xo_.assign(w=1.0), X3_.assign(w=1.0), S]), "B", [60, 120])
m.range_scale = {60: 1.0, 120: 1.0}; m.plan_min = 120; m.fiber_mode = "split"; m.exercise = True; m.family_shapes = SH["lit"]; m.family_map = F.FAMILY
m.tank = dict(liver_cap=LIV_CAP, muscle_cap=MUS_CAP, liver_start=LIV0, muscle_start=MUS0, liver_share=LIV_SHARE, muscle_share=MUS_SHARE, gng_per_g_protein=GNG, fast_drain_gph=FAST_DRAIN, fed_drain_gph=FED_DRAIN, walk_carb_frac=CARB_FRAC)
N = "GLU-B_glucometer-watch15-steps-cal-PLANNED2H-FOODFAMILY-GLYCOGENTANK_v15_2026-11-22"
os.makedirs(f"../out/{N}", exist_ok=True); joblib.dump(m, f"../out/{N}/{N}.joblib", compress=3)
json.dump({"model_id": N, "trained": "2026-11-22", "base": "GLU-B v14 food-family", "cgm_rows": int(len(Xo_) + len(X3_)), "fingerstick_pairs": int(len(S)), "fingerstick_weight": 5,
  "horizons_min": [60, 120], "planned_meal_window_min": 120, "tank_features": TANK, "tank_params": m.tank, "body_model_params": m.body.params()}, open(f"../out/{N}/{N}_card.json", "w"), indent=1)
print(m.body.params())
