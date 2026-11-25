import json, joblib, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from pairs import *
from whyb import WHybrid
Xo = pd.read_pickle("../Xo.pkl"); Xo["w"] = 1.0
S = pd.concat([pd.read_pickle(f"../{n}.pkl") for n in ("Po", "P2", "P3")]); S["w"] = 5.0
m = WHybrid().fit(pd.concat([Xo, S]), "B", [60, 120]); m.range_scale = {60: 1.0, 120: 1.0}; m.plan_min = 120; m.fiber_mode = "split"; m.exercise = True
N = "GLU-B_glucometer-watch15-steps-cal-PLANNED2H-FIBER-EXERCISE-STICKx5-SET3_v13_2026-11-21"
joblib.dump(m, f"../{N}.joblib", compress=3)
json.dump({"model_id": N, "trained": "2026-11-21", "base": "GLU-B v10/v11", "cgm_rows": int(len(Xo)), "fingerstick_pairs": int(len(S)), "fingerstick_weight": 5,
  "horizons_min": [60, 120], "planned_meal_window_min": 120, "body_model_params": m.body.params()}, open(f"../{N}_card.json", "w"), indent=1)
print(m.body.params())
R = pd.read_csv("../res13.csv"); R = R[R.cfg == "new w5"]
r = lambda e: round(float(np.sqrt((e**2).mean())), 1)
for (d, h), g in R.groupby(["day", "h"]): print(d[:10], h, len(g), "new", r(g.pred-g.true), "v10", r(g.v10-g.true), "persist", r(g.g0-g.true))
