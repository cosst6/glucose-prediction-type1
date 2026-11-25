import joblib, json, pandas as pd, warnings; warnings.filterwarnings("ignore")
from glucoforecast5 import load_log5, restrict_to_logged_days
from moe_model import MoE5
log = load_log5("log.csv")
for model, plan, name in [("A", 60, "GLU-A_cgm-watch-food-PLANNED-MOE_v6_2026-10-31"), ("A", 0, "GLU-A_cgm-watch-food-MOE_v6_2026-10-31"),
                          ("B", 0, "GLU-B_glucometer-watch-food-MOE_v6_2026-10-31")]:
    X = restrict_to_logged_days(pd.read_pickle(f"X5_p{plan}.pkl"), log); m = MoE5().fit(X, model); m.planned_meal_window_min = plan
    joblib.dump(m, f"{name}.joblib", compress=3); print(name, len(X))
