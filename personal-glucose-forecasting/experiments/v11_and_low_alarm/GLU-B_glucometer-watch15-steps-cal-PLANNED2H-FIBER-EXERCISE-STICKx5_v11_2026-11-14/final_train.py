"""Train final v11 forecaster (v10 + fingerstick pairs x5) and LOW-1 alarm on all data; write model folders."""
import os, glob, json, joblib, shutil, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from pairs import *
from whyb import WHybrid
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
D = "../D"; OUT = "../out"
N11 = "GLU-B_glucometer-watch15-steps-cal-PLANNED2H-FIBER-EXERCISE-STICKx5_v11_2026-11-14"
NL = "LOW-1_glucometer-optional-watch15-steps-cal-food-insulin-PLANNED_v1_2026-11-14"
cgm = load_cgm(glob.glob(f"{D}/*glucose*.csv")[0]); cgm = cgm[cgm.index >= "2026-08-01"]
logo = load_log9(f"{D}/log.csv"); wo = load_watch(f"{D}/noise hr logger")
Xo = restrict_to_logged_days(build_frame9(cgm, logo, wo, [60, 120]), logo); Xo["w"] = 1.0
z = pd.read_csv("../cgm_vs_bg_pairs.csv", parse_dates=["time"])[["time", "bg"]]
Bo = pd.concat([logo.dropna(subset=["bg"])[["time", "bg"]], z]); Bo["time"] = Bo.time.dt.round("5min"); Bo = Bo.drop_duplicates("time").sort_values("time").reset_index(drop=True)
S = pd.concat([make_pairs(logo, wo, Bo), pd.read_pickle("../pairs_nov.pkl")]); S["w"] = 5.0
m = WHybrid().fit(pd.concat([Xo, S]), "B", [60, 120]); m.range_scale = {60: 1.0, 120: 1.0}; m.plan_min = 120; m.fiber_mode = "split"; m.exercise = True
os.makedirs(f"{OUT}/{N11}", exist_ok=True); joblib.dump(m, f"{OUT}/{N11}/{N11}.joblib", compress=3)
json.dump({"model_id": N11, "trained": "2026-11-14", "base": "GLU-B v10", "cgm_rows": int(len(Xo)), "fingerstick_pairs": int(len(S)), "fingerstick_weight": 5,
           "horizons_min": [60, 120], "planned_meal_window_min": 120, "body_model_params": m.body.params()}, open(f"{OUT}/{N11}/{N11}_card.json", "w"), indent=1)
# LOW-1
T = pd.read_pickle("../low_blood.pkl")
T["ins0"]=T.n_ins_nextH; T["carb0"]=T.n_carb_next; T["fpu0"]=T.n_fpu_nextH; T["iob0"]=T.n_ins_ob - T.n_ins_nextH
W=["n_hr_15","n_steps_60","n_cal_60"]; CN=["ins0","carb0","fpu0","iob0","L"]+W; CO=["o_g0","ins","carb","fpu","H","o_age"]+W
LR=lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000))
low = {"no_glucose": LR().fit(T[CN], T.y), "one_stick": LR().fit(T[CO], T.y), "cols_no_glucose": CN, "cols_one_stick": CO, "threshold": 0.5, "low_mgdl": 70}
os.makedirs(f"{OUT}/{NL}", exist_ok=True); joblib.dump(low, f"{OUT}/{NL}/{NL}.joblib", compress=3)
coef = {k: dict(zip(low[f"cols_{k}"], [round(float(v), 3) for v in low[k][-1].coef_[0]])) for k in ("no_glucose", "one_stick")}
json.dump({"model_id": NL, "trained": "2026-11-14", "rows": int(len(T)), "lows": int(T[T.y==1].target.nunique()), "standardized_coefficients": coef}, open(f"{OUT}/{NL}/{NL}_card.json", "w"), indent=1)
shutil.copy("../low_alarm_results.csv", f"{OUT}/{NL}/")
for f in ["lowfeat.py","lowexp.py","lowblood.py","lowfinal.py","pairs.py","glucoforecast.py","glucoforecast8.py","glucoforecast9.py","watch7.py","parse_meal.py","food_db.csv"]: shutil.copy(f, f"{OUT}/{NL}/")
for f in ["whyb.py","pairs.py","exp11.py","test_v10.py","final_train.py","glucoforecast.py","glucoforecast8.py","glucoforecast9.py","watch7.py","parse_meal.py","food_db.csv"]: shutil.copy(f, f"{OUT}/{N11}/")
shutil.copy("../res11.csv", f"{OUT}/{N11}/walkforward_nov_results.csv"); shutil.copy("../D/log_nov.csv", f"{OUT}/{N11}/log_nov6-12.csv"); shutil.copy("../D/log_nov.csv", f"{OUT}/{NL}/log_nov6-12.csv")
shutil.copy("../cgm_vs_bg_pairs.csv", f"{OUT}/{N11}/libre_vs_fingerstick_pairs.csv")
print(coef)
