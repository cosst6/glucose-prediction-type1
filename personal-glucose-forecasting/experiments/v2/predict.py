"""Forecast glucose from now.

Model A (CGM running):   python predict.py DATA --mode cgm --now "2026-09-05 12:00"
Model B (glucometer):    python predict.py DATA --mode fingerstick --bg 145 --now "2026-09-05 12:00"
Add what you are eating/injecting right now (it is appended to the log for this forecast only):
    --food "3 boiled egg; 250 ml milk with coffee; 1 cup moong" --insulin 4
"""
import argparse, joblib, numpy as np, pandas as pd
from glucoforecast import *

ap = argparse.ArgumentParser()
ap.add_argument("data"); ap.add_argument("--mode", choices=["cgm", "fingerstick"], required=True)
ap.add_argument("--now", required=True); ap.add_argument("--bg", type=float)
ap.add_argument("--food", default=""); ap.add_argument("--insulin", type=float, default=0)
ap.add_argument("--model-file", help="a single saved model (.json from Drive, or .joblib); default models.joblib")
a = ap.parse_args()
now = pd.Timestamp(a.now).round("min")
key0 = "A" if a.mode == "cgm" else "B"
if a.model_file and a.model_file.endswith(".json"):
    models = {key0: load_json_model(a.model_file)}
elif a.model_file:
    models = {key0: joblib.load(a.model_file)}
else:
    models = joblib.load("models.joblib")
log = load_log(f"{a.data}/log.csv")
if a.food or a.insulin:
    extra = pd.DataFrame([{"time": now, "rapid_u": a.insulin, "lantus_u": np.nan, "bg": a.bg, "cgm_noted": np.nan, "food": a.food}])
    log = pd.concat([log, extra], ignore_index=True)
    items, tot = parse_meal(a.food)
    for i in items: print(f"  {i['item']:<45} -> {i['food']:<10} {i['grams'] or '':>6} g  carbs {i['carbs']:>5}  protein {i['protein']:>5}  fat {i['fat']:>5}")
    log = log.assign(**{k: [parse_meal(f)[1][k] for f in log.food] for k in ["carbs", "protein", "fat", "fiber"]})
    log["net_carbs"] = (log.carbs - 0.5 * log.fiber).clip(lower=0); log["fpu"] = (9 * log.fat + 4 * log.protein) / 100
    log = log.sort_values("time").reset_index(drop=True)
try:
    watch = load_watch(f"{a.data}/noise hr logger")
except Exception:
    watch = None
idx = pd.DatetimeIndex([now])
H = HORIZONS_A if a.mode == "cgm" else HORIZONS_B
if a.mode == "cgm":
    cgm = load_cgm(glob.glob(f"{a.data}/*glucose*.csv")[0])
    hist = cgm.loc[:now, "g"].dropna()
    if len(hist) < 5 or (now - hist.index[-1]) > pd.Timedelta("20min"):
        raise SystemExit("no recent CGM readings at --now; use --mode fingerstick")
    g = hist.iloc[-5:].values
    F = pd.DataFrame({"g0": [g[-1]], "d15": [g[-1] - g[-2]], "d30": [g[-1] - g[-3]], "d60": [g[-1] - g[-5]],
                      "accel": [(g[-1] - g[-2]) - (g[-2] - g[-3])]}, index=idx)
else:
    if a.bg is None: raise SystemExit("--bg is required in fingerstick mode")
    F = pd.DataFrame({"g0": [a.bg]}, index=idx)
F = F.join(event_features(idx, log, H)).join(time_features(idx)).join(watch_features(idx, watch))
for c in WATCH: F[c] = F[c].fillna(0 if c != "hr_30" else 70) if c in F else (0 if c != "hr_30" else 70)
key = "A" if a.mode == "cgm" else "B"
print(f"\nnow {now}  start {F.g0.iloc[0]:.0f} mg/dL  insulin on board {F.ins_ob.iloc[0]:.1f} u  carbs on board {F.carb_ob.iloc[0]:.0f} g")
for h in H:
    p = float(np.clip(predict(models[key][h], F).iloc[0], 40, 400))
    print(f"  +{h:>3} min  {(now + pd.Timedelta(minutes=h)).strftime('%H:%M')}  {p:6.0f} mg/dL")
