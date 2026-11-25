"""Fit SPIKE-2 v2 (ranges) and RISE40 on all meals; save joblibs + cards; write Kotlin params + golden test values."""
import numpy as np, pandas as pd, joblib, json, os, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
T = pd.read_pickle("meals_r.pkl"); R = pd.read_csv("ranges_results.csv")
T["y30"] = (T.rise >= 30).astype(int); T["y60"] = (T.rise >= 60).astype(int)
TIME = {"none": ["sug_share", "starch", "fiber", "fpu", "milk", "dose", "uncov", "iob"],
        "pre": ["sug_share", "starch", "fiber", "fpu", "milk", "dose", "uncov", "iob", "night", "hr15", "steps60", "pre"]}
SIZE = {"none": ["uncov", "iob", "steps60", "hr15"], "pre": ["uncov", "iob", "steps60", "hr15", "prec", "low"]}
LATE = ["sug_share", "starch", "fiber", "fpu", "milk", "dose", "uncov", "iob"]
RG = lambda: make_pipeline(StandardScaler(), Ridge(alpha=3))
LR = lambda **k: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, **k))
def fac(model, what, mode):
    r = R[(R.model.str.startswith(model)) & (R.what == what) & (R["mode"] == mode) & (R.target_cov == 0.7)].iloc[0]
    return [float(r.lo_factor), float(r.hi_factor)]
L = T[T.src == "Libre"]
M = {}
for mode in ("none", "pre"):
    S30 = L[L.rise >= 30]; S40 = L[L.rise >= 40]
    M[mode] = dict(
        p30=LR(C=1.0).fit(T[SIZE[mode]], T.y30), p60=LR(C=1.0).fit(T[SIZE[mode]], T.y60), p40=LR(C=1.0).fit(T[SIZE[mode]], T.y40),
        s2_start=RG().fit(S30[TIME[mode]], np.log(S30.t30_min)), s2_peak=RG().fit(S30[TIME[mode]], np.log(S30.peak_min)),
        late=LR(C=0.3, class_weight="balanced").fit(S30[LATE], (S30.t30_min >= 120).astype(int)),
        r_start=RG().fit(S40.dropna(subset=["t20"])[TIME[mode]], np.log(S40.dropna(subset=["t20"]).t20)),
        r_40=RG().fit(S40.dropna(subset=["t40"])[TIME[mode]], np.log(S40.dropna(subset=["t40"]).t40)),
        r_peak=RG().fit(S40[TIME[mode]], np.log(S40.peak_min)),
        f=dict(s2_start=fac("SPIKE-2", "t30_min", mode), s2_peak=fac("SPIKE-2", "peak_min", mode),
               r_start=fac("RISE40", "t20", mode), r_40=fac("RISE40", "t40", mode), r_peak=fac("RISE40", "peak_min", mode)))
os.makedirs("out2", exist_ok=True)
N2 = "SPIKE-2_rise-size-and-timing-RANGES_food-insulin-watch15-steps-PLANNED_v2_2026-11-15"
N4 = "RISE40-ALERT_rise-40plus-start-peak-RANGES_food-insulin-watch15-steps-PLANNED_v1_2026-11-15"
for n in (N2, N4): os.makedirs(f"out2/{n}", exist_ok=True)
joblib.dump(dict(models=M, SIZE=SIZE, TIME=TIME, LATE=LATE, coverage=0.7, carbs_per_unit=8), f"out2/{N2}/{N2}.joblib")
joblib.dump(dict(models=M, SIZE=SIZE, TIME=TIME, coverage=0.7, carbs_per_unit=8), f"out2/{N4}/{N4}.joblib")
# ---------- Kotlin export
def kt_log(p, cols):
    sc, lr = p[0], p[-1]
    return f'Logistic(arrayOf({", ".join(chr(34)+c+chr(34) for c in cols)}), doubleArrayOf({", ".join(map(repr, sc.mean_.tolist()))}), doubleArrayOf({", ".join(map(repr, sc.scale_.tolist()))}), doubleArrayOf({", ".join(map(repr, lr.coef_[0].tolist()))}), {float(lr.intercept_[0])!r})'
def kt_rg(p, cols):
    sc, rg = p[0], p[-1]
    return f'Linear(arrayOf({", ".join(chr(34)+c+chr(34) for c in cols)}), doubleArrayOf({", ".join(map(repr, sc.mean_.tolist()))}), doubleArrayOf({", ".join(map(repr, sc.scale_.tolist()))}), doubleArrayOf({", ".join(map(repr, rg.coef_.tolist()))}), {float(rg.intercept_)!r})'
out = ["package com.example.glucosealarms", "", "// AUTO-GENERATED (export2.py, 2026-11-15) from the trained joblib files. Do not edit by hand.",
       f"// {N2}", f"// {N4}", "object TimingParams {", f'    const val SPIKE2_ID = "{N2}"', f'    const val RISE40_ID = "{N4}"',
       "    const val COVERAGE = 0.7   // 7 of 10 real times fell inside the ranges on unseen days"]
for mode in ("none", "pre"):
    m = M[mode]; U = mode.upper()
    out.append(f"    val P40_{U} = {kt_log(m['p40'], SIZE[mode])}")
    out.append(f"    val LATE_{U} = {kt_log(m['late'], LATE)}")
    for k in ("s2_start", "s2_peak", "r_start", "r_40", "r_peak"):
        out.append(f"    val {k.upper()}_{U} = {kt_rg(m[k], TIME[mode])}")
        out.append(f"    val {k.upper()}_{U}_RANGE = doubleArrayOf({m['f'][k][0]}, {m['f'][k][1]})")
out.append("}")
open("TimingParams.kt", "w").write("\n".join(out) + "\n")
# golden values via a reference python function
def feats(sugar, starch, fiber, fat, protein, milk, dose, iob, hr15, steps60, hour, pre):
    avail = sugar + starch
    x = dict(uncov=avail - 8 * dose, iob=iob, hr15=hr15, steps60=steps60, sug_share=sugar / (avail + 1), starch=starch, fiber=fiber,
             fpu=(9 * fat + 4 * protein) / 100, milk=float(milk), dose=dose, night=float(hour >= 21 or hour < 6))
    mode = "none"
    if pre is not None: mode = "pre"; x.update(prec=float(np.clip(pre, 40, 250)), low=float(pre < 70), pre=float(pre))
    return x, mode
def ref(*a):
    x, mode = feats(*a); m = M[mode]; X = pd.DataFrame([x])
    g = lambda k, cols: float(np.exp(m[k].predict(X[cols])[0]))
    return dict(mode=mode, p40=float(m["p40"].predict_proba(X[SIZE[mode]])[0, 1]), late=float(m["late"].predict_proba(X[LATE])[0, 1]),
                **{k: g(k, TIME[mode]) for k in ("s2_start", "s2_peak", "r_start", "r_40", "r_peak")})
cases = [(5.6, 0.0, 0.0, 15.0, 8.0, 1, 0.0, 0.0, 70.0, 0.0, 23, None), (14.0, 20.0, 6.0, 30.0, 30.0, 1, 4.0, 0.5, 85.0, 400.0, 13, 110.0),
         (3.0, 3.0, 3.0, 35.0, 36.0, 0, 3.0, 1.2, 76.8, 1200.0, 21, 160.0), (12.2, 7.4, 2.6, 0.3, 1.0, 0, 0.0, 0.0, 90.0, 50.0, 10, 65.0)]
json.dump([dict(args=c, out=ref(*c)) for c in cases], open("golden2.json", "w"), indent=1)
print(open("TimingParams.kt").read()[:600]); print(json.dumps(json.load(open("golden2.json"))[1], indent=0))
