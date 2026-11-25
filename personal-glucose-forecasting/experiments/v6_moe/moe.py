"""Mixture-of-experts test: manager (gate) + spike expert + stable expert, vs current hybrid.
usage: python moe.py MODEL PLAN   (MODEL A or B, PLAN 0 or 60)
Same day-by-day walk-forward as eval5 (train on all data before the test day)."""
import os, sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler
from glucoforecast5 import *
from glucoforecast4 import gbm
MODEL, PLAN = sys.argv[1], int(sys.argv[2]); HS = [60, 120]; THR = {60: 40, 120: 60}   # "fast" = moves >=40 in 1h / >=60 in 2h
pk = f"X5_p{PLAN}.pkl"
log = load_log5("log.csv")
if os.path.exists(pk): X = pd.read_pickle(pk)
else:
    cgm = load_cgm("cgm_export_29-10-2026.csv"); cgm = cgm[cgm.index >= "2026-08-01"]
    X = build_frame5(cgm, log, load_watch("noise hr logger"), sorted(set(HORIZONS_A) | set(HORIZONS_B)), PLAN); X.to_pickle(pk)
X = restrict_to_logged_days(X, log)

def gate_cols(h):
    c = [f"{e}_{k}" for e in EV for k in ("ob", "act30")] + [f"{e}_next{h}" for e in EV] + ["g0", "min_since_meal", "uncovered_carbs", "body"]
    return c + (["d15", "d30", "d60", "accel"] if MODEL == "A" else [])

days = [d for d in sorted(X.index.normalize().unique()) if d >= pd.Timestamp("2026-08-30")]; out = []
for day in days:
    te = X[X.index.normalize() == day]; tr = X[X.index < day - pd.Timedelta(minutes=max(HS) + PLAN)]
    body = BodyModel5().fit(tr, HS)
    for h in HS:
        cols = feature_list5(MODEL, h); y = f"y{h}"
        R = tr.dropna(subset=cols + [y]); T = te.dropna(subset=cols + [y])
        if not len(T): continue
        FR = R[cols].assign(body=body.predict(R, h)); FT = T[cols].assign(body=body.predict(T, h))
        rR = R[y] - FR.body; s = (R[y].abs() >= THR[h]).astype(float)        # s = 1 on fast/spike rows
        base = gbm().fit(FR, rR)                                               # = current v5 hybrid middle
        # manager: soft probability that the next h minutes are "fast"
        gc = gate_cols(h); gate = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)).fit(FR[gc], s)
        p = gate.predict_proba(FT[gc])[:, 1]
        # experts: spike expert sees spike rows at full weight, stable rows at 0.2; stable expert the reverse
        spk = gbm().fit(FR, rR, sample_weight=0.2 + 0.8 * s); stb = gbm().fit(FR, rR, sample_weight=1.0 - 0.8 * s)
        ps, pt, pb = spk.predict(FT), stb.predict(FT), base.predict(FT)
        # alternative: one model, spike errors weighted 4x
        wtd = gbm().fit(FR, rR, sample_weight=1 + 3 * s).predict(FT)
        g0, b = T.g0.values, FT.body.values; sT = (T[y].abs() >= THR[h]).values.astype(float)
        out.append(pd.DataFrame({"h": h, "true": T.g0 + T[y], "g0": g0, "persist": g0,
            "hybrid": g0 + b + pb,
            "moe_soft": g0 + b + p * ps + (1 - p) * pt,
            "moe_hard": g0 + b + np.where(p >= 0.5, ps, pt),
            "moe_oracle": g0 + b + np.where(sT == 1, ps, pt),            # manager that is always right (upper bound)
            "weighted4x": g0 + b + wtd,
            "gate_p": p, "is_fast": sT}, index=T.index))
    print(MODEL, PLAN, day.date(), flush=True)
pd.concat(out).to_csv(f"moe_{MODEL}_p{PLAN}.csv")
