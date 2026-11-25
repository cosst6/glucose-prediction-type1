import sys, joblib, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from pairs import *
log = load_log9("../D/log_nov.csv"); watch = load_watch("../D/nov_watch")
X = make_pairs(log, watch); X.to_pickle("../pairs_nov.pkl")
m = joblib.load("glucose-model/v9_v10_fiber_exercise/GLU-B_glucometer-watch15-steps-cal-PLANNED2H-FIBER-EXERCISE_v10_2026-11-01/GLU-B_glucometer-watch15-steps-cal-PLANNED2H-FIBER-EXERCISE_v10_2026-11-01.joblib")
R = 0.72
r = lambda e: round(float(np.sqrt((e**2).mean())), 1)
for h in (60, 120):
    Z = X[X.h == h]; p = m.predict(Z, h).pred
    Zc = Z.copy(); Zc["g0"] = Z.g0 * R; pc = m.predict(Zc, h).pred / R
    Zs = Z.copy(); ps = Z.g0 + (m.predict(Z, h).pred - Z.g0) / R
    print(h, "n", len(Z), "persist", r(Z.g0 - Z.true), "v10", r(p - Z.true), "v10 cgm-scale", r(pc - Z.true), "v10 delta/0.72", r(ps - Z.true),
          "| wrong low<70 truth", int((Z.true < 70).sum()))
