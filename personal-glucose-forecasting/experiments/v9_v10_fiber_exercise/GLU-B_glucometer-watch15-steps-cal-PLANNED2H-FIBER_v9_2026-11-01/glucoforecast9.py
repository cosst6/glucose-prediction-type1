"""GLU-B v9/v10 (2026-11-01): v8 + fiber (v9) + exercise-rise features (v10).
Fiber: per meal, fiber fraction s = fiber / (available carbs + fiber). Fiber slows carbs, so a share of the fast parts
       (free sugar, GL) moves into the slow-carb channel; plus a fiber channel and the latest meal's fiber numbers.
Exercise: walking bouts found from the watch (>=350 steps per 5 min for >=15 min, pauses <=10 min allowed). Causal: a bout
       is only "seen" once 15 min of walking has happened. Features: minutes since bout start, in-bout flag, bout minutes and
       steps so far, HR above resting in the bout, and a rise bump (peak ~45 min after start, gone by 2 h) x intensity.
Switches (env): FIB=0|split|chan|both, EXF=0|1, FIBK = share of fiber fraction moved (default 1.0).
"""
import os, numpy as np, pandas as pd
import glucoforecast8 as g8
from glucoforecast8 import *
FIB = os.environ.get("FIB", "split"); EXF = os.environ.get("EXF", "0") == "1"; FIBK = float(os.environ.get("FIBK", "1.0"))
CAP = 0.6
def load_log9(path):
    L = load_log4(path)
    s = (L.fiber / (L.avail + L.fiber).replace(0, np.nan)).fillna(0).clip(0, CAP) * FIBK
    L["fib_frac"] = s
    if FIB in ("split", "both"):
        mv_s, mv_g = L.free_sugar * s, L.gl * s
        L["free_sugar"] = L.free_sugar - mv_s; L["gl"] = L.gl - mv_g; L["slow"] = L.slow + mv_s + mv_g
    return L
if FIB in ("chan", "both"):
    g8.KERNELS4["fib"] = ("fiber", lambda m: tri_cum(m, 240, 0.4))

# ---------------------------------------------------------------- exercise
STEP5, MIN_SEEN, GAP = 350, 15, 10
def bump_cum(m):  # cumulative of a triangle 0 -> peak 45 min -> 0 at 120 min (area 1)
    m = np.clip(m, 0, 120); a = np.where(m <= 45, m**2 / 2 / 45, 45 / 2 + (m - 45) - (m - 45)**2 / 2 / 75)
    return a / 60.0
def exercise_features(idx, w):
    st5 = w.steps.resample("5min", label="right", closed="right").sum()
    hr5 = w.hr.resample("5min", label="right", closed="right").mean()
    rest = hr5.rolling("24h", min_periods=50).quantile(0.1); up = (hr5 - rest)
    T = st5.index; act = (st5 >= STEP5).values
    start = pd.Series(pd.NaT, index=T); inb = np.zeros(len(T)); bmin = np.zeros(len(T)); bst = np.zeros(len(T)); bhr = np.zeros(len(T))
    cur, last, seen = None, None, None; acc_s, acc_h, n = 0.0, 0.0, 0; ls, lmin, lst, lhr = pd.NaT, 0, 0, 0
    for i, t in enumerate(T):
        if act[i]:
            if cur is None: cur, acc_s, acc_h, n = t - pd.Timedelta("5min"), 0.0, 0.0, 0
            last = t; acc_s += st5.iloc[i]; u = up.iloc[i]; acc_h += 0 if np.isnan(u) else u; n += 1
        elif cur is not None and (t - last) > pd.Timedelta(minutes=GAP):
            cur = None
        if cur is not None and (last - cur) >= pd.Timedelta(minutes=MIN_SEEN):
            ls, lmin, lst, lhr = cur, (last - cur).total_seconds() / 60, acc_s, acc_h / max(n, 1); inb[i] = 1.0 if act[i] or (t - last) <= pd.Timedelta(minutes=GAP) else 0
        start.iloc[i] = ls; bmin[i], bst[i], bhr[i] = lmin, lst, lhr
    f = pd.DataFrame(index=T)
    f["ex_min_since"] = ((pd.Series(T, index=T) - start).dt.total_seconds() / 60).fillna(1e4).values
    f["ex_in_bout"] = inb; f["ex_bout_min"] = bmin; f["ex_bout_steps"] = bst; f["ex_hrup"] = bhr
    f = f.reindex(idx, method="ffill")
    old = f.ex_min_since > 240
    for c in ("ex_bout_min", "ex_bout_steps", "ex_hrup"): f.loc[old, c] = 0
    f["ex_min_since"] = f.ex_min_since.clip(upper=240).fillna(240); f = f.fillna(0)
    inten = f.ex_hrup.clip(lower=0) / 40
    for h in (60, 120): f[f"ex_next{h}"] = inten * (bump_cum(f.ex_min_since.values + h) - bump_cum(f.ex_min_since.values)) * (~old)
    f["ex_now"] = inten * (bump_cum(f.ex_min_since.values + 15) - bump_cum(f.ex_min_since.values)) * (~old)
    return f
EXCOLS = ["ex_min_since", "ex_in_bout", "ex_bout_min", "ex_bout_steps", "ex_hrup", "ex_now"]

def build_frame9(cgm, log, watch, horizons):
    X = build_frame4(cgm, log, watch, horizons)
    # latest meal (>=5 g carbs, in last 4 h incl. planned) fiber numbers
    meals = log[log.avail >= 5]; mt = meals.time.values
    pos = np.searchsorted(mt, (X.index + pd.Timedelta(minutes=PLAN)).values, side="right") - 1
    ok = (pos >= 0) & ((X.index.values - mt[np.clip(pos, 0, None)]) < np.timedelta64(4, "h"))
    X["meal_fiber_g"] = np.where(ok, meals.fiber.values[np.clip(pos, 0, None)], 0)
    X["meal_fiber_frac"] = np.where(ok, meals.fib_frac.values[np.clip(pos, 0, None)], 0)
    if EXF: X = X.join(exercise_features(X.index, watch))
    return X

_base = g8.feature_list4
def feature_list9(model, h, use_watch=True):
    c = _base(model, h, use_watch)
    if FIB in ("chan", "both"): c = c + [f"fib_next{h}", "fib_ob", "fib_act30"]
    if FIB != "0": c = c + ["meal_fiber_g", "meal_fiber_frac"]
    if EXF: c = c + EXCOLS + [f"ex_next{h}"]
    return c
g8.feature_list4 = feature_list4 = feature_list9
EV9 = g8.EV + (["fib"] if FIB in ("chan", "both") else [])
