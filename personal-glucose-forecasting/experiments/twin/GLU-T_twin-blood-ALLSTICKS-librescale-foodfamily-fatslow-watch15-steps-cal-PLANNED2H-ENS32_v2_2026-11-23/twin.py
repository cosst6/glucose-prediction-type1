"""GLU-T digital twin: personal physiological glucose model (blood units inside, Libre via a sensor map).

State: blood glucose G. Inputs on a 5-min grid, integrated exactly from logged event times:
  insulin  NovoRapid -> 3-compartment chain (Erlang-3, time const ti) -> action; effect scales with glucose (Bergman-like).
  carbs    per food family (fast, milk, starch, legume, veg): 2-compartment gut (Erlang-2, own time const),
           each family slowed by the meal's fat (gastric emptying).  FPU (fat/protein) own slow channel.
  body     pull to basal Gb, time-of-day drift, counter-regulation below 85.
  watch    HR above rest (15-min mean), raw steps, raw calories at forecast time -> uptake that fades.
  trend    the twin's own miss over the last readings is carried forward and fades (data assimilation).
  sensor   Libre = a * blood + b (a ~0.72, learned jointly from sticks + Libre).
Meals/insulin inside the forecast window are known (planned-meal rule).
"""
import numpy as np, pandas as pd
from numba import njit

FAMS = ["fast", "milk", "starch", "legume", "veg"]
DT = 5.0
# name: (init, lo, hi, prior_mean, prior_sd, log-prior?)
P = {
    "SI":     (35.0, 5, 120, 35.0, 0.5, 1),     # mg/dL (blood) per unit at G=120
    "ti":     (30.0, 12, 70, 30.0, 0.3, 1),     # insulin chain time const (peak ~2*ti)
    "cf_fast": (3.0, 0.2, 10, 3.0, 0.6, 1), "cf_milk": (3.0, 0.2, 10, 3.0, 0.6, 1), "cf_starch": (2.5, 0.2, 10, 2.5, 0.6, 1),
    "cf_legume": (2.0, 0.1, 10, 2.0, 0.7, 1), "cf_veg": (1.0, 0.05, 8, 1.0, 0.8, 1),
    "tau_fast": (25.0, 8, 90, 25.0, 0.4, 1), "tau_milk": (35.0, 8, 120, 35.0, 0.4, 1), "tau_starch": (50.0, 10, 180, 50.0, 0.4, 1),
    "tau_legume": (80.0, 15, 240, 80.0, 0.4, 1), "tau_veg": (60.0, 10, 240, 60.0, 0.5, 1),
    "cf_fpu": (15.0, 0.5, 60, 15.0, 0.6, 1), "tau_fpu": (120.0, 40, 400, 120.0, 0.4, 1),
    "fat_slow": (0.3, 0.0, 3.0, 0.3, 0.5, 0),   # tau multiplier per 30 g fat
    "k":      (0.002, 0.0, 0.02, 0.002, 0.004, 0),  # 1/min pull to Gb
    "Gb":     (140.0, 60, 260, 140.0, 40.0, 0),
    "dr_s":   (0.0, -40, 40, 0.0, 10.0, 0), "dr_c": (0.0, -40, 40, 0.0, 10.0, 0),  # mg/dL per h
    "dr0":    (0.0, -40, 40, 0.0, 10.0, 0),
    "beta":   (0.3, 0.0, 2.0, 0.3, 0.4, 0),
    "c_low":  (0.01, 0.0, 0.2, 0.01, 0.03, 0),
    "w_d":    (0.6, 0.0, 1.5, 0.6, 0.4, 0), "tau_d": (40.0, 5, 300, 40.0, 0.6, 1), "w_ds": (0.3, 0.0, 1.5, 0.2, 0.4, 0),
    "e_hr":   (0.0, -0.05, 0.1, 0.0, 0.02, 0), "e_st": (0.0, -0.5, 1.0, 0.0, 0.2, 0), "e_cal": (0.0, -0.5, 1.0, 0.0, 0.2, 0),
    "si_s":   (0.0, -0.6, 0.6, 0.0, 0.2, 0), "si_c": (0.0, -0.6, 0.6, 0.0, 0.2, 0),   # insulin sensitivity by time of day
    "lowfast": (0.7, 0.2, 1.5, 0.7, 0.3, 0),   # gut time multiplier for food eaten while low
    "a":      (0.72, 0.4, 1.2, 0.72, 0.08, 0), "b": (0.0, -40, 40, 0.0, 12.0, 0),
    "a2":     (0.72, 0.4, 1.2, 0.72, 0.08, 0), "b2": (0.0, -40, 40, 0.0, 12.0, 0),   # 2nd Libre sensor (from Nov 18)
}
NAMES = list(P)
IDX = {n: i for i, n in enumerate(NAMES)}
X0 = np.array([P[n][0] for n in NAMES]); LO = np.array([P[n][1] for n in NAMES]); HI = np.array([P[n][2] for n in NAMES])
PM = np.array([P[n][3] for n in NAMES]); PS = np.array([P[n][4] for n in NAMES]); PL = np.array([P[n][5] for n in NAMES], bool)

# ------------------------------------------------------------------ data -> timeline
class Timeline:
    """One continuous stretch of logged data on a 5-min grid."""
    def __init__(self, log, watch, start, end):
        self.t0 = pd.Timestamp(start).floor("5min") - pd.Timedelta("12h")
        self.T = pd.date_range(self.t0, pd.Timestamp(end).ceil("5min") + pd.Timedelta("3h"), freq="5min")
        m = lambda ts: ((pd.DatetimeIndex(ts) - self.t0).total_seconds() / 60).values.astype(float)
        L = log[(log.time >= self.t0) & (log.time <= self.T[-1])]
        B = L.dropna(subset=["rapid_u"]); B = B[B.rapid_u > 0]
        self.bt, self.bu = m(B.time), B.rapid_u.values.astype(float)
        M = L[(L[[f"c_{f}" for f in FAMS]].sum(1) + L.fpu) > 0]
        self.mt = m(M.time); self.mc = M[[f"c_{f}" for f in FAMS]].values.astype(float)
        self.mfat = M.fat.values.astype(float); self.mfpu = M.fpu.values.astype(float)
        self.mtime = pd.DatetimeIndex(M.time); self.mlow = np.zeros(len(self.mt))
        mins = self.T.hour * 60 + self.T.minute
        self.tod_s, self.tod_c = np.sin(2 * np.pi * mins / 1440).values, np.cos(2 * np.pi * mins / 1440).values
        r = lambda s, how: getattr(s.resample("5min", label="right", closed="right"), how)()
        if watch is not None and len(watch):
            hr5, st5, ca5 = r(watch.hr, "mean"), r(watch.steps, "sum"), r(watch.cal, "sum")
            rest = hr5.rolling("24h", min_periods=50).quantile(0.1)
            up = (hr5 - rest).rolling(3, min_periods=1).mean().reindex(self.T).fillna(0).clip(lower=0)   # HR 15-min mean above rest
            st = st5.rolling(3, min_periods=1).sum().reindex(self.T).fillna(0)                          # raw steps, 15 min total
            ca = ca5.rolling(3, min_periods=1).sum().reindex(self.T).fillna(0)                          # raw calories, 15 min total
            self.hrup, self.st15, self.cal15 = up.values, st.values, ca.values
        else:
            z = np.zeros(len(self.T)); self.hrup, self.st15, self.cal15 = z, z, z
        self.n = len(self.T)
    def set_low(self, readings_blood, thr=80.0, win="45min"):
        """flag meals eaten within 45 min after a low reading (blood units)."""
        r = readings_blood.sort_index()
        self.mlow = np.array([float(((r.index <= t) & (r.index >= t - pd.Timedelta(win)) & (r.values < thr)).any()) for t in self.mtime])
    def idx(self, ts): return np.round(((pd.DatetimeIndex(ts) - self.t0).total_seconds() / 60).values / DT).astype(np.int64)

@njit(cache=True)
def _erl_cum(x, n):   # Erlang-n cumulative at x = t/tau (x>=0)
    if x <= 0: return 0.0
    e = np.exp(-x)
    if n == 2: return 1.0 - (1.0 + x) * e
    return 1.0 - (1.0 + x + 0.5 * x * x) * e

@njit(cache=True)
def _rates(n, bt, bu, mt, mc, mfat, mfpu, mlow, lowfast, ti, cf, tau, fat_slow, cf_fpu, tau_fpu):
    ins = np.zeros(n); ra = np.zeros(n)
    for e in range(len(bt)):
        j0 = int(bt[e] // DT);
        for j in range(max(j0, 0), min(n, j0 + 120)):   # 10 h
            t = j * DT - bt[e]
            ins[j] += bu[e] * (_erl_cum((t + DT) / ti, 3) - _erl_cum(t / ti, 3))
    for e in range(len(mt)):
        j0 = int(mt[e] // DT); slow = 1.0 + fat_slow * mfat[e] / 30.0
        if mlow[e] > 0: slow = slow * lowfast
        for j in range(max(j0, 0), min(n, j0 + 144)):  # 12 h
            t = j * DT - mt[e]; s = 0.0
            for f in range(5):
                if mc[e, f] > 0:
                    tf = tau[f] * slow
                    s += mc[e, f] * cf[f] * (_erl_cum((t + DT) / tf, 2) - _erl_cum(t / tf, 2))
            if mfpu[e] > 0:
                s += mfpu[e] * cf_fpu * (_erl_cum((t + DT) / tau_fpu, 2) - _erl_cum(t / tau_fpu, 2))
            ra[j] += s
    return ins, ra

@njit(cache=True)
def _sim(G, i0, nstep, ins, ra, drift, SI, k, Gb, beta, c_low, d0, tau_d, ex0):
    for s in range(nstep):
        j = i0 + s
        f = 1.0 + beta * (G - 120.0) / 120.0
        if f < 0.3: f = 0.3
        t = s * DT
        dist = d0 * tau_d * (np.exp(-t / tau_d) - np.exp(-(t + DT) / tau_d))
        ex = ex0 * 45.0 * (np.exp(-t / 45.0) - np.exp(-(t + DT) / 45.0))
        low = c_low * (85.0 - G) * DT if G < 85.0 else 0.0
        G = G - k * (G - Gb) * DT + ra[j] - SI * ins[j] * f + drift[j] * DT / 60.0 + dist - ex + low
        if G < 20.0: G = 20.0
    return G

@njit(cache=True)
def _forecast(ins, ra, drift, hrup, st15, cal15, A_i0, A_n, A_g0, A_lib, A_pi, A_pg, A_plib, th, A_olib):
    SI, k, Gb, beta, c_low = th[0], th[1], th[2], th[3], th[4]
    w_d, tau_d, w_ds, e_hr, e_st, e_cal, a, b = th[5], th[6], th[7], th[8], th[9], th[10], th[11], th[12]
    out = np.empty(len(A_i0)); dout = np.empty(len(A_i0))
    for q in range(len(A_i0)):
        i0 = A_i0[q]
        G0 = (A_g0[q] - b) / a if A_lib[q] else A_g0[q]
        d0 = 0.0
        if A_pi[q] >= 0:   # assimilation: twin's own miss from previous reading to now
            pi = A_pi[q]; Gp = (A_pg[q] - b) / a if A_plib[q] else A_pg[q]
            ex_p = e_hr * hrup[pi] + e_st * st15[pi] / 100.0 + e_cal * cal15[pi] / 10.0
            Gh = _sim(Gp, pi, i0 - pi, ins, ra, drift, SI, k, Gb, beta, c_low, 0.0, tau_d, ex_p)
            w = w_d if A_lib[q] else w_ds
            d0 = w * (G0 - Gh) / ((i0 - pi) * DT)
        ex0 = e_hr * hrup[i0] + e_st * st15[i0] / 100.0 + e_cal * cal15[i0] / 10.0
        G = _sim(G0, i0, A_n[q], ins, ra, drift, SI, k, Gb, beta, c_low, d0, tau_d, ex0)
        out[q] = a * G + b if A_olib[q] else G
        dout[q] = d0
    return out, dout

# ------------------------------------------------------------------ anchors
class Anchors:
    """Forecast questions: from a reading at t0 (Libre or stick) to a target n steps later."""
    def __init__(self, tl, t0, n, g0, lib, prev_t, prev_g, prev_lib, true, w, tag, olib=None):
        self.tl = tl; self.i0 = tl.idx(t0); self.n = np.asarray(n, np.int64)
        self.g0 = np.asarray(g0, float); self.lib = np.asarray(lib, np.bool_)
        self.olib = self.lib.copy() if olib is None else np.asarray(olib, np.bool_)
        pt = pd.DatetimeIndex(prev_t); ok = ~pd.isna(pt)
        self.pi = np.full(len(self.i0), -1, np.int64)
        if ok.any(): self.pi[ok] = tl.idx(pt[ok])
        self.pg = np.nan_to_num(np.asarray(prev_g, float)); self.plib = np.asarray(prev_lib, np.bool_)
        self.true = np.asarray(true, float); self.w = np.asarray(w, float); self.tag = list(tag)
        self.t = pd.DatetimeIndex(t0)
    def subset(self, m):
        A = object.__new__(Anchors); A.tl = self.tl
        for k in ("i0", "n", "g0", "lib", "olib", "pi", "pg", "plib", "true", "w"): setattr(A, k, getattr(self, k)[m])
        A.tag = [x for x, y in zip(self.tag, m) if y]; A.t = self.t[m]; return A
    def __len__(self): return len(self.i0)

def unpack(x):
    return {n: x[i] for i, n in enumerate(NAMES)}

import os
LAG = int(os.environ.get("LAG", 0))
SENS2 = int(os.environ.get("SENS2", 0))  # Libre lag in 5-min steps
_cache = {}
def predict(x, A, return_d=False):
    p = unpack(x); tl = A.tl
    cf = np.array([p[f"cf_{f}"] for f in FAMS]); tau = np.array([p[f"tau_{f}"] for f in FAMS])
    key = (id(tl), tuple(np.round(np.r_[p["ti"], cf, tau, p["fat_slow"], p["cf_fpu"], p["tau_fpu"], p["lowfast"]], 10)))
    if key in _cache: ins, ra = _cache[key]
    else:
        ins, ra = _rates(tl.n, tl.bt, tl.bu, tl.mt, tl.mc, tl.mfat, tl.mfpu, tl.mlow, p["lowfast"], p["ti"], cf, tau, p["fat_slow"], p["cf_fpu"], p["tau_fpu"])
        if len(_cache) > 64: _cache.clear()
        _cache[key] = (ins, ra)
    ins = ins * np.maximum(0.2, 1.0 + p["si_s"] * tl.tod_s + p["si_c"] * tl.tod_c)
    drift = p["dr0"] + p["dr_s"] * tl.tod_s + p["dr_c"] * tl.tod_c
    th = np.array([p["SI"], p["k"], p["Gb"], p["beta"], p["c_low"], p["w_d"], p["tau_d"], p["w_ds"], p["e_hr"], p["e_st"], p["e_cal"], p["a2"] if getattr(tl, "sensor2", False) and SENS2 else p["a"], p["b2"] if getattr(tl, "sensor2", False) and SENS2 else p["b"]])
    sh = (LAG * A.lib).astype(np.int64)
    out, d = _forecast(ins, ra, drift, tl.hrup, tl.st15, tl.cal15, A.i0 - sh, A.n, A.g0, A.lib, np.where(A.pi >= 0, A.pi - sh, -1), A.pg, A.plib, th, getattr(A, "olib", A.lib))
    return (out, d) if return_d else out

# ------------------------------------------------------------------ what-if: full curve for one scenario
def scenario_timeline(events, hours=6):
    """events: list of dict(t_min, units=0, food=dict(c_fast..c_veg, fat, fpu), low=0). t_min relative to 'now' (=0), may be negative (past)."""
    tl = object.__new__(Timeline)
    tl.t0 = pd.Timestamp("2000-01-01"); start = -12 * 60
    tl.n = int((hours * 60 - start) / DT) + 1
    shift = -start
    b = [e for e in events if e.get("units", 0) > 0]; m = [e for e in events if e.get("food")]
    tl.bt = np.array([e["t_min"] + shift for e in b], float); tl.bu = np.array([e["units"] for e in b], float)
    tl.mt = np.array([e["t_min"] + shift for e in m], float)
    tl.mc = np.array([[e["food"].get(f"c_{f}", 0.0) for f in FAMS] for e in m], float).reshape(-1, 5)
    tl.mfat = np.array([e["food"].get("fat", 0.0) for e in m], float); tl.mfpu = np.array([e["food"].get("fpu", 0.0) for e in m], float)
    tl.mlow = np.array([e.get("low", 0.0) for e in m], float)
    z = np.zeros(tl.n); tl.tod_s, tl.tod_c, tl.hrup, tl.st15, tl.cal15 = z.copy(), z.copy(), z.copy(), z.copy(), z.copy()
    tl.i_now = int(shift / DT)
    return tl

def curve(x, tl, g_now, minute_of_day=0, steps=48, libre=False):
    """glucose path every 5 min for `steps` steps after now (blood units unless libre=True)."""
    mins = (minute_of_day + (np.arange(tl.n) - tl.i_now) * DT) % 1440
    tl.tod_s, tl.tod_c = np.sin(2 * np.pi * mins / 1440), np.cos(2 * np.pi * mins / 1440)
    _cache.clear()
    A = object.__new__(Anchors); A.tl = tl
    A.i0 = np.full(steps, tl.i_now, np.int64); A.n = np.arange(1, steps + 1, dtype=np.int64)
    A.g0 = np.full(steps, float(g_now)); A.lib = np.full(steps, bool(libre)); A.pi = np.full(steps, -1, np.int64)
    A.pg = np.zeros(steps); A.plib = np.zeros(steps, np.bool_)
    out = predict(x, A); _cache.clear(); return out
