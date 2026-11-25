"""Filled blood curve: fingersticks set the level, Libre gives shape/trend, the twin bridges gaps with no Libre."""
import numpy as np, pandas as pd, pickle
from scipy.optimize import least_squares
from fam11 import load_cgm
import twin as T
cgm = load_cgm("../D/cgm_export_21-11-2026.csv").g
pk = {n: pd.read_pickle(f"../{n}.pkl") for n in ("logo", "l2", "l3")}
LOG = {"set1": pk["logo"], "set2": pk["l2"], "set3": pk["l3"]}
def sticks(k):
    B = LOG[k].dropna(subset=["bg"])[["time", "bg"]].copy(); B["time"] = B.time.dt.round("5min")
    return B.drop_duplicates("time").sort_values("time").set_index("time").bg
STK = {k: sticks(k) for k in LOG}
LIB = {"set1": cgm[(cgm.index >= "2026-08-27") & (cgm.index < "2026-09-18")], "set3": cgm[(cgm.index >= "2026-11-18 13:00") & (cgm.index < "2026-11-20 03:00")]}
def libre_at(k, t):
    s = LIB.get(k)
    if s is None or not len(s): return np.nan
    x = s.reindex(s.index.union([t])).interpolate(limit=2, limit_area="inside")
    v = x.get(t, np.nan)
    near = (abs(s.index - t) <= pd.Timedelta("20min")).any()
    return float(v) if near else np.nan
PAIRS = pd.DataFrame([dict(k=k, t=t, bg=v, L=libre_at(k, t)) for k in ("set1", "set3") for t, v in STK[k].items()]).dropna()
def fit_cal(before):
    """robust blood = c0 + c1 * Libre, from pairs before `before` (pooled sensors)."""
    P = PAIRS[PAIRS.t < before]
    if len(P) < 6: return np.array([20.0, 1.15])
    f = lambda c: (c[0] + c[1] * P.L.values - P.bg.values)
    return least_squares(f, [20, 1.1], loss="huber", f_scale=20).x
def offset(k, t, cal, causal=True, tau=180.0, w0=0.7, before=None):
    """stick-minus-calibrated-Libre residuals near t (only sticks <= t if causal), kernel weighted, shrunk to 0."""
    P = PAIRS[(PAIRS.k == k)]
    if before is not None: P = P[P.t < before]
    if causal: P = P[P.t <= t]
    if not len(P): return 0.0
    dt = np.abs((t - P.t).dt.total_seconds().values / 60); w = np.exp(-dt / tau)
    r = P.bg.values - (cal[0] + cal[1] * P.L.values)
    return float((w * r).sum() / (w.sum() + w0))
def libre_to_blood(k, t, L, cal, before):
    return cal[0] + cal[1] * L + offset(k, t, cal, True, before=before)

def curve(x, before, step="15min"):
    """blood curve for every timeline up to `before` (days whose outcome < before). Returns DataFrame t, tl, blood, src, w."""
    cal = fit_cal(before); out = []
    for k in ("set1", "set2", "set3"):
        tl_s = STK[k][STK[k].index < before]
        lib = LIB.get(k)
        if lib is not None:
            L = lib[lib.index < before]
            for t, v in L.items():
                if t.minute % 15: continue
                b = cal[0] + cal[1] * v + offset(k, t, cal, causal=False, before=before)
                dn = np.min(np.abs((tl_s.index - t).total_seconds())) / 60 if len(tl_s) else 1e9
                out.append(dict(t=t, tl=k, blood=b, src="libre", near=dn))
        # twin bridge between consecutive sticks where no Libre
        s = tl_s
        for (t1, v1), (t2, v2) in zip(s.items(), list(s.items())[1:]):
            gap = (t2 - t1).total_seconds() / 60
            if gap < 20 or gap > 300: continue
            if lib is not None and len(lib[(lib.index > t1) & (lib.index < t2)]) > gap / 15 * 0.7: continue
            tl = T.scenario_timeline([], 1)  # dummy, replaced below
            from tfit import TL, make
            A = T.Anchors(TL[k], pd.DatetimeIndex([t1] * int(gap // 5)), np.arange(1, int(gap // 5) + 1), np.full(int(gap // 5), v1), np.zeros(int(gap // 5), bool),
                          [pd.NaT] * int(gap // 5), np.full(int(gap // 5), np.nan), np.zeros(int(gap // 5), bool), np.zeros(int(gap // 5)), np.ones(int(gap // 5)), ["b"] * int(gap // 5))
            p = T.predict(x, A); e = v2 - p[-1]; fr = np.arange(1, len(p) + 1) / len(p); p = p + e * fr
            for j, val in enumerate(p[:-1]):
                tt = t1 + pd.Timedelta(minutes=5 * (j + 1))
                if tt.minute % 15 == 0: out.append(dict(t=tt, tl=k, blood=val, src="bridge", near=min(5 * (j + 1), gap - 5 * (j + 1))))
        for t, v in tl_s.items(): out.append(dict(t=t, tl=k, blood=v, src="stick", near=0.0))
    C = pd.DataFrame(out).sort_values(["tl", "t"])
    return C.drop_duplicates(["tl", "t"], keep="last").reset_index(drop=True), cal
