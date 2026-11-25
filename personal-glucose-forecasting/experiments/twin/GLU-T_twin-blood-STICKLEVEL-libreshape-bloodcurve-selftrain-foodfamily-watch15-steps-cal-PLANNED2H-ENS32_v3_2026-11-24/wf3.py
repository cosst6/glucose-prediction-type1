"""Walk-forward on every fingerstick with (a) denoised Libre anchors (calibrated + recent-stick offset, causal) and (b) self-training on a filled blood curve."""
import os, sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from tfit import *
import blood as BL
DEN = int(os.environ.get("DEN", 1)); SELF = int(os.environ.get("SELF", 1)); KEEPLL = float(os.environ.get("KEEPLL", 1.0))
WCS, WCL, WCB, WS = (float(os.environ.get(k, v)) for k, v in (("WCS", 2), ("WCL", 0.5), ("WCB", 1), ("WS", 5)))
FIX = {"si_s": 0, "si_c": 0, "a2": 0.72, "b2": 0.0}
S = pd.read_pickle("../allsticks.pkl").reset_index(drop=True)
LL = pd.concat([TABS["set1 Libre"], TABS["Nov18-19 Libre"]]).reset_index(drop=True); LL["olib"] = LL.lib
LL["tgt"] = LL.t + pd.to_timedelta(LL.n * 5, "min")
def denoise(df, before):
    """Libre anchors -> blood using only calibration pairs before `before` and sticks before the anchor."""
    df = df.copy(); k = df.lib.values
    if not k.any(): return df
    cal = BL.fit_cal(before); g = df.g0.values.copy(); pg = df.prev_g.values.copy()
    for i in np.where(k)[0]:
        t = df.t.iloc[i]; g[i] = BL.libre_to_blood(df.tl.iloc[i], t, g[i], cal, before=min(before, t + pd.Timedelta("1min")))
        if not np.isnan(pg[i]): pg[i] = cal[0] + cal[1] * pg[i] + (g[i] - cal[0] - cal[1] * df.g0.values[i])
    df["g0"] = g; df["prev_g"] = pg; df["lib"] = False; df["prev_lib"] = False
    return df
def curve_anchors(C):
    rows = []
    for k, g in C.groupby("tl"):
        s = g.set_index("t"); idx = s.index
        for t, r in s.iterrows():
            pv = s.blood.get(t - pd.Timedelta("30min"), np.nan)
            for h in (60, 120):
                tt = t + pd.Timedelta(minutes=h)
                if tt not in idx: continue
                tr = s.loc[tt]; src = tr.src
                w = WCS if src == "stick" else (WCB if src == "bridge" else (WCS if tr.near <= 30 else WCL))
                rows.append(dict(t=t, n=h // 5, g0=r.blood, lib=False, olib=False, prev_t=t - pd.Timedelta("30min") if not np.isnan(pv) else pd.NaT,
                                 prev_g=pv, prev_lib=False, true=tr.blood, tag="curve", tl=k, tgt=tt, w=w))
    return pd.DataFrame(rows)
def grp(df): return [make(g, g.w.values) for _, g in df.groupby("tl")]
def pred(x, df):
    p = np.empty(len(df))
    for _, g in df.groupby("tl"): p[df.index.get_indexer(g.index)] = T.predict(x, make(g))
    return p
res = []
for d in sorted(S.t.dt.normalize().unique()):
    ll = LL[LL.tgt < d].assign(w=KEEPLL); s = S[S.tgt < d]
    ss = s[s.tag == "stick->stick"].assign(w=WS); ls = s[s.tag == "Libre->stick"].assign(w=0.1)
    if DEN: ls = denoise(ls, d)
    parts = [p for p in (ll, ss, ls) if len(p)]
    if sum(len(p) for p in parts) < 300: continue
    x, _ = fit(grp(pd.concat(parts).reset_index(drop=True)), fixed=FIX)
    if SELF:
        C, _ = BL.curve(x, d); CA = curve_anchors(C).dropna(subset=["g0", "true"])
        CA = CA[[(TL[k].T[0] + pd.Timedelta("12h") <= t) and (t + pd.Timedelta("150min") <= TL[k].T[-1]) for k, t in zip(CA.tl, CA.t)]]
        x, _ = fit(grp(pd.concat(parts + [CA]).reset_index(drop=True)), x0=x, fixed=FIX)
    te = S[S.t.dt.normalize() == d].copy().reset_index(drop=True)
    if DEN: te = denoise(te, d)
    te["twin"] = pred(x, te); res.append(te); print(d.date(), len(te), flush=True)
R = pd.concat(res); tag = sys.argv[1]; R.to_pickle(f"../wf3_{tag}.pkl")
r = lambda e: round(float(np.sqrt(np.mean(np.square(e)))), 1)
print(tag, dict(DEN=DEN, SELF=SELF, KEEPLL=KEEPLL, WCS=WCS, WCL=WCL, WCB=WCB, WS=WS))
for (tg, hb), g in R.groupby(["tag", "hb"], observed=True):
    print(f"{tg:13s} {hb:20s} n={len(g):4d} twin {r(g.twin-g.true):5}")
