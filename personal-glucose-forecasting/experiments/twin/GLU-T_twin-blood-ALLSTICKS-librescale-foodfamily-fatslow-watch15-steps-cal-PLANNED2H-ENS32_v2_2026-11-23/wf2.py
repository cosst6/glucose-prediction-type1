"""Fair walk-forward on EVERY fingerstick: each day's forecasts come from a twin trained only on data whose outcome is before that day."""
import os, sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from tfit import *
WL = float(os.environ.get("WL", 1)); WS = float(os.environ.get("WS", 5)); WLS = float(os.environ.get("WLS", 0.5))
MODE = os.environ.get("MODE", "all")   # all = every stick pair in training; old = v1 training (Libre + 60/120 stick pairs)
FIX = eval(os.environ.get("FIX", '{"si_s":0,"si_c":0}'))
if not int(os.environ.get("SENS2", 0)): FIX.update({"a2": 0.72, "b2": 0.0})
S = pd.read_pickle("../allsticks.pkl").reset_index(drop=True)
LL = pd.concat([TABS["set1 Libre"], TABS["Nov18-19 Libre"]]).reset_index(drop=True)
LL["tgt"] = LL.t + pd.to_timedelta(LL.n * 5, "min")
OLD = pd.concat([TABS["set1 sticks"], TABS["set2 glucometer"], TABS["set3 glucometer"]]).reset_index(drop=True)
OLD["tgt"] = OLD.t + pd.to_timedelta(OLD.n * 5, "min")
HL = float(os.environ.get("HL", 0))
CUR = [None]
def wtab(df, w):
    ww = np.full(len(df), w, float)
    if HL > 0: ww *= 0.5 ** ((CUR[0] - df.tgt).dt.total_seconds().values / 86400 / HL)
    return df.assign(w=ww)
def train_parts(d):
    parts = [wtab(LL[LL.tgt < d], WL)]
    if MODE == "all":
        s = S[S.tgt < d]
        parts += [wtab(s[s.tag == "stick->stick"], WS), wtab(s[s.tag == "Libre->stick"], WS * WLS / 10)]
    else:
        parts += [wtab(OLD[OLD.tgt < d], 5.0)]
    return [p for p in parts if len(p)]
def grp(df):
    out = []
    for _, g in df.groupby("tl"):
        out.append(make(g, g.w.values))
    return out
def pred(x, df):
    p = np.empty(len(df))
    for _, g in df.groupby("tl"):
        p[df.index.get_indexer(g.index)] = T.predict(x, make(g))
    return p
res = []
for d in sorted(S.t.dt.normalize().unique()):
    CUR[0] = d; parts = train_parts(d)
    if sum(len(p) for p in parts) < 300: continue   # need a few days of history
    tr = sum([grp(p) for p in parts], [])
    x, _ = fit(tr, lam=1.0, fixed=FIX)
    te = S[S.t.dt.normalize() == d].copy(); te["twin"] = pred(x, te)
    res.append(te); print(d.date(), len(te), flush=True)
R = pd.concat(res); tag = sys.argv[1] if len(sys.argv) > 1 else "x"; R.to_pickle(f"../wf2_{tag}.pkl")
r = lambda e: round(float(np.sqrt(np.mean(np.square(e)))), 1)
print(tag, "WL", WL, "WS", WS, "WLS", WLS, MODE)
for (tg, hb), g in R.groupby(["tag", "hb"], observed=True):
    print(f"{tg:13s} {hb:20s} n={len(g):4d} sticks={g.tgt.nunique():3d} twin {r(g.twin-g.true):5}  no-change {r(g.g0-g.true):5}")
