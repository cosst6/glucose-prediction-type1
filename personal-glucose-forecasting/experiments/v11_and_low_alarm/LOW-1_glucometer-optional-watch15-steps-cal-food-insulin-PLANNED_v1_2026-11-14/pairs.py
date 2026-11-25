"""Fingerstick anchor -> later fingerstick target pairs, with v10 features computed for the exact gap."""
import os, numpy as np, pandas as pd
os.environ.setdefault("WSET","new"); os.environ.setdefault("FIB","split"); os.environ.setdefault("EXF","1")
from glucoforecast9 import *
NOM = {60: (40, 80), 120: (95, 145)}
def bg_points(log):
    B = log.dropna(subset=["bg"])[["time", "bg"]].copy()
    B["time"] = B.time.dt.round("5min"); return B.drop_duplicates("time").sort_values("time").reset_index(drop=True)
def make_pairs(log, watch, B=None):
    B = bg_points(log) if B is None else B
    rows = []
    for i, a in B.iterrows():
        for h, (lo, hi) in NOM.items():
            c = B[(B.time > a.time + pd.Timedelta(minutes=lo)) & (B.time <= a.time + pd.Timedelta(minutes=hi))]
            if len(c) == 0: continue
            j = (c.time - a.time - pd.Timedelta(minutes=h)).abs().idxmin(); b = c.loc[j]
            rows.append(dict(t=a.time, h=h, gap=int((b.time - a.time).total_seconds() // 60), g0=a.bg, true=b.bg))
    P = pd.DataFrame(rows)
    out = []
    for (h, gap), grp in P.groupby(["h", "gap"]):
        idx = pd.DatetimeIndex(grp.t.values)
        fake = pd.DataFrame({"g": grp.g0.values, "seg": range(len(grp))}, index=idx)
        X = build_frame9(fake, log, watch, [gap])
        X = X.rename(columns={f"{e}_next{gap}": f"{e}_next{h}" for e in EV9})
        X[f"y{h}"] = grp.true.values - grp.g0.values
        X["h"] = h; X["gap"] = gap; X["true"] = grp.true.values
        if f"ex_next{h}" not in X: X[f"ex_next{h}"] = 0.0
        out.append(X)
    return pd.concat(out).sort_index()
