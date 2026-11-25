import os, sys, glob, numpy as np, pandas as pd, joblib, warnings; warnings.filterwarnings("ignore")
import fam11 as F
from fam11 import *
from whyb import WHybrid
SH = {"lit": {"fast": (90, 1/3), "milk": (120, 1/3), "starch": (180, 1/3), "legume": (300, .5), "veg": (240, .4)},
      "user": {"fast": (150, .4), "milk": (180, .5), "starch": (300, .6), "legume": (330, .55), "veg": (240, .4)}}
Xo, X3, Po, P2, P3 = [pd.read_pickle(f"../{n}.pkl") for n in ("Xo", "X3", "Po", "P2", "P3")]
logs = {n: pd.read_pickle(f"../{n}.pkl") for n in ("logo", "l2", "l3")}
def refam(X, L, shape):  # recompute family channel features for a shape set
    F.set_shapes(shape); idx = X.index
    from glucoforecast8 import event_features4
    sub = {k: v for k, v in g8.KERNELS4.items() if k.startswith("F")}
    keep = g8.KERNELS4.copy(); g8.KERNELS4.clear(); g8.KERNELS4.update(sub)
    out = []
    if "gap" in X:   # stick pairs: feature window = actual gap, renamed to h
        for (h, gap), grp in X.groupby(["h", "gap"]):
            E = event_features4(grp.index, L, [gap]).rename(columns=lambda c: c.replace(f"next{gap}", f"next{h}"))
            out.append(E.set_index(grp.index))
        E = pd.concat(out)
    else: E = event_features4(idx, L, [60, 120])
    g8.KERNELS4.clear(); g8.KERNELS4.update(keep)
    E = E[[c for c in E if c.startswith("F")]]
    Y = X.copy()
    if "gap" in X:
        Y = Y.reset_index(); E = E.reset_index(drop=True)
        Y = Y.sort_values(["h", "gap"], kind="stable"); E.index = Y.index
        for c in E: Y[c] = E[c]
        Y = Y.sort_index().set_index(Y.columns[0]); Y.index.name = None
    else:
        for c in E: Y[c] = E[c].values
    return Y
v10 = joblib.load(glob.glob("glucose-model/v9_v10_fiber_exercise/*v10*/*.joblib")[0])
def fam_tag(X, L, h):
    tags = []
    lt = L.time.values
    for t in X.index:
        s = L[(L.time > t - pd.Timedelta("120min")) & (L.time <= t + pd.Timedelta(minutes=h))]
        v = {f: s[f"c_{f}"].sum() for f in FAMS}; tot = sum(v.values()); fpu = s.fpu.sum()
        if tot < 5: tags.append("fat/protein only" if fpu >= 1.5 else "no food")
        else: tags.append(max(v, key=v.get))
    return tags
r = lambda e: round(float(np.sqrt(np.mean(np.square(e)))), 1)
res = []
shape = sys.argv[1] if len(sys.argv) > 1 else "lit"; tree = os.environ.get("FTREE", "add")
Xo_ = refam(Xo, logs["logo"], SH[shape]); X3_ = refam(X3, logs["l3"], SH[shape])
Po_ = refam(Po, logs["logo"], SH[shape]); P2_ = refam(P2, logs["l2"], SH[shape]); P3_ = refam(P3, logs["l3"], SH[shape])
# A) set 1 Libre walk-forward (>=6 training days)
days = sorted(Xo_.index.normalize().unique())
for d in days[6:]:
    tr = Xo_[Xo_.index < d - pd.Timedelta("150min")].assign(w=1.0); te = Xo_[Xo_.index.normalize() == d]
    mf = FamHybrid().fit(tr, "B", [60, 120]); mb = WHybrid().fit(tr, "B", [60, 120])
    for h in (60, 120):
        T = te.dropna(subset=[f"y{h}"])
        if not len(T): continue
        tg = fam_tag(T, logs["logo"], h)
        res.append(pd.DataFrame({"test": "set1 Libre", "h": h, "true": T.g0 + T[f"y{h}"], "g0": T.g0, "fam": mf.predict(T, h).pred, "v10cfg": mb.predict(T, h).pred, "v10live": np.nan, "tag": tg}, index=T.index))
# B/C) Nov: sticks (sets 2,3) and Libre Nov 18-19; train = all set1 Libre (w1) + stick pairs before test day (w5)
S_all = pd.concat([Po_.assign(set=1), P2_.assign(set=2), P3_.assign(set=3)])
for name, TE, L in (("set2 glucometer", P2_, logs["l2"]), ("set3 glucometer", P3_, logs["l3"]), ("Nov18-19 Libre", X3_, logs["l3"])):
    for d in sorted(TE.index.normalize().unique()):
        S = S_all[S_all.index < d - pd.Timedelta("150min")].assign(w=5.0)
        tr = pd.concat([Xo_.assign(w=1.0), S]);
        if name.endswith("Libre"): tr = pd.concat([tr, X3_[X3_.index < d - pd.Timedelta("150min")].assign(w=1.0)])
        mf = FamHybrid().fit(tr, "B", [60, 120]); mb = WHybrid().fit(tr, "B", [60, 120])
        te = TE[TE.index.normalize() == d]
        for h in (60, 120):
            T = te[te.h == h] if "h" in te else te.dropna(subset=[f"y{h}"])
            if not len(T): continue
            true = T.true if "true" in T and "gap" in T else T.g0 + T[f"y{h}"]
            res.append(pd.DataFrame({"test": name, "h": h, "true": true.values, "g0": T.g0.values, "fam": mf.predict(T, h).pred.values, "v10cfg": mb.predict(T, h).pred.values,
                                     "v10live": v10.predict(T, h).pred.values, "tag": fam_tag(T, L, h)}, index=T.index))
R = pd.concat(res); R.to_csv(f"../res14_{shape}_{tree}.csv")
for (t, h), g in R.groupby(["test", "h"], sort=False):
    print(f"{shape}/{tree} {t:16s} h{h} n={len(g):4d} fam {r(g.fam-g.true)}  v10-same-data {r(g.v10cfg-g.true)}  v10-live {r(g.v10live-g.true) if g.v10live.notna().any() else '-'}  no-change {r(g.g0-g.true)}")
