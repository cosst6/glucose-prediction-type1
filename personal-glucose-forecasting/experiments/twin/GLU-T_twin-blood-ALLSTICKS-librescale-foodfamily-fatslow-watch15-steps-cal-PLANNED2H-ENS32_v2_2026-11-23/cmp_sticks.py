"""v10 / v14 walk-forward on every stick->stick pair 30-150 min (same pairs as the twin), with every stick pair in training too."""
import sys, numpy as np, pandas as pd, joblib, glob, warnings; warnings.filterwarnings("ignore")
sys.argv = ["x", "lit"]
exec(open("exp14.py").read().split("v10 = joblib")[0])
from glucoforecast9 import build_frame9, EV9
from watch7 import load_watch7 as LW
W = {"set1": load_watch("../D/noise hr logger"), "set2": load_watch("../D/nov_watch"), "set3": load_watch("../D/set3_watch")}
LG = {"set1": logs["logo"], "set2": logs["l2"], "set3": logs["l3"]}
S = pd.read_pickle("../allsticks.pkl"); S = S[(S.tag == "stick->stick") & (S.gap <= 150)].reset_index(drop=True)
S["h"] = np.where(S.gap <= 90, 60, 120)
fr = []
for (k, h, gap), g in S.groupby(["tl", "h", "gap"]):
    fake = pd.DataFrame({"g": g.g0.values, "seg": range(len(g))}, index=pd.DatetimeIndex(g.t.values))
    X = build_frame9(fake, LG[k], W[k], [gap]).rename(columns={f"{e}_next{gap}": f"{e}_next{h}" for e in EV9})
    X[f"y{h}"] = g.true.values - g.g0.values; X["h"] = h; X["gap"] = gap; X["true"] = g.true.values
    if f"ex_next{h}" not in X: X[f"ex_next{h}"] = 0.0
    X["row"] = g.index.values; X["tl"] = k; fr.append(refam(X, LG[k], SH["lit"]))
P = pd.concat(fr).sort_index(); P["tgt"] = P.index + pd.to_timedelta(P.gap, "min")
v10 = joblib.load(glob.glob("glucose-model/v9_v10_fiber_exercise/*v10*/*.joblib")[0])
Xo_ = refam(Xo, logs["logo"], SH["lit"]); X3_ = refam(X3, logs["l3"], SH["lit"])
X3_["tgt"] = X3_.index + pd.Timedelta("120min"); Xo_["tgt"] = Xo_.index + pd.Timedelta("120min")
out = []
for d in sorted(P.index.normalize().unique()):
    tr = pd.concat([Xo_[Xo_.tgt < d].assign(w=1.0), X3_[X3_.tgt < d].assign(w=1.0), P[P.tgt < d].assign(w=5.0)])
    if len(tr) < 300: continue
    mf = FamHybrid().fit(tr, "B", [60, 120]); mb = WHybrid().fit(tr, "B", [60, 120])
    te = P[P.index.normalize() == d]
    for h in (60, 120):
        T_ = te[te.h == h]
        if not len(T_): continue
        out.append(pd.DataFrame({"row": T_.row.values, "v14": mf.predict(T_, h).pred.values, "v10cfg": mb.predict(T_, h).pred.values, "v10live": v10.predict(T_, h).pred.values}))
    print(d.date(), flush=True)
pd.concat(out).to_pickle("../cmp_sticks.pkl")
