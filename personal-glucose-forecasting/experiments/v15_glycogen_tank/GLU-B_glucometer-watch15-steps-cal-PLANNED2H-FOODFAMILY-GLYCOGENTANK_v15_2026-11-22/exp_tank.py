"""Fair walk-forward test of the glycogen tank vs v10 / v14. usage: TBODY=0|1 python exp_tank.py set1|nov"""
import os, sys, glob, numpy as np, pandas as pd, joblib, warnings; warnings.filterwarnings("ignore")
part = sys.argv[1]; sys.argv = ["x", "lit"]
exec(open("exp14.py").read().split("v10 = joblib")[0])
shape = "lit"
Xo_ = refam(Xo, logs["logo"], SH[shape]); X3_ = refam(X3, logs["l3"], SH[shape])
Po_ = refam(Po, logs["logo"], SH[shape]); P2_ = refam(P2, logs["l2"], SH[shape]); P3_ = refam(P3, logs["l3"], SH[shape])
from tank import add_tank, TankHybrid, TankFamHybrid, BODY
from watch7 import load_watch7 as LW
W = {"logo": LW("../D/noise hr logger"), "l2": LW("../D/nov_watch"), "l3": LW("../D/set3_watch")}
Xo_ = add_tank(Xo_, logs["logo"], W["logo"]); X3_ = add_tank(X3_, logs["l3"], W["l3"])
Po_ = add_tank(Po_, logs["logo"], W["logo"]); P2_ = add_tank(P2_, logs["l2"], W["l2"]); P3_ = add_tank(P3_, logs["l3"], W["l3"])
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
MODELS = {"tank10": TankHybrid, "tank14": TankFamHybrid}
if not BODY: MODELS = {"v10cfg": WHybrid, "fam": FamHybrid, **MODELS}
sfx = "b" if BODY else ""
res = []
def run(name, TE, L, trainf, live):
    for d in sorted(TE.index.normalize().unique()):
        tr = trainf(d); te = TE[TE.index.normalize() == d]
        ms = {k: M().fit(tr, "B", [60, 120]) for k, M in MODELS.items()}
        for h in (60, 120):
            T = te[te.h == h] if "gap" in te else te.dropna(subset=[f"y{h}"])
            if not len(T): continue
            true = T.true if "gap" in T else T.g0 + T[f"y{h}"]
            row = {"test": name, "h": h, "true": true.values, "g0": T.g0.values, "tag": fam_tag(T, L, h),
                   "liv_frac": T.liv_frac.values, "mus_frac": T.mus_frac.values, "fast_h": T.fast_h.values}
            for k, m in ms.items(): row[k + sfx] = m.predict(T, h).pred.values
            if live and not BODY: row["v10live"] = v10.predict(T, h).pred.values
            res.append(pd.DataFrame(row, index=T.index))
        print(name, d.date(), flush=True)
if part == "set1":
    days = sorted(Xo_.index.normalize().unique())
    run("set1 Libre", Xo_[Xo_.index.normalize() >= days[6]], logs["logo"], lambda d: Xo_[Xo_.index < d - pd.Timedelta("150min")].assign(w=1.0), False)
else:
    S_all = pd.concat([Po_.assign(set=1), P2_.assign(set=2), P3_.assign(set=3)])
    for name, TE, L in (("set2 glucometer", P2_, logs["l2"]), ("set3 glucometer", P3_, logs["l3"]), ("Nov18-19 Libre", X3_, logs["l3"])):
        def trf(d, name=name):
            tr = pd.concat([Xo_.assign(w=1.0), S_all[S_all.index < d - pd.Timedelta("150min")].assign(w=5.0)])
            if name.endswith("Libre"): tr = pd.concat([tr, X3_[X3_.index < d - pd.Timedelta("150min")].assign(w=1.0)])
            return tr
        run(name, TE, L, trf, True)
R = pd.concat(res); R.to_csv(f"../rt_{part}{sfx}.csv")
r = lambda e: round(float(np.sqrt(np.mean(np.square(e)))), 1)
for (t, h), g in R.groupby(["test", "h"], sort=False):
    print(t, h, len(g), {k: r(g[k] - g.true) for k in g.columns if k not in ("test", "h", "true", "tag", "liv_frac", "mus_frac", "fast_h")})
