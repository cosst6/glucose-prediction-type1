"""v13 test: v10 hybrid trained on Libre set 1 + fingerstick pairs from sets 1,2,3 (weighted). Walk-forward on set 3 days."""
import glob, sys, numpy as np, pandas as pd, warnings, joblib; warnings.filterwarnings("ignore")
from pairs import *
from whyb import WHybrid
D = "../D"
cgm = load_cgm(glob.glob(f"{D}/*glucose*.csv")[0]); cgm = cgm[cgm.index >= "2026-08-01"]
logo = load_log9(f"{D}/log.csv"); wo = load_watch(f"{D}/noise hr logger")
Xo = restrict_to_logged_days(build_frame9(cgm, logo, wo, [60, 120]), logo)
z = pd.read_csv("../cgm_vs_bg_pairs.csv", parse_dates=["time"])[["time", "bg"]]
Bo = pd.concat([logo.dropna(subset=["bg"])[["time", "bg"]], z]); Bo["time"] = Bo.time.dt.round("5min"); Bo = Bo.drop_duplicates("time").sort_values("time").reset_index(drop=True)
Po = make_pairs(logo, wo, Bo)
l2 = load_log9(f"{D}/log_nov.csv"); w2 = load_watch(f"{D}/nov_watch"); P2 = make_pairs(l2, w2)
l3 = load_log9(f"{D}/log_set3.csv"); w3 = load_watch(f"{D}/set3_watch"); P3 = make_pairs(l3, w3)
for n, P in (("Po", Po), ("P2", P2), ("P3", P3)): P.to_pickle(f"../{n}.pkl")
Xo.to_pickle("../Xo.pkl")
print("CGM rows", len(Xo), "pairs set1", len(Po), "set2", len(P2), "set3", len(P3))
v10 = joblib.load(glob.glob("glucose-model/v9_v10_fiber_exercise/*v10*/*.joblib")[0])
v11 = joblib.load(glob.glob("glucose-model/v11_and_low_alarm/GLU-B*v11*/*.joblib")[0])
days = sorted(P3.index.normalize().unique())
res = []
for W in (5, 10):
    for day in days:
        te = P3[P3.index.normalize() == day]
        tr3 = P3[P3.index < day - pd.Timedelta(minutes=150)]
        S = pd.concat([Po, P2, tr3]); S["w"] = float(W); base = Xo.copy(); base["w"] = 1.0
        m = WHybrid().fit(pd.concat([base, S]), "B", [60, 120])
        for h in (60, 120):
            T = te[te.h == h]
            if not len(T): continue
            res.append(pd.DataFrame({"cfg": f"new w{W}", "day": day, "h": h, "true": T.true, "g0": T.g0, "pred": m.predict(T, h).pred,
                                     "v10": v10.predict(T, h).pred, "v11": v11.predict(T, h).pred}, index=T.index))
R = pd.concat(res); R.to_csv("../res13.csv")
r = lambda e: round(float(np.sqrt((e**2).mean())), 1)
for (c, h), d in R.groupby(["cfg", "h"], sort=False):
    print(f"{c} h{h} n={len(d)} new {r(d.pred-d.true)} v11 {r(d.v11-d.true)} v10 {r(d.v10-d.true)} persist {r(d.g0-d.true)}")
