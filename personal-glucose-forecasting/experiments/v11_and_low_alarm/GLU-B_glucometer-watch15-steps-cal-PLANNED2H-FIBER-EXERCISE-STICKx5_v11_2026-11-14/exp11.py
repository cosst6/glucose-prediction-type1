import glob, numpy as np, pandas as pd, warnings, joblib; warnings.filterwarnings("ignore")
from pairs import *
from whyb import WHybrid
D = "../D"
cgm = load_cgm(glob.glob(f"{D}/*glucose*.csv")[0]); cgm = cgm[cgm.index >= "2026-08-01"]
logo = load_log9(f"{D}/log.csv"); wo = load_watch(f"{D}/noise hr logger")
Xo = restrict_to_logged_days(build_frame9(cgm, logo, wo, [60, 120]), logo)
# old fingersticks: log bg + readings found in food notes
z = pd.read_csv("../cgm_vs_bg_pairs.csv", parse_dates=["time"])[["time", "bg"]]
Bo = pd.concat([logo.dropna(subset=["bg"])[["time", "bg"]], z]); Bo["time"] = Bo.time.dt.round("5min"); Bo = Bo.drop_duplicates("time").sort_values("time").reset_index(drop=True)
Po = make_pairs(logo, wo, Bo); Pn = pd.read_pickle("../pairs_nov.pkl")
print("old CGM rows", len(Xo), "old stick pairs", len(Po), "nov pairs", len(Pn))
# CGM -> blood calibration (bg regressed on cgm)
cp = pd.read_csv("../cgm_vs_bg_pairs.csv"); A, Bc = np.polyfit(cp.g, cp.bg, 1); print("blood = %.2f*cgm %+.1f" % (A, Bc))
def to_blood(X):
    X = X.copy()
    for h in (60, 120): X[f"y{h}"] = X[f"y{h}"] * A
    X["g0"] = X.g0 * A + Bc; return X
v10 = joblib.load(glob.glob("glucose-model/v9_v10_fiber_exercise/*v10*/*.joblib")[0])
days = sorted(Pn.index.normalize().unique())[1:]
res = []
for cfg in [("cgm+stick w5", False, 5), ("cgm+stick w20", False, 20), ("blood-scaled cgm+stick w5", True, 5), ("blood-scaled cgm+stick w20", True, 20), ("blood-scaled cgm only", True, 0)]:
    name, sc, w = cfg; base = to_blood(Xo) if sc else Xo.copy(); base["w"] = 1.0
    for day in days:
        te = Pn[Pn.index.normalize() == day]
        tr_n = Pn[Pn.index < day - pd.Timedelta(minutes=150)]
        S = pd.concat([Po, tr_n]); S["w"] = float(w)
        tr = pd.concat([base, S]) if w else base
        m = WHybrid().fit(tr, "B", [60, 120])
        for h in (60, 120):
            T = te[te.h == h]
            if not len(T): continue
            res.append(pd.DataFrame({"cfg": name, "h": h, "true": T.true, "g0": T.g0, "pred": m.predict(T, h).pred, "v10": v10.predict(T, h).pred}, index=T.index))
R = pd.concat(res); R.to_csv("../res11.csv")
r = lambda e: round(float(np.sqrt((e**2).mean())), 1)
for (c, h), d in R.groupby(["cfg", "h"], sort=False):
    print(f"{c:28s} h{h} n={len(d)} new {r(d.pred-d.true)}  v10 {r(d.v10-d.true)}  persist {r(d.g0-d.true)}")
