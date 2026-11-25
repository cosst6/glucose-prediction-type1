import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import Ridge; from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler
from glucoforecast5 import load_log5, restrict_to_logged_days, CLK
from glucoforecast4 import feature_list4
log = load_log5("log.csv"); X5 = restrict_to_logged_days(pd.read_pickle("X5_p0.pkl"), log); X4 = pd.read_pickle("X4.pkl").loc[X5.index.intersection(pd.read_pickle("X4.pkl").index)]
W = ["hr_30", "hr_above_rest_60", "steps_60", "steps_180", "watch_on"]
def wf(X, cols, h):
    y = f"y{h}"; Z = X.dropna(subset=cols + [y]); e = []
    for day in sorted(d for d in Z.index.normalize().unique() if d >= pd.Timestamp("2026-08-30")):
        te = Z[Z.index.normalize() == day]; tr = Z[Z.index < day - pd.Timedelta(minutes=h)]
        if len(te) == 0: continue
        e.append(make_pipeline(StandardScaler(), Ridge(alpha=30)).fit(tr[cols], tr[y]).predict(te[cols]) - te[y].values)
    return np.sqrt(np.mean(np.concatenate(e)**2))
for h in (60, 120):
    base = feature_list4("A", h); print(f"\n== h={h}  v4 features: {wf(X4, base, h):.2f}")
    def k(e): return [f"{e}_next{h}", f"{e}_ob", f"{e}_act30"]
    core = ["g0", "d15", "d30", "d60", "accel", "h_since_lantus", "lantus_ob", "uncovered_carbs", "min_since_meal", "tod_s", "tod_c"] + W + k("ins") + k("fsug")
    v = {"new kernels (milk fast, late fpu/slow)": core + k("dairy") + k("gl") + k("slow") + k("fpu")}
    n = v["new kernels (milk fast, late fpu/slow)"]
    v["+ night food"] = n + k("ncarb") + k("nfpu")
    v["+ clock blocks"] = n + CLK
    v["+ dawn + lantus_old"] = n + ["dawn", "lantus_old"]
    v["+ low rebound / high start"] = n + ["below70", "above140", "low_recent"]
    v["all"] = n + k("ncarb") + k("nfpu") + CLK + ["dawn", "lantus_old", "below70", "above140", "low_recent"]
    for name, cols in v.items(): print(f"{name:45s} {wf(X5, cols, h):.2f}")
