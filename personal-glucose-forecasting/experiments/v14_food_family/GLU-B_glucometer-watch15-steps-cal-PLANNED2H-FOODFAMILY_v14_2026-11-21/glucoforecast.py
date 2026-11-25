"""Personal glucose forecaster.

Model A ("cgm"):        CGM history + watch (HR/steps) + logged food/insulin  -> glucose at t+h
Model B ("fingerstick"): one glucometer reading + watch + logged food/insulin -> glucose at t+h

Both use the same physiological features: insulin activity/IOB (exponential rapid-acting curve),
carb absorption (net carbs), slow fat+protein absorption, and how much of each will act between
now and t+h given what has already been logged. One ridge regression per horizon.
"""
import glob, os
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from parse_meal import parse_meal

STEP = 15
HORIZONS_A = [15, 30, 45, 60, 90, 120, 180]
HORIZONS_B = [30, 60, 90, 120, 180, 240]
HORIZONS_C = HORIZONS_B

# ---------------------------------------------------------------- loading
def load_cgm(path):
    raw = pd.read_csv(path, skiprows=1)
    raw["t"] = pd.to_datetime(raw["Device Timestamp"], format="%d-%m-%Y %I:%M %p")
    s = raw[raw["Record Type"] == 0].set_index("t")["Historic Glucose mg/dL"].sort_index()
    s = s[~s.index.duplicated()]
    s.index = s.index.round(f"{STEP}min"); s = s[~s.index.duplicated()]
    seg = (s.index.to_series().diff() > pd.Timedelta("45min")).cumsum()
    parts = [pd.DataFrame({"g": p.resample(f"{STEP}min").mean().interpolate(limit=2), "seg": k}) for k, p in s.groupby(seg)]
    return pd.concat(parts)

def load_watch(folder):
    frames = []
    for p in glob.glob(os.path.join(folder, "*.csv")):
        try:
            x = pd.read_csv(p, skiprows=2)
        except Exception:
            continue
        x.columns = [c.strip().lower() for c in x.columns]
        hcol = next((c for c in x.columns if "heart" in c), None)
        if hcol is None:
            continue
        y = pd.DataFrame({"t": pd.to_datetime(x[x.columns[0]], errors="coerce"), "hr": pd.to_numeric(x[hcol], errors="coerce")})
        y["steps"] = pd.to_numeric(x["steps"], errors="coerce") if "steps" in x.columns else np.nan  # steps per sample interval
        frames.append(y.dropna(subset=["t"]))
    w = pd.concat(frames).sort_values("t").drop_duplicates("t").set_index("t")
    w.loc[(w.hr < 35) | (w.hr > 220), "hr"] = np.nan
    return w

def load_log(path):
    L = pd.read_csv(path, parse_dates=["time"])
    for c in ["rapid_u", "lantus_u", "bg", "cgm_noted"]:
        L[c] = pd.to_numeric(L[c], errors="coerce")
    macros = [parse_meal(f)[1] for f in L["food"]]
    for k in ["carbs", "protein", "fat", "fiber"]:
        L[k] = [m[k] for m in macros]
    L["net_carbs"] = (L["carbs"] - 0.5 * L["fiber"]).clip(lower=0)
    L["fpu"] = (9 * L["fat"] + 4 * L["protein"]) / 100  # Warsaw fat-protein units
    return L.sort_values("time").reset_index(drop=True)

# ---------------------------------------------------------------- kernels (cumulative fraction absorbed after m minutes)
def insulin_cum(m, peak=75, dur=300):
    m = np.asarray(m, float)
    tau = peak * (1 - peak / dur) / (1 - 2 * peak / dur)
    a = 2 * tau / dur; S = 1 / (1 - a + (1 + a) * np.exp(-dur / tau))
    t = np.clip(m, 0, dur)
    iob = 1 - S * (1 - a) * ((t**2 / (tau * dur * (1 - a)) - t / tau - 1) * np.exp(-t / tau) + 1)
    return np.where(m < 0, 0.0, 1 - np.clip(iob, 0, 1))

def tri_cum(m, dur, peak_frac=1 / 3):
    m = np.asarray(m, float); p = dur * peak_frac; t = np.clip(m, 0, dur)
    c = np.where(t < p, t**2 / (p * dur), 1 - (dur - t) ** 2 / ((dur - p) * dur))
    return np.where(m < 0, 0.0, c)

KERNELS = {  # (log column, cumulative absorption fn)
    "ins": ("rapid_u", insulin_cum),
    "carb": ("net_carbs", lambda m: tri_cum(m, 180)),
    "fpu": ("fpu", lambda m: tri_cum(m, 300, 0.5)),
}

def event_features(idx, log, horizons):
    """For each time t: amount acting in the last 30 min, amount still on board, and amount that will act in (t, t+h]."""
    tt = (idx.values.astype("datetime64[m]").astype(np.int64))[:, None].astype(float)
    et = log["time"].values.astype("datetime64[m]").astype(np.int64)[None, :].astype(float)
    m = tt - et  # minutes since each event, (n_times, n_events); events after t get m<0 -> ignored
    known = m >= 0
    f = {}
    for name, (col, cum) in KERNELS.items():
        amt = np.nan_to_num(log[col].values.astype(float))[None, :] * known
        c_now = cum(m)
        f[f"{name}_ob"] = (amt * (1 - c_now)).sum(1)
        f[f"{name}_act30"] = (amt * (c_now - cum(m - 30))).sum(1)
        for h in horizons:
            f[f"{name}_next{h}"] = (amt * (cum(m + h) - c_now)).sum(1)
    lt = log.loc[log.lantus_u > 0, "time"].values
    pos = np.searchsorted(lt, idx.values, side="right") - 1
    f["h_since_lantus"] = np.where(pos >= 0, (idx.values - lt[np.clip(pos, 0, None)]) / np.timedelta64(1, "h"), 24.0) if len(lt) else np.full(len(idx), 24.0)  # fix Nov 21: no Lantus logged -> no crash
    # Lantus: ~flat 24 h action; units still active, linearly fading over 24 h
    lant = np.nan_to_num(log["lantus_u"].values.astype(float))[None, :] * known
    f["lantus_ob"] = (lant * np.clip(1 - m / 1440, 0, 1)).sum(1)
    return pd.DataFrame(f, index=idx)

def watch_features(idx, w):
    f = pd.DataFrame(index=idx)
    if w is None or len(w) == 0:
        return f
    r = lambda s, how: getattr(s.resample("5min", label="right", closed="right"), how)()
    hr5, st5 = r(w["hr"], "mean"), r(w["steps"], "sum")
    rest = hr5.rolling("24h", min_periods=50).quantile(0.1)
    f["hr_30"] = hr5.rolling(6, min_periods=2).mean().reindex(idx)
    f["hr_above_rest_60"] = (hr5 - rest).rolling(12, min_periods=3).mean().reindex(idx)
    f["steps_60"] = st5.rolling(12, min_periods=1).sum().reindex(idx)
    f["steps_180"] = st5.rolling(36, min_periods=1).sum().reindex(idx)
    f["watch_on"] = f["hr_30"].notna().astype(float)
    return f

def time_features(idx):
    h = idx.hour + idx.minute / 60
    return pd.DataFrame({"tod_s": np.sin(2 * np.pi * h / 24), "tod_c": np.cos(2 * np.pi * h / 24)}, index=idx)

# ---------------------------------------------------------------- design
WATCH = ["hr_30", "hr_above_rest_60", "steps_60", "steps_180", "watch_on"]
BASE_EVENT = ["ins_ob", "ins_act30", "carb_ob", "carb_act30", "fpu_ob", "fpu_act30", "h_since_lantus"]

def feature_list(model, h, use_watch=True):
    fut = [f"ins_next{h}", f"carb_next{h}", f"fpu_next{h}"]
    core = ["g0"] + fut + BASE_EVENT + ["tod_s", "tod_c"] + (WATCH if use_watch else [])
    if model == "A":
        core += ["d15", "d30", "d60", "accel"]
    if model == "C":  # glucometer + food + insulin only (no watch, no CGM trend)
        core = ["g0"] + fut + BASE_EVENT + ["lantus_ob", "tod_s", "tod_c"]
    return core

def build_frame(cgm, log, watch, horizons):
    rows = []
    for _, seg in cgm.groupby("seg"):
        g = seg["g"]; d = pd.DataFrame(index=seg.index)
        d["g0"] = g
        for k in (1, 2, 4): d[f"d{k*STEP}"] = g - g.shift(k)
        d["accel"] = d["d15"] - (g.shift(1) - g.shift(2))
        for h in horizons: d[f"y{h}"] = g.shift(-h // STEP) - g
        rows.append(d)
    X = pd.concat(rows)
    X = X.join(event_features(X.index, log, horizons)).join(time_features(X.index)).join(watch_features(X.index, watch))
    for c in WATCH:
        if c in X: X[c] = X[c].fillna(0 if c != "hr_30" else 70)
    return X

def fit(X, model, h, alpha=30, use_watch=True):
    cols = feature_list(model, h, use_watch); y = f"y{h}"
    D = X.dropna(subset=cols + [y])
    m = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(D[cols], D[y])
    m.cols = cols
    return m

def predict(m, X):
    return X["g0"] + m.predict(X[m.cols])

def restrict_to_logged_days(X, log):
    """Keep only CGM points on days with a food/insulin log AND the watch worn in the previous 30 min."""
    logged = set(log["time"].dt.normalize())
    keep = X.index.normalize().isin(list(logged)) & (X.get("watch_on", 0) == 1)
    return X[keep]

# ---------------------------------------------------------------- portable model files (plain JSON, no sklearn needed to run)
class JsonLinear:
    def __init__(self, d):
        self.cols = d["features"]; self.mean = np.array(d["mean"]); self.scale = np.array(d["scale"])
        self.coef = np.array(d["coef"]); self.intercept = d["intercept"]
    def predict(self, X):
        return ((X[self.cols].values - self.mean) / self.scale) @ self.coef + self.intercept

def export_json(models_for_one_type):
    out = {}
    for h, m in models_for_one_type.items():
        sc, rg = m[0], m[1]
        out[str(h)] = {"features": list(m.cols), "mean": [float(f"{v:.6g}") for v in sc.mean_],
                       "scale": [float(f"{v:.6g}") for v in sc.scale_], "coef": [float(f"{v:.6g}") for v in rg.coef_],
                       "intercept": float(f"{rg.intercept_:.6g}")}
    return out

def load_json_model(path):
    import json
    d = json.load(open(path))
    return {int(h): JsonLinear(v) for h, v in d["horizons"].items()}
