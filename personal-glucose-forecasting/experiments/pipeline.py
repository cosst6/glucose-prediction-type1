"""Glucose forecasting pipeline: FreeStyle Libre CGM + insulin/carb log (+ optional HR/steps)."""
import sys, glob, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = sys.argv[1]
STEP = 15  # minutes, Libre historic interval

# ---------- glucose ----------
raw = pd.read_csv(f"{DATA}/glucose.csv", skiprows=1)
raw["t"] = pd.to_datetime(raw["Device Timestamp"], format="%d-%m-%Y %I:%M %p")
hist = raw[raw["Record Type"] == 0].set_index("t")["Historic Glucose mg/dL"].sort_index()
hist = hist[~hist.index.duplicated()]

def to_grid(s):
    """Snap to a 15-min grid, split into segments at gaps > 45 min, interpolate small holes."""
    s = s.copy(); s.index = s.index.round(f"{STEP}min"); s = s[~s.index.duplicated()]
    seg = (s.index.to_series().diff() > pd.Timedelta("45min")).cumsum()
    out = []
    for k, part in s.groupby(seg):
        g = part.resample(f"{STEP}min").mean().interpolate(limit=2)
        out.append(pd.DataFrame({"g": g, "seg": k}))
    return pd.concat(out)

G = to_grid(hist)

# ---------- events (insulin / carbs) ----------
ev = pd.read_csv(f"{DATA}/events.csv", parse_dates=["time"])

def insulin_activity(minutes, peak=75, dur=300):
    # exponential insulin model (as used by Loop/OpenAPS); returns activity (fraction/min) and IOB fraction
    tau = peak * (1 - peak / dur) / (1 - 2 * peak / dur)
    a = 2 * tau / dur; S = 1 / (1 - a + (1 + a) * np.exp(-dur / tau))
    t = np.clip(minutes, 0, dur)
    act = (S / tau**2) * t * (1 - t / dur) * np.exp(-t / tau)
    iob = 1 - S * (1 - a) * ((t**2 / (tau * dur * (1 - a)) - t / tau - 1) * np.exp(-t / tau) + 1)
    act[(minutes < 0) | (minutes > dur)] = 0
    iob[minutes < 0] = 0; iob[minutes > dur] = 0
    return act, iob

def carb_curve(minutes, dur=180):
    # triangular absorption peaking at dur/3; returns rate (fraction/min) and carbs-on-board fraction
    p = dur / 3; t = np.asarray(minutes, float)
    rate = np.where(t < p, 2 * t / (p * dur), 2 * (dur - t) / ((dur - p) * dur))
    rate[(t < 0) | (t > dur)] = 0
    absorbed = np.where(t < p, t**2 / (p * dur), 1 - (dur - t) ** 2 / ((dur - p) * dur))
    cob = 1 - np.clip(absorbed, 0, 1); cob[t < 0] = 0; cob[t > dur] = 0
    return rate, cob

def event_features(idx):
    f = pd.DataFrame(index=idx, dtype=float)
    for c in ["ins_act", "iob", "carb_rate", "cob", "ins_1h", "carbs_1h", "mins_since_lantus"]:
        f[c] = 0.0
    tt = idx.values.astype("datetime64[m]").astype(float)
    for _, e in ev.iterrows():
        m = tt - np.datetime64(e.time, "m").astype(float)
        if e.rapid_u:
            a, i = insulin_activity(m); f["ins_act"] += e.rapid_u * a * 60; f["iob"] += e.rapid_u * i
            f["ins_1h"] += e.rapid_u * ((m >= 0) & (m < 60))
        if e.carbs_g:
            r, c = carb_curve(m); f["carb_rate"] += e.carbs_g * r * 60; f["cob"] += e.carbs_g * c
            f["carbs_1h"] += e.carbs_g * ((m >= 0) & (m < 60))
    lt = ev[ev.lantus_u > 0].time.sort_values()
    pos = np.searchsorted(lt.values, idx.values, side="right") - 1
    f["mins_since_lantus"] = np.where(pos >= 0, (idx.values - lt.values[np.clip(pos, 0, None)]) / np.timedelta64(1, "m"), np.nan)
    f["net_effect"] = f["carb_rate"] - 3 * f["ins_act"]  # crude: ~1 u per 3 g, matching the log's "apple worth 2 iu"
    return f

# ---------- HR / steps (optional) ----------
def load_hr():
    frames = []
    for p in glob.glob(f"{DATA}/hr/*.csv"):
        try:
            x = pd.read_csv(p, skiprows=2)
        except Exception:
            continue
        x.columns = [c.strip().lower() for c in x.columns]
        tcol = x.columns[0]; hcol = [c for c in x.columns if "heart" in c][0]
        y = pd.DataFrame({"t": pd.to_datetime(x[tcol], errors="coerce"), "hr": pd.to_numeric(x[hcol], errors="coerce")})
        y["steps"] = pd.to_numeric(x["steps"], errors="coerce") if "steps" in x.columns else np.nan
        frames.append(y.dropna(subset=["t"]))
    if not frames:
        return None
    hr = pd.concat(frames).sort_values("t").drop_duplicates("t").set_index("t")
    hr.loc[(hr.hr < 35) | (hr.hr > 220), "hr"] = np.nan
    return hr

def hr_features(idx, hr):
    f = pd.DataFrame(index=idx)
    if hr is None:
        return f
    m15 = hr["hr"].resample("15min", label="right", closed="right").mean()
    m60 = hr["hr"].resample("15min", label="right", closed="right").mean().rolling(4, min_periods=1).mean()
    # steps column is cumulative within a session in some exports; use positive diffs
    st = hr["steps"].diff().clip(lower=0).resample("15min", label="right", closed="right").sum(min_count=1)
    f["hr_15"] = m15.reindex(idx); f["hr_60"] = m60.reindex(idx)
    f["steps_60"] = st.rolling(4, min_periods=1).sum().reindex(idx)
    f["hr_cov"] = f["hr_15"].notna().astype(float)
    return f

# ---------- design matrix ----------
def build(G, use_events, hr=None):
    rows = []
    for _, seg in G.groupby("seg"):
        g = seg["g"]
        d = pd.DataFrame(index=seg.index)
        d["g"] = g
        for k in (1, 2, 4): d[f"d{k*STEP}"] = g - g.shift(k)
        d["accel"] = d["d15"] - (g.shift(1) - g.shift(2))
        hrs = d.index.hour + d.index.minute / 60
        d["tod_s"] = np.sin(2 * np.pi * hrs / 24); d["tod_c"] = np.cos(2 * np.pi * hrs / 24)
        for h in (2, 4): d[f"y{h*STEP}"] = g.shift(-h) - g
        rows.append(d)
    X = pd.concat(rows)
    if use_events:
        X = X.join(event_features(X.index))
    if hr is not None:
        X = X.join(hr_features(X.index, hr))
    return X
