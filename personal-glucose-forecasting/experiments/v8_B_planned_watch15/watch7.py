"""v7 watch features (user rules 31 Oct): HR mean over 15 min; steps and calories as raw totals (sums, no averaging)."""
import os, glob, numpy as np, pandas as pd
def load_watch7(folder):
    fr = []
    for p in glob.glob(os.path.join(folder, "*.csv")):
        try: x = pd.read_csv(p, skiprows=2)
        except Exception: continue
        x.columns = [c.strip().lower() for c in x.columns]
        if "heart_rate" not in x.columns: continue
        y = pd.DataFrame({"t": pd.to_datetime(x.iloc[:, 0], errors="coerce"), "hr": pd.to_numeric(x.heart_rate, errors="coerce"),
                          "steps": pd.to_numeric(x.get("steps"), errors="coerce"), "cal": pd.to_numeric(x.get("calories"), errors="coerce")})
        fr.append(y.dropna(subset=["t"]))
    w = pd.concat(fr).sort_values("t").drop_duplicates("t").set_index("t")
    w.loc[(w.hr < 35) | (w.hr > 220), "hr"] = np.nan
    return w
SET = os.environ.get("WSET", "new")   # old | new | none | newex
NEW = ["hr_15", "hr_above_rest_15", "steps_15", "steps_60", "steps_180", "cal_15", "cal_60", "cal_180", "watch_on"]
EX = ["steps_6h", "steps_24h", "cal_6h", "cal_24h"]
WATCH7 = {"new": NEW, "newex": NEW + EX, "none": [], "old": ["hr_30", "hr_above_rest_60", "steps_60", "steps_180", "watch_on"]}[SET]
def watch_features7(idx, w):
    f = pd.DataFrame(index=idx)
    r = lambda s, how: getattr(s.resample("5min", label="right", closed="right"), how)()
    hr5, st5, ca5 = r(w.hr, "mean"), r(w.steps, "sum"), r(w.cal, "sum")
    rest = hr5.rolling("24h", min_periods=50).quantile(0.1)
    f["hr_15"] = hr5.rolling(3, min_periods=1).mean().reindex(idx)
    f["hr_above_rest_15"] = (hr5 - rest).rolling(3, min_periods=1).mean().reindex(idx)
    f["hr_30"] = hr5.rolling(6, min_periods=2).mean().reindex(idx)
    f["hr_above_rest_60"] = (hr5 - rest).rolling(12, min_periods=3).mean().reindex(idx)
    for n, k in (("15", 3), ("60", 12), ("180", 36), ("6h", 72), ("24h", 288)):
        f[f"steps_{n}"] = st5.rolling(k, min_periods=1).sum().reindex(idx)
        f[f"cal_{n}"] = ca5.rolling(k, min_periods=1).sum().reindex(idx)
    f["watch_on"] = f["hr_15"].notna().astype(float)
    for c in f:
        if c.startswith("hr"): f[c] = f[c].fillna(70 if c in ("hr_15", "hr_30") else 0)
        else: f[c] = f[c].fillna(0)
    return f
