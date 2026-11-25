"""v5 glucose forecaster = v4 + fixes found in the 30 Oct error analysis.

 1. Milk sugar is its own fast channel (peaks ~70 min), not 'slow carbs'.
 2. Fat/protein peak moved from 2.5 h to 3.5 h; slow carbs from 1.6 h to 2.5 h.
 3. Time of day: 3-hour clock blocks + night food (eaten 22:00-03:00) as its own channels.
 4. Lantus gap: flag when the last Lantus is >22 h old (old dose fading) and dawn window 03-08.
 5. Low rebound: how far below 70 now, and (CGM) lowest value in last 2 h.
 6. Starting high: distance above 140.
 7. Planned-meal mode: food + insulin logged up to h minutes ahead can be fed in.
"""
import numpy as np, pandas as pd
from glucoforecast4 import (load_cgm, load_watch, insulin_cum, tri_cum, time_features, watch_features, WATCH, STEP,
                            HORIZONS_A, HORIZONS_B, HORIZONS_C, restrict_to_logged_days, gbm, BodyModel, load_log4)
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.optimize import lsq_linear
from parse_meal import parse_meal, DAIRY

def load_log5(path):
    L = load_log4(path)
    dairy = []
    for f in L.food:
        items, _ = parse_meal(f)
        m = [i for i in items if i["food"] in DAIRY and not i["food"].startswith("paneer")]
        av = sum(max(i["carbs"] - i["fiber"], 0) for i in m); gl = sum(i["gl"] for i in m)
        dairy.append((av, gl))
    d = np.array(dairy) if len(dairy) else np.zeros((0, 2))
    L["dairy"] = d[:, 0]
    L["gl"] = (L.gl - d[:, 1]).clip(lower=0)                     # milk no longer counted in GL / slow
    L["slow"] = (L.avail - L.dairy - L.gl).clip(lower=0)
    night = (L.time.dt.hour >= 22) | (L.time.dt.hour < 3)
    L["night_carb"] = np.where(night, L.avail, 0.0); L["night_fpu"] = np.where(night, L.fpu, 0.0)
    return L

KERNELS5 = {
    "ins": ("rapid_u", insulin_cum),
    "fsug": ("free_sugar", lambda m: tri_cum(m, 90)),
    "dairy": ("dairy", lambda m: tri_cum(m, 210, 0.33)),        # milk: peak ~70 min, tail to 3.5 h
    "gl": ("gl", lambda m: tri_cum(m, 150)),
    "slow": ("slow", lambda m: tri_cum(m, 300, 0.5)),           # peak 2.5 h
    "fpu": ("fpu", lambda m: tri_cum(m, 420, 0.5)),             # peak 3.5 h, done by 7 h
    "ncarb": ("night_carb", lambda m: tri_cum(m, 360, 0.5)),    # extra channel for food eaten 22-03 h
    "nfpu": ("night_fpu", lambda m: tri_cum(m, 480, 0.5)),
}
import os
LEAN = os.environ.get("LEAN") == "1"
EV = [e for e in KERNELS5 if not (LEAN and e in ("ncarb", "nfpu"))]

def event_features5(idx, log, horizons, plan=0):
    """plan>0: food/insulin logged up to `plan` minutes after the forecast time is treated as known (planned meal)."""
    tt = idx.values.astype("datetime64[m]").astype(np.int64)[:, None].astype(float)
    et = log["time"].values.astype("datetime64[m]").astype(np.int64)[None, :].astype(float)
    m = tt - et; known = m >= -plan; f = {}
    for name, (col, cum) in KERNELS5.items():
        amt = np.nan_to_num(log[col].values.astype(float))[None, :] * known
        c = cum(m)
        f[f"{name}_ob"] = (amt * (1 - c)).sum(1)
        f[f"{name}_act30"] = (amt * (c - cum(m - 30))).sum(1)
        for h in horizons:
            f[f"{name}_next{h}"] = (amt * (cum(m + h) - c)).sum(1)
    lt = log.loc[log.lantus_u > 0, "time"].values
    pos = np.searchsorted(lt, idx.values, side="right") - 1
    hs = np.where(pos >= 0, (idx.values - lt[np.clip(pos, 0, None)]) / np.timedelta64(1, "h"), 24.0)
    f["h_since_lantus"] = hs; f["lantus_old"] = np.clip(hs - 22, 0, 6)
    lant = np.nan_to_num(log["lantus_u"].values.astype(float))[None, :] * (m >= 0)
    f["lantus_ob"] = (lant * np.clip(1 - m / 1440, 0, 1)).sum(1)
    meals = log[log.avail >= 5]; doses = log[log.rapid_u > 0]; unc, since = [], []
    for t in idx:
        mm = meals[(meals.time <= t + pd.Timedelta(minutes=plan)) & (meals.time > t - pd.Timedelta("4h"))]
        if len(mm) == 0: unc.append(0.0); since.append(240.0); continue
        r = mm.iloc[-1]; since.append(max((t - r.time).total_seconds() / 60, 0))
        d = doses[(doses.time >= r.time - pd.Timedelta("60min")) & (doses.time <= r.time + pd.Timedelta("90min"))]
        unc.append(0.0 if len(d) else r.avail)
    f["uncovered_carbs"] = unc; f["min_since_meal"] = since
    return pd.DataFrame(f, index=idx)

def clock_features(idx):
    h = idx.hour + idx.minute / 60; T = time_features(idx)
    for a in range(0, 24, 3): T[f"clk{a:02d}"] = ((h >= a) & (h < a + 3)).astype(float)
    T["dawn"] = ((h >= 3) & (h < 8)).astype(float)
    return T

def build_frame5(cgm, log, watch, horizons, plan=0):
    rows = []
    for _, seg in cgm.groupby("seg"):
        g = seg["g"]; d = pd.DataFrame(index=seg.index); d["g0"] = g
        for k in (1, 2, 4): d[f"d{k*STEP}"] = g - g.shift(k)
        d["accel"] = d["d15"] - (g.shift(1) - g.shift(2))
        d["gmin2h"] = g.rolling(8, min_periods=1).min()
        for h in horizons: d[f"y{h}"] = g.shift(-h // STEP) - g
        fut = pd.concat([g.shift(-k) for k in range(1, 9)], axis=1)
        d["rise2h"] = fut.max(axis=1) - g; d.loc[fut.isna().any(axis=1), "rise2h"] = np.nan
        rows.append(d)
    X = pd.concat(rows)
    X["below70"] = np.clip(70 - X.g0, 0, None); X["above140"] = np.clip(X.g0 - 140, 0, None)
    X["low_recent"] = np.clip(70 - X.gmin2h, 0, None)
    X = X.join(event_features5(X.index, log, horizons, plan)).join(clock_features(X.index)).join(watch_features(X.index, watch))
    for c in WATCH:
        if c in X: X[c] = X[c].fillna(0 if c != "hr_30" else 70)
    if os.environ.get("CTX48") == "1": X = X.join(context48(X, cgm, log, horizons))
    return X

CLK = [f"clk{a:02d}" for a in range(0, 24, 3)]
def feature_list5(model, h, use_watch=True):
    fut = [f"{e}_next{h}" for e in EV]; cur = [f"{e}_{k}" for e in EV for k in ("ob", "act30")]
    if LEAN:
        core = ["g0"] + fut + cur + ["h_since_lantus", "lantus_ob", "uncovered_carbs", "min_since_meal", "tod_s", "tod_c"]
    else:
        core = ["g0", "below70", "above140"] + fut + cur + ["h_since_lantus", "lantus_old", "lantus_ob", "uncovered_carbs", "min_since_meal",
                                                         "tod_s", "tod_c", "dawn"] + CLK
    if model in ("A", "B") and use_watch: core += WATCH
    if model == "A": core += ["d15", "d30", "d60", "accel"] + ([] if LEAN else ["low_recent"])
    if model == "A" and os.environ.get("CTX48") == "1": core += CTX_COLS + [f"yday_move{h}", f"d2_move{h}"]
    return core

PHYS = ["ins", "fsug", "dairy", "gl", "slow", "fpu", "ncarb", "nfpu"]
def _design(X, h):
    k = h / 60
    return np.column_stack([X[f"{e}_next{h}"].values for e in PHYS] +
                           [np.full(len(X), k)] + [k * X[c].values for c in CLK[1:]] +
                           [k * X["lantus_old"].values, k * X["below70"].values / 10, k * (X["g0"].values - 110) / 100])

class BodyModel5(BodyModel):
    names = ["per_unit_insulin(-)", "per_g_free_sugar", "per_g_milk_sugar", "per_g_GL", "per_g_slow_carb", "per_FPU", "night_extra_per_g_carb", "night_extra_per_FPU",
             "drift_per_h"] + [f"drift_{c}_per_h" for c in CLK[1:]] + ["lantus_old_per_h_per_h", "low_rebound_per_h_per_10below70", "pull_to_110_per_h"]
    def fit(self, X, horizons):
        Ds, ys = [], []
        for h in horizons:
            Z = X.dropna(subset=[f"y{h}"]); Ds.append(_design(Z, h)); ys.append(Z[f"y{h}"].values)
        D, y = np.vstack(Ds), np.concatenate(ys); n = D.shape[1]
        lo = [-np.inf] + [0] * 7 + [-np.inf] * (n - 11) + [0, 0, -np.inf]
        hi = [0] + [np.inf] * 7 + [np.inf] * (n - 11) + [np.inf, np.inf, 0]
        self.coef = lsq_linear(D, y, bounds=(lo, hi)).x; return self
    def predict(self, X, h): return _design(X, h) @ self.coef

class Hybrid5:
    def fit(self, X, model, horizons, use_watch=True):
        self.model, self.horizons = model, horizons
        self.body = BodyModel5().fit(X, horizons); self.m = {}
        for h in horizons:
            cols = feature_list5(model, h, use_watch); Z = X.dropna(subset=cols + [f"y{h}"])
            F = Z[cols].assign(body=self.body.predict(Z, h)); y = Z[f"y{h}"]
            self.m[h] = {"cols": cols, "mid": gbm().fit(F, y - F.body), "lo": gbm("quantile", 0.1).fit(F, y), "hi": gbm("quantile", 0.9).fit(F, y)}
        h2 = 120 if 120 in horizons else horizons[-1]; cols = feature_list5(model, h2, use_watch)
        Z = X.dropna(subset=cols + ["rise2h"])
        self.spike = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=25, random_state=0)
        self.spike.fit(Z[cols].assign(body=self.body.predict(Z, h2)), (Z.rise2h >= 50).astype(int)); self.spike_cols = (cols, h2)
        return self
    def predict(self, X, h):
        r = self.m[h]; F = X[r["cols"]].assign(body=self.body.predict(X, h))
        mid = F.body + r["mid"].predict(F); lo, hi = r["lo"].predict(F), r["hi"].predict(F)
        return pd.DataFrame({"pred": X.g0 + mid, "lo": X.g0 + np.minimum(lo, mid), "hi": X.g0 + np.maximum(hi, mid), "body": X.g0 + F.body}, index=X.index)
    def spike_risk(self, X):
        cols, h2 = self.spike_cols
        return self.spike.predict_proba(X[cols].assign(body=self.body.predict(X, h2)))[:, 1]

# ---------------------------------------------------------------- 2-day context (CTX48=1)
CTX48 = os.environ.get("CTX48") == "1"
CTX_COLS = ["ctx_hours", "g_mean6h", "g_mean24h", "g_mean48h", "g_sd24h", "g_sd48h", "low_frac24h", "high_frac24h", "g_yday", "g_2dago",
            "carbs24h", "carbs48h", "rapid24h", "rapid48h", "fpu24h", "steps24h", "lantus_gap_prev"]
def context48(X, cgm, log, horizons):
    """Summaries of up to the last 48 h. Uses whatever exists (less if the user just started), never more than 48 h."""
    g = cgm["g"].resample("15min").mean()          # regular grid, NaN where no reading
    r = lambda w, f: getattr(g.rolling(w, min_periods=1), f)().reindex(X.index)
    C = pd.DataFrame(index=X.index)
    C["ctx_hours"] = g.notna().rolling("48h").sum().reindex(X.index) / 4
    C["g_mean6h"], C["g_mean24h"], C["g_mean48h"] = r("6h", "mean"), r("24h", "mean"), r("48h", "mean")
    C["g_sd24h"], C["g_sd48h"] = r("24h", "std"), r("48h", "std")
    C["low_frac24h"] = (g < 70).astype(float).where(g.notna()).rolling("24h", min_periods=1).mean().reindex(X.index)
    C["high_frac24h"] = (g > 180).astype(float).where(g.notna()).rolling("24h", min_periods=1).mean().reindex(X.index)
    gi = lambda off: g.reindex(X.index - pd.Timedelta(off)).values
    C["g_yday"], C["g_2dago"] = gi("24h"), gi("48h")
    for h in horizons:   # what glucose did over the same h minutes yesterday / 2 days ago
        C[f"yday_move{h}"] = gi(pd.Timedelta("24h") - pd.Timedelta(minutes=h)) - C["g_yday"].values
        C[f"d2_move{h}"] = gi(pd.Timedelta("48h") - pd.Timedelta(minutes=h)) - C["g_2dago"].values
    t = log.time.values
    def tot(col, hrs):
        v = np.nan_to_num(log[col].values.astype(float)); out = []
        for x in X.index.values:
            m = (t <= x) & (t > x - np.timedelta64(hrs, "h")); out.append(v[m].sum())
        return np.array(out)
    C["carbs24h"], C["carbs48h"], C["rapid24h"], C["rapid48h"], C["fpu24h"] = tot("avail", 24), tot("avail", 48), tot("rapid_u", 24), tot("rapid_u", 48), tot("fpu", 24)
    C["steps24h"] = X["steps_60"].rolling("24h", min_periods=1).sum() / 4 if "steps_60" in X else 0.0
    lt = log.loc[log.lantus_u > 0, "time"].values; gap = []
    for x in X.index.values:
        p = lt[lt <= x]; gap.append((p[-1] - p[-2]) / np.timedelta64(1, "h") if len(p) >= 2 else 24.0)
    C["lantus_gap_prev"] = gap
    # nothing available yet -> neutral values, ctx_hours tells the model how much history it really had
    for c in ["g_mean6h", "g_mean24h", "g_mean48h", "g_yday", "g_2dago"]: C[c] = C[c].fillna(X["g0"])
    return C.fillna(0.0)
