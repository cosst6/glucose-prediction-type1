"""GLU-B v8 (2026-11-01): v4 hybrid + planned meals (food/insulin logged up to PLAN min ahead are known) + v7 watch rules.
v4 glucose forecaster: built to catch spikes better than the v3 straight-line models.

Changes vs v3 (see NOTE files for the why):
 1. Carbs split by speed: free sugar (fruit/sweets, fastest), glycemic load (GI x carbs, fast), slow carbs (carbs - GL), plus fat/protein.
 2. Insulin timing vs meal: minutes between the last meal and its NovoRapid dose, and carbs eaten without a dose.
 3. Curved model (gradient boosting) that can learn combinations ("fast sugar + little insulin + low start").
 4. Body-based model: a few personal numbers (mg/dL per unit, per g sugar/carb/FPU, basal drift) fitted with sign limits.
    Its forecast is fed to the curved model, which learns what the body model gets wrong (hybrid).
 5. Likely range (10th-90th percentile) and spike-risk probability (rise of 50+ within 2 h).
 6. Fast-rise alert from CGM (model A only).
"""
import numpy as np, pandas as pd
from scipy.optimize import lsq_linear
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from glucoforecast import (load_cgm, load_watch, insulin_cum, tri_cum, time_features, watch_features, WATCH, STEP,
                           HORIZONS_A, HORIZONS_B, HORIZONS_C, restrict_to_logged_days)
from parse_meal import parse_meal

def load_log4(path):
    L = pd.read_csv(path, parse_dates=["time"])
    for c in ["rapid_u", "lantus_u", "bg", "cgm_noted"]:
        L[c] = pd.to_numeric(L[c], errors="coerce")
    tot = [parse_meal(f)[1] for f in L["food"]]
    for k in ["carbs", "protein", "fat", "fiber", "sugar", "starch", "gl", "free_sugar"]:
        L[k] = [t[k] for t in tot]
    L["avail"] = (L["carbs"] - L["fiber"]).clip(lower=0)
    L["slow"] = (L["avail"] - L["gl"]).clip(lower=0)
    L["fpu"] = (9 * L["fat"] + 4 * L["protein"]) / 100
    return L.sort_values("time").reset_index(drop=True)

KERNELS4 = {
    "ins": ("rapid_u", insulin_cum),
    "fsug": ("free_sugar", lambda m: tri_cum(m, 90)),       # fruit / sweets: peak ~30 min, done by 1.5 h
    "gl": ("gl", lambda m: tri_cum(m, 150)),                 # glycemic load: peak ~50 min
    "slow": ("slow", lambda m: tri_cum(m, 240, 0.4)),        # low-GI part of carbs: peak ~95 min
    "fpu": ("fpu", lambda m: tri_cum(m, 300, 0.5)),          # fat + protein: peak 2.5 h
}

PLAN = 120   # minutes of planned food/insulin known ahead
def event_features4(idx, log, horizons):
    tt = idx.values.astype("datetime64[m]").astype(np.int64)[:, None].astype(float)
    et = log["time"].values.astype("datetime64[m]").astype(np.int64)[None, :].astype(float)
    m = tt - et; known = m >= -PLAN; f = {}
    for name, (col, cum) in KERNELS4.items():
        amt = np.nan_to_num(log[col].values.astype(float))[None, :] * known
        c = cum(m)
        f[f"{name}_ob"] = (amt * (1 - c)).sum(1)
        f[f"{name}_act30"] = (amt * (c - cum(m - 30))).sum(1)
        for h in horizons:
            f[f"{name}_next{h}"] = (amt * (cum(m + h) - c)).sum(1)
    # Lantus
    lt = log.loc[log.lantus_u > 0, "time"].values
    pos = np.searchsorted(lt, idx.values, side="right") - 1
    f["h_since_lantus"] = np.where(pos >= 0, (idx.values - lt[np.clip(pos, 0, None)]) / np.timedelta64(1, "h"), 24.0) if len(lt) else np.full(len(idx), 24.0)  # fix Nov 21: no Lantus logged -> no crash
    lant = np.nan_to_num(log["lantus_u"].values.astype(float))[None, :] * (m >= 0)
    f["lantus_ob"] = (lant * np.clip(1 - m / 1440, 0, 1)).sum(1)
    # insulin timing vs meal: for the latest meal (>=5 g available carbs) in the last 4 h
    meals = log[log.avail >= 5]; doses = log[log.rapid_u > 0]
    gap, uncovered, since = [], [], []
    for t in idx:
        mm = meals[(meals.time <= t + pd.Timedelta(minutes=PLAN)) & (meals.time > t - pd.Timedelta("4h"))]
        if len(mm) == 0:
            gap.append(0.0); uncovered.append(0.0); since.append(240.0); continue
        r = mm.iloc[-1]; since.append((t - r.time).total_seconds() / 60)
        d = doses[(doses.time >= r.time - pd.Timedelta("60min")) & (doses.time <= min(t, r.time + pd.Timedelta("90min")))]
        if len(d) == 0:
            gap.append(0.0); uncovered.append(r.avail)
        else:
            gap.append((d.time.iloc[0] - r.time).total_seconds() / 60); uncovered.append(0.0)  # + = dose after eating
    f["bolus_gap"] = gap; f["uncovered_carbs"] = uncovered; f["min_since_meal"] = since
    return pd.DataFrame(f, index=idx)

def build_frame4(cgm, log, watch, horizons):
    rows = []
    for _, seg in cgm.groupby("seg"):
        g = seg["g"]; d = pd.DataFrame(index=seg.index); d["g0"] = g
        for k in (1, 2, 4): d[f"d{k*STEP}"] = g - g.shift(k)
        d["accel"] = d["d15"] - (g.shift(1) - g.shift(2))
        for h in horizons: d[f"y{h}"] = g.shift(-h // STEP) - g
        fut = pd.concat([g.shift(-k) for k in range(1, 9)], axis=1)  # next 2 h
        d["rise2h"] = fut.max(axis=1) - g
        d.loc[fut.isna().any(axis=1), "rise2h"] = np.nan
        rows.append(d)
    X = pd.concat(rows)
    X = X.join(event_features4(X.index, log, horizons)).join(time_features(X.index)).join(watch_features(X.index, watch))
    for c in WATCH:
        if c in X: X[c] = X[c].fillna(0 if c != "hr_30" else 70)
    return X

EV = ["ins", "fsug", "gl", "slow", "fpu"]
def feature_list4(model, h, use_watch=True):
    fut = [f"{e}_next{h}" for e in EV]
    cur = [f"{e}_{k}" for e in EV for k in ("ob", "act30")]
    core = ["g0"] + fut + cur + ["h_since_lantus", "lantus_ob", "bolus_gap", "uncovered_carbs", "min_since_meal", "tod_s", "tod_c"]
    if model in ("A", "B") and use_watch: core += WATCH
    if model == "A": core += ["d15", "d30", "d60", "accel"]
    return core

# ---------------------------------------------------------------- body-based model (one parameter set for all horizons)
PHYS_COLS = ["ins", "fsug", "gl", "slow", "fpu"]
def _phys_design(X, h):
    D = np.column_stack([X[f"{e}_next{h}"].values for e in PHYS_COLS] +
                        [np.full(len(X), h / 60), h / 60 * X["tod_s"].values, h / 60 * X["tod_c"].values,
                         h / 60 * (X["g0"].values - 110) / 100])
    return D

class BodyModel:
    """dG(h) = -ISF*insulin_acting + k_s*free_sugar + k_gl*GL + k_slow*slow_carbs + k_f*FPU + drift(h, time of day) - pull back toward ~110."""
    names = ["mgdl_per_unit_insulin(-)", "mgdl_per_g_free_sugar", "mgdl_per_g_GL", "mgdl_per_g_slow_carb", "mgdl_per_FPU",
             "drift_per_h", "drift_tod_s", "drift_tod_c", "pull_to_110_per_h"]
    def fit(self, X, horizons):
        Ds, ys = [], []
        for h in horizons:
            Z = X.dropna(subset=[f"y{h}"]); Ds.append(_phys_design(Z, h)); ys.append(Z[f"y{h}"].values)
        D, y = np.vstack(Ds), np.concatenate(ys)
        lo = [-np.inf, 0, 0, 0, 0, -np.inf, -np.inf, -np.inf, -np.inf]; hi = [0, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 0]
        self.coef = lsq_linear(D, y, bounds=(lo, hi)).x
        return self
    def predict(self, X, h):
        return _phys_design(X, h) @ self.coef
    def params(self):
        return dict(zip(self.names, [round(float(v), 3) for v in self.coef]))

def gbm(loss="squared_error", q=None):
    kw = dict(max_depth=3, learning_rate=0.05, max_iter=250, min_samples_leaf=25, l2_regularization=1.0, random_state=0)
    return HistGradientBoostingRegressor(loss=loss, quantile=q, **kw) if q else HistGradientBoostingRegressor(**kw)

class Hybrid:
    """Body model forecast + gradient boosting on what it gets wrong. Also 10th/90th percentile range and spike risk."""
    def fit(self, X, model, horizons, use_watch=True):
        self.model, self.horizons, self.use_watch = model, horizons, use_watch
        self.body = BodyModel().fit(X, horizons); self.m = {}
        for h in horizons:
            cols = feature_list4(model, h, use_watch); Z = X.dropna(subset=cols + [f"y{h}"])
            F = Z[cols].assign(body=self.body.predict(Z, h)); y = Z[f"y{h}"]
            self.m[h] = {"cols": cols, "mid": gbm().fit(F, y - F.body),
                         "lo": gbm("quantile", 0.1).fit(F, y), "hi": gbm("quantile", 0.9).fit(F, y)}
        h2 = 120 if 120 in horizons else horizons[-1]; cols = feature_list4(model, h2, use_watch)
        Z = X.dropna(subset=cols + ["rise2h"])
        self.spike = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=25, random_state=0)
        self.spike.fit(Z[cols].assign(body=self.body.predict(Z, h2)), (Z.rise2h >= 50).astype(int)); self.spike_cols = (cols, h2)
        return self
    def predict(self, X, h):
        r = self.m[h]; F = X[r["cols"]].assign(body=self.body.predict(X, h))
        mid = F.body + r["mid"].predict(F); lo, hi = r["lo"].predict(F), r["hi"].predict(F)
        return pd.DataFrame({"pred": X.g0 + mid, "lo": X.g0 + np.minimum(lo, mid), "hi": X.g0 + np.maximum(hi, mid),
                             "body": X.g0 + F.body}, index=X.index)
    def spike_risk(self, X):
        cols, h2 = self.spike_cols
        return self.spike.predict_proba(X[cols].assign(body=self.body.predict(X, h2)))[:, 1]

def rise_alert(X, per_min=2.0):
    """CGM-only alert: rising faster than per_min mg/dL per minute over the last 15 min."""
    return X["d15"] >= per_min * 15
from watch7 import WATCH7 as WATCH, watch_features7 as watch_features, load_watch7 as load_watch
