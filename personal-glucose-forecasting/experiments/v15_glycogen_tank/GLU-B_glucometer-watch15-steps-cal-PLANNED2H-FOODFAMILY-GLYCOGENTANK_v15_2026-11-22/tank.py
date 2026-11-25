"""Glycogen tank (2026-11-22): estimated liver + muscle glycogen, simulated every 5 min from the past only (causal).
Liver tank 100 g, muscle tank 400 g (start of each data set: liver 60, muscle 250).
Fill: available carbs (carbs - fiber) absorbed over a 3 h triangle; of each gram absorbed, 40% goes to the liver
      and 30% to muscle (if room); the rest is burned right away. Protein: 0.5 g glucose per g (gluconeogenesis, over 5 h) -> liver.
Drain: liver 4.5 g/h while fasting (absorption < 5 g/h), 1 g/h while carbs are coming in;
       muscle = watch active calories x CARB_FRAC / 4 kcal per g (easy walking on a low-carb, high-fat diet ~15% from carbs).
Features: tank grams, fullness (0-1), grams drained in last 6 h, hours fasting.
"""
import os, numpy as np, pandas as pd
from scipy.optimize import lsq_linear
import glucoforecast9 as G
from glucoforecast9 import feature_list4, gbm
import glucoforecast8 as g8
from whyb import WHybrid
import fam11 as F

LIV_CAP, MUS_CAP, LIV0, MUS0 = 100.0, 400.0, 60.0, 250.0
LIV_SHARE, MUS_SHARE = float(os.environ.get("LIVS", 0.4)), float(os.environ.get("MUSS", 0.3))
GNG = 0.5   # g glucose made per g protein (absorbed over 5 h) -> liver
FAST_DRAIN, FED_DRAIN, CARB_FRAC = 4.5, 1.0, 0.15
TANK = ["liv_g", "mus_g", "liv_frac", "mus_frac", "liv_drain6h", "mus_drain6h", "fast_h"]

def tank_series(log, watch):
    t0 = log.time.min().floor("5min"); t1 = log.time.max().ceil("1D") + pd.Timedelta("1D")
    T = pd.date_range(t0, t1, freq="5min"); n = len(T)
    # carb absorption per 5-min step (g), triangle over 180 min peak at 1/3
    tm = T.values.astype("datetime64[m]").astype(np.int64).astype(float)
    et = log.time.values.astype("datetime64[m]").astype(np.int64).astype(float); amt = np.nan_to_num(log.avail.values.astype(float))
    ab = np.zeros(n)
    for e, a in zip(et, amt):
        if a <= 0: continue
        m = tm - e; c1 = G.tri_cum(m, 180, 1/3); c0 = G.tri_cum(m - 5, 180, 1/3); ab += a * (c1 - c0)
    gng = np.zeros(n)
    for e, p in zip(et, np.nan_to_num(log.protein.values.astype(float))):
        if p <= 0: continue
        m = tm - e; gng += GNG * p * (G.tri_cum(m, 300, 0.5) - G.tri_cum(m - 5, 300, 0.5))
    cal = np.zeros(n)
    if watch is not None and len(watch):
        c = watch.cal.clip(upper=10).resample("5min", label="right", closed="right").sum().clip(upper=60)
        cal = c.reindex(T).fillna(0).values
    liv, mus = np.zeros(n), np.zeros(n); ld, md = np.zeros(n), np.zeros(n); fh = np.zeros(n)
    L, M, fast = LIV0, MUS0, 0.0
    for i in range(n):
        a = ab[i]
        L += min(a * LIV_SHARE + gng[i], LIV_CAP - L); M += min(a * MUS_SHARE, MUS_CAP - M)
        rate = a * 12  # g/h
        fast = fast + 5 / 60 if rate < 5 else 0.0
        dl = (FAST_DRAIN if rate < 5 else FED_DRAIN) * 5 / 60; dm = cal[i] * CARB_FRAC / 4
        dl, dm = min(dl, L - 5), min(dm, M - 20)
        L -= max(dl, 0); M -= max(dm, 0)
        liv[i], mus[i], ld[i], md[i], fh[i] = L, M, max(dl, 0), max(dm, 0), fast
    S = pd.DataFrame({"liv_g": liv, "mus_g": mus, "liv_frac": liv / LIV_CAP, "mus_frac": mus / MUS_CAP,
                      "liv_drain6h": pd.Series(ld).rolling(72, min_periods=1).sum().values,
                      "mus_drain6h": pd.Series(md).rolling(72, min_periods=1).sum().values, "fast_h": np.minimum(fh, 24)}, index=T)
    return S

def add_tank(X, log, watch):
    S = tank_series(log, watch)
    Y = X.copy()
    k = S.reindex(pd.DatetimeIndex(X.index).floor("5min"), method="ffill")
    for c in TANK: Y[c] = k[c].values
    return Y

BODY = os.environ.get("TBODY", "0") == "1"   # also give the body model a liver-fullness drift term
def _tank_cols(Z, h): return [h / 60 * (Z["liv_frac"].values - 0.5), h / 60 * (Z["mus_frac"].values - 0.6)]

class TWBody(G.g8.BodyModel):
    def fit(self, X, horizons):
        Ds, ys, ws = [], [], []
        for h in horizons:
            Z = X.dropna(subset=[f"y{h}"]); Ds.append(self.design(Z, h)); ys.append(Z[f"y{h}"].values); ws.append(Z["w"].values)
        D, y, w = np.vstack(Ds), np.concatenate(ys), np.sqrt(np.concatenate(ws))
        lo = [-np.inf, 0, 0, 0, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]; hi = [0, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 0, np.inf, np.inf]
        self.coef = lsq_linear(D * w[:, None], y * w, bounds=(lo, hi)).x; return self
    def design(self, Z, h): return np.column_stack([g8._phys_design(Z, h)] + _tank_cols(Z, h))
    def predict(self, X, h): return self.design(X, h) @ self.coef
    def params(self): return dict(zip(self.names + ["liver_fullness_drift_per_h", "muscle_fullness_drift_per_h"], [round(float(v), 3) for v in self.coef]))

class TFamBody(F.FamBody):
    def fit(self, X, horizons):
        Ds, ys, ws = [], [], []
        for h in horizons:
            Z = X.dropna(subset=[f"y{h}"]); Ds.append(self.design(Z, h)); ys.append(Z[f"y{h}"].values); ws.append(Z["w"].values)
        D, y, w = np.vstack(Ds), np.concatenate(ys), np.sqrt(np.concatenate(ws))
        nf = len(F.FEV) + 1
        lo = [-np.inf] + [0] * nf + [-np.inf] * 4 + [-np.inf] * 2; hi = [0] + [np.inf] * nf + [np.inf] * 3 + [0] + [np.inf] * 2
        self.coef = lsq_linear(D * w[:, None], y * w, bounds=(lo, hi)).x; return self
    def design(self, Z, h): return np.column_stack([F.fam_design(Z, h)] + _tank_cols(Z, h))
    def predict(self, X, h): return self.design(X, h) @ self.coef
    def params(self): return dict(zip(self.names + ["liver_fullness_drift_per_h", "muscle_fullness_drift_per_h"], [round(float(v), 3) for v in self.coef]))

def _fit_trees(self, X, extra):
    self.m = {}
    for h in self.horizons:
        cols = feature_list4(self.model, h, self.use_watch) + extra(h) + TANK
        Z = X.dropna(subset=cols + [f"y{h}"]); w = Z.w.values
        Fm = Z[cols].assign(body=self.body.predict(Z, h)); y = Z[f"y{h}"]
        self.m[h] = {"cols": cols, "mid": gbm().fit(Fm, y - Fm.body, sample_weight=w),
                     "lo": gbm("quantile", 0.1).fit(Fm, y, sample_weight=w), "hi": gbm("quantile", 0.9).fit(Fm, y, sample_weight=w)}
    return self

class TankHybrid(WHybrid):      # v10 + tank
    def fit(self, X, model, horizons, use_watch=True):
        self.model, self.horizons, self.use_watch = model, horizons, use_watch
        self.body = (TWBody() if BODY else __import__("whyb").WBody()).fit(X, horizons)
        return _fit_trees(self, X, lambda h: [])

class TankFamHybrid(F.FamHybrid):   # v14 + tank
    def fit(self, X, model, horizons, use_watch=True):
        self.model, self.horizons, self.use_watch = model, horizons, use_watch
        self.body = (TFamBody() if BODY else F.FamBody()).fit(X, horizons)
        return _fit_trees(self, X, F.fam_feats)
