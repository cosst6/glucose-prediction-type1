"""Weighted Hybrid: same as v10 Hybrid but rows carry weights (fingersticks heavier)."""
import numpy as np, pandas as pd
from scipy.optimize import lsq_linear
from sklearn.ensemble import HistGradientBoostingClassifier
import glucoforecast9 as G
from glucoforecast9 import *
class WBody(BodyModel):
    def fit(self, X, horizons):
        Ds, ys, ws = [], [], []
        for h in horizons:
            Z = X.dropna(subset=[f"y{h}"]); Z = Z[(Z.get("h", h) == h) if "h" in Z else slice(None)] if False else Z
            Ds.append(G.g8._phys_design(Z, h)); ys.append(Z[f"y{h}"].values); ws.append(Z["w"].values)
        D, y, w = np.vstack(Ds), np.concatenate(ys), np.sqrt(np.concatenate(ws))
        lo = [-np.inf, 0, 0, 0, 0, -np.inf, -np.inf, -np.inf, -np.inf]; hi = [0, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 0]
        self.coef = lsq_linear(D * w[:, None], y * w, bounds=(lo, hi)).x; return self
class WHybrid(Hybrid):
    def fit(self, X, model, horizons, use_watch=True):
        self.model, self.horizons, self.use_watch = model, horizons, use_watch
        self.body = WBody().fit(X, horizons); self.m = {}
        for h in horizons:
            cols = feature_list4(model, h, use_watch); Z = X.dropna(subset=cols + [f"y{h}"])
            F = Z[cols].assign(body=self.body.predict(Z, h)); y = Z[f"y{h}"]; w = Z.w.values
            self.m[h] = {"cols": cols, "mid": gbm().fit(F, y - F.body, sample_weight=w),
                         "lo": gbm("quantile", 0.1).fit(F, y, sample_weight=w), "hi": gbm("quantile", 0.9).fit(F, y, sample_weight=w)}
        return self
