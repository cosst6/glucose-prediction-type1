"""Hybrid layer: gradient boosting on what the twin gets wrong (v10 inputs + twin outputs)."""
import numpy as np, pandas as pd
import fam11 as F
from glucoforecast8 import gbm
import glucoforecast8 as g8
PK = {"set1 Libre": "Xo", "Nov18-19 Libre": "X3", "set1 sticks": "Po", "set2 glucometer": "P2", "set3 glucometer": "P3"}
FR = {k: pd.read_pickle(f"../{v}.pkl") for k, v in PK.items()}
def cols(h): return g8.feature_list4("B", h, True) + [f"{e}_next{h}" for e in F.FEV]
def feats(df):
    out = []
    for (tag, h), g in df.groupby(["tag", "h"], sort=False):
        X = FR[tag]
        if "gap" in X: X = X[X.h == h]
        X = X[~X.index.duplicated()]
        c = cols(h); Z = X.reindex(pd.DatetimeIndex(g.t))[c]; Z.columns = [x.replace(f"_next{h}", "_nextH") for x in c]
        Z.index = g.index; out.append(Z)
    return pd.concat(out).loc[df.index]
class Resid:
    def fit(self, df, pred, d0, w):
        self.m = {}
        Fm = feats(df).assign(twin=pred - df.g0.values, d0=d0)
        for h in (60, 120):
            k = (df.h == h).values
            self.m[h] = gbm().fit(Fm[k], (df.true.values - pred)[k], sample_weight=w[k])
        return self
    def predict(self, df, pred, d0):
        Fm = feats(df).assign(twin=pred - df.g0.values, d0=d0); out = pred.copy()
        for h in (60, 120):
            k = (df.h == h).values
            if k.any(): out[k] = pred[k] + self.m[h].predict(Fm[k])
        return out
