"""GLU-B v14 food-family model (2026-11-21): v10 hybrid, but carbs go into food-family channels, each with its own timing.
Families (available carbs, g):  fast = fruit, sweets, biscuits, chocolate, candy, "x gm carb";  milk = milk, curd;
starch = rice, roti, puri, poha, idli, pumpkin;  legume = dal, moong, chana/chole, rajma, peas, besan, soy, bhujiya;
veg = cooked veg, sauce, salad.  Fat/protein stays one FPU channel (paneer, egg, oil from CIIS cooking).
A new food in a known family borrows that family's learned rise and timing.
"""
import os
os.environ.setdefault("WSET", "new"); os.environ.setdefault("FIB", "split"); os.environ.setdefault("EXF", "1")
import re, numpy as np, pandas as pd
from scipy.optimize import lsq_linear
import glucoforecast8 as g8
from glucoforecast9 import *
from parse_meal import parse_item, DB

FAMILY = {"fast": ("apple", "banana", "guava", "pomegranate", "grape", "melody", "explicit_carbs", "biscuit_hideseek", "biscuit_goodday", "chocolate", "sugar"),
          "milk": ("milk", "milk_buffalo", "milk_toned", "milk_cow", "curd"),
          "starch": ("rice", "roti", "puri", "poha", "idli", "pumpkin", "potato"),
          "legume": ("moong", "dal", "chickpea", "rajma", "peas", "besan", "soy", "bhujiya")}
FAMS = ["fast", "milk", "starch", "legume", "veg"]
# timing: (duration min, peak fraction) of a triangle absorption curve
SHAPE = {"fast": (120, 0.4), "milk": (150, 0.5), "starch": (240, 0.5), "legume": (300, 0.55), "veg": (240, 0.4)}

def fam_of(key):
    for f, keys in FAMILY.items():
        if key in keys: return f
    return "veg"

def family_carbs(food):
    d = dict.fromkeys(FAMS, 0.0)
    if isinstance(food, str):
        for p in re.split(r"[;\n]", food):
            it = parse_item(p)
            if not it or it["food"] in ("UNKNOWN", "paneer", "paneer_from_milk", "egg", "almond", "walnut", "fish_oil"): continue
            key = it["food"]
            if key == "explicit_carbs":  # "30 gm carb roti" -> family of the named food
                row = next((r.key for _, r in DB.iterrows() if re.search(r.pattern, re.sub(r".*carb", "", p.lower()))), None)
                key = row if row else "apple"
            d[fam_of(key)] += max(it["carbs"] - it["fiber"], 0)
    return d

def load_logF(path):
    L = load_log9(path)
    F = pd.DataFrame([family_carbs(f) for f in L.food])
    for f in FAMS: L[f"c_{f}"] = F[f].values
    return L

def set_shapes(shape):
    for f in FAMS:
        d, p = shape[f]; g8.KERNELS4[f"F{f}"] = (f"c_{f}", (lambda m, d=d, p=p: tri_cum(m, d, p)))
set_shapes(SHAPE)
FEV = [f"F{f}" for f in FAMS]

def strip_fam(X):  # v10 frames must not depend on family columns
    return X

FBODY = ["ins"] + FEV + ["fpu"]
def fam_design(X, h):
    return np.column_stack([X[f"{e}_next{h}"].values for e in FBODY] +
                           [np.full(len(X), h / 60), h / 60 * X["tod_s"].values, h / 60 * X["tod_c"].values, h / 60 * (X["g0"].values - 110) / 100])

class FamBody:
    names = ["mgdl_per_unit_insulin(-)"] + [f"mgdl_per_g_{f}_carb" for f in FAMS] + ["mgdl_per_FPU", "drift_per_h", "drift_tod_s", "drift_tod_c", "pull_to_110_per_h"]
    def fit(self, X, horizons):
        Ds, ys, ws = [], [], []
        for h in horizons:
            Z = X.dropna(subset=[f"y{h}"]); Ds.append(fam_design(Z, h)); ys.append(Z[f"y{h}"].values)
            ws.append(Z["w"].values if "w" in Z else np.ones(len(Z)))
        D, y, w = np.vstack(Ds), np.concatenate(ys), np.sqrt(np.concatenate(ws))
        nf = len(FEV) + 1
        lo = [-np.inf] + [0] * nf + [-np.inf] * 4; hi = [0] + [np.inf] * nf + [np.inf] * 3 + [0]
        self.coef = lsq_linear(D * w[:, None], y * w, bounds=(lo, hi)).x; return self
    def predict(self, X, h): return fam_design(X, h) @ self.coef
    def params(self): return dict(zip(self.names, [round(float(v), 3) for v in self.coef]))

TREE_MODE = os.environ.get("FTREE", "add")   # add = v10 tree inputs + family inputs; v10 = v10 tree inputs only
def fam_feats(h):
    return [f"{e}_next{h}" for e in FEV] + [f"{e}_ob" for e in FEV]

class FamHybrid(Hybrid):
    def fit(self, X, model, horizons, use_watch=True):
        self.model, self.horizons, self.use_watch = model, horizons, use_watch
        self.body = FamBody().fit(X, horizons); self.m = {}
        w_all = X["w"] if "w" in X else pd.Series(1.0, index=X.index)
        for h in horizons:
            cols = feature_list4(model, h, use_watch) + (fam_feats(h) if TREE_MODE == "add" else [])
            Z = X.dropna(subset=cols + [f"y{h}"]); w = w_all.loc[Z.index].values if "w" in X else np.ones(len(Z))
            if "w" in X: w = Z["w"].values
            F = Z[cols].assign(body=self.body.predict(Z, h)); y = Z[f"y{h}"]
            self.m[h] = {"cols": cols, "mid": gbm().fit(F, y - F.body, sample_weight=w),
                         "lo": gbm("quantile", 0.1).fit(F, y, sample_weight=w), "hi": gbm("quantile", 0.9).fit(F, y, sample_weight=w)}
        return self
