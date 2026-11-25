import os, glob, numpy as np, pandas as pd, joblib, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("WSET","new"); os.environ.setdefault("FIB","split"); os.environ.setdefault("EXF","1")
import glucoforecast9 as G
from glucoforecast9 import *
R = 0.72
v10 = joblib.load(glob.glob("glucose-model/v9_v10_fiber_exercise/*v10*/*.joblib")[0])
def feats(idx, log, watch, H, g0=None):
    """features at times idx for a window of H minutes; meals/insulin planned up to H ahead are known"""
    G.g8.PLAN = H - 1  # events AT or after the target time are not known (e.g. low treatment)
    E = G.g8.event_features4(idx, log, [H]); W = watch_features(idx, watch); T = time_features(idx)
    X = E.join(W).join(T)
    X["g0"] = 110.0 if g0 is None else g0
    X["body"] = G.g8._phys_design(X, H) @ v10.body.coef / R  # body-model change, blood scale
    X["H"] = H
    for e in ["ins", "fsug", "gl", "slow", "fpu"]: X[f"{e}_nextH"] = X.pop(f"{e}_next{H}")
    X["carb_ob"] = X.fsug_ob + X.gl_ob + X.slow_ob
    X["carb_next"] = X.fsug_nextH + X.gl_nextH + X.slow_nextH
    return X
NOG = ["H", "body", "ins_ob", "ins_nextH", "carb_ob", "carb_next", "fpu_ob", "fpu_nextH", "lantus_ob", "h_since_lantus", "tod_s", "tod_c", "hr_15", "steps_60", "cal_60"]
ONE = NOG + ["g0", "pred", "age"]
