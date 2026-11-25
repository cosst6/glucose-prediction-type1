"""Same features for public people and the user, from the shared 15-min grid."""
import numpy as np, pandas as pd
from glucoforecast import insulin_cum, tri_cum
H = [15, 30, 45, 60, 90, 120, 180]
KER = {"ins": ("bolus", insulin_cum), "carb": ("carbs", lambda m: tri_cum(m, 180))}

def person_feats(f):
    f = f.copy(); n = len(f); X = pd.DataFrame(index=f.index)
    g = f.g; X["g0"] = g
    for k in (1, 2, 3, 4, 6, 8): X[f"dg{k*15}"] = g - g.shift(k)
    for name, (col, cum) in KER.items():
        a = f[col].fillna(0).values; ob = np.zeros(n); a30 = np.zeros(n); nxt = {h: np.zeros(n) for h in H}
        for lag in range(0, 33):  # up to 8 h back
            s = np.roll(a, lag); s[:lag] = 0; m = lag * 15
            c = float(cum(m)); ob += s * (1 - c); a30 += s * (c - float(cum(m - 30)))
            for h in H: nxt[h] += s * (float(cum(m + h)) - c)
        X[f"{name}_ob"] = ob; X[f"{name}_act30"] = a30
        for h in H: X[f"{name}_next{h}"] = nxt[h]
    X["basal_3h"] = f.basal.fillna(0).rolling(12, min_periods=1).sum(); X["basal_24h"] = f.basal.fillna(0).rolling(96, min_periods=1).sum()
    X["hr_1h"] = (f.hr * f.hr_mask).rolling(4, min_periods=1).mean(); X["hr_mask"] = f.hr_mask.rolling(4, min_periods=1).max()
    X["steps_1h"] = (f.steps * f.steps_mask).rolling(4, min_periods=1).sum(); X["steps_3h"] = (f.steps * f.steps_mask).rolling(12, min_periods=1).sum()
    X["steps_mask"] = f.steps_mask
    hh = f.index.hour + f.index.minute / 60; X["tod_s"] = np.sin(2 * np.pi * hh / 24); X["tod_c"] = np.cos(2 * np.pi * hh / 24)
    for h in H: X[f"y{h}"] = g.shift(-h // 15) - g
    X["pid"] = f.pid.values; X["src"] = f.src.values
    return X

def all_feats(df):
    return pd.concat([person_feats(x) for _, x in df.groupby("pid")])

EVENT = ["ins_ob", "ins_act30", "carb_ob", "carb_act30"] + [f"{e}_next{h}" for e in ("ins", "carb") for h in H]
OTHER = ["basal_3h", "basal_24h", "hr_1h", "hr_mask", "steps_1h", "steps_3h", "steps_mask", "tod_s", "tod_c"]
FAMILY = {"cgm": ["g0"] + [f"dg{k*15}" for k in (1, 2, 3, 4, 6, 8)] + EVENT + OTHER,
          "stick": ["g0"] + EVENT + OTHER}
YS = [f"y{h}" for h in H]
