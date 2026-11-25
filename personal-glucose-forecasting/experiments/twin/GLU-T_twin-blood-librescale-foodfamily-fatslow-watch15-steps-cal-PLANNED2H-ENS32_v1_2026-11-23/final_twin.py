"""Train the final GLU-T twin on ALL data: MAP + 32-member day-bootstrap ensemble + out-of-bag noise model."""
import os, json, numpy as np, pandas as pd
os.environ.setdefault("ENS", "32")
import wf
from wf import *
S1 = TABS["set1 Libre"]; X3 = TABS["Nov18-19 Libre"]
ST = pd.concat([TABS["set1 sticks"], TABS["set2 glucometer"], TABS["set3 glucometer"]])
parts = [(S1, 1.0), (ST, STICK_W), (X3, 1.0)]
tr = sum([groups(p, w) for p, w in parts], [])
x, sol = fit(tr, lam=LAM, fixed=FIX)
print("MAP", {n: round(float(v), 4) for n, v in T.unpack(x).items()})
# ensemble members
rng = np.random.default_rng(7)
TR = pd.concat([q.assign(w=w) for q, w in parts]).reset_index(drop=True)
day = TR.t.dt.strftime("%Y-%m-%d") + TR.lib.astype(str); ud = day.unique()
members, oob_r, oob_i = [], [], []
for b in range(int(os.environ["ENS"])):
    cnt = pd.Series(rng.choice(ud, len(ud))).value_counts(); m = day.map(cnt).fillna(0).values
    sub = TR[m > 0]; ww = (sub.w * m[m > 0]).values
    xb, _ = fit([make(g, ww[np.searchsorted(sub.index.values, g.index.values)]) for _, g in sub.groupby("tl")], x0=x, lam=LAM, fixed=FIX)
    members.append(xb); oo = TR[m == 0]
    if len(oo): oob_r.append(tpred(xb, oo)[0] - oo.true.values); oob_i.append(oo.index.values)
    print("member", b, flush=True)
r = np.concatenate(oob_r); O = TR.loc[np.concatenate(oob_i)].reset_index(drop=True); Do = noise_design(O)
noise = {}
for h in (60, 120):
    k = ((O.h == h) & O.lib).values
    coef = np.linalg.lstsq(Do[k], np.abs(r[k]) * np.sqrt(np.pi / 2), rcond=None)[0]
    zq = np.quantile(np.abs(r[k]) / np.maximum(Do[k] @ coef, 8.0), 0.8)
    ks = ((O.h == h) & ~O.lib).values
    noise[h] = {"libre_coef": coef.tolist(), "libre_z80": float(zq), "stick_q80": float(np.quantile(np.abs(r[ks]), 0.8)) if ks.sum() >= 15 else None}
    print("OOB rmse h", h, "libre", round(rmse(r[k]), 1), "sticks", round(rmse(r[ks]), 1) if ks.any() else None, noise[h])
M = np.array(members)
card = {"name": os.environ.get("TNAME", "GLU-T"), "param_names": T.NAMES, "map": x.tolist(), "members": M.tolist(),
        "noise_sigma_coef": {"design": ["1", "insulin_units_2h_before_to_end", "carbs_g_/10", "g0_/100"], **{str(h): v for h, v in noise.items()}},
        "families": T.FAMS, "units": "blood mg/dL inside; Libre = a*blood + b", "fixed": FIX, "stick_weight": STICK_W, "prior_lambda": LAM}
json.dump(card, open("../twin_final.json", "w"), indent=1)
pd.DataFrame(M, columns=T.NAMES).describe(percentiles=[.1, .5, .9]).T.round(3).to_csv("../twin_param_posterior.csv")
print(pd.read_csv("../twin_param_posterior.csv", index_col=0)[["10%", "50%", "90%"]])
