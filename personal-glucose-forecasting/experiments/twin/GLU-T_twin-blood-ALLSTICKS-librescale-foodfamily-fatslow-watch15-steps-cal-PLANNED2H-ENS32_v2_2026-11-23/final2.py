"""GLU-T v2: final twin on EVERY data point (all Libre windows + every stick->stick and Libre->stick pair), 32 day-bootstrap members."""
import json, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from tfit import *
FIX = {"si_s": 0, "si_c": 0, "a2": 0.72, "b2": 0.0}
S = pd.read_pickle("../allsticks.pkl")
LL = pd.concat([TABS["set1 Libre"], TABS["Nov18-19 Libre"]]); LL["olib"] = LL.lib
TR = pd.concat([LL.assign(w=1.0), S[S.tag == "stick->stick"].assign(w=5.0), S[S.tag == "Libre->stick"].assign(w=0.1)]).reset_index(drop=True)
def grp(df): return [make(g, g.w.values) for _, g in df.groupby("tl")]
def pred(x, df):
    p = np.empty(len(df))
    for _, g in df.groupby("tl"): p[df.index.get_indexer(g.index)] = T.predict(x, make(g))
    return p
x, _ = fit(grp(TR), fixed=FIX)
rng = np.random.default_rng(11); day = TR.t.dt.strftime("%Y-%m-%d"); ud = day.unique()
members, rr, ii = [], [], []
for b in range(32):
    cnt = pd.Series(rng.choice(ud, len(ud))).value_counts(); m = day.map(cnt).fillna(0).values
    sub = TR[m > 0].assign(w=TR.w[m > 0] * m[m > 0]); xb, _ = fit(grp(sub), x0=x, fixed=FIX); members.append(xb)
    oo = TR[m == 0]
    if len(oo): rr.append(pred(xb, oo) - oo.true.values); ii.append(oo.index.values)
r = np.concatenate(rr); O = TR.loc[np.concatenate(ii)].reset_index(drop=True); gap = O.n * 5
q = {}
for kind, sel in (("stick->stick", (O.tag == "stick->stick")), ("Libre->stick", (O.tag == "Libre->stick")), ("Libre->Libre", O.olib)):
    for lo, hi in ((0, 90), (90, 150), (150, 240)):
        k = (sel & (gap > lo) & (gap <= hi)).values
        if k.sum() > 10: q[f"{kind} {lo}-{hi}"] = dict(q80=float(np.quantile(np.abs(r[k]), .8)), rmse=float(rmse(r[k])), n=int(k.sum()))
print(json.dumps(q, indent=1))
c0 = json.load(open("../twin_final.json"))
noise = c0["noise_sigma_coef"]
noise["60"]["stick_q80"] = q["stick->stick 0-90"]["q80"]; noise["120"]["stick_q80"] = q["stick->stick 90-150"]["q80"]
card = {**c0, "map": x.tolist(), "members": np.array(members).tolist(), "noise_sigma_coef": noise, "oob_bands": q, "fixed": FIX,
        "training": "all Libre windows (w1) + every stick->stick pair 30-240 min (w5) + every Libre->stick pair (w0.1)"}
json.dump(card, open("../twin_final_v2.json", "w"), indent=1)
M = pd.DataFrame(members, columns=T.NAMES); M.describe(percentiles=[.1, .5, .9]).T.round(3).to_csv("../twin_param_posterior_v2.csv")
print(M.describe(percentiles=[.1, .5, .9]).T[["10%", "50%", "90%"]].round(3).to_string())
