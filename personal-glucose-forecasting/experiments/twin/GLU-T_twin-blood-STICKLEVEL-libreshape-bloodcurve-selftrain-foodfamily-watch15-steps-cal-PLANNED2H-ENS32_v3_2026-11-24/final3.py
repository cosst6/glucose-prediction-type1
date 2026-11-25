"""GLU-T v3 final: every data point, denoised Libre, self-trained on the filled blood curve, 32 bootstrap members."""
import os, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
os.environ.update(KEEPLL="0.3", WCL="1")
import sys; sys.argv = ["x", "final"]
src = open("wf3.py").read().split("res = []")[0]; exec(src)
D = pd.Timestamp("2030-01-01")
parts = [LL.assign(w=KEEPLL), S[S.tag == "stick->stick"].assign(w=WS), denoise(S[S.tag == "Libre->stick"].assign(w=0.1), D)]
x, _ = fit(grp(pd.concat(parts).reset_index(drop=True)), fixed=FIX)
C, cal = BL.curve(x, D); CA = curve_anchors(C); CA = CA.dropna(subset=["g0", "true"]); CA = CA[[(TL[k].T[0] + pd.Timedelta("12h") <= t) and (t + pd.Timedelta("150min") <= TL[k].T[-1]) for k, t in zip(CA.tl, CA.t)]]; print("curve anchors", len(CA), flush=True)
TR = pd.concat(parts + [CA]).reset_index(drop=True)
x, _ = fit(grp(TR), x0=x, fixed=FIX)
C.to_csv("../blood_curve_v3.csv", index=False)
rng = np.random.default_rng(5); day = TR.t.dt.strftime("%Y-%m-%d"); ud = day.unique(); members, rr, ii = [], [], []
for b in range(32):
    cnt = pd.Series(rng.choice(ud, len(ud))).value_counts(); m = day.map(cnt).fillna(0).values
    sub = TR[m > 0].assign(w=TR.w[m > 0] * m[m > 0]); xb, _ = fit(grp(sub), x0=x, fixed=FIX); members.append(xb)
    oo = TR[(m == 0) & TR.tag.isin(["stick->stick"])]
    if len(oo): rr.append(pred(xb, oo) - oo.true.values); ii.append(oo.index.values)
    print("member", b, flush=True)
r = np.concatenate(rr); O = TR.loc[np.concatenate(ii)]; gap = O.n.values * 5
q60 = float(np.quantile(np.abs(r[gap <= 90]), .8)); q120 = float(np.quantile(np.abs(r[(gap > 90) & (gap <= 150)]), .8))
c0 = json.load(open("../twin_final_v2.json")); noise = c0["noise_sigma_coef"]; noise["60"]["stick_q80"] = q60; noise["120"]["stick_q80"] = q120
card = {**c0, "map": x.tolist(), "members": np.array(members).tolist(), "noise_sigma_coef": noise, "libre_to_blood_cal": list(map(float, cal)),
        "training": "Libre windows x0.3 + every stick->stick pair x5 + denoised Libre->stick pairs x0.1 + filled blood curve (sticks x2, Libre-derived x1, twin bridge x1)"}
json.dump(card, open("../twin_final_v3.json", "w"), indent=1)
M = pd.DataFrame(members, columns=T.NAMES); M.describe(percentiles=[.1, .5, .9]).T.round(3).to_csv("../twin_param_posterior_v3.csv")
print("q80", q60, q120, "cal", cal); print(M.describe(percentiles=[.1, .5, .9]).T[["10%", "50%", "90%"]].round(3).to_string())
print("curve rows", C.src.value_counts().to_dict())
