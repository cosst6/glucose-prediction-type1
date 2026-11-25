import glob, joblib, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score as AUC
S = "glucose-model/spike_alarm"
T = pd.read_pickle("meals.pkl").reset_index(drop=True)
T["hr15"] = T.hr15.fillna(T.hr15.median()); T["steps60"] = T.steps60.fillna(0)
T["uncov"] = T.avail - 8 * T.dose; T["low"] = (T.pre < 70).astype(float); T["prec"] = T.pre.clip(40, 250)
T["y30"] = (T.rise >= 30).astype(int); T["y60"] = (T.rise >= 60).astype(int); T["y40"] = (T.rise >= 40).astype(int); T["band"] = T.y30 + T.y60
B = ["uncov", "iob", "steps60", "hr15"]; S1 = B + ["prec", "low"]
LR = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
sp1 = joblib.load(glob.glob(f"{S}/SPIKE-1*/*.joblib")[0]); r40 = joblib.load(glob.glob(f"{S}/RISE40*/*.joblib")[0])
T["old_b1"] = (sp1["step1_pre_stick"]["m"]["y30"].predict_proba(T[S1])[:, 1] > .5).astype(int) + (sp1["step1_pre_stick"]["m"]["y60"].predict_proba(T[S1])[:, 1] > .5).astype(int)
T["old_p40"] = r40["models"]["pre"]["p40"].predict_proba(T[r40["SIZE"]["pre"]])[:, 1]
T["old_p40n"] = r40["models"]["none"]["p40"].predict_proba(T[r40["SIZE"]["none"]])[:, 1]
sw = lambda d: np.where(d.src != "Libre", 3.0, 1.0)
for c in ("new_b1", "new_p40", "new_p40n"): T[c] = np.nan
for d in T.day.unique():
    tr, te = T.day != d, T.day == d; W = sw(T[tr])
    p30 = LR().fit(T.loc[tr, S1], T.loc[tr, "y30"], logisticregression__sample_weight=W).predict_proba(T.loc[te, S1])[:, 1]
    p60 = LR().fit(T.loc[tr, S1], T.loc[tr, "y60"], logisticregression__sample_weight=W).predict_proba(T.loc[te, S1])[:, 1]
    T.loc[te, "new_b1"] = (p30 > .5).astype(int) + (np.minimum(p60, p30) > .5).astype(int)
    T.loc[te, "new_p40"] = LR().fit(T.loc[tr, r40["SIZE"]["pre"]], T.loc[tr, "y40"]).predict_proba(T.loc[te, r40["SIZE"]["pre"]])[:, 1]
    T.loc[te, "new_p40n"] = LR().fit(T.loc[tr, r40["SIZE"]["none"]], T.loc[tr, "y40"]).predict_proba(T.loc[te, r40["SIZE"]["none"]])[:, 1]
T["rule"] = (T.avail > 8 * T.dose).astype(int)
rows = []
for src in ("Glucometer3", "Glucometer", "Libre"):
    d = T[T.src == src]; s = d.band >= 1
    for nm, b in (("SPIKE-1 old (pre-meal stick)", d.old_b1), ("SPIKE-1 new (pre-meal stick)", d.new_b1), ("your rule <1u per 8g", d.rule)):
        rows.append(dict(data=src, model=nm, meals=len(d), band_right=f"{round(100*(b==d.band).mean())}%" if "rule" not in nm else "-", spikes_caught=f"{int(((b>=1)&s).sum())}/{int(s.sum())}", false_alarms=f"{int(((b>=1)&~s).sum())}/{int((~s).sum())}"))
    for nm, p in (("RISE40 old (pre reading)", d.old_p40), ("RISE40 new (pre reading)", d.new_p40), ("RISE40 old (no reading)", d.old_p40n), ("RISE40 new (no reading)", d.new_p40n)):
        a = p > .5; y = d.y40 == 1
        rows.append(dict(data=src, model=nm, meals=len(d), band_right=f"{round(100*(a==y).mean())}%", spikes_caught=f"{int((a&y).sum())}/{int(y.sum())}", false_alarms=f"{int((a&~y).sum())}/{int((~y).sum())}"))
R = pd.DataFrame(rows); pd.set_option("display.width", 200); print(R.to_string(index=False)); R.to_csv("spike13_results.csv", index=False)
T.to_pickle("meals13_scored.pkl")
# final models on all meals
W = sw(T)
M = {"step1_pre_stick": {"cols": S1, "m": {y: LR().fit(T[S1], T[y], logisticregression__sample_weight=W) for y in ("y30", "y60")}},
     "step0_no_stick": {"cols": B, "m": {y: LR().fit(T[B], T[y], logisticregression__sample_weight=W) for y in ("y30", "y60")}},
     "step2_45min_stick": sp1["step2_45min_stick"], "bands": sp1["bands"], "ask_step2_if_within": 0.1, "carbs_per_unit": 8, "units": sp1["units"]}
joblib.dump(M, "spike1_v2.joblib")
r = {k: dict(v) for k, v in r40["models"].items()}
for mode in ("none", "pre"): r[mode]["p40"] = LR().fit(T[r40["SIZE"][mode]], T.y40)
R40 = dict(r40); R40["models"] = r; joblib.dump(R40, "rise40_v2.joblib")
print(T.groupby("src").size())
