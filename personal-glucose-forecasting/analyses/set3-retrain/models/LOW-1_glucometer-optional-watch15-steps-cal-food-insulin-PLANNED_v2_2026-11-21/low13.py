"""LOW-1 v2: same features as LOW-1 v1, trained on fingersticks of set 2 + set 3. Fair tests: (a) old v1 on set 3; (b) leave-one-day-out."""
import sys, glob, joblib, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from lowfeat import *
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
D = "../D"; HS = [30, 60, 90, 120, 180, 240, 300, 360]
def rows(logf, watchf, setno):
    logn = load_log9(f"{D}/{logf}"); wn = load_watch(f"{D}/{watchf}")
    B = logn.dropna(subset=["bg"])[["time", "bg"]].sort_values("time").reset_index(drop=True); te = []
    for L in HS:
        for _, r in B.iterrows():
            t = r.time - pd.Timedelta(minutes=L); prev = B[B.time <= t]
            nog = feats(pd.DatetimeIndex([t]), logn, wn, L).iloc[0]
            row = dict(set=setno, target=r.time, L=L, bg=r.bg, y=int(r.bg < 70), **{f"n_{k}": nog[k] for k in NOG})
            if len(prev):
                a = prev.iloc[-1]; age = (t - a.time).total_seconds() / 60
                one = feats(pd.DatetimeIndex([a.time]), logn, wn, int(age + L), g0=np.array([a.bg])).iloc[0]
                row.update({f"o_{k}": one[k] for k in NOG}, o_g0=a.bg, o_age=age, o_pred=a.bg + one["body"])
            te.append(row)
    return pd.DataFrame(te)
T = pd.concat([rows("log_nov.csv", "nov_watch", 2), rows("log_set3.csv", "set3_watch", 3)], ignore_index=True)
T = T[T.o_g0.notna()].copy(); T["day"] = T.target.dt.normalize()
T["ins"] = T.o_ins_nextH; T["carb"] = T.o_carb_next; T["fpu"] = T.o_fpu_nextH; T["H"] = T.o_H
T["ins0"] = T.n_ins_nextH; T["carb0"] = T.n_carb_next; T["fpu0"] = T.n_fpu_nextH; T["iob0"] = T.n_ins_ob - T.n_ins_nextH
W = ["n_hr_15", "n_steps_60", "n_cal_60"]; CN = ["ins0", "carb0", "fpu0", "iob0", "L"] + W; CO = ["o_g0", "ins", "carb", "fpu", "H", "o_age"] + W
T.to_pickle("../low13_rows.pkl")
LR = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000))
old = joblib.load(glob.glob("glucose-model/v11_and_low_alarm/LOW-1*/*.joblib")[0])
T["old_N"] = old["no_glucose"].predict_proba(T[old["cols_no_glucose"]])[:, 1]; T["old_O"] = old["one_stick"].predict_proba(T[old["cols_one_stick"]])[:, 1]
for c in ("pN", "pO", "fN", "fO"): T[c] = np.nan
for d in T.day.unique():   # leave one day out (all sets)
    tr, te = T[T.day != d], T[T.day == d]
    T.loc[te.index, "pN"] = LR().fit(tr[CN], tr.y).predict_proba(te[CN])[:, 1]; T.loc[te.index, "pO"] = LR().fit(tr[CO], tr.y).predict_proba(te[CO])[:, 1]
for d in sorted(T[T.set == 3].day.unique()):   # forward: only days before
    tr, te = T[T.day < d], T[T.day == d]
    T.loc[te.index, "fN"] = LR().fit(tr[CN], tr.y).predict_proba(te[CN])[:, 1]; T.loc[te.index, "fO"] = LR().fit(tr[CO], tr.y).predict_proba(te[CO])[:, 1]
res = []
for L in HS:
    d = T[(T.L == L) & (T.o_age <= 480) & (T.set == 3)]; p, n = d.y == 1, d.y == 0
    for name, s in [("old v1 one stick", d.old_O), ("new forward one stick", d.fO), ("new LODO one stick", d.pO), ("old v1 no glucose", d.old_N), ("new forward no glucose", d.fN)]:
        f = s > 0.5
        res.append(dict(warn_min=L, model=name, caught=f"{int((f&p).sum())}/{int(p.sum())}", false_alarm_pct=round(100 * (f & n).sum() / max(n.sum(), 1)),
                        auc=round(roc_auc_score(d.y, s), 2) if p.sum() and n.sum() else None))
R = pd.DataFrame(res); R.to_csv("../low13_results.csv", index=False); print(R.to_string(index=False))
print("rows", len(T), "low readings set2", T[(T.y == 1) & (T.set == 2)].target.nunique(), "set3", T[(T.y == 1) & (T.set == 3)].target.nunique())
low = {"no_glucose": LR().fit(T[CN], T.y), "one_stick": LR().fit(T[CO], T.y), "cols_no_glucose": CN, "cols_one_stick": CO, "threshold": 0.5, "low_mgdl": 70}
joblib.dump(low, "../low13_final.joblib", compress=3)
