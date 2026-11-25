"""LOW-1 v2: same features as LOW-1 v1, trained on fingersticks of set 2 + set 3. Fair tests: (a) old v1 on set 3; (b) leave-one-day-out."""
import sys, glob, joblib, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from lowfeat import *
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
D = "../D"; HS = [30, 60, 90, 120, 180, 240, 300, 360]
from tank import tank_series, TANK
for _k in [k for k in G.g8.KERNELS4 if k.startswith("F")]: del G.g8.KERNELS4[_k]
def rows(logf, watchf, setno):
    logn = load_log9(f"{D}/{logf}"); wn = load_watch(f"{D}/{watchf}"); TS = tank_series(logn, wn)
    B = logn.dropna(subset=["bg"])[["time", "bg"]].sort_values("time").reset_index(drop=True); te = []
    for L in HS:
        for _, r in B.iterrows():
            t = r.time - pd.Timedelta(minutes=L); prev = B[B.time <= t]
            nog = feats(pd.DatetimeIndex([t]), logn, wn, L).iloc[0]
            tk = TS.reindex([t.floor("5min")], method="ffill").iloc[0]
            row = dict(**{f"t_{k}": tk[k] for k in TANK}, set=setno, target=r.time, L=L, bg=r.bg, y=int(r.bg < 70), **{f"n_{k}": nog[k] for k in NOG})
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

TK = ["t_liv_frac", "t_mus_frac", "t_fast_h", "t_liv_drain6h"]
LR = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000))
res = []
for tag, cn, co in (("LOW-1 v2", CN, CO), ("LOW-1 + tank", CN + TK, CO + TK)):
    for c in ("fN", "fO"): T[c] = np.nan
    for d in sorted(T[T.set == 3].day.unique()):
        tr, te = T[T.day < d], T[T.day == d]
        T.loc[te.index, "fN"] = LR().fit(tr[cn], tr.y).predict_proba(te[cn])[:, 1]; T.loc[te.index, "fO"] = LR().fit(tr[co], tr.y).predict_proba(te[co])[:, 1]
    for L in HS:
        d = T[(T.L == L) & (T.o_age <= 480) & (T.set == 3)]; p, n = d.y == 1, d.y == 0
        for name, s in (("one stick", d.fO), ("no glucose", d.fN)):
            f = s > 0.5
            res.append(dict(model=tag, kind=name, warn_min=L, caught=f"{int((f&p).sum())}/{int(p.sum())}", false_alarm_pct=round(100 * (f & n).sum() / max(n.sum(), 1)), auc=round(roc_auc_score(d.y, s), 2)))
R = pd.DataFrame(res); R.to_csv("../low_tank_results.csv", index=False); print(R.to_string(index=False))
print(T.groupby("y")[TK].mean().round(2))
