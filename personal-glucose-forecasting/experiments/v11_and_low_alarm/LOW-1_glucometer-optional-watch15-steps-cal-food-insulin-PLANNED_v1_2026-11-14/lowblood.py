import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
T = pd.read_pickle("../low_scored.pkl"); T = T[T.o_g0.notna()].copy()
T["ins"] = T.o_ins_nextH; T["carb"] = T.o_carb_next; T["fpu"] = T.o_fpu_nextH; T["H"] = T.o_H
# rule from earlier data (Oct 30 findings): 1u ~ -18 mg/dL, 1 g carb ~ +2 mg/dL, FPU ~ +10
T["phys"] = T.o_g0 - 18 * T.ins + 2 * T.carb + 10 * T.fpu
T["day"] = T.target.dt.normalize()
C = ["o_g0", "ins", "carb", "fpu", "H", "o_age"]
T["pL"] = np.nan
for d in T.day.unique():  # leave one day out
    tr, te = T[T.day != d], T[T.day == d]
    if tr.y.sum() == 0: continue
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000)).fit(tr[C], tr.y)
    T.loc[te.index, "pL"] = m.predict_proba(te[C])[:, 1]
T.to_pickle("../low_blood.pkl")
def rep(name, d, flag, s):
    p, n = d.y == 1, d.y == 0
    return f"{name:30s} caught {int((flag&p).sum())}/{int(p.sum())}  false alarms {int((flag&n).sum())}/{int(n.sum())} ({100*(flag&n).sum()/n.sum():.0f}%)  AUC {roc_auc_score(d.y, s):.2f}"
for amax in (240, 480):
    print(f"===== last fingerstick at most {amax//60}h old")
    for L in (30, 60, 90, 120, 180, 240):
        d = T[(T.L == L) & (T.o_age <= amax)]; print(f"-- {L} min ahead")
        print(rep("rule g0-18u+2g carbs <90", d, d.phys < 90, -d.phys))
        print(rep("blood-trained LR p>0.5", d, d.pL > 0.5, d.pL))
