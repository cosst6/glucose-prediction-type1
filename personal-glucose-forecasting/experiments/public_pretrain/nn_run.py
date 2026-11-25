import numpy as np, pandas as pd, copy, json, warnings, sys; warnings.filterwarnings("ignore")
from sklearn.neural_network import MLPRegressor; from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge; from sklearn.pipeline import make_pipeline
from nn_feats import *
pub = pd.read_pickle("pubX.pkl"); me = pd.read_pickle("meX.pkl")
rm = lambda a, b: float(np.sqrt(np.nanmean((a - b) ** 2)))
SEEDS = (0, 1, 2); YSC = 50.0
def mlp(seed): return MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-3, max_iter=200, early_stopping=True, n_iter_no_change=10, random_state=seed)
def finetune(base, X, Y, epochs=25):
    m = copy.deepcopy(base); m.early_stopping = False; m._optimizer.learning_rate_init = 3e-4; m.best_loss_ = np.inf; m._no_improvement_count = 0
    for _ in range(epochs): m.partial_fit(X, Y)
    return m
report = {}
for fam, cols in FAMILY.items():
    P = pub.dropna(subset=cols + YS); sc = StandardScaler().fit(P[cols])
    # --- stage 1: general model, tested on public people it never saw
    people = sorted(P.pid.unique()); rng = np.random.RandomState(0); test_p = set(rng.choice(people, 6, replace=False))
    tr, te = P[~P.pid.isin(test_p)], P[P.pid.isin(test_p)]
    preds = np.mean([mlp(s).fit(sc.transform(tr[cols]), tr[YS] / YSC).predict(sc.transform(te[cols])) for s in SEEDS], axis=0) * YSC
    rid = Ridge(10).fit(sc.transform(tr[cols]), tr[YS]).predict(sc.transform(te[cols]))
    report[f"{fam}_public_heldout"] = {h: {"no_change": rm(0, te[f"y{h}"]), "ridge": rm(rid[:, i], te[f"y{h}"]), "nn": rm(preds[:, i], te[f"y{h}"])} for i, h in enumerate(H)}
    print(fam, "STAGE1 public held-out people", json.dumps({h: {k: round(v, 1) for k, v in d.items()} for h, d in report[f'{fam}_public_heldout'].items()}), flush=True)
    # general model on all public people
    base = [mlp(s).fit(sc.transform(P[cols]), P[YS] / YSC) for s in SEEDS]
    # --- stage 2/3: on the user, walk-forward by day
    M = me.dropna(subset=cols + YS); days = [d for d in sorted(M.index.normalize().unique()) if d >= pd.Timestamp("2026-08-30")]
    rows = []
    for d in days:
        te = M[M.index.normalize() == d]; tr = M[M.index < d - pd.Timedelta("3h")]
        Xte, Xtr = sc.transform(te[cols]), sc.transform(tr[cols])
        zero = np.mean([b.predict(Xte) for b in base], 0) * YSC
        fine = np.mean([finetune(b, Xtr, tr[YS] / YSC).predict(Xte) for b in base], 0) * YSC
        own = np.mean([mlp(s).fit(Xtr, tr[YS] / YSC).predict(Xte) for s in SEEDS], 0) * YSC
        rr = Ridge(30).fit(Xtr, tr[YS]).predict(Xte)
        for i, h in enumerate(H):
            rows.append(pd.DataFrame({"h": h, "true": te[f"y{h}"].values, "zero_shot": zero[:, i], "fine_tuned": fine[:, i], "own_only_nn": own[:, i], "ridge_own": rr[:, i], "g0": te.g0.values}, index=te.index))
        print(fam, d.date(), flush=True)
    R = pd.concat(rows); R.to_csv(f"nn_preds_{fam}.csv")
    S = R.groupby("h").apply(lambda x: pd.Series({"no_change": rm(0, x.true), **{c: rm(x[c], x.true) for c in ("ridge_own", "own_only_nn", "zero_shot", "fine_tuned")}})).round(1)
    print(fam, "USER\n", S.to_string(), flush=True); S.to_csv(f"nn_user_{fam}.csv")
json.dump(report, open("nn_public_report.json", "w"))
