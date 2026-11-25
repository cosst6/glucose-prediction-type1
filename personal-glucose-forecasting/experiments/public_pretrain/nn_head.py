"""General net body (frozen) + a small head refitted on the user's data with extra user-only inputs (fat/protein, fast sugar)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.neural_network import MLPRegressor; from sklearn.preprocessing import StandardScaler; from sklearn.linear_model import Ridge
from nn_feats import *
from glucoforecast4 import build_frame4, load_log4, load_watch
import glob
pub = pd.read_pickle("pubX.pkl"); me = pd.read_pickle("meX.pkl")
X4 = pd.read_pickle("X4.pkl")  # v4 frame: has fpu/fsug/gl/slow kernels on the 15-min grid
rm = lambda a, b: float(np.sqrt(np.nanmean((a - b) ** 2)))
def hidden(m, X):
    a = X
    for W, b in zip(m.coefs_[:-1], m.intercepts_[:-1]): a = np.maximum(a @ W + b, 0)
    return a
rows = []
for fam, cols in FAMILY.items():
    P = pub.dropna(subset=cols + YS); sc = StandardScaler().fit(P[cols])
    base = [MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True, n_iter_no_change=10, max_iter=200, random_state=s).fit(sc.transform(P[cols]), P[YS] / 50) for s in (0, 1, 2)]
    M = me.dropna(subset=cols + YS).join(X4[[c for c in X4.columns if c.startswith(("fpu_", "fsug_", "gl_", "slow_"))]], how="inner").dropna()
    extra = [f"{e}_{k}" for e in ("fpu", "fsug", "gl", "slow") for k in ("ob", "act30")]
    days = [d for d in sorted(M.index.normalize().unique()) if d >= pd.Timestamp("2026-08-30")]
    out = []
    for d in days:
        te = M[M.index.normalize() == d]; tr = M[M.index < d - pd.Timedelta("3h")]
        Htr = np.mean([hidden(b, sc.transform(tr[cols])) for b in base], 0); Hte = np.mean([hidden(b, sc.transform(te[cols])) for b in base], 0)
        es = StandardScaler().fit(tr[extra]); Etr, Ete = es.transform(tr[extra]), es.transform(te[extra])
        hs = StandardScaler().fit(Htr)
        A = Ridge(30).fit(hs.transform(Htr), tr[YS]).predict(hs.transform(Hte))                                   # last layer only
        B = Ridge(30).fit(np.hstack([hs.transform(Htr), Etr]), tr[YS]).predict(np.hstack([hs.transform(Hte), Ete]))  # + fat/protein/sugar speeds
        C = Ridge(30).fit(np.hstack([sc.transform(tr[cols]), Etr]), tr[YS]).predict(np.hstack([sc.transform(te[cols]), Ete]))  # own ridge + same extras (reference)
        D = Ridge(30).fit(sc.transform(tr[cols]), tr[YS]).predict(sc.transform(te[cols]))
        for i, h in enumerate(H):
            out.append(pd.DataFrame({"h": h, "true": te[f"y{h}"].values, "head_only": A[:, i], "head_plus_extra": B[:, i], "own_ridge_plus_extra": C[:, i], "own_ridge": D[:, i], "g0": te.g0.values}))
    R = pd.concat(out); S = R.groupby("h").apply(lambda x: pd.Series({"no_change": rm(0, x.true), **{c: rm(x[c], x.true) for c in ("own_ridge", "own_ridge_plus_extra", "head_only", "head_plus_extra")}})).round(1)
    print(fam, "\n", S.to_string(), flush=True); S.to_csv(f"nn_head_{fam}.csv")
