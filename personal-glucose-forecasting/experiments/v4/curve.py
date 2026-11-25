import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import Ridge; from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler
from glucoforecast4 import *
X = pd.read_pickle("X4.pkl"); L = load_log4("log.csv"); logged = list(set(L.time.dt.normalize()))
X = X[X.index.normalize().isin(logged)]
days = sorted(X.index.normalize().unique()); test = [d for d in days if d >= pd.Timestamp("2026-09-12")]
rows = []
for N in (2, 4, 6, 8, 10, 12, 14):
    for h in (60, 120):
        cols = feature_list4("C", h, False); y = f"y{h}"; err = []; errh = []
        for d in test:
            prior = [p for p in days if p < d][-N:]
            if len(prior) < N: continue
            tr = X[X.index.normalize().isin(prior) & (X.index < d - pd.Timedelta(minutes=h))].dropna(subset=cols + [y]); te = X[X.index.normalize() == d].dropna(subset=cols + [y])
            if len(te) == 0: continue
            m = make_pipeline(StandardScaler(), Ridge(alpha=30)).fit(tr[cols], tr[y]); err.append(te[y] - m.predict(te[cols]))
            hb = Hybrid().fit(X[X.index.normalize().isin(prior) & (X.index < d - pd.Timedelta(minutes=240))], "C", HORIZONS_C, use_watch=False)
            errh.append(te.g0 + te[y] - hb.predict(te, h).pred)
        e = pd.concat(err); eh = pd.concat(errh)
        rows.append({"days": N, "h": h, "ridge": np.sqrt((e**2).mean()), "curved": np.sqrt((eh**2).mean())}); print(rows[-1], flush=True)
pd.DataFrame(rows).round(1).to_csv("curve.csv", index=False)
