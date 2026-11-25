import sys, numpy as np, pandas as pd
M = pd.read_csv(sys.argv[1]); M = M[M.gru.notna()]
iso = lambda p, t: float(np.mean(np.where(t < 100, np.abs(p - t) <= 15, np.abs(p - t) <= 0.15 * t)) * 100)
rows = []
for (grp, g) in [("sets 2+3", M), *M.groupby("test")]:
    for h in (60, 120):
        x = g[g.h == h]
        for name, col in (("GRU", "gru"), ("v10 live", "v10live"), ("v14", "fam"), ("no change", "g0")):
            e = x[col] - x.true
            rows.append(dict(test=grp, h=h, model=name, n=len(x), MAE=round(e.abs().mean(), 1), RMSE=round(np.sqrt((e**2).mean()), 1), ISO=round(iso(x[col], x.true), 0)))
T = pd.DataFrame(rows); print(T.to_string(index=False))
if len(sys.argv) > 2: T.to_csv(sys.argv[2], index=False)
