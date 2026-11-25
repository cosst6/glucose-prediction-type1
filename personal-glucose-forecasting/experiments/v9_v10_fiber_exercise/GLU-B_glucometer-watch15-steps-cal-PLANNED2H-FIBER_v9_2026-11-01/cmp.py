import sys, numpy as np, pandas as pd
tags = sys.argv[1:]; base = "preds8_B.csv"
L = {"v8": pd.read_csv(base, index_col=0, parse_dates=True)}
for t in tags: L[t] = pd.read_csv(f"preds_{t}.csv", index_col=0, parse_dates=True)
rng = np.random.default_rng(0)
for h in (60, 120):
    D = {k: v[v.h == h].loc[lambda d: ~d.index.duplicated()] for k, v in L.items()}
    idx = sorted(set.intersection(*[set(d.index) for d in D.values()]))
    e8 = (D["v8"].loc[idx].hybrid - D["v8"].loc[idx].true); days = pd.Index(idx).normalize(); ud = days.unique()
    for k, d in D.items():
        d = d.loc[idx]; e = d.hybrid - d.true; big = (d.true - d.persist).abs() >= 40
        # day-bootstrap of RMSE difference vs v8
        diffs = []
        for _ in range(500):
            s = rng.choice(ud, len(ud)); m = np.concatenate([np.where(days == x)[0] for x in s])
            diffs.append(np.sqrt((e.values[m]**2).mean()) - np.sqrt((e8.values[m]**2).mean()))
        lo, hi = np.percentile(diffs, [5, 95])
        print(f"{h:4d} {k:14s} n={len(d)} rmse {np.sqrt((e**2).mean()):.2f}  big-move {np.sqrt((e[big]**2).mean()):.1f}  vs v8 90%CI [{lo:+.2f},{hi:+.2f}]")
