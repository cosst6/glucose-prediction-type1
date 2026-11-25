import sys, numpy as np, pandas as pd
M = ["persist", "hybrid", "moe_soft", "moe_hard", "weighted4x", "moe_oracle"]
def lag(d, col, h):
    """minutes the forecast trails reality: shift that best lines up predicted vs actual curve (on for-time axis)."""
    ft = d.index + pd.Timedelta(minutes=h); p = pd.Series(d[col].values, ft); a = pd.Series(d.true.values, ft)
    best = max(range(0, 121, 15), key=lambda k: p.corr(a.shift(freq=pd.Timedelta(minutes=k)).reindex(p.index)))
    return best
rows = []
for f in sys.argv[1:]:
    P = pd.read_csv(f, index_col=0, parse_dates=True); tag = f[4:-4]
    for h in (60, 120):
        d = P[P.h == h]; fast = d.is_fast == 1
        for m in M:
            e = d[m] - d.true; mv = d[m] - d.g0; tv = d.true - d.g0
            rows.append({"set": tag, "h": h, "model": m, "rmse": np.sqrt((e**2).mean()), "rmse_fast": np.sqrt((e[fast]**2).mean()),
                         "rmse_calm": np.sqrt((e[~fast]**2).mean()), "move_ratio_fast": (mv[fast] * np.sign(tv[fast])).mean() / tv[fast].abs().mean(),
                         "lag_min": lag(d, m, h), "n_fast": int(fast.sum()), "n": len(d)})
R = pd.DataFrame(rows).round(2); R.to_csv("moe_results.csv", index=False); print(R.to_string())
