import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
old = {"A": "preds_A_{}.csv", "B": "preds_B_{}.csv", "C": "preds_C_{}.csv"}
rows = []
for M in "ABC":
    P = pd.read_csv(f"preds4_{M}.csv", index_col=0, parse_dates=True)
    for h, d in P.groupby("h"):
        o = pd.read_csv(old[M].format(h), index_col=0, parse_dates=True)["pred"].rename("v3")
        d = d.join(o, how="inner")
        r = lambda c, s=slice(None): float(np.sqrt(((d[c][s] - d.true[s]) ** 2).mean()))
        sp = (d.true - d.persist) >= 40; hi = d.true > 200; lo = d.true < 80
        row = {"model": M, "h": h, "n": len(d), "persist": r("persist"), "v3": r("v3"), "ridge4": r("ridge4"), "body": r("body"), "hybrid": r("hybrid"),
               "v3_spikes": r("v3", sp), "hyb_spikes": r("hybrid", sp), "v3_bias_hi": float((d.v3 - d.true)[hi].mean()), "hyb_bias_hi": float((d.hybrid - d.true)[hi].mean()),
               "v3_bias_lo": float((d.v3 - d.true)[lo].mean()), "hyb_bias_lo": float((d.hybrid - d.true)[lo].mean()),
               "in_range_%": float(100 * ((d.true >= d.lo) & (d.true <= d.hi)).mean()), "range_width": float((d.hi - d.lo).mean()),
               "hyb_within30_%": float(100 * ((d.hybrid - d.true).abs() <= 30).mean()), "v3_within30_%": float(100 * ((d.v3 - d.true).abs() <= 30).mean()),
               "hyb_off50_%": float(100 * ((d.hybrid - d.true).abs() > 50).mean()), "v3_off50_%": float(100 * ((d.v3 - d.true).abs() > 50).mean())}
        rows.append(row)
R = pd.DataFrame(rows).round(1); R.to_csv("results4.csv", index=False); pd.set_option("display.width", 250); print(R.to_string())
for M in "ABC":
    P = pd.read_csv(f"preds4_{M}.csv", index_col=0, parse_dates=True); P = P[~P.index.duplicated()].dropna(subset=["risk", "rise2h"])
    y = P.rise2h >= 50
    for th in (0.3, 0.5):
        a = P.risk >= th
        print(M, "spike-risk AUC %.2f | thr %.1f: caught %.0f%% of spikes, false alarms %.0f%% of warnings | spikes %d" % (roc_auc_score(y, P.risk), th, 100 * (a & y).sum() / y.sum(), 100 * (a & ~y).sum() / max(a.sum(), 1), y.sum()))
P = pd.read_csv("preds4_A.csv", index_col=0, parse_dates=True); P = P[~P.index.duplicated()]
g = P.persist
for pm in (1.5, 2.0):
    al = P.d15 >= pm * 15; y = P.rise2h >= 50
    print("rise alert %.1f/min: fires %d times, %.0f%% followed by more rise>=20 in next hour" % (pm, al.sum(), 0))
