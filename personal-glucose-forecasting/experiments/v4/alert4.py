import pandas as pd, numpy as np
X = pd.read_pickle("X4.pkl"); X = X[X.index >= "2026-08-30"]
P = pd.read_csv("preds4_A.csv", index_col=0, parse_dates=True); X = X[X.index.isin(P.index)]
g = X.g0
for pm in (1.5, 2.0):
    al = X.d15 >= pm * 15
    first = al & ~al.shift(1, fill_value=False)  # start of each alert episode
    nxt = pd.concat([g.shift(-k) for k in range(1, 5)], axis=1).max(axis=1) - g
    useful = (nxt[first] >= 20).mean()
    # spikes: rises of 50+ within 2h; was there an alert before the peak?
    peaks = []
    for t in X.index[X.rise2h >= 50]:
        w = g[t:t + pd.Timedelta("2h")]; pk = w.idxmax()
        if not peaks or pk != peaks[-1][1]: peaks.append((t, pk))
    lead = []
    for t0, pk in peaks:
        a = al[t0:pk]; a = a[a]
        lead.append((pk - a.index[0]).total_seconds() / 60 if len(a) else np.nan)
    lead = pd.Series(lead)
    print(f"{pm}/min: {first.sum()} alerts, {100*useful:.0f}% followed by 20+ more rise in the next hour; spikes {len(peaks)}, caught {100*lead.notna().mean():.0f}%, median warning {lead.median():.0f} min before the peak")
