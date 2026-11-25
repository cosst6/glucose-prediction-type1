"""All-fingerstick tables: every stick -> every later stick 30-240 min; every Libre reading (15-min) -> every later stick 30-240 min."""
import os, numpy as np, pandas as pd, pickle
from fam11 import load_cgm
TL, TABS = pickle.load(open("../twin_data.pkl", "rb"))
pk = {n: pd.read_pickle(f"../{n}.pkl") for n in ("logo", "l2", "l3")}
cgm = load_cgm("../D/cgm_export_21-11-2026.csv").g
LOG = {"set1": pk["logo"], "set2": pk["l2"], "set3": pk["l3"]}
LIBDAYS = {"set1": set(TABS["set1 Libre"].t.dt.normalize()), "set3": set(TABS["Nov18-19 Libre"].t.dt.normalize())}
GMIN, GMAX = 30, 240
rows = []
for k, L in LOG.items():
    B = L.dropna(subset=["bg"])[["time", "bg"]].copy(); B["time"] = B.time.dt.round("5min"); B = B.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    for i, a in B.iterrows():
        prev = B[(B.time < a.time - pd.Timedelta("19min")) & (B.time >= a.time - pd.Timedelta("180min"))]
        pt, pg = (prev.time.iloc[-1], prev.bg.iloc[-1]) if len(prev) else (pd.NaT, np.nan)
        for _, b in B[(B.time >= a.time + pd.Timedelta(minutes=GMIN)) & (B.time <= a.time + pd.Timedelta(minutes=GMAX))].iterrows():
            gap = int((b.time - a.time).total_seconds() // 60)
            rows.append(dict(t=a.time, n=gap // 5, gap=gap, g0=a.bg, lib=False, olib=False, prev_t=pt, prev_g=pg, prev_lib=False, true=b.bg, tag="stick->stick", tl=k, tgt=b.time))
        if k in LIBDAYS:   # Libre anchors (every 15 min) -> this stick as target
            pass
    if k in LIBDAYS:
        g = cgm[(cgm.index >= TL[k].T[0]) & (cgm.index <= TL[k].T[-1])]
        g = g[g.index.normalize().isin(LIBDAYS[k]) & (g.index.minute % 15 == 0)]
        for _, b in B.iterrows():
            for t0, v in g[(g.index >= b.time - pd.Timedelta(minutes=GMAX)) & (g.index <= b.time - pd.Timedelta(minutes=GMIN))].items():
                pv = cgm.get(t0 - pd.Timedelta("30min"), np.nan); gap = int((b.time - t0).total_seconds() // 60)
                rows.append(dict(t=t0, n=gap // 5, gap=gap, g0=v, lib=True, olib=False, prev_t=t0 - pd.Timedelta("30min") if not pd.isna(pv) else pd.NaT, prev_g=pv, prev_lib=True, true=b.bg, tag="Libre->stick", tl=k, tgt=b.time))
S = pd.DataFrame(rows)
S["hb"] = pd.cut(S.gap, [29, 90, 150, 240], labels=["~1h (30-90 min)", "~2h (90-150 min)", "3-4h (150-240 min)"])
print(S.groupby(["tag", "tl", "hb"]).size().unstack())
print("distinct target sticks", S.groupby("tag").tgt.nunique().to_dict())
S.to_pickle("../allsticks.pkl")
