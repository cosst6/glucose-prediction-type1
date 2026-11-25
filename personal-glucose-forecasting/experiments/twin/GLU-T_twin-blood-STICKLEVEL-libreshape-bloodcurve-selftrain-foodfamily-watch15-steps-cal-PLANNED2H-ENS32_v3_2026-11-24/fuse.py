"""Fusion with learned weights, walk-forward: weights for day d are fit only on earlier days' out-of-sample forecasts."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from scipy.optimize import nnls
import blood as BL
r = lambda e: round(float(np.sqrt(np.mean(np.square(e)))), 1)
R = pd.read_pickle("../wf3_a_wcl1.pkl").reset_index(drop=True)
S = pd.read_pickle("../allsticks.pkl")
R0 = pd.read_pickle("../wf2_wls02.pkl")   # v2 twin (before)
def wf_weights(D, cols, min_days=3):
    """convex-ish weights (nnls, renormalised) per day from earlier days."""
    out = np.full(len(D), np.nan); D = D.reset_index(drop=True); day = D.t.dt.normalize()
    for d in sorted(day.unique()):
        tr = D[day < d]; te = day == d
        if tr.t.dt.normalize().nunique() < min_days: out[te.values] = D.loc[te, cols[0]].values; continue
        w, _ = nnls(tr[cols].values, tr.true.values); w = w / w.sum() if w.sum() > 0 else np.eye(len(cols))[0]
        out[te.values] = D.loc[te, cols].values @ w
    return out
# ---- stick -> stick (<=150 min): twin + v14 + v10
C = pd.read_pickle("../cmp_sticks.pkl").set_index("row")
SS = S[(S.tag == "stick->stick") & (S.gap <= 150)].reset_index(drop=True).join(C, how="inner")
SS = SS.merge(R[["t", "tgt", "tag", "twin"]], on=["t", "tgt", "tag"]).merge(R0[["t", "tgt", "tag", "twin"]].rename(columns={"twin": "twin_v2"}), on=["t", "tgt", "tag"])
SS["hb"] = np.where(SS.gap <= 90, "~1h", "~2h")
for h, g in SS.groupby("hb"):
    g = g.sort_values("t").reset_index(drop=True)
    g["fuse"] = wf_weights(g, ["twin", "v14", "v10live"]); g["avg"] = (g.twin + g.v14 + g.v10live) / 3
    print("stick->stick", h, len(g), {c: r(g[c] - g.true) for c in ["twin", "twin_v2", "v14", "v10live", "avg", "fuse", "g0"]})
# ---- Libre -> stick: twin + Chronos-2 (Libre forecast turned into blood with the causal calibration)
CH = pd.read_pickle("../chronos_preds.pkl")
L = R[R.tag == "Libre->stick"].copy()
L = L.merge(CH, on=["t", "tl"], how="inner")
def chr_blood(row):
    h = int(np.clip(round(row.gap / 15) * 15, 15, 240)); lv = row[f"c2_{h}"]
    cal = BL.fit_cal(row.t.normalize()); return BL.libre_to_blood(row.tl, row.t, lv, cal, before=row.t + pd.Timedelta("1min"))
L["chronos"] = L.apply(chr_blood, axis=1)
L = L.merge(R0[["t", "tgt", "tag", "twin"]].rename(columns={"twin": "twin_v2"}), on=["t", "tgt", "tag"])
for h, g in L.groupby("hb", observed=True):
    g = g.sort_values("t").reset_index(drop=True); g["fuse"] = wf_weights(g, ["twin", "chronos"]); g["avg"] = (g.twin + g.chronos) / 2
    print("Libre->stick", h, len(g), {c: r(g[c] - g.true) for c in ["twin", "twin_v2", "chronos", "avg", "fuse", "g0"]})
SS.to_pickle("../fuse_ss.pkl"); L.to_pickle("../fuse_ls.pkl")
