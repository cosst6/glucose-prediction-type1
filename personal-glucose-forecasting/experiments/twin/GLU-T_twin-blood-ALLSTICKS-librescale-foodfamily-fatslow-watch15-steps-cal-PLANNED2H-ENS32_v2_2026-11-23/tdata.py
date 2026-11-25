"""Build twin timelines + anchor tables from the rebuilt working data (../*.pkl, ../D)."""
import os, numpy as np, pandas as pd, pickle, warnings; warnings.filterwarnings("ignore")
from fam11 import load_cgm
from watch7 import load_watch7 as LW
from twin import Timeline

D = "../D"
def libre_anchor_table(X, cgm, tag):
    g = cgm.g
    rows = []
    for t, r in X.iterrows():
        prev = t - pd.Timedelta(minutes=int(os.environ.get("PREVMIN", 30)))
        pg = g.get(prev, np.nan)
        for h in (60, 120):
            y = r.get(f"y{h}")
            if pd.isna(y): continue
            rows.append(dict(t=t, h=h, n=h // 5, g0=r.g0, lib=True, prev_t=prev if not pd.isna(pg) else pd.NaT, prev_g=pg, prev_lib=True, true=r.g0 + y, tag=tag))
    return pd.DataFrame(rows)

def stick_anchor_table(P, B, tag):
    bt = B.set_index("time").bg
    rows = []
    for t, r in P.iterrows():
        c = bt[(bt.index < t - pd.Timedelta("19min")) & (bt.index >= t - pd.Timedelta("180min"))]
        pt, pg = (c.index[-1], c.iloc[-1]) if len(c) else (pd.NaT, np.nan)
        rows.append(dict(t=t, h=int(r.h), n=int(r.gap) // 5, g0=r.g0, lib=False, prev_t=pt, prev_g=pg, prev_lib=False, true=r.true, tag=tag))
    return pd.DataFrame(rows)

def bg_points(log, extra=None):
    B = log.dropna(subset=["bg"])[["time", "bg"]].copy()
    if extra is not None: B = pd.concat([B, extra])
    B["time"] = B.time.dt.round("5min"); return B.drop_duplicates("time").sort_values("time").reset_index(drop=True)

if __name__ == "__main__":
    pk = {n: pd.read_pickle(f"../{n}.pkl") for n in ("Xo", "X3", "Po", "P2", "P3", "logo", "l2", "l3")}
    cgm = load_cgm(f"{D}/cgm_export_21-11-2026.csv")
    W = {"set1": LW(f"{D}/noise hr logger"), "set2": LW(f"{D}/nov_watch"), "set3": LW(f"{D}/set3_watch")}
    LOG = {"set1": pk["logo"], "set2": pk["l2"], "set3": pk["l3"]}
    TL = {}
    for k, L in LOG.items():
        TL[k] = Timeline(L, W[k], L.time.min(), L.time.max())
        print(k, TL[k].T[0], TL[k].T[-1], TL[k].n, "boluses", len(TL[k].bt), "meals", len(TL[k].mt))
    # low flags for meals: Libre (converted ~/0.72) or stick below 80 in the 45 min before eating
    for k, L in LOG.items():
        st = L.dropna(subset=["bg"]).set_index("time").bg
        lib = cgm.g[(cgm.index >= TL[k].T[0]) & (cgm.index <= TL[k].T[-1])] / 0.72
        if k == "set3": lib = lib[lib.index >= "2026-11-18 13:00"]
        TL[k].set_low(pd.concat([st, lib])); print(k, "meals after a low", int(TL[k].mlow.sum()))
    z = pd.read_csv("../cgm_vs_bg_pairs.csv", parse_dates=["time"])[["time", "bg"]]
    tabs = {
        "set1 Libre": libre_anchor_table(pk["Xo"], cgm, "set1 Libre").assign(tl="set1"),
        "Nov18-19 Libre": libre_anchor_table(pk["X3"], cgm, "Nov18-19 Libre").assign(tl="set3"),
        "set1 sticks": stick_anchor_table(pk["Po"], bg_points(pk["logo"], z), "set1 sticks").assign(tl="set1"),
        "set2 glucometer": stick_anchor_table(pk["P2"], bg_points(pk["l2"]), "set2 glucometer").assign(tl="set2"),
        "set3 glucometer": stick_anchor_table(pk["P3"], bg_points(pk["l3"]), "set3 glucometer").assign(tl="set3"),
    }
    for k, v in tabs.items(): print(k, len(v), "with prev", v.prev_t.notna().sum())
    pickle.dump((TL, tabs), open(os.environ.get("TDATA", "../twin_data.pkl"), "wb"))
