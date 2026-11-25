"""Build 5-min grids for sets 1-3: events (food families, insulin), watch (HR, steps, cal), Libre, sticks."""
import sys, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from fam11 import load_logF, FAMS
from glucoforecast import insulin_cum, tri_cum
from watch7 import load_watch7
R = sys.argv[1]; OUT = sys.argv[2]
SH = {"fast": (90, 1/3), "milk": (120, 1/3), "starch": (180, 1/3), "legume": (300, .5), "veg": (240, .4)}  # v14 "lit"
sets = {1: (f"{R}/log_set1.csv", f"{R}/fms/FMS/noise hr logger"),
        2: (f"{R}/log_set2.csv", f"{R}/s2/glucometer diabetes data/heart rate steps"),
        3: (f"{R}/log_set3.csv", f"{R}/s3/gkucometer-set3-15nov onwards/noise watch data")}
raw = pd.read_csv(f"{R}/libre_21-11.csv", skiprows=1)
raw["t"] = pd.to_datetime(raw["Device Timestamp"], format="%d-%m-%Y %I:%M %p")
lib = raw[raw["Record Type"] == 0].groupby("t")["Historic Glucose mg/dL"].mean().sort_index()
lib = lib[lib.index >= "2026-08-01"]
pairs = pd.read_csv(f"{R}/cgm_vs_bg_pairs.csv", parse_dates=["time"])
grids = []
for s, (lp, wp) in sets.items():
    L = load_logF(lp); W = load_watch7(wp)
    days = sorted(set(L.time.dt.normalize()) & set(W.index.normalize()))
    idx = pd.date_range(days[0], days[-1] + pd.Timedelta("1D") + pd.Timedelta("3h"), freq="5min")
    G = pd.DataFrame(index=idx); G["set"] = s; G["day_ok"] = idx.normalize().isin(days)
    # watch: 5-min bins; HR mean, steps/cal raw sums; NaN where watch off
    w5 = W.resample("5min").agg({"hr": "mean", "steps": "sum", "cal": "sum"}); n5 = W.hr.resample("5min").count()
    w5.loc[n5 == 0, ["steps", "cal"]] = np.nan
    G = G.join(w5)
    # Libre (raw, 15-min historic -> placed on grid, not filled here)
    G["libre"] = lib.reindex(idx.round("5min")).values if False else lib.resample("5min").mean().reindex(idx).values
    # sticks
    B = L.dropna(subset=["bg"])[["time", "bg"]]
    if s == 1: B = pd.concat([B, pairs[["time", "bg"]]])
    B = B.assign(time=B.time.dt.round("5min")).groupby("time").bg.mean()
    G["stick"] = B.reindex(idx).values
    # events: impulses + causal activity curves (only events <= grid time)
    m = lambda t: (idx.values[:, None] - np.asarray(t, "datetime64[ns]")[None, :]) / np.timedelta64(1, "m")
    rap = L.dropna(subset=["rapid_u"]); lan = L.dropna(subset=["lantus_u"])
    def act(times, amt, cum, step=5):
        M = m(times); return ((cum(M) - cum(M - step)) * np.asarray(amt)[None, :]).sum(1)
    def imp(times, amt):
        return pd.Series(np.asarray(amt, float), index=pd.DatetimeIndex(times).round("5min")).groupby(level=0).sum().reindex(idx).fillna(0).values
    G["ins_imp"] = imp(rap.time, rap.rapid_u); G["lan_imp"] = imp(lan.time, lan.lantus_u)
    G["ins_act"] = act(rap.time, rap.rapid_u, insulin_cum)
    G["ins_ob"] = (rap.rapid_u.values[None, :] * (1 - insulin_cum(m(rap.time)))*(m(rap.time) >= 0)).sum(1)
    G["lan_h"] = np.nan
    for f in FAMS:
        d, p = SH[f]; G[f"c_{f}_imp"] = imp(L.time, L[f"c_{f}"]); G[f"c_{f}_act"] = act(L.time, L[f"c_{f}"], lambda x, d=d, p=p: tri_cum(x, d, p))
    G["fpu_imp"] = imp(L.time, L.fpu); G["fpu_act"] = act(L.time, L.fpu, lambda x: tri_cum(x, 360, .5))
    G["fib_imp"] = imp(L.time, L.fiber)
    lt = lan.time.values
    last = np.array([lt[lt <= t].max() if (lt <= t).any() else np.datetime64("NaT") for t in idx.values])
    G["lan_h"] = np.clip((idx.values - last) / np.timedelta64(1, "h"), 0, 30); G["lan_h"] = G.lan_h.fillna(24)
    grids.append(G)
    print(s, "days", len(days), "sticks", int(G.stick.notna().sum()), "libre pts", int(G.libre.notna().sum()), "watch cover", round(G.hr.notna().mean(), 2))
G = pd.concat(grids); G = G[~G.index.duplicated()]
G.to_pickle(OUT)
