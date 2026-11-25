import glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from fam11 import *
import glucoforecast9 as G9; G9.EV9.extend(FEV)
from pairs import make_pairs
D = "../D"
cgm = load_cgm(f"{D}/cgm_export_21-11-2026.csv"); cgm = cgm[cgm.index >= "2026-08-01"]
logo = load_logF(f"{D}/log.csv"); wo = load_watch(f"{D}/noise hr logger")
Xo = restrict_to_logged_days(build_frame9(cgm[cgm.index < "2026-10-01"], logo, wo, [60, 120]), logo)
l3 = load_logF(f"{D}/log_set3.csv"); w3 = load_watch(f"{D}/set3_watch")
X3 = restrict_to_logged_days(build_frame9(cgm[cgm.index >= "2026-11-01"], l3, w3, [60, 120]), l3)
z = pd.read_csv("../cgm_vs_bg_pairs.csv", parse_dates=["time"])[["time", "bg"]]
Bo = pd.concat([logo.dropna(subset=["bg"])[["time", "bg"]], z]); Bo["time"] = Bo.time.dt.round("5min"); Bo = Bo.drop_duplicates("time").sort_values("time").reset_index(drop=True)
Po = make_pairs(logo, wo, Bo)
l2 = load_logF(f"{D}/log_nov.csv"); w2 = load_watch(f"{D}/nov_watch"); P2 = make_pairs(l2, w2)
P3 = make_pairs(l3, w3)
for n, v in dict(Xo=Xo, X3=X3, Po=Po, P2=P2, P3=P3, logo=logo, l2=l2, l3=l3).items(): v.to_pickle(f"../{n}.pkl")
print("Xo", len(Xo), Xo.index.normalize().nunique(), "X3 (Libre Nov)", len(X3), sorted(set(X3.index.date)), "Po", len(Po), "P2", len(P2), "P3", len(P3))
for n, L in (("set1", logo), ("set2", l2), ("set3", l3)): print(n, {f: round(L[f"c_{f}"].sum()) for f in FAMS}, "meals>5g", {f: int((L[f"c_{f}"] > 5).sum()) for f in FAMS})
