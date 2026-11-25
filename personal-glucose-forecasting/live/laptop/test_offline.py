"""Replays history through the exact cloud -> model-folder -> forecast path (no internet needed). Dev-only test."""
import sys, json, numpy as np, pandas as pd
from zoneinfo import ZoneInfo
sys.path.insert(0, "models/GLU-B_glucometer-watch15-steps-cal-PLANNED2H_v8_2026-11-01")
from watch7 import load_watch7
import forecaster as fc
tz = ZoneInfo("Asia/Kolkata"); ms = lambda t: int(pd.Timestamp(t).tz_localize(tz).timestamp() * 1000)
raw = pd.read_csv("/home/user/gf/data/cgm_export_29-10-2026.csv", skiprows=1)
raw["ts"] = pd.to_datetime(raw["Device Timestamp"], format="%d-%m-%Y %I:%M %p")
G = [{"t": ms(r.ts), "mgdl": float(r["Historic Glucose mg/dL"]), "kind": "hist"} for _, r in raw[raw["Record Type"] == 0].iterrows()]
L = pd.read_csv("/home/user/gf/log.csv", parse_dates=["time"]); E = []
for _, r in L.iterrows():
    if isinstance(r.food, str) and r.food.strip(): E.append({"t": ms(r.time), "type": "food", "text": r.food})
    if r.rapid_u == r.rapid_u and r.rapid_u: E.append({"t": ms(r.time), "type": "rapid", "units": float(r.rapid_u)})
    if r.lantus_u == r.lantus_u and r.lantus_u: E.append({"t": ms(r.time), "type": "lantus", "units": float(r.lantus_u)})
    if r.bg == r.bg and r.bg: E.append({"t": ms(r.time), "type": "stick", "units": float(r.bg)})
W = load_watch7("/home/user/gf/data/noise hr logger").reset_index(); W["b"] = W.t.dt.floor("15min"); B = []
nz = lambda v, d=0: d if v != v else int(v)
for b, g in W.groupby("b"):
    B.append({"t": ms(b), "ts": [int(pd.Timestamp(x).tz_localize(tz).timestamp()) for x in g.t], "hr": [None if v != v else int(v) for v in g.hr],
              "steps": [nz(v) for v in g.steps], "cal": [nz(v) for v in g.cal]})
truth = pd.Series({pd.Timestamp(r.ts).round("15min"): r["Historic Glucose mg/dL"] for _, r in raw[raw["Record Type"] == 0].iterrows()})
specs = fc.enabled({"active_model": "GLU-B_glucometer-watch15-steps-cal-PLANNED2H-FIBER-EXERCISE_v10_2026-11-01"})
for now in ["2026-09-14 18:30", "2026-09-12 18:25", "2026-09-16 16:10"]:
    n = ms(now); H = 3600 * 1000
    for libre in (True, False):   # False = glucometer only: no Libre data at all, just logged fingersticks
        data = {"glucose": [g for g in G if n - 30 * H <= g["t"] <= n] if libre else [], "watch": [b for b in B if n - 26 * H <= b["t"] <= n],
                "events": [e for e in E if n - 30 * H <= e["t"] < n + 7 * H], "now_ms": n, "tz": "Asia/Kolkata"}
        for s in specs:
            f = fc.run_model(s, data)
            if f.get("status") == "error": print("ERROR", f); continue
            t0 = pd.Timestamp(f["based_on_local"]) if f.get("based_on_local") else None
            print(now, "with Libre" if libre else "sticks only", "|", f["model_name"], f["status"], "g0", f.get("g_now"), f.get("g_source"), "@", f.get("based_on_local"), "risk", f.get("spike_risk"), "|", f["message"])
            if f.get("preds"): print("    ", " | ".join(f"+{p['h']}m {p['mid']} [{p['lo']}-{p['hi']}] Libre true {truth.get(t0.round('15min') + pd.Timedelta(minutes=p['h']), np.nan):.0f}" for p in f["preds"]))
