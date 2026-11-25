"""Adapter for GLU-A v5 planned-6h. Every model folder has an adapter.py with the same forecast() signature."""
import joblib, numpy as np, pandas as pd
from fc_common import cgm_frame, log_frame, watch_frame, make_preds, to_local, to_ms
from glucoforecast5 import build_frame5, load_log5


def forecast(glucose, events, buckets, now_ms, tz, spec):
    HS = spec["horizons_min"]; now_local = to_local(now_ms, tz)
    cgm = cgm_frame(glucose, tz)
    if cgm is None or cgm.g.dropna().empty or (now_local - cgm.g.dropna().index[-1]) > pd.Timedelta("12h"):
        return {"status": "no_recent_glucose", "message": "No Libre reading in the last 12 h. Plug in the reader."}
    log = log_frame(events, tz, load_log5); watch = watch_frame(buckets, tz)[["hr", "steps"]]
    X = build_frame5(cgm, log, watch, HS, plan=spec["planned_window_min"]); X = X[X.g0.notna()]
    x = X.iloc[[-1]]; t0 = x.index[0]; m = joblib.load(spec["id"] + ".joblib")
    age = (now_local - t0).total_seconds() / 60
    warming = bool(pd.isna(x["d60"].iloc[0])); msg = []
    if warming: msg.append("Warming up: less than 1 h of sensor data, range is wider than usual.")
    if age > 60: msg.append(f"Last Libre reading is {age/60:.1f} h old. Plug in the reader for a fresher forecast.")
    if not (x["watch_on"].iloc[0] > 0): msg.append("No watch data in the last 30 min.")
    return {"status": "ok" if age <= 60 else "stale", "based_on_local": t0.strftime("%Y-%m-%d %H:%M"), "data_age_min": round(age),
            "g_now": round(float(x.g0.iloc[0])), "g_source": "libre",
            "trend_15min": None if pd.isna(x.d15.iloc[0]) else round(float(x.d15.iloc[0])),
            "preds": make_preds(t0, m, x, HS), "spike_risk": round(float(m.spike_risk(x)[0]), 2), "warming_up": warming,
            "planned_events": sum(1 for e in events if not e.get("deleted") and e["t"] > to_ms(t0, tz)),
            "message": " ".join(msg) or "OK"}
