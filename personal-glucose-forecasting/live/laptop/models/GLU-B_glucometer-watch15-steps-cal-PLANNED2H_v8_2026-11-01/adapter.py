"""Adapter for GLU-B v8: one glucose reading + food/insulin (planned 2 h ahead) + watch (HR 15-min avg, raw steps and calories)."""
import joblib, numpy as np, pandas as pd
from fc_common import readings, log_frame, watch_frame, make_preds, to_local, to_ms
from glucoforecast8 import load_log4, event_features4, time_features, watch_features


def forecast(glucose, events, buckets, now_ms, tz, spec):
    HS = spec["horizons_min"]; now_local = to_local(now_ms, tz)
    R = [r for r in readings(glucose, events, tz) if r[0] <= now_local + pd.Timedelta("5min")]
    if not R:
        return {"status": "no_recent_glucose", "message": "No reading yet. Take a fingerstick and log it (or plug in the reader)."}
    t0, g0, src = R[-1]
    sticks = [r for r in R if r[2] == "fingerstick" and t0 - r[0] <= pd.Timedelta("15min")]
    if sticks: t0, g0, src = sticks[-1]           # prefer a fingerstick taken within 15 min of the newest Libre value
    age = (now_local - t0).total_seconds() / 60
    if age > spec.get("max_reading_age_min", 120):
        return {"status": "no_recent_glucose", "based_on_local": t0.strftime("%Y-%m-%d %H:%M"), "data_age_min": round(age),
                "message": f"Last reading is {age/60:.1f} h old. Take a fingerstick and log it for a new forecast."}
    log = log_frame(events, tz, load_log4); watch = watch_frame(buckets, tz)
    idx = pd.DatetimeIndex([t0])
    x = (pd.DataFrame({"g0": [g0]}, index=idx).join(event_features4(idx, log, HS)).join(time_features(idx))
         .join(watch_features(pd.DatetimeIndex([t0.floor("5min")]), watch).set_axis(idx)))   # watch features live on a 5-min grid
    m = joblib.load(spec["id"] + ".joblib")
    msg = []
    if not (x["watch_on"].iloc[0] > 0): msg.append("No watch data around this reading.")
    if src == "fingerstick" and g0 >= 180: msg.append("Trained on Libre values (about 50 lower than sticks): may forecast too low when high.")
    return {"status": "ok", "based_on_local": t0.strftime("%Y-%m-%d %H:%M"), "data_age_min": round(age),
            "g_now": round(g0), "g_source": src, "trend_15min": None,
            "preds": make_preds(t0, m, x, HS), "spike_risk": round(float(m.spike_risk(x)[0]), 2), "warming_up": False,
            "planned_events": sum(1 for e in events if not e.get("deleted") and e["t"] > to_ms(t0, tz)),
            "message": " ".join(msg) or "OK"}
