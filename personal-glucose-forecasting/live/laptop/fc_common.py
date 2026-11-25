"""Shared helpers for model adapters: turn the cloud documents into the tables the training code expects."""
import os, tempfile
import numpy as np, pandas as pd


def to_local(ms, tz):
    return pd.Timestamp(ms, unit="ms", tz="UTC").tz_convert(tz).tz_localize(None)


def to_ms(local_ts, tz):
    return int(pd.Timestamp(local_ts).tz_localize(tz).timestamp() * 1000)


def cgm_frame(glucose, tz, kinds=("hist", "scan")):
    """Same cleaning as training: 15-min grid, gaps > 45 min split segments, short gaps filled."""
    g = [(to_local(d["t"], tz), d["mgdl"]) for d in glucose if d.get("kind") in kinds]
    if not g: return None
    s = pd.Series([v for _, v in g], index=[t for t, _ in g]).sort_index()
    s.index = s.index.round("15min"); s = s.groupby(level=0).mean()
    seg = (s.index.to_series().diff() > pd.Timedelta("45min")).cumsum()
    parts = [pd.DataFrame({"g": p.resample("15min").mean().interpolate(limit=2), "seg": k}) for k, p in s.groupby(seg)]
    return pd.concat(parts)


def readings(glucose, events, tz):
    """Every glucose value with its source: Libre history/scan, reader strip, or fingerstick typed in the app."""
    r = [(to_local(d["t"], tz), float(d["mgdl"]), {"hist": "libre", "scan": "libre", "stick": "fingerstick"}.get(d.get("kind"), "libre")) for d in glucose]
    r += [(to_local(e["t"], tz), float(e["units"]), "fingerstick") for e in events
          if e.get("type") == "stick" and not e.get("deleted") and e.get("units")]
    return sorted(r, key=lambda x: x[0])


def dedupe_reader(events):
    """If the same insulin/food was logged on the phone AND typed on the reader, keep only the phone entry."""
    group = lambda ty: "insulin" if ty in ("rapid", "correction") else ty
    phone = [e for e in events if e.get("src") != "reader" and not e.get("deleted")]
    return [e for e in events if not (e.get("src") == "reader" and any(
        group(p.get("type")) == group(e.get("type")) and abs(p["t"] - e["t"]) <= 20 * 60 * 1000 for p in phone))]


def log_frame(events, tz, loader):
    """Events -> the log.csv layout used in training, loaded with the model's own loader."""
    rows = []
    for e in dedupe_reader(events):
        if e.get("deleted"): continue
        t, ty, u, txt = to_local(e["t"], tz), e.get("type"), e.get("units"), e.get("text") or ""
        r = {"time": t, "rapid_u": None, "lantus_u": None, "bg": None, "cgm_noted": None, "food": None}
        if ty in ("food", "low_treat"): r["food"] = txt
        elif ty in ("rapid", "correction"): r["rapid_u"] = u
        elif ty == "lantus": r["lantus_u"] = u
        elif ty == "stick": r["bg"] = u
        else: continue
        rows.append(r)
    df = pd.DataFrame(rows, columns=["time", "rapid_u", "lantus_u", "bg", "cgm_noted", "food"])
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
        df.to_csv(f, index=False); path = f.name
    try:
        return loader(path)
    finally:
        os.remove(path)


def watch_frame(buckets, tz):
    """Watch buckets -> one row per 30 s sample: hr, steps, cal (as in the Noise CSV)."""
    ts, hr, st, ca = [], [], [], []
    for b in buckets:
        n = len(b.get("ts", []))
        ts += b.get("ts", []); hr += b.get("hr", []); st += b.get("steps", [])
        c = b.get("cal") or []
        ca += list(c) + [np.nan] * (n - len(c))
    if not ts: return pd.DataFrame(columns=["hr", "steps", "cal"], index=pd.DatetimeIndex([], name="t"))
    w = pd.DataFrame({"t": [to_local(x * 1000, tz) for x in ts], "hr": pd.to_numeric(hr, errors="coerce"),
                      "steps": pd.to_numeric(st, errors="coerce"), "cal": pd.to_numeric(ca, errors="coerce")})
    w = w.sort_values("t").drop_duplicates("t").set_index("t")
    w.loc[(w.hr < 35) | (w.hr > 220), "hr"] = np.nan
    return w


def make_preds(t0, model, x, horizons, lo_clip=40, hi_clip=400):
    """Standard output rows: mid forecast and 80% range (raw 10-90% range widened by the model's range_scale)."""
    out = []
    for h in horizons:
        p = model.predict(x, h).iloc[0]; s = getattr(model, "range_scale", {}).get(h, 2.0)
        mid = float(p.pred); lo = mid - s * (mid - float(p.lo)); hi = mid + s * (float(p.hi) - mid)
        c = lambda v: round(float(np.clip(v, lo_clip, hi_clip)))
        out.append({"h": h, "at_local": (t0 + pd.Timedelta(minutes=h)).strftime("%H:%M"), "mid": c(mid), "lo": c(lo), "hi": c(hi)})
    return out
