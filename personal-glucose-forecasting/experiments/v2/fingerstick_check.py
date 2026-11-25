"""Real-world check of Model B: start from a logged glucometer reading, predict the next logged glucometer reading."""
import sys, numpy as np, pandas as pd
from glucoforecast import *
DATA = sys.argv[1]
cgm = load_cgm(glob.glob(f"{DATA}/*glucose*.csv")[0]); cgm = cgm[cgm.index >= "2026-08-01"]
log = load_log(f"{DATA}/log.csv"); watch = load_watch(f"{DATA}/noise hr logger")
X = restrict_to_logged_days(build_frame(cgm, log, watch, HORIZONS_B), log)
bg = log.dropna(subset=["bg"])[["time", "bg"]].reset_index(drop=True)
pairs = [(a, b) for i, a in bg.iterrows() for j, b in bg.iterrows()
         if j > i and 45 <= (b.time - a.time).total_seconds() / 60 <= 300][:]
# keep only the first later reading for each anchor
seen = set(); P = []
for a, b in pairs:
    if a.time in seen: continue
    seen.add(a.time); P.append((a, b))
rows = []
for a, b in P:
    gap = (b.time - a.time).total_seconds() / 60
    idx = pd.DatetimeIndex([a.time])
    F = pd.DataFrame({"g0": [a.bg]}, index=idx).join(event_features(idx, log, HORIZONS_B)).join(time_features(idx)).join(watch_features(idx, watch))
    for c in WATCH: F[c] = F[c].fillna(0 if c != "hr_30" else 70)
    train = X[X.index < a.time.normalize()] if a.time >= pd.Timestamp("2026-08-30") else None
    if train is None or len(train) < 200: train = X  # early anchors: fall back to all CGM data (flagged)
    preds = {h: float(predict(fit(train, "B", h), F).iloc[0]) for h in HORIZONS_B}
    hs = np.array(HORIZONS_B); p = np.interp(gap, hs, [preds[h] for h in hs])
    rows.append({"anchor": a.time, "bg0": a.bg, "target_time": b.time, "gap_min": round(gap), "bg_true": b.bg,
                 "model_B": round(p), "persistence": a.bg, "out_of_sample": a.time >= pd.Timestamp("2026-08-30")})
R = pd.DataFrame(rows); print(R.to_string())
for lab, S in (("all", R), ("out-of-sample", R[R.out_of_sample])):
    print(lab, len(S), "MAE model", (S.model_B - S.bg_true).abs().mean().round(1), "MAE persistence", (S.persistence - S.bg_true).abs().mean().round(1))
R.to_csv("fingerstick_check.csv", index=False)
