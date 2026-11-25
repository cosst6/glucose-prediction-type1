"""Detect walking/exercise bouts from Noise watch data (5-min bins). Causal: uses only data up to each time."""
import numpy as np, pandas as pd
STEP5 = 350      # >=350 steps in 5 min (~70/min) = brisk walking
MIN_LEN = 20     # bout must last >=20 min
GAP = 10         # allow up to 10 min pause inside a bout
def bouts(w):
    st5 = w.steps.resample("5min", label="right", closed="right").sum()
    hr5 = w.hr.resample("5min", label="right", closed="right").mean()
    rest = hr5.rolling("24h", min_periods=50).quantile(0.1)
    act = st5 >= STEP5
    out = []; start = None; last = None
    for t, a in act.items():
        if a:
            if start is None: start = t
            last = t
        elif start is not None and (t - last) > pd.Timedelta(minutes=GAP):
            if (last - start) >= pd.Timedelta(minutes=MIN_LEN - 5): out.append((start - pd.Timedelta("5min"), last))
            start = None
    if start is not None and (last - start) >= pd.Timedelta(minutes=MIN_LEN - 5): out.append((start - pd.Timedelta("5min"), last))
    B = pd.DataFrame(out, columns=["start", "end"])
    B["min"] = (B.end - B.start).dt.total_seconds() / 60
    B["steps"] = [st5[(st5.index > s) & (st5.index <= e)].sum() for s, e in zip(B.start, B.end)]
    B["hr_above_rest"] = [(hr5 - rest)[(hr5.index > s) & (hr5.index <= e)].mean() for s, e in zip(B.start, B.end)]
    B["spm"] = B.steps / B["min"]
    return B
