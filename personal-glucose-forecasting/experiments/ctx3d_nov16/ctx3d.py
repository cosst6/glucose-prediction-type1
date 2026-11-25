"""Flexible past context, 0-3 days (2026-11-16).
Uses ALL history available up to 72 h before time t; works with 0 h. Each rate is a 'shrunk' average:
  rate = (sum over the window + PRIOR * K) / (days covered + K)
so with 3 days of data the user's own numbers dominate (more data preferred), with a few hours it leans on the prior.
Glucose context uses only logged glucometer readings (log.bg), same as the phone app.
"""
import numpy as np, pandas as pd
MAXH = 72.0; K = 0.5  # prior worth half a day
PRIOR = dict(ins=16.0, carb=85.0, fpu=15.0, steps=12000.0, lows=0.3)   # per 24 h (set from Aug-Nov medians)
BG_PRIOR, BG_K = 140.0, 2.0
CTX = ["ctx_h", "ctx_ins24", "ctx_carb24", "ctx_fpu24", "ctx_steps24", "ctx_steps3h", "ctx_bg_mean", "ctx_lows24", "ctx_h_since_low"]
def context(idx, log, watch, start=None):
    """idx: times. log: needs time, rapid_u, avail, fpu, bg. watch: index t, steps. start: first time any data exists (default first log/watch)."""
    lt = log.time.values; ins = np.nan_to_num(log.rapid_u.values.astype(float)); av = np.nan_to_num(log.avail.values.astype(float))
    fp = np.nan_to_num(log.fpu.values.astype(float)); bg = log.bg.values.astype(float)
    # history starts at the first log entry of the stretch containing t (a gap of > 24 h with no log = new stretch)
    gaps = np.diff(lt).astype("timedelta64[m]").astype(float) > 24 * 60
    seg_start = lt[np.r_[True, gaps]]
    ws = watch.steps.fillna(0) if len(watch) else pd.Series(dtype=float)
    wcum = ws.cumsum(); wt = ws.index.values
    out = []
    for t in pd.DatetimeIndex(idx):
        st = seg_start[max(np.searchsorted(seg_start, np.datetime64(t), side='right') - 1, 0)] if start is None else np.datetime64(start)
        a = max(t - pd.Timedelta(hours=MAXH), pd.Timestamp(min(st, np.datetime64(t)))); hrs = max((t - a).total_seconds() / 3600, 0.0); days = hrs / 24
        m = (lt >= np.datetime64(a)) & (lt < np.datetime64(t))
        r = lambda tot, k: (tot + PRIOR[k] * K) / (days + K)
        # watch: steps over covered samples (30 s each)
        if len(wt):
            i0, i1 = np.searchsorted(wt, np.datetime64(a)), np.searchsorted(wt, np.datetime64(t))
            n = i1 - i0; tot = float(ws.values[i0:i1].sum()); wd = n * 30 / 86400
            st24 = (tot + PRIOR["steps"] * K) / (wd + K)
            j0 = np.searchsorted(wt, np.datetime64(t - pd.Timedelta("3h"))); st3 = float(ws.values[j0:i1].sum())
        else: st24, st3 = PRIOR["steps"], 0.0
        b = bg[m]; b = b[~np.isnan(b)]
        lows = (b < 70).sum()
        lowt = lt[m][~np.isnan(bg[m])][b < 70] if len(b) else []
        hsl = (t - pd.Timestamp(lowt[-1])).total_seconds() / 3600 if len(lowt) else MAXH
        out.append(dict(ctx_h=hrs, ctx_ins24=r(ins[m].sum(), "ins"), ctx_carb24=r(av[m].sum(), "carb"), ctx_fpu24=r(fp[m].sum(), "fpu"),
                        ctx_steps24=st24, ctx_steps3h=st3, ctx_bg_mean=(b.sum() + BG_PRIOR * BG_K) / (len(b) + BG_K),
                        ctx_lows24=r(lows, "lows"), ctx_h_since_low=min(hsl, MAXH)))
    return pd.DataFrame(out, index=pd.DatetimeIndex(idx))
# Keep values inside what the models were trained on (2-98% of Aug-Nov data): stops wild extrapolation on unusual days.
CLIP = dict(ctx_ins24=(8.5, 21), ctx_carb24=(62, 130), ctx_fpu24=(9, 18), ctx_steps24=(10000, 20500), ctx_steps3h=(0, 10000),
            ctx_bg_mean=(110, 172), ctx_lows24=(0, 2))
_context_raw = context
def context(idx, log, watch, start=None, clip=True):
    c = _context_raw(idx, log, watch, start)
    if clip:
        for k, (a, b) in CLIP.items(): c[k] = c[k].clip(a, b)
    return c
SM2 = ["ctx_ins24", "ctx_steps24", "ctx_steps3h", "ctx_bg_mean"]   # used by the context models (no 'hours of history' input: avoids learning "short history = odd day")
