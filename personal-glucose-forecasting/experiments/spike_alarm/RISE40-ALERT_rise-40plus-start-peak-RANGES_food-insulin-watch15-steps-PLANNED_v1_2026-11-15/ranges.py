"""Timing ranges for SPIKE-2 and the new RISE40 alert (2026-11-15). Leave-one-day-out, ranges from other days' errors."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from glucoforecast import load_cgm
T = pd.read_pickle("meals.pkl").reset_index(drop=True)
C0 = load_cgm("libre.csv"); cg = pd.concat([s.g.rolling(3, center=True, min_periods=1).mean() for _, s in C0.groupby("seg")]).dropna() / 0.72
H = lambda m: pd.Timedelta(minutes=m)
def first_cross(t0, end, lvl):
    w = cg[t0 + H(15):end]; a = w[w >= lvl]
    return (a.index[0] - t0).total_seconds() / 60 if len(a) else np.nan
T["end"] = [r.t + H(r.win_min) for r in T.itertuples()]
for k in (20, 40):
    T[f"t{k}"] = [first_cross(r.t, r.end, r.pre + k) if r.src == "Libre" else np.nan for r in T.itertuples()]
T["fpu"] = (9*T.fat + 4*T.protein)/100; T["sug_share"] = T.sugar/(T.avail+1); T["night"] = ((T.t.dt.hour>=21)|(T.t.dt.hour<6)).astype(float)
T["hr15"] = T.hr15.fillna(T.hr15.median()); T["steps60"] = T.steps60.fillna(0); T["uncov"] = T.avail - 8*T.dose
T["prec"] = T.pre.clip(40, 250); T["low"] = (T.pre < 70).astype(float)
T["y40"] = (T.rise >= 40).astype(int)
T.to_pickle("meals_r.pkl")
print("rises >=40:", T.groupby("src").y40.agg(["sum", "size"]).to_dict())
TIME = {"none": ["sug_share", "starch", "fiber", "fpu", "milk", "dose", "uncov", "iob"],
        "pre": ["sug_share", "starch", "fiber", "fpu", "milk", "dose", "uncov", "iob", "night", "hr15", "steps60", "pre"]}
SIZE = {"none": ["uncov", "iob", "steps60", "hr15"], "pre": ["uncov", "iob", "steps60", "hr15", "prec", "low"]}
RG = lambda: make_pipeline(StandardScaler(), Ridge(alpha=3))
LR = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
def lodo_time(S, tgt, cols):
    p = pd.Series(index=S.index, dtype=float)
    for d in S.day.unique():
        tr, te = S.day != d, S.day == d
        p[te] = np.exp(RG().fit(S.loc[tr, cols], np.log(S.loc[tr, tgt])).predict(S.loc[te, cols]))
    return p
def range_eval(S, tgt, pred, cov):
    """range = pred * exp(quantiles of other days' log-errors); coverage + width on each held-out day."""
    lo_q, hi_q = (1 - cov) / 2, 1 - (1 - cov) / 2
    r = np.log(S[tgt]) - np.log(pred); inside, width, lo_f, hi_f = [], [], [], []
    for d in S.day.unique():
        o = S.day != d; a, b = np.quantile(r[o], [lo_q, hi_q])
        te = S.day == d; L, U = pred[te] * np.exp(a), pred[te] * np.exp(b)
        inside += list((S.loc[te, tgt] >= L) & (S.loc[te, tgt] <= U)); width += list(U - L)
    a, b = np.quantile(r, [lo_q, hi_q])
    return np.mean(inside), np.median(width), np.exp(a), np.exp(b)
def typical_eval(S, tgt, cov):
    lo_q, hi_q = (1 - cov) / 2, 1 - (1 - cov) / 2; inside, width = [], []
    for d in S.day.unique():
        o = S.day != d; a, b = np.quantile(S.loc[o, tgt], [lo_q, hi_q]); te = S.day == d
        inside += list((S.loc[te, tgt] >= a) & (S.loc[te, tgt] <= b)); width += [b - a] * te.sum()
    return np.mean(inside), np.median(width)
rows = []
for model, thr, tgts in (("SPIKE-2 (rise >=30)", 30, ("t30_min", "peak_min")), ("RISE40 (rise >=40)", 40, ("t20", "t40", "peak_min"))):
    S = T[(T.src == "Libre") & (T.rise >= thr)].copy()
    for tgt in tgts:
        S2 = S.dropna(subset=[tgt])
        for mode in ("none", "pre"):
            pred = lodo_time(S2, tgt, TIME[mode])
            for cov in (0.5, 0.7, 0.8):
                c, w, a, b = range_eval(S2, tgt, pred, cov); tc, tw = typical_eval(S2, tgt, cov)
                rows.append(dict(model=model, what=tgt, mode=mode, n=len(S2), target_cov=cov, inside=round(100*c), width_min=round(w),
                                 typical_inside=round(100*tc), typical_width=round(tw), lo_factor=round(a, 2), hi_factor=round(b, 2)))
R = pd.DataFrame(rows); pd.set_option("display.width", 250); print(R.to_string(index=False)); R.to_csv("ranges_results.csv", index=False)
# RISE40 detection (yes/no), LODO
for mode in ("none", "pre"):
    p = pd.Series(index=T.index, dtype=float)
    for d in T.day.unique():
        tr, te = T.day != d, T.day == d
        p[te] = LR().fit(T.loc[tr, SIZE[mode]], T.loc[tr, "y40"]).predict_proba(T.loc[te, SIZE[mode]])[:, 1]
    T[f"p40_{mode}"] = p
    for src in ("Glucometer", "Libre"):
        m = T.src == src; y = T.y40[m]; a = p[m] > .5
        print(mode, src, "AUC", round(roc_auc_score(y, p[m]), 2), "acc", round(100*((a == (y == 1)).mean())), "caught", int((a & (y == 1)).sum()), "/", int(y.sum()),
              "warn-real", int((a & (y == 1)).sum()), "/", int(a.sum()), "false", int((a & (y == 0)).sum()), "/", int((y == 0).sum()))
T.to_pickle("meals_r.pkl")
