"""SPIKE-2 (2026-11-14): at meal log, predict rise size band AND when it starts (+30) and peaks. No glucose after the meal.
Modes: no glucose at all, or pre-meal baseline only. Leave-one-day-out scoring."""
import numpy as np, pandas as pd, warnings, joblib, json, os; warnings.filterwarnings("ignore")
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
T = pd.read_pickle("meals.pkl").reset_index(drop=True)
T["fpu"] = (9*T.fat + 4*T.protein)/100; T["sug_share"] = T.sugar/(T.avail+1); T["night"] = ((T.t.dt.hour>=21)|(T.t.dt.hour<6)).astype(float)
T["hr15"] = T.hr15.fillna(T.hr15.median()); T["steps60"] = T.steps60.fillna(0); T["uncov"] = T.avail - 8*T.dose
T["prec"] = T.pre.clip(40, 250); T["low"] = (T.pre < 70).astype(float)
T["y30"] = (T.rise >= 30).astype(int); T["y60"] = (T.rise >= 60).astype(int); T["band"] = T.y30 + T.y60
SIZE = {"none": ["uncov", "iob", "steps60", "hr15"], "pre": ["uncov", "iob", "steps60", "hr15", "prec", "low"]}
TIME = {"none": ["sug_share", "starch", "fiber", "fpu", "milk", "dose", "uncov", "iob"],
        "pre": ["sug_share", "starch", "fiber", "fpu", "milk", "dose", "uncov", "iob", "night", "hr15", "steps60", "pre"]}
LATE = ["sug_share", "starch", "fiber", "fpu", "milk", "dose", "uncov", "iob"]
LR = lambda **k: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, **k))
RG = lambda: make_pipeline(StandardScaler(), Ridge(alpha=3))
def fit_all(tr, mode):
    S = tr[(tr.src == "Libre") & (tr.rise >= 30)]
    return dict(size={y: LR(C=1.0).fit(tr[SIZE[mode]], tr[y]) for y in ("y30", "y60")},
                start=RG().fit(S[TIME[mode]], np.log(S.t30_min)), peak=RG().fit(S[TIME[mode]], np.log(S.peak_min)),
                late=LR(C=0.3, class_weight="balanced").fit(S[LATE], (S.t30_min >= 120).astype(int)))
def predict(M, X, mode):
    p30 = M["size"]["y30"].predict_proba(X[SIZE[mode]])[:, 1]; p60 = np.minimum(M["size"]["y60"].predict_proba(X[SIZE[mode]])[:, 1], p30)
    return pd.DataFrame(dict(p30=p30, p60=p60, band=(p30 > .5).astype(int) + (p60 > .5), start=np.exp(M["start"].predict(X[TIME[mode]])),
                             peak=np.exp(M["peak"].predict(X[TIME[mode]])), late=M["late"].predict_proba(X[LATE])[:, 1]), index=X.index)
rows = []; out = {}
for mode in ("none", "pre"):
    P = []
    for d in T.day.unique():
        tr, te = T[T.day != d], T[T.day == d]; P.append(predict(fit_all(tr, mode), te, mode))
    P = pd.concat(P).sort_index(); out[mode] = P
    for src in ("Glucometer", "Libre"):
        m = T.src == src; t = T.band[m]; b = P.band[m]
        r = dict(mode="no glucose" if mode == "none" else "pre-meal baseline", data=src, meals=int(m.sum()), band_right=f"{round(100*(b==t).mean())}%",
                 spikes_caught=f"{int(((b>=1)&(t>=1)).sum())}/{int((t>=1).sum())}", big_named=f"{int(((b==2)&(t==2)).sum())}/{int((t==2).sum())}",
                 false_alarms=f"{int(((b>=1)&(t==0)).sum())}/{int((t==0).sum())}")
        if src == "Libre":
            c = m & (P.band >= 1) & (T.band >= 1)
            es, ep = (P.start - T.t30_min)[c].abs(), (P.peak - T.peak_min)[c].abs()
            late = c & (T.t30_min >= 120)
            base_s = (T.t30_min[c] - T[(T.src=="Libre")&(T.rise>=30)].t30_min.median()).abs()
            r.update(start_err_min=round(es.median()), start_within_45=f"{round(100*(es<=45).mean())}%", peak_err_min=round(ep.median()),
                     start_err_if_typical_40min=round(base_s.median()), warning_before_rise_min=round(T.t30_min[c].median()),
                     late_rises_flagged=f"{int((P.late[late]>.5).sum())}/{int(late.sum())}", late_start_err_min=round(es[late].median()) if late.any() else None,
                     late_false_flags=f"{int(((P.late>.5)&c&(T.t30_min<120)).sum())}/{int((c&(T.t30_min<120)).sum())}")
        rows.append(r)
R = pd.DataFrame(rows); pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30); print(R.T.to_string())
R.to_csv("spike2_results.csv", index=False)
T.join(out["pre"].add_prefix("pre_")).join(out["none"].add_prefix("none_")).to_pickle("spike2_scored.pkl")
name = "SPIKE-2_food-insulin-watch15-steps-PLANNED-premeal-optional_v1_2026-11-14"; os.makedirs(f"out/{name}", exist_ok=True)
joblib.dump(dict(none=fit_all(T, "none"), pre=fit_all(T, "pre"), SIZE=SIZE, TIME=TIME, LATE=LATE, carbs_per_unit=8,
                 bands=["no rise (<+30)", "rise +30-60", "big rise +60 or more"]), f"out/{name}/{name}.joblib")
