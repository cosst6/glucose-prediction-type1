import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
T = pd.read_pickle("meals.pkl").reset_index(drop=True)
T["fpu"] = (9*T.fat + 4*T.protein)/100; T["sug_share"] = T.sugar/(T.avail+1); T["night"] = ((T.t.dt.hour>=21)|(T.t.dt.hour<6)).astype(float)
T["hr15"] = T.hr15.fillna(T.hr15.median()); T["steps60"] = T.steps60.fillna(0); T["uncov"] = T.avail - 8*T.dose
T["hour_s"] = np.sin(2*np.pi*T.t.dt.hour/24); T["hour_c"] = np.cos(2*np.pi*T.t.dt.hour/24)
S = T[(T.src=="Libre") & (T.rise>=30)].copy()
S["capped"] = (S.win_min - S.peak_min) <= 15
S["late"] = (S.peak_min >= 150).astype(int)
SETS = {"food speed": ["sug_share","starch","fiber","fpu","milk"],
        "+insulin": ["sug_share","starch","fiber","fpu","milk","dose","uncov","iob"],
        "+insulin+night+watch": ["sug_share","starch","fiber","fpu","milk","dose","uncov","iob","night","hr15","steps60"],
        "+all+pre": ["sug_share","starch","fiber","fpu","milk","dose","uncov","iob","night","hr15","steps60","pre"]}
for tgt in ("t30_min","peak_min"):
    y = np.log(S[tgt]); base = []; 
    for d in S.day.unique(): base += list(np.abs(S[S.day==d][tgt] - S[S.day!=d][tgt].median()))
    print(tgt, "baseline (median) MAE", round(np.mean(base)), "min")
    for nm, cols in SETS.items():
        for a in (3, 10, 30):
            p = pd.Series(index=S.index, dtype=float)
            for d in S.day.unique():
                tr, te = S.day!=d, S.day==d
                p[te] = np.exp(make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(S.loc[tr,cols], y[tr]).predict(S.loc[te,cols]))
            e = np.abs(p - S[tgt]); w60 = (e<=60).mean()
            print(f"  {nm:24s} a={a:2d} MAE {e.mean():5.0f}  within 60 min {100*w60:3.0f}%  late-peak MAE {e[S.late==1].mean():4.0f}")
print("late (peak>=150) AUC:")
for nm, cols in SETS.items():
    p = pd.Series(index=S.index, dtype=float)
    for d in S.day.unique():
        tr, te = S.day!=d, S.day==d
        if S.loc[tr,"late"].nunique()<2: continue
        p[te] = make_pipeline(StandardScaler(), LogisticRegression(C=0.3)).fit(S.loc[tr,cols], S.loc[tr,"late"]).predict_proba(S.loc[te,cols])[:,1]
    print(" ", nm, round(roc_auc_score(S.late, p.fillna(.5)),2), "caught", int(((p>.5)&(S.late==1)).sum()), "/", int(S.late.sum()), "false", int(((p>.5)&(S.late==0)).sum()))
print(S[S.late==1][["t","food","dose","rise","t30_min","peak_min","capped"]].to_string())
print("\nLATE START (t30>=120) among Libre spikes:", int((S.t30_min>=120).sum()), "/", len(S))
S["lstart"] = (S.t30_min >= 120).astype(int)
S["lowcarb_ins"] = ((S.avail < 15) & (S.dose >= 2)).astype(float)
for nm, cols in list(SETS.items()) + [("rule: carbs<15g & dose>=2u", ["lowcarb_ins"])]:
    p = pd.Series(index=S.index, dtype=float)
    for d in S.day.unique():
        tr, te = S.day!=d, S.day==d
        if S.loc[tr,"lstart"].nunique()<2: p[te]=0; continue
        p[te] = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced")).fit(S.loc[tr,cols], S.loc[tr,"lstart"]).predict_proba(S.loc[te,cols])[:,1]
    print(" ", nm, "AUC", round(roc_auc_score(S.lstart, p),2), "caught", int(((p>.5)&(S.lstart==1)).sum()), "/", int(S.lstart.sum()), "false", int(((p>.5)&(S.lstart==0)).sum()))
print(" raw rule caught", int(((S.lowcarb_ins==1)&(S.lstart==1)).sum()), "false", int(((S.lowcarb_ins==1)&(S.lstart==0)).sum()))
# among ALL Libre meals: low-carb + insulin meals, how many rise late?
A = T[T.src=="Libre"]; m = (A.avail<15)&(A.dose>=2)
print(" all Libre low-carb+insulin meals:", int(m.sum()), "late rise >=30 after 2h:", int((m&(A.t30_min>=120)).sum()), "any spike:", int((m&(A.rise>=30)).sum()))
