import numpy as np, pandas as pd, warnings, itertools; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
T = pd.read_pickle("meals.pkl").reset_index(drop=True)
T["night"] = ((T.t.dt.hour >= 22) | (T.t.dt.hour < 6)).astype(float)
T["hr15"] = T.hr15.fillna(T.hr15.median()); T["steps60"] = T.steps60.fillna(0)
T["uncov"] = T.avail - 8 * T.dose; T["low"] = (T.pre < 70).astype(float); T["prec"] = T.pre.clip(40, 250)
T["fast"] = T.sugar; T["y30"] = (T.rise >= 30).astype(int); T["y60"] = (T.rise >= 60).astype(int)
T["band"] = T.y30 + T.y60
FOOD = ["sugar", "starch", "fiber", "fat", "protein"]; INS = ["dose", "iob"]; W = ["hr15", "steps60"]
SETS = {
 "food+ins+watch": FOOD + INS + W,
 "+uncov": FOOD + INS + W + ["uncov"],
 "+uncov+night+cob": FOOD + INS + W + ["uncov", "night", "cob", "fpu_ob"],
 "+uncov+milk+night": FOOD + INS + W + ["uncov", "milk", "night"],
}
STICK = ["prec", "low"]
LR = lambda C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
def lodo(cols, C=0.3, gw=3.0):
    P = np.zeros((len(T), 2))
    for d in T.day.unique():
        tr, te = T.day != d, T.day == d
        sw = np.where(T.src[tr] == "Glucometer", gw, 1.0)
        for j, y in enumerate(("y30", "y60")):
            P[te, j] = LR(C).fit(T.loc[tr, cols], T.loc[tr, y], logisticregression__sample_weight=sw).predict_proba(T.loc[te, cols])[:, 1]
    P[:, 1] = np.minimum(P[:, 1], P[:, 0]); return P
def score(P, m=None):
    m = np.ones(len(T), bool) if m is None else m
    pb = (P[:, 0] > .5).astype(int) + (P[:, 1] > .5).astype(int)
    t = T.band.values; s = t >= 1; a = pb >= 1
    return dict(n=int(m.sum()), band_acc=round(100 * (pb == t)[m].mean()), spike_caught=f"{(a & s & m).sum()}/{(s & m).sum()}",
                big_caught=f"{((pb == 2) & (t == 2) & m).sum()}/{((t == 2) & m).sum()}", false_alarm=f"{(a & ~s & m).sum()}/{(~s & m).sum()}",
                off2=int(((abs(pb - t) == 2) & m).sum()))
rows = []
for name, cols in SETS.items():
    for st in (False, True):
        for C in (0.1, 0.3, 1.0):
            c = cols + (STICK if st else [])
            P = lodo(c, C)
            for src in ("Libre", "Glucometer"):
                rows.append(dict(set=name, stick=st, C=C, src=src, **score(P, (T.src == src).values)))
R = pd.DataFrame(rows); pd.set_option("display.width", 250); print(R.to_string(index=False))
# baselines
for src in ("Libre", "Glucometer"):
    m = (T.src == src).values; t = T.band.values
    rule = (T.avail > 8 * T.dose).values  # user's data rule: <1u per 8 g -> spike
    print(src, "rule <1u/8g: caught", (rule & (t >= 1) & m).sum(), "/", ((t >= 1) & m).sum(), "false", (rule & (t == 0) & m).sum(), "/", ((t == 0) & m).sum())
