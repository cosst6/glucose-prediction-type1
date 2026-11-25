from lowfeat import *
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
D = "../D"; HS = [30, 60, 90, 120, 180, 240, 300, 360]
# ---- training set: old CGM period (Libre converted to blood scale: blood = cgm / 0.72); low = cgm < 52 (~blood < 72)
cgm = load_cgm(glob.glob(f"{D}/*glucose*.csv")[0]); cgm = cgm[cgm.index >= "2026-08-27"]
logo = load_log9(f"{D}/log.csv"); wo = load_watch(f"{D}/noise hr logger")
g = cgm.g; logged = set(logo.time.dt.normalize()); base = g.index[g.index.normalize().isin(list(logged))]
tr = []
for H in HS:
    fut = g.reindex(base + pd.Timedelta(minutes=H)).values
    ok = ~np.isnan(fut) & ~np.isnan(g.reindex(base).values)
    idx = base[ok]
    X = feats(idx, logo, wo, H, g0=g.reindex(idx).values / R)
    X["pred"] = X.g0 + X.body; X["age"] = 0.0
    X["y"] = (fut[ok] < 52).astype(int); tr.append(X)
TR = pd.concat(tr); print("train rows", len(TR), "low rows", TR.y.sum())
# ---- test: every Nov fingerstick r, decisions made L minutes before it
logn = load_log9(f"{D}/log_nov.csv"); wn = load_watch(f"{D}/nov_watch")
B = logn.dropna(subset=["bg"])[["time", "bg"]].sort_values("time").reset_index(drop=True)
te = []
for L in HS:
    for _, r in B.iterrows():
        t = r.time - pd.Timedelta(minutes=L); prev = B[B.time <= t]
        nog = feats(pd.DatetimeIndex([t]), logn, wn, L).iloc[0]
        row = dict(target=r.time, L=L, bg=r.bg, y=int(r.bg < 70), **{f"n_{k}": nog[k] for k in NOG})
        if len(prev):
            a = prev.iloc[-1]; age = (t - a.time).total_seconds() / 60
            one = feats(pd.DatetimeIndex([a.time]), logn, wn, int(age + L), g0=np.array([a.bg])).iloc[0]
            row.update({f"o_{k}": one[k] for k in NOG}, o_g0=a.bg, o_age=age, o_pred=a.bg + one["body"])
        te.append(row)
TE = pd.DataFrame(te); TE.to_pickle("../low_test.pkl"); TR.to_pickle("../low_train.pkl")
print("test rows", len(TE), "lows", TE.y.sum(), "distinct low readings", TE[TE.y == 1].target.nunique())
