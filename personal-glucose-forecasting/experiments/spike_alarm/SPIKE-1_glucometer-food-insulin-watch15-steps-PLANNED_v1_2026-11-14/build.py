"""Build one row per meal/snack: food split, insulin, watch at meal time, and the rise that followed (blood scale)."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from glucoforecast import load_cgm, insulin_cum, tri_cum
from glucoforecast8 import load_log4
from watch7 import load_watch7
R = 0.72  # Libre = 0.72 x blood (measured Nov 14)
H = lambda m: pd.Timedelta(minutes=m)
C = load_cgm("libre.csv"); C = C[C.index >= "2026-08-27"]
cgm = pd.concat([s.g.rolling(3, center=True, min_periods=1).mean() for _, s in C.groupby("seg")]).dropna() / R
w = pd.concat([load_watch7("/home/user/sp/data/fms/FMS/noise hr logger"), load_watch7("/home/user/sp/data/nov/glucometer diabetes data/heart rate steps")]).sort_index()
w = w[~w.index.duplicated()]
def watch_at(t):
    hr = w.hr[t - H(15):t]; st = w.steps[t - H(60):t]
    return (float(hr.mean()) if hr.notna().any() else np.nan, float(st.sum()) if len(st) else np.nan)
def meals(L):
    F = L[L.food.notna()]; out = []; used = set()
    for i, r in F.iterrows():
        if i in used: continue
        grp = F[(F.time >= r.time) & (F.time <= r.time + H(45))]; used |= set(grp.index)
        D = L[(L.time >= r.time - H(50)) & (L.time <= grp.time.max() + H(30))]
        prev = L[L.time < r.time - H(50)]
        m = (r.time - prev.time).dt.total_seconds().values / 60
        iob = float((np.nan_to_num(prev.rapid_u.values) * (1 - insulin_cum(m))).sum())
        cob = float((prev.avail.values * (1 - tri_cum(m, 180))).sum())
        fpu_ob = float((prev.fpu.values * (1 - tri_cum(m, 300, .5))).sum())
        lt = L[(L.lantus_u > 0) & (L.time <= r.time)].time
        out.append(dict(t=r.time, end_food=grp.time.max(), food="; ".join(grp.food), sugar=grp.sugar.sum(), starch=grp.starch.sum(), fiber=grp.fiber.sum(),
                        fat=grp.fat.sum(), protein=grp.protein.sum(), avail=grp.avail.sum(), dose=np.nan_to_num(D.rapid_u).sum(), iob=iob, cob=cob, fpu_ob=fpu_ob,
                        milk=float(grp.food.str.contains("milk").any() and not grp.food.str.contains("paneer from").all()),
                        h_lantus=(r.time - lt.iloc[-1]).total_seconds() / 3600 if len(lt) else 24))
    M = pd.DataFrame(out); M["next"] = list(M.t.iloc[1:]) + [pd.NaT]
    M["end"] = [min(n, t + H(300)) if pd.notna(n) else t + H(300) for t, n in zip(M.t, M.next)]
    return M
rows = []
for src, logf in (("Libre", "log_aug.csv"), ("Glucometer", "log_nov.csv")):
    L = load_log4(logf); L = L.drop_duplicates(subset=["time", "rapid_u", "lantus_u", "bg", "food"])
    M = meals(L); sticks = L.dropna(subset=["bg"]).set_index("time").bg
    for _, m in M.iterrows():
        d = m.to_dict(); d["src"] = src; d["hr15"], d["steps60"] = watch_at(m.t)
        if src == "Libre":
            pre = cgm[m.t - H(20):m.t + H(10)]; win = cgm[m.t + H(15):m.end]
            if len(pre) < 2 or len(win) < 4: continue
            d["pre"] = pre.mean(); d["pre_src"] = "libre"
        else:
            pre = sticks[m.t - H(45):m.t + H(10)]; win = sticks[m.t + H(15):m.end + H(10)]
            if len(pre) == 0 or len(win) == 0: continue
            d["pre"] = pre.iloc[-1]; d["n_post"] = len(win)
        d["peak"] = win.max(); d["rise"] = win.max() - d["pre"]; d["peak_min"] = (win.idxmax() - m.t).total_seconds() / 60
        above = win[win >= d["pre"] + 30]
        d["t30_min"] = (above.index[0] - m.t).total_seconds() / 60 if len(above) else np.nan  # when the rise passed +30
        d["win_min"] = (m.end - m.t).total_seconds() / 60
        rows.append(d)
T = pd.DataFrame(rows); T["day"] = T.t.dt.date.astype(str); T.to_pickle("meals.pkl")
print(T.groupby("src").size()); print(T.groupby("src").rise.describe())
print(pd.crosstab(T.src, pd.cut(T.rise, [-999, 0, 15, 30, 45, 60, 90, 120, 999])))
print(T[T.src == "Glucometer"][["t", "food", "dose", "pre", "peak", "rise", "n_post", "peak_min"]].to_string())
