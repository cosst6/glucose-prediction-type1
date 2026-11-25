import json, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from feats import *; from phys import *
NJ = json.load(open(glob.glob("glucose-model/nightlow/NIGHT*/NIGHT*.json")[0]))
OLD = {int(h): (np.array(NJ["coef"][h]), NJ["sd"][h]) for h in NJ["coef"]}
def prep(D):
    D = D.copy(); D["fpu"] = (9*D.fat4h + 4*D.prot4h)/100; D["hr15"] = D.hr15.fillna(70); D["steps3h"] = np.log1p(D.steps3h.fillna(0)); return D
DL = prep(pd.read_csv("nights_libre.csv", parse_dates=["bed"])); DL["src"] = "libre"
rows = []
for s, lf, wf in ((2, "log_nov.csv", "nov_watch"), (3, "log_set3.csv", "set3_watch")):
    L = prep_log(pd.read_csv(f"/home/user/work/D/{lf}")); W = load_watch7(f"/home/user/work/D/{wf}")
    for _, r in L[L.lantus_u.notna()].iterrows():
        t = r.time; b = L[(L.bg.notna()) & ((L.time - t).dt.total_seconds().abs() <= 60*60)]
        if not len(b): continue
        b = b.iloc[(b.time - t).abs().argsort().values[:1]]
        f = bed_features(t, L, W, float(b.bg.iloc[0])); f.update(bed=t, src=f"set{s}")
        ev = L[(L.time > t + pd.Timedelta("90min")) & (L.food.notna() | L.rapid_u.notna())]
        end = min(ev.time.min() if len(ev) else t + pd.Timedelta("10.5h"), t + pd.Timedelta("10.5h"))
        x = L[(L.time > t + pd.Timedelta("90min")) & (L.time <= end) & L.bg.notna()]
        if not len(x): continue
        for _, xx in x.iterrows():
            hh = int(min(max(round((xx.time - t).total_seconds()/3600), 1), 9)); f[f"g{hh}"] = xx.bg
        rows.append(f)
DB = prep(pd.DataFrame(rows)); print(DB[["bed", "src", "g_bed", "iob", "milk_bed"] + [f"g{h}" for h in H if f"g{h}" in DB]].to_string())
for h in H:
    if f"g{h}" not in DB: DB[f"g{h}"] = np.nan
def score(Mfun, name):
    out = []
    for i in DB.index:
        M = Mfun(i); p, P = predict(M, DB.loc[i], 70)
        for h in H:
            a = DB.loc[i, f"g{h}"]
            if pd.notna(a): out.append(dict(model=name, bed=DB.loc[i, "bed"], src=DB.loc[i, "src"], h=h, actual=a, pred=round(p[h-1]), p_low_max=round(float(P.max()), 2), real_low=int(a < 70)))
    return out
oldM = lambda i: OLD
ALL = pd.concat([DL, DB], ignore_index=True)
newM = lambda i: fit(pd.concat([DL, DB.drop(i)], ignore_index=True))
O = pd.DataFrame(score(oldM, "old NIGHTLOW-1") + score(newM, "new (Libre + other blood nights)"))
O["alarm"] = (O.p_low_max > 0.3).astype(int); O.to_csv("night13_results.csv", index=False); print(O.to_string())
for m, d in O.groupby("model"):
    e = d.pred - d.actual; print(m, "RMSE", round(float(np.sqrt((e**2).mean())), 1), "no-change", round(float(np.sqrt(((d.actual - DB.set_index("bed").loc[d.bed, "g_bed"].values)**2).mean())), 1),
          "lows caught", int(((d.alarm == 1) & (d.real_low == 1)).sum()), "/", int(d.real_low.sum()), "false alarms (nights)", d[(d.real_low == 0)].groupby("bed").alarm.max().sum(), "/", d[d.real_low == 0].bed.nunique())
Mf = fit(ALL); json.dump({h: [round(float(v), 4) for v in Mf[h][0]] for h in H}, open("night13_coef.json", "w"))
DB.to_csv("blood_nights.csv", index=False)
