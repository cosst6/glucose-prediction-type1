import os, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
os.environ.update(WSET="new", FIB="split", EXF="1")
from glucoforecast9 import *
X = pd.read_pickle("X10.pkl"); hs=[60,120]
days = [d for d in sorted(X.index.normalize().unique()) if d >= pd.Timestamp("2026-08-30")]
G = {"glucose now": ["g0"], "insulin (NovoRapid)": ["ins_"], "fast carbs (sugar+GL)": ["fsug_", "gl_"], "slow carbs": ["slow_"],
     "fat+protein": ["fpu_"], "Lantus": ["lantus", "h_since_lantus"], "meal/dose timing": ["bolus_gap", "uncovered_carbs", "min_since_meal"],
     "time of day": ["tod_"], "watch HR/steps/cal": ["hr_", "steps_", "cal_", "watch_on"], "walk features": ["ex_"], "fiber": ["meal_fiber"]}
rng = np.random.default_rng(0); res = {h: {g: [] for g in ["base"] + list(G)} for h in hs}; allp=[]
for d in days:
    te = X[X.index.normalize() == d]; tr = X[X.index < d - pd.Timedelta(minutes=240)]
    hyb = Hybrid().fit(tr, "B", hs)
    for h in hs:
        cols = feature_list4("B", h); t2 = te.dropna(subset=cols + [f"y{h}"])
        if not len(t2): continue
        tru = t2.g0 + t2[f"y{h}"]; P = hyb.predict(t2, h)
        res[h]["base"].append(tru - P.pred); allp.append(pd.DataFrame({"h":h,"true":tru,"pred":P.pred,"body":P.body,"g0":t2.g0},index=t2.index))
        pool = tr.dropna(subset=cols)
        for g, pre in G.items():
            gc = [c for c in cols if any(c.startswith(p) for p in pre)]
            t3 = t2.copy()
            # replace group with values from random other rows (breaks link, keeps realistic values); g0 kept in target scale
            s = pool.sample(len(t3), replace=True, random_state=int(rng.integers(1e9)))
            for c in gc: t3[c] = s[c].values
            if g == "glucose now":
                P3 = hyb.predict(t3, h); err = tru - (t2.g0 + (P3.pred - t3.g0))  # forecast change applied to true start
            else:
                P3 = hyb.predict(t3, h); err = tru - P3.pred
            res[h][g].append(err)
    print(d.date(), flush=True)
pd.concat(allp).to_csv("wf_preds.csv")
out=[]
for h in hs:
    b = np.sqrt((pd.concat(res[h]["base"])**2).mean())
    for g in G: r = np.sqrt((pd.concat(res[h][g])**2).mean()); out.append({"h":h,"group":g,"rmse_base":round(b,1),"rmse_scrambled":round(r,1),"worse_by":round(r-b,1)})
o=pd.DataFrame(out); print(o.sort_values(["h","worse_by"],ascending=[True,False]).to_string()); o.to_csv("imp.csv",index=False)
