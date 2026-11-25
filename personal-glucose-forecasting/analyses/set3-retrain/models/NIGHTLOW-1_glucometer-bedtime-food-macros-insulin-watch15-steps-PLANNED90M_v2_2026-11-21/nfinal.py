import json, joblib, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from phys import *
exec(open("night13.py").read().split("def score")[0].split("DB = prep(pd.DataFrame(rows))")[0])
DB = prep(pd.DataFrame(rows))
for h in H:
    if f"g{h}" not in DB: DB[f"g{h}"] = np.nan
D = pd.concat([DL, DB], ignore_index=True); E = {h: [] for h in H}
for i in D.index:
    M = fit(D.drop(i)); p, _ = predict(M, D.loc[i], 70)
    for h in H:
        if pd.notna(D.loc[i, f"g{h}"]): E[h].append(p[h-1] - D.loc[i, f"g{h}"])
sd = {h: float(np.sqrt(np.mean(np.square(E[h])))) for h in H}; M = fit(D)
NAME = "NIGHTLOW-1_glucometer-bedtime-food-macros-insulin-watch15-steps-PLANNED90M_v2_2026-11-21"
old = json.load(open(glob.glob("glucose-model/nightlow/NIGHT*/NIGHT*.json")[0]))
model = dict(old); model.update(name=NAME, coef={h: [round(float(v), 4) for v in M[h][0]] for h in H}, sd={h: round(max(sd[h], 20.0), 1) for h in H},
    trained="2026-11-21: 18 Libre nights (/0.72) + 7 glucometer nights (set 2: 5, set 3: 2)")
json.dump(model, open(f"{NAME}.json", "w"), indent=1); joblib.dump(model, f"{NAME}.joblib"); print(model["sd"])
