import json
src = open('final.py').read(); exec(src[:src.index("rows = []\nfor src")])
ask2 = ((abs(P1[:, 0] - .5) < .1) | (abs(P1[:, 1] - .5) < .1)) & Lm.values
T["ask2"] = ask2; T["bfin"] = np.where(ask2, T.b2, T.b1)
import joblib, glob
M = joblib.load(glob.glob("out/*/*.joblib")[0])
def contrib(step, row):
    s = M[step]; out = {}
    for y in ("y30",):
        pipe = s["m"][y]; sc, lr = pipe[0], pipe[-1]
        x = np.array([row[c] for c in s["cols"]], float); z = (x - sc.mean_) / sc.scale_
        out = {c: round(float(v), 2) for c, v in zip(s["cols"], lr.coef_[0] * z)}
    return out
from glucoforecast8 import load_log4
Ln = load_log4("log_nov.csv").drop_duplicates(subset=["time", "rapid_u", "lantus_u", "bg", "food"]); sticks = Ln.dropna(subset=["bg"]).set_index("time").bg
def ftype(t):
    t = t.lower(); items = [x for x in t.split(";") if x.strip()]
    if all("milk" in x for x in items): return "milk only"
    if re.search(r"apple|banana|dragon|pomegranate|grape|guava|melody|candy", t): return "has fruit"
    if re.search(r"roti|rice|dal|moong|besan|peas|soy", t): return "roti/rice/dal"
    if re.search("paneer|egg|omelette|almond", t): return "paneer/egg + veg"
    return "veg/other"
import re
meals = []
for i, r in T.iterrows():
    t0 = r.t
    if r.src == "Libre":
        s = cg[t0 - pd.Timedelta("60min"): t0 + pd.Timedelta("300min")]
    else:
        s = sticks[t0 - pd.Timedelta("120min"): t0 + pd.Timedelta("330min")]
    curve = [[round((k - t0).total_seconds() / 60), round(float(v))] for k, v in s.items()]
    end = round(r.win_min)
    meals.append(dict(id=i, src=r.src, t=t0.strftime("%Y-%m-%d %H:%M"), food=r.food, type=ftype(r.food), hour=int(t0.hour),
        avail=round(r.avail, 1), sugar=round(r.sugar, 1), starch=round(r.starch, 1), fiber=round(r.fiber, 1), fat=round(r.fat, 1), protein=round(r.protein, 1),
        dose=float(r.dose), uncov=round(r.uncov, 1), iob=round(r.iob, 2), hr15=round(r.hr15), steps60=int(r.steps60), pre=round(r.pre),
        rise=round(r.rise), band=int(r.band), peak_min=round(r.peak_min), t30=None if pd.isna(r.t30_min) else round(r.t30_min), win=end,
        p0=[round(P0[i, 0], 2), round(P0[i, 1], 2)], b0=int(r.b0), p1=[round(P1[i, 0], 2), round(P1[i, 1], 2)], b1=int(r.b1),
        p2=None if np.isnan(P2[i, 0]) else [round(P2[i, 0], 2), round(P2[i, 1], 2)], b2=int(r.b2), d45=None if pd.isna(r.d45) else round(r.d45),
        ask2=bool(r.ask2), bfin=int(r.bfin), rule=int(r.rule), why=contrib("step1_pre_stick", r), curve=curve))
json.dump(meals, open("explorer_data.json", "w"))
print(len(meals), len(json.dumps(meals)) // 1024, "KB")
print(pd.Series([m["type"] for m in meals]).value_counts())
