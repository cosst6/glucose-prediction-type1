"""Insulin-to-food timing (Nov 19 2026). For every meal: gap = food time - nearest NovoRapid dose time (+ = dose first).
Libre Aug27-Sep17 (15-min, smoothed) gives curves; glucometer Nov 6-12 gives sparse checks."""
import sys, re, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
G = "glucose-model"
sys.path.insert(0, f"{G}/v8_B_planned_watch15")
from glucoforecast import load_cgm, load_log
C = load_cgm(f"{G}/v3/cgm_export_29-10-2026.csv"); C = C[C.index >= "2026-08-27"]
cgm = pd.concat([s.g.rolling(3, center=True, min_periods=1).mean() for _, s in C.groupby("seg")]).dropna()
H = lambda m: pd.Timedelta(minutes=m)
FAST = r"banana|apple|grape|guava|pomegranate|candy|melody|rice|biscuit|cake|sugar|dragon"
def fam(t):
    t = t.lower(); it = [x for x in t.split(";") if x.strip()]
    if all("milk" in x for x in it): return "milk only"
    if re.search(FAST, t) and re.search("paneer|egg|omelette|dal|moong", t): return "fruit/rice + protein"
    if re.search(FAST, t): return "fruit/fast carb only"
    if re.search("roti|dal|moong|besan|peas|soy", t): return "roti/dal/moong"
    return "paneer/egg + veg"
def meals(L):
    F = L[L.food.notna()]; used = set(); out = []
    for i, r in F.iterrows():
        if i in used: continue
        grp = F[(F.time >= r.time) & (F.time <= r.time + H(20))]; used |= set(grp.index)
        txt = "; ".join(grp.food)
        D = L[(L.time >= r.time - H(60)) & (L.time <= r.time + H(40)) & (L.rapid_u > 0)]
        # dose rows that also carry food from another meal are excluded except this meal's own rows
        D = D[(D.food.isna()) | D.index.isin(grp.index)]
        if len(D):
            dt = D.time.iloc[(D.time - r.time).abs().argsort().iloc[0]]
            gap = (r.time - dt).total_seconds() / 60; dose = D.rapid_u.sum()
        else: gap, dose = np.nan, 0
        out.append(dict(t=r.time, food=txt, fam=fam(txt), carbs=round(grp.net_carbs.sum()), fpu=round(grp.fpu.sum(), 1), dose=dose, gap=gap))
    return pd.DataFrame(out)
def gcat(g, d):
    if d == 0: return "no insulin"
    if g >= 10: return "dose 10-60 min BEFORE food"
    if g <= -10: return "dose 10-40 min AFTER food"
    return "dose WITH food (0-9 min)"
L = load_log(f"{G}/v2/log.csv"); M = meals(L); rows = []
for k, m in M.iterrows():
    nxt = M.t.iloc[k + 1] if k + 1 < len(M) else m.t + H(300); end = min(nxt, m.t + H(240))
    pre = cgm[m.t - H(20):m.t + H(5)]; win = cgm[m.t:end]
    if len(pre) == 0 or len(win) < 6: continue
    p = pre.mean(); extra = L[(L.time > m.t + H(40)) & (L.time < end) & (L.rapid_u > 0) & L.food.isna()].rapid_u.sum()
    r = dict(m); r.update(pre=round(p), rise=round(win.max() - p), peak_min=int((win.idxmax() - m.t).total_seconds() / 60),
        nadir=round(win.min()), drop=round(p - win.min()), r60=round(cgm[m.t + H(50):m.t + H(70)].mean() - p) if len(cgm[m.t + H(50):m.t + H(70)]) else np.nan,
        win_min=int((end - m.t).total_seconds() / 60), corr_after=extra, low70=bool(win.min() < 70), cat=gcat(m.gap, m.dose))
    rows.append(r)
R = pd.DataFrame(rows)
# Libre reads low (~28%); a low on Libre <70 ~ blood <~95. Keep Libre scale but flag.
R.to_csv("insulin-timing/meals_timing_libre.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
print(R.cat.value_counts())
print(R.groupby(["fam", "cat"]).agg(n=("rise", "size"), rise=("rise", "median"), peak=("peak_min", "median"), r60=("r60", "median"), drop=("drop", "median"), lows=("low70", "sum"), pre=("pre", "median"), carbs=("carbs", "median"), dose=("dose", "median")))
print(R[R.cat != "dose WITH food (0-9 min)"][["t", "food", "fam", "dose", "gap", "pre", "rise", "peak_min", "nadir", "corr_after", "win_min"]].to_string())
