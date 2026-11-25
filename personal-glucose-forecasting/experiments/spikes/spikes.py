"""Spike report (Nov 14 2026). Every meal/snack window + every no-food rise, Libre (Aug27-Sep17) and glucometer (Nov 6-12).
Spike = glucose rises >=30 mg/dL above the pre-meal value inside the meal window (meal -> next meal, max 5h),
or a climb >=30 with no food in the previous 5h. Usage: python spikes.py <FMS folder> <Nov watch csv folder>"""
import sys, re, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "../v8_B_planned_watch15")
from glucoforecast import load_cgm, load_log
from watch7 import load_watch7
FMS, ZIP = sys.argv[1], sys.argv[2]
C = load_cgm("../v3/cgm_export_29-10-2026.csv"); C = C[C.index >= "2026-08-27"]
cgm = pd.concat([seg.g.rolling(3, center=True, min_periods=1).mean() for _, seg in C.groupby("seg")]).dropna()
logA = load_log("../v2/log.csv"); logB = load_log("nov_log.csv")
w = pd.concat([load_watch7(f"{FMS}/noise hr logger"), load_watch7(ZIP)]).sort_index(); w = w[~w.index.duplicated()]
FR = r"apple|banana|dragon|pomegranate|grape|guava|melody|candy"; ST = r"roti|rice|dal|moong|besan|peas|soy"
def ftype(t):
    t = t.lower(); items = [x for x in t.split(";") if x.strip()]
    if all("milk" in x for x in items): return "milk only"
    if re.search(FR, t): return "has fruit"
    if re.search(ST, t): return "has roti/rice/dal"
    if re.search("paneer|egg|omelette|almond", t): return "paneer/egg + veg"
    return "veg/other"
def tod(h): return "night 0-6" if h < 6 else "morning 6-12" if h < 12 else "afternoon 12-18" if h < 18 else "evening 18-24"
def steps(a, b): x = w.steps[a:b]; return float(x.sum()) if len(x) else np.nan
H = lambda m: pd.Timedelta(minutes=m)
def meals_of(L):
    F = L[L.food.notna()]; out = []; used = set()
    for i, r in F.iterrows():
        if i in used: continue
        grp = F[(F.time >= r.time) & (F.time <= r.time + H(45))]; used |= set(grp.index)
        D = L[(L.time >= r.time - H(50)) & (L.time <= grp.time.max() + H(30))]
        txt = "; ".join(grp.food)
        out.append(dict(t=r.time, food=txt, type=ftype(txt), carbs=round(grp.net_carbs.sum()), fpu=round(grp.fpu.sum(), 1), dose=D.rapid_u.sum()))
    M = pd.DataFrame(out)
    # drop doses double-counted by two adjacent meals: keep as is (rare), fine for tagging
    return M
def zig(s, th=30, rev=15):
    v, t = s.values, s.index; legs = []; lo = hi = 0; mode = None
    for i in range(1, len(v)):
        if mode == "up":
            if v[i] > v[hi]: hi = i
            if v[hi] - v[i] >= rev:
                if v[hi] - v[lo] >= th: legs.append((t[lo], v[lo], t[hi], v[hi]))
                mode = "down"; lo = i
        else:
            if v[i] < v[lo]: lo = i
            if v[i] - v[lo] >= rev: mode = "up"; hi = i
    if mode == "up" and v[hi] - v[lo] >= th: legs.append((t[lo], v[lo], t[hi], v[hi]))
    return legs
rows = []
def cls(r):
    if r["kind"] == "no food": return "no food: night/dawn (0-7h)" if r["peak_t"].hour < 7 else "no food: daytime"
    if r["pre"] < 70 and r["dose"] == 0: return "low treated (no insulin) -> rebound"
    if r["pre"] < 70 and r["sensor"] == "Glucometer": return "meal started low -> rebound + meal"
    if r["dose"] == 0: return "food, no insulin"
    if r["carbs"] > 15 * r["dose"]: return "under-dosed (1u per 15g rule)"
    if r["peak_min"] < 60: return "covered, fast peak <1h"
    if r["peak_min"] <= 180: return "covered, peak 1-3h (UNEXPLAINED)"
    return "covered, late peak >3h"
def meal_windows(M, L, sensor, series):
    for k, m in M.iterrows():
        nxt = M.t.iloc[k + 1] if k + 1 < len(M) else m.t + H(300)
        end = min(nxt, m.t + H(300))
        if sensor == "Libre":
            pre = series[m.t - H(20):m.t + H(10)]; win = series[m.t + H(15):end]
            if len(pre) == 0 or len(win) < 4: continue
            pre = pre.mean(); cover = (end - m.t).total_seconds() / 60
        else:
            pre = series[m.t - H(45):m.t + H(10)]; win = series[m.t + H(15):end + H(10)]
            if len(pre) == 0 or len(win) == 0: continue
            pre = pre.iloc[-1]; cover = (end - m.t).total_seconds() / 60
        r = dict(sensor=sensor, kind="meal", meal_t=m.t, food=m.food, type=m.type, carbs=m.carbs, fpu=m.fpu, dose=m.dose, pre=round(pre),
                 peak=round(win.max()), rise=round(win.max() - pre), peak_t=win.idxmax(), peak_min=round((win.idxmax() - m.t).total_seconds() / 60),
                 window_min=round(cover), n_points=len(win), hour=m.t.hour, tod=tod(m.t.hour), steps_3h=steps(m.t, m.t + H(180)),
                 corr_in_window=L[(L.time > m.t + H(30)) & (L.time < end) & L.food.isna()].rapid_u.sum(),
                 lantus_before_6h=bool(L[(L.time >= m.t - H(360)) & (L.time <= m.t)].lantus_u.sum() > 0))
        r["cls"] = cls(r); rows.append(r)
def nofood(M, L, sensor, series, legs):
    for t0, g0, tp, gp in legs:
        if len(M[(M.t > min(tp - H(300), t0 - H(180))) & (M.t <= tp)]): continue
        r = dict(sensor=sensor, kind="no food", pre=round(g0), peak=round(gp), rise=round(gp - g0), peak_t=tp, start_t=t0, hour=tp.hour, tod=tod(tp.hour),
                 steps_3h=steps(tp - H(180), tp), corr_in_window=L[(L.time > t0 - H(180)) & (L.time < tp)].rapid_u.sum(),
                 lantus_before_6h=bool(L[(L.time >= tp - H(360)) & (L.time <= tp)].lantus_u.sum() > 0))
        r["cls"] = cls(r); rows.append(r)
MA, MB = meals_of(logA), meals_of(logB)
B = logB[logB.bg.notna()].set_index("time").bg
meal_windows(MA, logA, "Libre", cgm); meal_windows(MB, logB, "Glucometer", B)
legsA = [l for _, seg in cgm.groupby(C.loc[cgm.index, "seg"]) for l in zig(seg)]
legsB = [(t0, g0, tp, gp) for (t0, g0), (tp, gp) in zip(B.items(), list(B.items())[1:]) if gp - g0 >= 30 and tp - t0 <= H(240)]
nofood(MA, logA, "Libre", cgm, legsA); nofood(MB, logB, "Glucometer", B, legsB)
R = pd.DataFrame(rows).sort_values("peak_t")
R["spike"] = R.rise >= 30
R["size"] = pd.cut(R.rise, [-999, 29.9, 60, 120, 999], labels=["<30 (no spike)", "+30-60", "+60-120", "+120+"])
R.to_csv("all_windows.csv", index=False)
S = R[R.spike]; S.to_csv("spikes_list.csv", index=False)
print("windows", len(R), "spikes", len(S)); print(pd.crosstab(S.cls, S.sensor, margins=True)); print(pd.crosstab(S.cls, S["size"], margins=True))
