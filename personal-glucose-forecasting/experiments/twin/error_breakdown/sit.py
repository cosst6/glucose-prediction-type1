import numpy as np, pandas as pd, pickle
TL, TABS = pickle.load(open("../twin_data.pkl", "rb"))
def tag(df):
    out = []
    for _, r in df.iterrows():
        tl = TL[r.tl]; a = (r.t - tl.t0).total_seconds() / 60; b = (r.tgt - tl.t0).total_seconds() / 60
        carbs = tl.mc[(tl.mt >= a - 60) & (tl.mt <= b)].sum(); fpu = tl.mfpu[(tl.mt >= a - 60) & (tl.mt <= b)].sum(); ins = tl.bu[(tl.bt >= a - 60) & (tl.bt <= b)].sum()
        hr = r.t.hour
        s = "night (10pm-6am)" if (hr >= 22 or hr < 6) else ("meal in window" if carbs >= 5 or fpu >= 1.5 else "no food (fasting)")
        lvl = "start low (<80)" if r.g0 < 80 else ("start high (>180)" if r.g0 > 180 else "start 80-180")
        out.append((s, lvl, "insulin in window" if ins > 0 else "no insulin"))
    return pd.DataFrame(out, columns=["when", "level", "ins"], index=df.index)
r = lambda e: round(float(np.sqrt(np.mean(np.square(e)))), 1)
w30 = lambda e: round(100 * float((np.abs(e) <= 30).mean()))
SS = pd.read_pickle("../fuse_ss.pkl"); SS["avg"] = (SS.twin + SS.v14 + SS.v10live) / 3; SS = SS.join(tag(SS))
L = pd.read_pickle("../wf3_a_wcl1.pkl"); L = L[(L.tag == "Libre->stick") & (L.gap > 90) & (L.gap <= 150)].copy(); L = L.join(tag(L))
for name, D, cols in (("stick->stick ~2h", SS[SS.hb == "~2h"], ["twin", "v14", "v10live", "avg"]), ("stick->stick ~1h", SS[SS.hb == "~1h"], ["twin", "avg"]), ("Libre->stick ~2h", L, ["twin"])):
    print("==", name)
    for k in ("when", "level", "ins"):
        for v, g in D.groupby(k):
            print(f"  {v:20s} n={len(g):3d} " + "  ".join(f"{c}: rmse {r(g[c]-g.true)} within30 {w30(g[c]-g.true)}% bias {round((g[c]-g.true).mean())}" for c in cols))
# biggest misses twin v3 stick 2h
g = SS[SS.hb == "~2h"].assign(err=lambda x: x.twin - x.true).sort_values("err", key=abs, ascending=False).head(8)
print(g[["t", "gap", "g0", "true", "twin", "v14", "v10live", "when", "level", "tl"]].round(0).to_string())
