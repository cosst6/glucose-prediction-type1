import numpy as np, pandas as pd, pickle
def clarke(ref, pred):
    z = []
    for r, p in zip(ref, pred):
        if (r <= 70 and p <= 70) or (p <= 1.2 * r and p >= 0.8 * r): z.append("A")
        elif (r >= 180 and p <= 70) or (r <= 70 and p >= 180): z.append("E")
        elif (70 <= r <= 290 and p >= r + 110) or (130 <= r <= 180 and p <= (7 / 5) * r - 182): z.append("C")
        elif (r >= 240 and 70 <= p <= 180) or (r <= 175 / 3 and 70 <= p <= 180) or (175 / 3 <= r <= 70 and p >= (6 / 5) * r): z.append("D")
        else: z.append("B")
    return np.array(z)
def stats(true, pred):
    e = pred - true; a = np.abs(e); z = clarke(true, pred)
    iso = np.where(true < 100, a <= 15, a <= 0.15 * true)
    return {"n": len(e), "RMSE": np.sqrt(np.mean(e**2)), "MAE": a.mean(), "median": np.median(a), "p90": np.quantile(a, .9),
            "<=10": 100*(a<=10).mean(), "<=20": 100*(a<=20).mean(), "<=30": 100*(a<=30).mean(), "<=50": 100*(a<=50).mean(), ">80": 100*(a>80).mean(),
            "ISO15%": 100*iso.mean(), "ClarkeA": 100*(z=="A").mean(), "ClarkeA+B": 100*np.isin(z,["A","B"]).mean(), "ClarkeD+E": 100*np.isin(z,["D","E"]).mean(),
            "bias": e.mean()}
SS = pd.read_pickle("../fuse_ss.pkl"); SS["avg"] = (SS.twin + SS.v14 + SS.v10live) / 3
rows = []
for h, g in SS.groupby("hb"):
    for c, name in (("twin", "Twin v3"), ("v14", "v14"), ("v10live", "v10"), ("avg", "Average of 3"), ("g0", "No change")):
        rows.append({"test": "stick->stick", "h": h, "model": name, **stats(g.true.values, g[c].values)})
L = pd.read_pickle("../wf3_a_wcl1.pkl"); L = L[L.tag == "Libre->stick"]
L["hb2"] = np.where(L.gap <= 90, "~1h", np.where(L.gap <= 150, "~2h", "3-4h"))
import blood as BL
raw = pd.read_pickle("../allsticks.pkl"); raw = raw[raw.tag == "Libre->stick"][["t", "tgt", "g0"]].rename(columns={"g0": "libre_raw"})
L = L.merge(raw, on=["t", "tgt"])
for h, g in L.groupby("hb2"):
    for c, name in (("twin", "Twin v3"), ("g0", "Corrected Libre, no change"), ("libre_raw", "Raw Libre, no change")):
        rows.append({"test": "Libre->stick", "h": h, "model": name, **stats(g.true.values, g[c].values)})
D = pd.DataFrame(rows).round(1); D.to_csv("../error_distribution.csv", index=False)
pd.set_option("display.width", 250); print(D.to_string(index=False))
