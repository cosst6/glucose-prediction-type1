"""Zero-shot pretrained forecasters on your Libre history: Chronos-2 (with meals/insulin as known future covariates) and Chronos-Bolt (glucose only).
One forecast per Libre reading time (15-min grid), 16 steps = 4 h. Uses only data up to the forecast time (plus planned meals/insulin)."""
import sys, numpy as np, pandas as pd, pickle, torch, time
from chronos import BaseChronosPipeline, Chronos2Pipeline
from fam11 import load_cgm
torch.set_num_threads(4)
TL, TABS = pickle.load(open("../twin_data.pkl", "rb"))
cgm = load_cgm("../D/cgm_export_21-11-2026.csv").g
H, CTX = 16, 192
def grid_cov(tl, idx):
    """15-min covariate bins aligned to idx (bin = (t-15, t])."""
    t0 = tl.t0; edges = ((idx - t0).total_seconds() / 60).values
    def binsum(et, ev):
        out = np.zeros(len(idx))
        if len(et): 
            k = np.searchsorted(edges, et, side="left")
            ok = (k < len(idx)) & (et > edges[np.clip(k, 0, len(idx) - 1)] - 15)
            np.add.at(out, k[ok], ev[ok])
        return out
    cov = {"insulin": binsum(tl.bt, tl.bu)}
    for f, name in enumerate(["fast", "milk", "starch", "legume", "veg"]): cov[f"carb_{name}"] = binsum(tl.mt, tl.mc[:, f])
    cov["fpu"] = binsum(tl.mt, tl.mfpu)
    T5 = pd.Series(np.arange(tl.n), index=tl.T)
    j = np.clip(np.searchsorted(tl.T.values, idx.values), 0, tl.n - 1)
    past = {"hr_up": tl.hrup[j], "steps": tl.st15[j], "cal": tl.cal15[j]}
    return cov, past
def run(name, tl_key, start, end):
    tl = TL[tl_key]
    idx = pd.date_range(pd.Timestamp(start).floor("15min") - pd.Timedelta("48h"), pd.Timestamp(end) + pd.Timedelta("5h"), freq="15min")
    g = cgm.reindex(idx).values.astype(float)
    if tl_key == "set3": g[idx < "2026-11-18 13:00"] = np.nan
    cov, past = grid_cov(tl, idx)
    anchors = TABS[name].t.drop_duplicates().sort_values()
    c2 = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cpu", torch_dtype=torch.float32)
    cb = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-base", device_map="cpu", torch_dtype=torch.float32)
    pos = {t: i for i, t in enumerate(idx)}
    rows = []; inp2, inpb, keep = [], [], []
    for t in anchors:
        i = pos.get(t)
        if i is None or np.isnan(g[i]): continue
        lo = max(0, i - CTX + 1); y = g[lo:i + 1]
        if np.isfinite(y).sum() < 8: continue
        pc = {k: v[lo:i + 1] for k, v in {**cov, **past}.items()}
        fc = {k: v[i + 1:i + 1 + H] for k, v in cov.items()}
        inp2.append({"target": y, "past_covariates": pc, "future_covariates": fc}); inpb.append(torch.tensor(y, dtype=torch.float32)); keep.append(t)
    t1 = time.time()
    P2 = c2.predict(inp2, prediction_length=H, batch_size=64)   # list of [1, n_q, H]
    q2 = np.stack([p[0].numpy() for p in P2])                   # (N, n_q, H)
    qb, mb = cb.predict_quantiles(inpb, prediction_length=H, quantile_levels=[0.1, 0.5, 0.9])
    print(name, len(keep), "anchors", round(time.time() - t1), "s", q2.shape, flush=True)
    nq = q2.shape[1]; mid = nq // 2
    out = pd.DataFrame({"t": keep})
    for s in range(H):
        out[f"c2_{(s+1)*15}"] = q2[:, mid, s]; out[f"c2lo_{(s+1)*15}"] = q2[:, 0 if nq == 3 else 1, s]; out[f"c2hi_{(s+1)*15}"] = q2[:, -1 if nq == 3 else -2, s]
        out[f"cb_{(s+1)*15}"] = qb[:, s, 1].numpy()
    out["c2_nq"] = nq; out["tl"] = tl_key
    return out
R = pd.concat([run("set1 Libre", "set1", "2026-08-27", "2026-09-17"), run("Nov18-19 Libre", "set3", "2026-11-18", "2026-11-20")])
R.to_pickle("../chronos_preds.pkl")
