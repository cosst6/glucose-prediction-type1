import sys, os, numpy as np, pandas as pd, torch, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__)); from gru import *
G = pd.read_pickle(sys.argv[1]); out = sys.argv[2]
RT = pd.read_csv(sys.argv[3], index_col=0, parse_dates=True); RT = RT[RT.test.str.contains("glucometer")]
res = []
for d in sorted(RT.index.normalize().unique()):
    cut = d - pd.Timedelta("150min")
    a, b, n = calib(G, G.index < cut)                 # Libre->blood relation from training days only
    gv, real, syn = glucose_series(G, a, b)
    A = anchors(G, gv, real, syn); A["d"] = A.t.dt.normalize()
    tr = A[(A.t < cut) & (A.t + pd.to_timedelta(A.gap, "m") < d)]
    te = A[(A.d == d) & (A.real == 1)]
    X, St, tr = tensors(G, gv, real, syn, tr)
    g0 = gv[tr.i.values]; y = torch.tensor(((tr.y.values - g0) / 50).astype(np.float32)); w = np.where(tr.real == 1, 5.0, 1.0)
    ms = fit(X, St, y, w)
    Xt, Stt, te = tensors(G, gv, real, syn, te)
    pr = gv[te.i.values] + 50 * predict(ms, Xt, Stt)
    te = te.assign(gru=pr, g0=gv[te.i.values]); res.append(te)
    print(d.date(), f"cal blood={a:.2f}*libre{b:+.0f} (n={n})", "train", len(tr), "real", int(tr.real.sum()), "test", len(te), flush=True)
R = pd.concat(res)
M = RT.reset_index().rename(columns={"index": "t"}).merge(R[["t", "h", "gru", "y", "gap"]], on=["t", "h"], how="left")
M.to_csv(out, index=False)
print("matched", M.gru.notna().sum(), "of", len(M), "true diff max", (M.y - M.true).abs().max())
