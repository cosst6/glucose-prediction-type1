"""Fair walk-forward for the twin, on the exact benchmark points of v10/v14 (rt_set1.csv, rt_nov.csv)."""
import sys, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from tfit import *
HYB = int(os.environ.get("HYB", 0))
if HYB: from hyb import Resid
CUT = pd.Timedelta("150min")
ENS = int(os.environ.get("ENS", 0))
STICK_W = float(os.environ.get("STICK_W", 5)); LAM = float(os.environ.get("LAM", 1)); FIX = eval(os.environ.get("FIX", '{"si_s":0,"si_c":0}'))
def tpred(x, df):
    p = np.empty(len(df)); d = np.empty(len(df))
    for _, g in df.groupby("tl"):
        k = df.index.get_indexer(g.index); a, b = T.predict(x, make(g), True); p[k] = a; d[k] = b
    return p, d
def activity(df):
    """insulin units and carbs eaten from 2h before the forecast to its end (for the noise model)."""
    u = np.zeros(len(df)); c = np.zeros(len(df))
    for _, g in df.groupby("tl"):
        tl = TL[g.tl.iloc[0]]; k = df.index.get_indexer(g.index)
        t0 = ((pd.DatetimeIndex(g.t) - tl.t0).total_seconds() / 60).values; t1 = t0 + g.n.values * 5
        for i, (a, b) in enumerate(zip(t0 - 120, t1)):
            u[k[i]] = tl.bu[(tl.bt >= a) & (tl.bt <= b)].sum(); c[k[i]] = tl.mc[(tl.mt >= a) & (tl.mt <= b)].sum()
    return u, c
def noise_design(df):
    u, c = activity(df)
    return np.column_stack([np.ones(len(df)), u, c / 10, df.g0.values / 100])
def ensemble(x, parts, te):
    """day-block bootstrap of the MAP fit -> members; out-of-bag errors -> noise model (sigma by horizon & activity)."""
    rng = np.random.default_rng(len(te) + int(te.t.iloc[0].value % 1000))
    tr = pd.concat([q.assign(w=w) for q, w in parts if len(q)]).reset_index(drop=True)
    day = tr.t.dt.strftime("%Y-%m-%d") + tr.lib.astype(str); ud = day.unique()
    P = []; oob_r = []; oob_i = []
    for b in range(ENS):
        draw = rng.choice(ud, len(ud)); cnt = pd.Series(draw).value_counts()
        m = day.map(cnt).fillna(0).values
        sub = tr[m > 0]; ww = (sub.w * m[m > 0]).values
        xb, _ = fit([make(g, ww[sub.index.get_indexer(g.index) - 0] if False else ww[np.searchsorted(sub.index.values, g.index.values)]) for _, g in sub.groupby("tl")], x0=x, lam=LAM, fixed=FIX)
        P.append(tpred(xb, te)[0])
        oo = tr[m == 0]
        if len(oo): oob_r.append(tpred(xb, oo)[0] - oo.true.values); oob_i.append(oo.index.values)
    P = np.array(P)
    r = np.concatenate(oob_r); idx = np.concatenate(oob_i); O = tr.loc[idx].reset_index(drop=True)
    sig = np.zeros(len(te)); Dt = noise_design(te.reset_index(drop=True)); Do = noise_design(O)
    for h, lb in [(60, True), (60, False), (120, True), (120, False)]:
        k = ((O.h == h) & (O.lib == lb)).values; kt = ((te.h == h) & (te.lib == lb)).values
        if not kt.any(): continue
        if lb:   # Libre: plenty of rows -> sigma grows with insulin, carbs and level
            coef = np.linalg.lstsq(Do[k], np.abs(r[k]) * np.sqrt(np.pi / 2), rcond=None)[0]
            s_o = np.maximum(Do[k] @ coef, 8.0); zq = np.quantile(np.abs(r[k]) / s_o, 0.8)
            sig[kt] = np.maximum(Dt[kt] @ coef, 8.0) * zq / 1.2816
        else:    # sticks: few rows -> one 80% width per horizon (Libre-based, rescaled to blood, if too few)
            q = np.quantile(np.abs(r[k]), 0.8) if k.sum() >= 15 else np.quantile(np.abs(r[((O.h == h) & O.lib).values]), 0.8) / 0.72
            sig[kt] = q / 1.2816
    return P, sig
FB = int(os.environ.get("FB", 0)); FBH = float(os.environ.get("FBH", 6))
def recent_err(x, df_all, df):
    """mean twin error of forecasts made in the last FBH hours whose target time has already passed (per row of df)."""
    p, _ = tpred(x, df_all); e = p - df_all.true.values
    tt = df_all.t + pd.to_timedelta(df_all.n * 5, "min"); out = np.zeros(len(df))
    for i, (t, tl, lb) in enumerate(zip(df.t, df.tl, df.lib)):
        k = (df_all.tl.values == tl) & (df_all.lib.values == lb) & (tt.values <= np.datetime64(t)) & (df_all.t.values >= np.datetime64(t - pd.Timedelta(hours=FBH)))
        if k.any(): out[i] = e[k].mean()
    return out
ALLT = None
def finish(x, parts, te, name):
    te = te.reset_index(drop=True); p, d = tpred(x, te); te = te.assign(twin=p, test=name)
    if FB:
        allt = pd.concat([TABS[k] for k in TABS]).reset_index(drop=True)
        tr = pd.concat([q.assign(w=w) for q, w in parts if len(q)]).reset_index(drop=True)
        ptr, _ = tpred(x, tr); etr = recent_err(x, allt, tr); ete = recent_err(x, allt, te)
        te["fb"] = p.copy()
        for h in (60, 120):
            k = (tr.h == h).values; kt = (te.h == h).values
            a = np.sum(etr[k] * (tr.true.values[k] - ptr[k]) * tr.w.values[k]) / max(np.sum(etr[k] ** 2 * tr.w.values[k]), 1e-9)
            a = float(np.clip(a, -1.5, 0)); te.loc[kt, "fb"] = p[kt] + a * ete[kt]
            print("fb alpha", h, round(a, 2), flush=True)
    if ENS:
        P, sig = ensemble(x, parts, te)
        sd = sig
        te["ens"] = P.mean(0); te["lo80"] = te.ens - 1.2816 * sd; te["hi80"] = te.ens + 1.2816 * sd; te["sd_par"] = P.std(0)
    if HYB:
        tr = pd.concat([q.assign(w=w) for q, w in parts if len(q)]).reset_index(drop=True)
        pt, dt = tpred(x, tr)
        R = Resid().fit(tr, pt, dt, tr.w.values)
        te["hyb"] = R.predict(te, p, d)
    return te
def run(part, fitkw=None, verbose=False):
    fitkw = fitkw or {}
    out = []
    S1 = TABS["set1 Libre"]
    if part == "set1":
        days = sorted(S1.t.dt.normalize().unique())
        for d in days[6:]:
            tr = S1[S1.t < d - CUT]; te = S1[S1.t.dt.normalize() == d]
            x, _ = fit(groups(tr), lam=LAM, fixed=FIX, **fitkw)
            out.append(finish(x, [(tr, 1.0)], te, "set1 Libre"))
            if verbose: print(d.date(), flush=True)
    else:
        ST = pd.concat([TABS["set1 sticks"], TABS["set2 glucometer"], TABS["set3 glucometer"]])
        X3 = TABS["Nov18-19 Libre"]
        for name in ("set2 glucometer", "set3 glucometer", "Nov18-19 Libre"):
            TE = TABS[name]
            for d in sorted(TE.t.dt.normalize().unique()):
                parts = [(S1, 1.0), (ST[ST.t < d - CUT], STICK_W)]
                if name.endswith("Libre"):
                    q = X3[X3.t < d - CUT]
                    if len(q): parts.append((q, 1.0))
                tr = sum([groups(p, w) for p, w in parts if len(p)], [])
                x, _ = fit(tr, lam=LAM, fixed=FIX, **fitkw)
                te = TE[TE.t.dt.normalize() == d]
                out.append(finish(x, parts, te, name))
                if verbose: print(name, d.date(), flush=True)
    return pd.concat(out)
def score(R, bench):
    B = pd.read_csv(bench, index_col=0, parse_dates=True).reset_index().rename(columns={"index": "t", "t": "t"})
    B.columns = ["t"] + list(B.columns[1:])
    M = B.merge(R[["t", "h", "test", "twin"] + [c for c in ("hyb", "fb", "ens", "lo80", "hi80", "sd_par") if c in R]], on=["t", "h", "test"], how="left")
    rows = []
    for (t, h), g in M.groupby(["test", "h"], sort=False):
        g = g.dropna(subset=["twin"])
        rows.append(dict(test=t, h=h, n=len(g), twin=rmse(g.twin - g.true), hyb=rmse(g.hyb - g.true) if "hyb" in g else np.nan, fb=rmse(g.fb - g.true) if "fb" in g else np.nan, blend=rmse((g.twin + g.fam) / 2 - g.true), ens=rmse(g.ens - g.true) if "ens" in g else np.nan, cover80=round(100 * ((g.true >= g.lo80) & (g.true <= g.hi80)).mean()) if "lo80" in g else np.nan, width80=(g.hi80 - g.lo80).mean() if "lo80" in g else np.nan, v10=rmse(g.v10cfg - g.true), v14=rmse(g.fam - g.true),
                         v10live=rmse(g.v10live - g.true) if "v10live" in g else np.nan, nochange=rmse(g.g0 - g.true)))
    return pd.DataFrame(rows).round(1), M
if __name__ == "__main__":
    part = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else "base"
    R = run(part, eval(os.environ.get("FK", "{}")))
    S, M = score(R, f"../rt_{'set1' if part=='set1' else 'nov'}.csv")
    M.to_csv(f"../wf_{part}_{tag}.csv", index=False)
    print(tag, part); print(S.to_string(index=False))
