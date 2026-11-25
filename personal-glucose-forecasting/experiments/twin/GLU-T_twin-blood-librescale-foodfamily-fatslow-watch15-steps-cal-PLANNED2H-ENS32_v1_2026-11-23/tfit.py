"""MAP fit of the twin (least squares + physiological priors)."""
import numpy as np, pandas as pd, pickle
from scipy.optimize import least_squares
import twin as T

TL, TABS = pickle.load(open(__import__("os").environ.get("TDATA", "../twin_data.pkl"), "rb"))

def make(df, w=1.0):
    """DataFrame of anchors (one timeline) -> twin.Anchors"""
    tl = TL[df.tl.iloc[0]]
    return T.Anchors(tl, df.t, df.n, df.g0, df.lib, df.prev_t, df.prev_g, df.prev_lib, df.true, np.full(len(df), w) if np.isscalar(w) else w, df.tag)

def groups(df, w=1.0):
    return [make(g, w) for _, g in df.groupby("tl")]

def fit(train, x0=None, lam=1.0, fixed=None, sig=25.0, loss="soft_l1", fscale=40.0, max_nfev=300, tol=1e-8):
    """train: list of Anchors. lam scales prior strength. fixed: dict name->value held constant."""
    fixed = fixed or {}
    free = [i for i, n in enumerate(T.NAMES) if n not in fixed]
    base = (T.X0 if x0 is None else x0).copy()
    for n, v in fixed.items(): base[T.IDX[n]] = v
    def full(z):
        x = base.copy(); x[free] = z; return x
    pm, ps, pl = T.PM[free], T.PS[free], T.PL[free]
    def res(z):
        x = full(z); r = []
        for A in train:
            r.append(np.sqrt(A.w) * (T.predict(x, A) - A.true))
        zz = np.where(pl, (np.log(np.maximum(z, 1e-9)) - np.log(pm)) / ps, (z - pm) / ps)
        r.append(lam * sig * zz)
        return np.concatenate(r)
    z0 = np.clip(base[free], T.LO[free] + 1e-9, T.HI[free] - 1e-9)
    sol = least_squares(res, z0, bounds=(T.LO[free], T.HI[free]), loss=loss, f_scale=fscale, max_nfev=max_nfev, diff_step=1e-3, ftol=tol, xtol=tol, gtol=tol)
    return full(sol.x), sol

rmse = lambda e: float(np.sqrt(np.mean(np.square(e))))
