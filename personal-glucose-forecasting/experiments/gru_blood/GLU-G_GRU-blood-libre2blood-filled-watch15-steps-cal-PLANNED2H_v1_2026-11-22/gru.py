"""GLU-G GRU: blood-glucose forecaster. Causal gap filling per sample, Libre->blood relation learned on training days only."""
import os, sys, numpy as np, pandas as pd, torch, torch.nn as nn, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import HuberRegressor
torch.set_num_threads(int(os.environ.get("NT", 4)))
HIST, FUT, STEP = 72, 24, 5          # 6 h history, 2 h planned meals/insulin
NOM = {60: (40, 80), 120: (95, 145)}
FILL = os.environ.get("FILL", "exp")  # exp | lin  (glucose gap interpolation)
EV = ["ins_imp", "lan_imp", "ins_act", "ins_ob", "lan_h"] + [f"c_{f}_{k}" for f in ["fast", "milk", "starch", "legume", "veg"] for k in ("imp", "act")] + ["fpu_imp", "fpu_act", "fib_imp"]
EVS = np.array([2, 10, .05, 3, 12] + [20, .3] * 5 + [1, .02, 5], float)

def fill_window(v, log=False):
    """interpolate interior gaps, forward-fill tail, back-fill head; uses only the values given (all <= anchor)."""
    v = np.array(v, float); ok = ~np.isnan(v)
    if ok.sum() == 0: return v, ok
    x = np.arange(len(v)); y = np.log(v[ok]) if log else v[ok]
    f = np.interp(x, x[ok], y)        # np.interp = linear between points, flat (ffill/bfill) outside
    return (np.exp(f) if log else f), ok

def calib(G, train_mask):
    """blood = a*libre + b from stick/Libre pairs on training days (Libre within +-10 min)."""
    lib = G.libre.interpolate(limit=3, limit_area="inside")
    P = G[train_mask & G.stick.notna() & lib.notna()]
    x = lib[P.index].values.reshape(-1, 1); y = P.stick.values
    h = HuberRegressor().fit(x, y); return float(h.coef_[0]), float(h.intercept_), len(y)

def glucose_series(G, a, b):
    """real sticks where present, else synthetic blood from Libre (gaps <=15 min bridged) ; flags."""
    lib = G.libre.interpolate(limit=3, limit_area="inside")
    syn = a * lib + b
    g = G.stick.where(G.stick.notna(), syn)
    return g.values, G.stick.notna().values, syn.notna().values & G.stick.isna().values

def anchors(G, gv, real, syn):
    t = G.index; rows = []
    ok = G.day_ok.values; pos = {x: i for i, x in enumerate(t)}
    S = np.where(real)[0]
    for i in S:   # real stick pairs
        if not ok[i]: continue
        for h, (lo, hi) in NOM.items():
            c = [j for j in S if lo < (j - i) * STEP <= hi]
            if not c: continue
            j = min(c, key=lambda j: abs((j - i) * STEP - h))
            rows.append(dict(i=i, h=h, gap=(j - i) * STEP, y=gv[j], real=1))
    for i in np.where(syn | real)[0]:   # synthetic (Libre-derived) targets, every 15 min
        if not ok[i] or t[i].minute % 15: continue
        for h in (60, 120):
            j = i + h // STEP
            if j < len(gv) and syn[j] and (t[j] - t[i]) == pd.Timedelta(minutes=h):
                rows.append(dict(i=i, h=h, gap=h, y=gv[j], real=0))
    A = pd.DataFrame(rows); A["t"] = t[A.i.values]; return A

def make_x(G, gv, real, syn, i):
    lo = i - HIST + 1
    if lo < 0 or i + FUT >= len(G) or (G.index[i + FUT] - G.index[lo]) != pd.Timedelta(minutes=STEP * (HIST + FUT - 1)): return None
    seg = slice(lo, i + FUT + 1); n = HIST + FUT
    g = gv[lo:i + 1].copy(); g0 = g[-1]
    gf, gok = fill_window(g, log=(FILL == "exp"))
    since = np.zeros(HIST); last = -1e9
    for k in range(HIST):
        if real[lo + k]: last = k
        since[k] = min((k - last) * STEP / 60, 6)
    W = G[["hr", "steps", "cal"]].values[lo:i + 1]
    wcols = []
    for c, dflt, sc in ((0, 80, 20), (1, 0, 100), (2, 0, 5)):
        f, okc = fill_window(W[:, c]); f = np.nan_to_num(f, nan=dflt)
        if c == 0: f = pd.Series(f).rolling(3, min_periods=1).mean().values   # 15-min HR average
        wcols += [np.r_[(f - dflt) / sc, np.zeros(FUT)]]
    wmask = np.r_[~np.isnan(W[:, 0]), np.zeros(FUT)]
    E = G[EV].values[seg] / EVS
    tod = G.index[seg].hour.values + G.index[seg].minute.values / 60
    X = np.column_stack([np.r_[(gf - g0) / 50, np.zeros(FUT)], np.r_[real[lo:i + 1], np.zeros(FUT)], np.r_[syn[lo:i + 1], np.zeros(FUT)],
                         np.r_[gok, np.zeros(FUT)], np.r_[since, np.zeros(FUT)], *wcols, wmask, E,
                         np.sin(2 * np.pi * tod / 24), np.cos(2 * np.pi * tod / 24), np.r_[np.zeros(HIST), np.ones(FUT)]])
    return X.astype(np.float32), g0

class Net(nn.Module):
    def __init__(s, c, h=48):
        super().__init__(); s.gru = nn.GRU(c, h, batch_first=True); s.drop = nn.Dropout(.2)
        s.head = nn.Sequential(nn.Linear(h + 3, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(s, x, st):
        o, hN = s.gru(x); return s.head(torch.cat([s.drop(hN[-1]), st], 1)).squeeze(1)

def tensors(G, gv, real, syn, A):
    xs, st, keep = [], [], []
    for k, r in enumerate(A.itertuples()):
        z = make_x(G, gv, real, syn, r.i)
        if z is None: continue
        xs.append(z[0]); st.append([z[1] / 100 - 1.5, r.gap / 120, r.h / 120]); keep.append(k)
    return torch.tensor(np.stack(xs)), torch.tensor(np.array(st, np.float32)), A.iloc[keep].reset_index(drop=True)

def fit(X, St, y, w, seeds=3, epochs=int(os.environ.get("EP", 30))):
    ms = []
    for sd in range(seeds):
        torch.manual_seed(sd); np.random.seed(sd)
        m = Net(X.shape[2]); opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=1e-3)
        n = len(y); p = w / w.sum()
        for ep in range(epochs):
            m.train()
            for _ in range(max(n // 128, 1)):
                b = torch.tensor(np.random.choice(n, 128, p=p))   # weighted sampling: real sticks x5
                loss = nn.functional.huber_loss(m(X[b], St[b]), y[b], delta=1.0)
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        m.eval(); ms.append(m)
    return ms

def predict(ms, X, St):
    with torch.no_grad(): return np.mean([m(X, St).numpy() for m in ms], 0)
