exec(open('exp.py').read().split('rows = []')[0])
from glucoforecast import load_cgm
C0 = load_cgm("libre.csv"); cg = pd.concat([s.g.rolling(3, center=True, min_periods=1).mean() for _, s in C0.groupby("seg")]).dropna() / 0.72
def at(t, m):  # Libre value (blood scale) m minutes after meal
    x = cg[t + pd.Timedelta(minutes=m - 8):t + pd.Timedelta(minutes=m + 8)]; return x.mean() if len(x) else np.nan
T["d45"] = [at(t, 45) - p if s == "Libre" else np.nan for t, p, s in zip(T.t, T.pre, T.src)]
B = ["uncov", "iob", "steps60", "hr15"]; S1 = B + ["prec", "low"]; S2 = S1 + ["d45"]
P0, P1 = lodo(B, 1.0, 1), lodo(S1, 1.0, 1)
# stage 2 (Libre only: no blood readings at +45 min exist yet)
Lm = (T.src == "Libre") & T.d45.notna(); P2 = np.full((len(T), 2), np.nan)
for d in T[Lm].day.unique():
    tr, te = Lm & (T.day != d), Lm & (T.day == d)
    for j, y in enumerate(("y30", "y60")):
        P2[te.values, j] = LR(1.0).fit(T.loc[tr, S2], T.loc[tr, y]).predict_proba(T.loc[te, S2])[:, 1]
P2[:, 1] = np.minimum(P2[:, 1], P2[:, 0])
band = lambda P: (P[:, 0] > .5).astype(int) + (P[:, 1] > .5).astype(int)
T["b0"], T["b1"] = band(P0), band(P1); T["p0_30"] = P0[:, 0]; T["p0_60"] = P0[:, 1]
T["b2"] = np.where(Lm, band(np.nan_to_num(P2)), -1)
T["rule"] = (T.avail > 8 * T.dose).astype(int)
# smart: no stick when model 0 is sure (p30 < lo or p30 > hi and band clear); else ask stick and use model 1
def smart(lo, hi):
    sure = (T.p0_30 < lo) | ((T.p0_30 > hi) & ((T.p0_60 < lo) | (T.p0_60 > hi)))
    return np.where(sure, T.b0, T.b1), (~sure).astype(int)
days = T.groupby("src").day.nunique()
def summ(name, pb, src, ask=None, m=None):
    m = (T.src == src).values if m is None else m; t = T.band.values; pb = np.asarray(pb)
    s, a = t >= 1, pb >= 1
    r = dict(model=name, data=src, meals=int(m.sum()), band_right=f"{round(100*(pb==t)[m].mean())}%",
             spikes_caught=f"{(a&s&m).sum()}/{(s&m).sum()}", **{"+30-60 right": f"{((pb==1)&(t==1)&m).sum()}/{((t==1)&m).sum()}", ">=60 right": f"{((pb==2)&(t==2)&m).sum()}/{((t==2)&m).sum()}"},
             false_alarms=f"{(a&~s&m).sum()}/{(~s&m).sum()}", sticks_per_day=round((np.asarray(ask)[m].sum() if ask is not None else 0) / days[src], 1))
    return r
rows = []
for src in ("Glucometer", "Libre"):
    n = T.src == src
    rows.append(summ("rule: <1u per 8g carbs (yes/no only)", T.rule, src))
    rows.append(summ("0 sticks", T.b0, src))
    rows.append(summ("pre-meal stick every meal", T.b1, src, ask=np.ones(len(T))))
    for lo, hi in ((0.2, 0.8), (0.3, 0.7)):
        b, ask = smart(lo, hi); rows.append(summ(f"smart: stick only if unsure ({lo}-{hi})", b, src, ask=ask))
rows.append(summ("pre stick + 2nd stick 45 min after meal", T.b2, "Libre", ask=2 * np.ones(len(T)), m=Lm.values))
R = pd.DataFrame(rows); pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30); print(R.to_string(index=False))
R.to_csv("spike_alarm_results.csv", index=False)
# warning time: alarm at meal log for caught spikes
for col in ("b0", "b1"):
    c = (T[col] >= 1) & (T.band >= 1) & (T.src == "Libre")
    print(col, "Libre caught: min to +30 median", T[c].t30_min.median(), "IQR", T[c].t30_min.quantile([.25, .75]).tolist(), "to peak median", T[c].peak_min.median())
c = (T.b2 >= 1) & (T.band >= 1) & Lm; print("stage2 warn before peak (peak-45) median", (T[c].peak_min - 45).median(), "share peak already passed", ((T[c].peak_min <= 45)).mean())
G = T[(T.src == "Glucometer") & (T.band >= 1)]; print("Glucometer spikes: peak stick min after meal median", G.peak_min.median())
T.to_pickle("meals_scored.pkl")
from sklearn.metrics import roc_auc_score as A
for src in ("Libre", "Glucometer"):
    d = T[T.src == src]; i = d.index
    print(src, "AUC spike>=30: uncov alone", round(A(d.y30, d.uncov), 2), "model0", round(A(d.y30, P0[i, 0]), 2), "model1", round(A(d.y30, P1[i, 0]), 2),
          "| >=60: uncov", round(A(d.y60, d.uncov), 2), "m0", round(A(d.y60, P0[i, 1]), 2), "m1", round(A(d.y60, P1[i, 1]), 2))
    if src == "Libre":
        k = Lm[Lm].index; print("  stage2 AUC30", round(A(T.y30[k], P2[k, 0]), 2), "AUC60", round(A(T.y60[k], P2[k, 1]), 2))
    # catches at the rule's false-alarm count
    fa = ((d.rule == 1) & (d.y30 == 0)).sum()
    for nm, p in (("model1", P1[i, 0]), ("uncov", d.uncov.values)):
        thr = np.sort(p[d.y30.values == 0])[::-1][fa] if fa < (d.y30 == 0).sum() else -1e9
        print("  at rule FA", fa, nm, "caught", int(((p > thr) & (d.y30.values == 1)).sum()), "/", int(d.y30.sum()))
# smart stage 2: 45-min stick only when step 1 is borderline (P within 0.4-0.6 for >=30 or >=60)
ask2 = ((abs(P1[:, 0] - .5) < .1) | (abs(P1[:, 1] - .5) < .1)) & Lm.values
b2s = np.where(ask2, T.b2, T.b1)
print(pd.DataFrame([summ("pre stick + 45-min stick only if borderline", b2s, "Libre", ask=1 + ask2, m=Lm.values)]).to_string(index=False))
R = pd.concat([R, pd.DataFrame([summ("pre stick + 45-min stick only if borderline", b2s, "Libre", ask=1 + ask2, m=Lm.values)])]); R.to_csv("spike_alarm_results.csv", index=False)
T["b2s"] = b2s; T.to_pickle("meals_scored.pkl")
