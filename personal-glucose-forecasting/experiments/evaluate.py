import sys, json, numpy as np, pandas as pd
sys.argv = [sys.argv[0], sys.argv[1]] + sys.argv[2:]
exec(open(__file__.replace("evaluate.py", "pipeline.py")).read())
USE_HR = "--hr" in sys.argv
hr = load_hr() if USE_HR else None

GF = ["g", "d15", "d30", "d60", "accel", "tod_s", "tod_c"]
EF = ["ins_act", "iob", "carb_rate", "cob", "ins_1h", "carbs_1h", "net_effect"]
HF = ["hr_15", "hr_60", "steps_60", "hr_cov"]

Xall = build(G, use_events=False)
X26 = build(G[G.index >= "2026-01-01"], use_events=True, hr=hr)
if hr is not None:
    X26[HF] = X26[HF].fillna(X26[HF].median()).fillna(0); X26["hr_cov"] = X26["hr_cov"].fillna(0)

ridge = lambda a=10: make_pipeline(StandardScaler(), Ridge(alpha=a))
hgb = lambda: HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=20, l2_regularization=1.0)

test_days = sorted({d for d in X26.index.normalize().unique() if d >= pd.Timestamp("2026-08-30")})
res = {h: [] for h in (30, 60)}
for H in (30, 60):
    y = f"y{H}"
    for day in test_days:
        te = X26[(X26.index.normalize() == day)].dropna(subset=GF + [y])
        tr26 = X26[X26.index < day - pd.Timedelta(minutes=H)].dropna(subset=GF + [y])  # no target leakage across boundary
        trall = Xall[Xall.index < day - pd.Timedelta(minutes=H)].dropna(subset=GF + [y])
        if len(te) == 0: continue
        out = pd.DataFrame({"true": te["g"] + te[y], "g": te["g"]}, index=te.index)
        out["persistence"] = te["g"]
        out["linear_trend"] = (te["g"] + te["d30"] * H / 30).clip(40, 500)
        m = ridge().fit(trall[GF], trall[y]); out["ridge_glucose_allyears"] = te["g"] + m.predict(te[GF])
        m26 = ridge().fit(tr26[GF], tr26[y]); out["ridge_glucose_2026"] = te["g"] + m26.predict(te[GF])
        feats = GF + EF + (HF if hr is not None else [])
        m2 = ridge(30).fit(tr26[feats], tr26[y]); out["ridge_glucose+insulin+carbs" + ("+hr" if hr is not None else "")] = te["g"] + m2.predict(te[feats])
        # two-stage: glucose-only model from all years, then residual model on 2026 events
        r = tr26[y] - m.predict(tr26[GF])
        m3 = ridge(30).fit(tr26[feats], r)
        out["two_stage" + ("+hr" if hr is not None else "")] = te["g"] + m.predict(te[GF]) + m3.predict(te[feats])
        m4 = hgb().fit(tr26[feats], tr26[y]); out["gbm_all_features"] = te["g"] + m4.predict(te[feats])
        res[H].append(out)

summary = {}
for H in (30, 60):
    R = pd.concat(res[H]); models = [c for c in R.columns if c not in ("true", "g")]
    rows = []
    for c in models:
        e = R[c].clip(40, 500) - R["true"]
        rows.append({"model": c, "RMSE": np.sqrt((e**2).mean()), "MAE": e.abs().mean(),
                     "within_20pct": ((e.abs() / R["true"]) <= 0.2).mean() * 100})
    T = pd.DataFrame(rows).set_index("model").round(1)
    T["skill_vs_persistence_%"] = (100 * (1 - T["RMSE"] / T.loc["persistence", "RMSE"])).round(1)
    print(f"\n=== horizon {H} min, n={len(R)} test points over {R.index.normalize().nunique()} days ===")
    print(T.sort_values("RMSE").to_string())
    summary[H] = T
    R.to_csv(f"preds_{H}{'_hr' if hr is not None else ''}.csv")
    # per-day
    pd_ = R.groupby(R.index.normalize()).apply(lambda d: pd.Series({c: np.sqrt(((d[c] - d.true) ** 2).mean()) for c in ["persistence", "linear_trend", "ridge_glucose_allyears", models[-2], models[-1]]}))
    print(pd_.round(1).to_string())
pd.concat(summary, names=["horizon"]).to_csv(f"summary{'_hr' if hr is not None else ''}.csv")
