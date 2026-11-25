import numpy as np, pandas as pd, os, warnings; warnings.filterwarnings("ignore")
os.environ["CTX48"] = "1"
from sklearn.linear_model import Ridge; from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler
from glucoforecast5 import load_log5, restrict_to_logged_days, feature_list5, CTX_COLS
log = load_log5("log.csv"); X = restrict_to_logged_days(pd.read_pickle("X5C1_p0.pkl"), log)
def wf(cols, h):
    y = f"y{h}"; Z = X.dropna(subset=cols + [y]); e = []
    for day in sorted(d for d in Z.index.normalize().unique() if d >= pd.Timestamp("2026-08-30")):
        te = Z[Z.index.normalize() == day]; tr = Z[Z.index < day - pd.Timedelta(minutes=h)]
        if len(te): e.append(make_pipeline(StandardScaler(), Ridge(alpha=30)).fit(tr[cols], tr[y]).predict(te[cols]) - te[y].values)
    return np.sqrt(np.mean(np.concatenate(e)**2))
for h in (60, 120):
    base = [c for c in feature_list5("A", h) if c not in CTX_COLS and "move" not in c]
    print(f"\nh={h} base {wf(base, h):.2f}")
    for c in CTX_COLS + [f"yday_move{h}", f"d2_move{h}"]:
        print(f"  + {c:16s} {wf(base + [c], h):.2f}")
