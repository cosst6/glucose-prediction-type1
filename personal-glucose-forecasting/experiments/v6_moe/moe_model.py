"""v6 = v5 hybrid + manager/spike/stable experts (mixture of experts).
Manager: logistic gate -> chance the next h min are 'fast' (>=40 in 1h, >=60 in 2h).
Spike expert: GBM on body-model residual, weight 1 on fast rows, 0.2 on calm. Stable expert: the reverse.
Forecast = g0 + body + p*spike + (1-p)*stable."""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler
from glucoforecast5 import BodyModel5, feature_list5, EV
from glucoforecast4 import gbm
THR = {60: 40, 120: 60}
def gate_cols(model, h):
    c = [f"{e}_{k}" for e in EV for k in ("ob", "act30")] + [f"{e}_next{h}" for e in EV] + ["g0", "min_since_meal", "uncovered_carbs", "body"]
    return c + (["d15", "d30", "d60", "accel"] if model == "A" else [])
class MoE5:
    def fit(self, X, model, horizons=(60, 120)):
        self.model, self.horizons = model, list(horizons); self.body = BodyModel5().fit(X, self.horizons); self.m = {}
        for h in self.horizons:
            cols = feature_list5(model, h); y = f"y{h}"; R = X.dropna(subset=cols + [y])
            F = R[cols].assign(body=self.body.predict(R, h)); r = R[y] - F.body; s = (R[y].abs() >= THR[h]).astype(float)
            gc = gate_cols(model, h)
            self.m[h] = {"cols": cols, "gcols": gc,
                "gate": make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)).fit(F[gc], s),
                "spike": gbm().fit(F, r, sample_weight=0.2 + 0.8 * s), "stable": gbm().fit(F, r, sample_weight=1 - 0.8 * s)}
        return self
    def predict(self, X, h):
        m = self.m[h]; F = X[m["cols"]].assign(body=self.body.predict(X, h)); p = m["gate"].predict_proba(F[m["gcols"]])[:, 1]
        sp, st = m["spike"].predict(F), m["stable"].predict(F)
        return pd.DataFrame({"pred": X.g0 + F.body + p * sp + (1 - p) * st, "spike_chance": p,
                             "spike_expert": X.g0 + F.body + sp, "stable_expert": X.g0 + F.body + st}, index=X.index)
