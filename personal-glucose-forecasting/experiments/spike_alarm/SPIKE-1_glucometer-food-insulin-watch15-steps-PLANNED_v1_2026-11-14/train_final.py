"""Train SPIKE-1 on all 120 meals and save. Run after build.py (meals.pkl)."""
import joblib, json
src = open('final.py').read(); exec(src[:src.index("P0, P1 = lodo")])
fit = lambda cols, rows: {y: LR(1.0).fit(T.loc[rows, cols], T.loc[rows, y]) for y in ("y30", "y60")}
allr = T.index == T.index; lib = (T.src == "Libre") & T.d45.notna()
M = {"step0_no_stick": {"cols": B, "m": fit(B, allr)}, "step1_pre_stick": {"cols": S1, "m": fit(S1, allr)},
     "step2_45min_stick": {"cols": S2, "m": fit(S2, lib)},
     "bands": ["no spike (<30)", "spike +30-60", "big spike >=60"], "ask_step2_if_within": 0.1, "carbs_per_unit": 8,
     "units": "mg/dL blood (glucometer) scale; Libre training rises divided by 0.72"}
name = "SPIKE-1_glucometer-food-insulin-watch15-steps-PLANNED_v1_2026-11-14"
import os; os.makedirs(f"out/{name}", exist_ok=True); joblib.dump(M, f"out/{name}/{name}.joblib")
card = {"name": name, "trained": "2026-11-14", "meals": int(len(T)), "bands": M["bands"]}
for k in ("step0_no_stick", "step1_pre_stick", "step2_45min_stick"):
    card[k] = {y: dict(zip(M[k]["cols"], [round(float(c), 3) for c in M[k]["m"][y][-1].coef_[0]]), intercept=round(float(M[k]["m"][y][-1].intercept_[0]), 3)) for y in ("y30", "y60")}
card["note"] = "Coefficients are on standardized inputs (StandardScaler). y30 = P(rise>=30), y60 = P(rise>=60)."
json.dump(card, open(f"out/{name}/{name}_card.json", "w"), indent=1); print(json.dumps(card, indent=1))
