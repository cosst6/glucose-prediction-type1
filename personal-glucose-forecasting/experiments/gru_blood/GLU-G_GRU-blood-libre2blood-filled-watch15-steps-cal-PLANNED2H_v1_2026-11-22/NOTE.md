# GLU-G v1: GRU blood-glucose model (2026-11-22) — research only, NOT live

**Problem -> solution, why:** Old GRU had no glucose history ("blind encoder") and was trained on Libre, not blood. -> GRU that reads 6 h of glucose history + 2 h of planned food/insulin, trained to predict fingerstick (blood) glucose, with Libre turned into "synthetic blood" by a learned Libre->blood relation. Why: blood is the real target, and Libre days (Sept) have few sticks.

**Gap filling (causal, no future data):** glucose gaps = exponential (log-linear) interpolation between past points, forward-fill up to now, back-fill window start (linear fill tested: same result). Libre->blood: blood = a*Libre + b (robust fit on stick/Libre pairs from training days only; all data: 1.03*Libre + 29). Watch HR/steps/calories gaps: linear interpolation + forward/back fill inside the history window, with a "watch off" flag. Meals/insulin never invented.
**Inputs:** glucose history (sticks + synthetic blood, flags, time since last stick), NovoRapid + Lantus, carbs in 5 food families (fast, milk, starch, legume, veg), fat/protein, fiber, planned 2 h, HR 15-min avg, raw steps, raw calories, time of day.
**Date:** 22 Nov 2026. Trained on sets 1-3 (34 logged days): 3,919 samples, 146 real stick pairs (weight 5).
**Outputs:** blood glucose in 60 and 120 min.

**Fair test vs fingersticks (sets 2+3, 12 unseen days, walk-forward, meals known 2 h):**
| | 1 h MAE / RMSE / ISO% | 2 h MAE / RMSE / ISO% |
|---|---|---|
| GRU (this) | 28.3 / 36.5 / 42% | 37.1 / 44.7 / 30% |
| v10 live | 26.6 / 37.2 / 51% | 33.9 / 41.7 / 31% |
| v14 | 25.9 / 37.6 / 51% | 32.6 / 39.8 / 33% |
| no change | 30.4 / 42.1 / 49% | 45.6 / 55.0 / 21% |
ISO 15197 = within ±15 mg/dL below 100, ±15% above (meters need 95%).

**Verdict:** no gain. Ties v10/v14 on 1 h RMSE, worse at 2 h. v10 stays live.
**Pros:** predicts blood directly; uses glucose history; fills gaps without leakage; learns Libre->blood itself.
**Cons:** only 146 real stick pairs, too few for a neural net; worse 2 h than v14; needs PyTorch.
**Files:** .pt (3 models), card.json, gru.py (model + filling), build_gru.py (data grid), run_gru.py (test), final_gru.py (train), test_scores.csv, test_predictions.csv.
