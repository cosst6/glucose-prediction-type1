# GLU-B v9: v8 + fiber (2026-11-01)

**Problem -> solution, why:** v8 only subtracted fiber from carbs; it did not use the fact that fiber slows how fast carbs hit. -> For each meal, fiber share s = fiber / (available carbs + fiber) (max 0.6). That share of the fast parts (free sugar, glycemic load) is moved into the slow-carb channel; the latest meal's fiber grams and fiber share are also given to the model. -> High-fiber meals (dal, moong, veg, guava, roti) should rise slower and later.

**Inputs:** same as v8 (one glucometer reading, food + NovoRapid + Lantus, planned food/insulin for the next 2h, watch: HR 15-min avg, HR above rest 15 min, raw steps 15/60/180 min, raw calories 15/60/180 min) + fiber split + meal fiber g + meal fiber share.
**Date:** trained 1 Nov 2026 on 27 Aug-17 Sep 2026 (logged days with watch data only; Libre values stand in for the glucometer reading).
**Outputs:** glucose in 60 and 120 min, 80% range, spike risk.

**Test (walk-forward 30 Aug-17 Sep, meals + insulin in the window known, same rows), RMSE mg/dL 1h / 2h:**
- v9 (fiber): 20.7 / 28.5
- v8: 20.7 / 28.6
- Other fiber designs tried: fiber as its own channel 20.9 / 28.8; channel + split 20.7 / 28.6; half split 20.8 / 28.5; double split 20.8 / 28.2.
- After high-fiber meals (fiber share >= 0.2, 497 rows): 20.0 / 26.8 vs v8 20.2 / 26.8.

**Verdict:** no real change. Fiber is only ~4 g per meal here (low-carb diet; milk, eggs, paneer have none), so there is little for the model to learn. v8 and v9 are equal.
**Pros:** more realistic food handling; ready to help if you eat more high-fiber carb meals.
**Cons:** no measurable gain with 19 days of data; fiber values are from a small food table.
**Files:** joblib + code (glucoforecast9.py with FIB=split, EXF=0 defaults; train9.py; eval9.py). Project files glucose-model/v9_v10_fiber_exercise/<this name>/.
