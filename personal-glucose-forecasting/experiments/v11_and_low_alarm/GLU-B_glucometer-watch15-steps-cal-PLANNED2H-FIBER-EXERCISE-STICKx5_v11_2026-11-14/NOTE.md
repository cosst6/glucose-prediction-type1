# GLU-B v11: v10 + fingerstick pairs weighted x5 (2026-11-14)

**Problem -> solution, why:** v10 learned only from Libre, which reads ~28% low and moves less than blood. -> Added every fingerstick-to-fingerstick pair (one reading -> the next reading 40-80 or 95-145 min later; 87 from Nov 6-12, 11 from Aug-Sep) to the training set at 5x weight, so the model learns real blood moves.

**Inputs:** same as v10 (one glucose reading, planned food + insulin for the next 2 h, fiber split, walk features, HR 15 min, raw steps, raw calories).
**Date:** trained 14 Nov 2026 on 27 Aug-17 Sep (Libre) + Nov 6-12 fingersticks.
**Outputs:** glucose in 60 and 120 min (blood scale), range.

**Fair test (Nov 7-11, each day predicted by a model trained only on earlier data; truth = glucometer):**
- v11 (sticks x5): 1h 21.8, 2h 35.7
- v10: 1h 22.0, 2h 37.3
- no change (persistence): 31.5 / 51.7
- sticks x20: 23.0 / 38.2 (worse); Libre rescaled to blood: no better.
- Day-bootstrap 90% range of v11 - v10: 1h -0.8 to +0.5, 2h -3.6 to +0.3 -> could be noise.
- v10 on all Nov 6-12 pairs vs blood: 29.2 / 37.0 (persistence 33.9 / 51.3).
- Both miss lows badly: 2h error on readings <70 is ~48.

**Libre check (37 paired readings):** Libre reads 29% MARD, ~43 low on average: ~20 low under 120, ~44 at 120-180, ~59 above 180. Fits "~28% low" (Libre ~ 0.72 x blood), not "10% low" or a fixed offset.

**Verdict:** maybe slightly better at 2h, not proven with 5 days. Keep v10 live; switch to v11 after ~2 more weeks of fingersticks if the gain holds.
**Pros:** learns from real blood readings; small 2h gain.
**Cons:** only ~100 fingerstick pairs; same weak spots as v10 (lows, late peaks).
**Files:** joblib + card + code (whyb.py = weighted hybrid, pairs.py = fingerstick pairs, exp11.py = test), log_nov6-12.csv (your text files transcribed), walkforward_nov_results.csv, libre_vs_fingerstick_pairs.csv.
