# GLU-B v15: glycogen tank (2026-11-22) — research only, NOT live

**Problem -> solution, why:** Models don't know how "full" the body's sugar stores are, which might explain hard bounces after lows and 4-5am rises. -> Added an estimated glycogen tank (liver 100 g, muscle 400 g) on top of v14 food-family. Why: a full liver could release more sugar after a low or overnight.

**Tank:** fills from available carbs (40% liver, 30% muscle, rest burned) + protein (0.5 g glucose per g, to liver). Drains: liver 4.5 g/h when fasting, 1 g/h when eating; muscle = watch active calories x 15% / 4. Past data only. Features: liver/muscle grams and fullness, grams drained in 6 h, hours fasting.
**Inputs:** v14 inputs (one glucose reading, planned food + insulin for next 2 h, family carbs, fat/protein, fiber, walks, HR 15-min avg, raw steps, raw calories) + tank features.
**Date:** 22 Nov 2026, trained on set 1 Libre + Libre Nov 18-19 + all fingerstick pairs sets 1-3 (x5).
**Outputs:** glucose in 60 and 120 min, with a range.

**Fair test (each day predicted by a model trained only on earlier days, meals known). 2 h RMSE mg/dL:**
| Test | v10 live | v14 | v10 + tank | v14 + tank (this) |
|---|---|---|---|---|
| Set 1 Libre | 27.6* | 27.1 | 28.4 | 27.7 |
| Set 2 glucometer | 37.0 | 38.8 | 36.9 | 38.8 |
| Set 3 glucometer | 48.7 | 41.3 | 50.2 | 43.2 |
| Libre Nov 18-19 | 43.9 | 38.8 | 42.9 | 39.9 |
*same-data v10. Also tried tank in the body model: worse on most tests.
**After lows (start <80):** set 1 Libre 26.9-27.7 vs v14 28.7 (small help); Nov 29.5 vs 27.4 (worse).
**2-4am starts (4-5am rises):** worse (24.7 vs 22.1).
**LOW-1 + tank:** fewer false alarms but fewer lows caught; AUC still <0.5 on set 3.

**Verdict:** no gain. Tank numbers are guesses (glycogen cannot be measured), and 4 weeks of data is too little for the model to learn from them. v10 stays live.
**Pros:** physiology-based; small help after lows on set 1. **Cons:** worse at night and on most tests; many guessed numbers.
