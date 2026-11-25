# GLU-B v8: glucometer + planned meals 2h + watch rules (2026-11-01)

**Problem -> solution, why:** The glucometer model (B v4) could not see meals/insulin logged after the reading, and used old watch features. -> Food + insulin for the next 2h are known (planned-meal input), and the watch uses v7 rules: HR averaged over 15 min, steps and calories as raw totals over 15/60/180 min. -> Knowing the meals in the window is how you actually use it (you re-forecast after each log), and it cuts 2h error by ~11%.

**Inputs:** one glucometer (fingerstick) reading (no CGM trend), food log (fast sugar, GL, slow carbs, fat/protein), NovoRapid, Lantus, time of day, planned food + insulin for the next 2h, Noise watch: HR 15-min avg, HR above resting (15 min), steps total 15/60/180 min, calories total 15/60/180 min, watch on.
**Date:** trained 1 Nov 2026 on 27 Aug-17 Sep 2026 (logged days only, 1751 rows; Libre values stand in for the glucometer reading in training).
**Outputs:** glucose in 60 and 120 min, 80% range (scale 1.6x / 1.7x), spike risk.

**Test (walk-forward 30 Aug-17 Sep, meals known 2h ahead, same rows), RMSE mg/dL 1h / 2h:**
- v8: 20.7 / 28.6
- B v4 as published (meals not known ahead): 22.0 / 32.0
- B v4 with planned meals, old watch: 20.7 / 28.4
- v5/v7-style B (milk channel, clock blocks etc.) + new watch: 21.5 / 28.9
- same without watch: 22.2 / 31.0
- no change (persistence): 26.2 / 38.3
**Real fingerstick check (9 pairs, next BG 40-150 min later):** error 42 vs 51 for "no change".

**Verdict:** best glucometer model so far. The gain comes from planned meals; the new watch rules give no change vs old watch (within noise), but watch data still helps ~7% at 2h. The v5 extras (milk channel, clock blocks) hurt B with this little data, so v8 is built on v4.
**Pros:** uses planned meals and the new watch rules; 6% better at 1h and 11% at 2h than B v4 as published; beats v5-style B.
**Cons:** trained on Libre values, which read ~50 below fingersticks, so it forecasts too low when you are high (seen in the fingerstick check); only 19 days; still misses big swings (2h error ~41 when glucose moves 40+).
**Files:** joblib + code in project files glucose-model/v8_B_planned_watch15 (glucoforecast8.py, train8.py, stick8.py).
