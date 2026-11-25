# GLU-T v1 digital twin (2026-11-23): research only, NOT live

**Problem -> solution, why:** v10/v14 are linear sums with fixed absorption shapes. They lag, and they treat every food the same way over time. -> A personal physiological "twin": blood glucose state + insulin chain + per-food-family gut absorption (learned speed and size), fat slows the meal, food eaten while low absorbs differently, insulin works harder at high glucose, counter-regulation below 85, a Libre-to-blood scale map, and a carried-forward trend correction. Fitted with physiology priors (MAP), then 32 day-bootstrap fits for uncertainty. Why: a few meaningful body numbers generalise better from 4 weeks of data than many free features.

**Inputs:** last glucose reading (Libre or fingerstick) and the one before it; NovoRapid doses (past 10 h + planned in window); meals in 5 food families + fat + FPU (past 12 h + planned in window, full 2 h horizon); HR 15-min average above rest, raw steps and raw calories (15-min totals) at forecast time; time of day.
**Date:** 23 Nov 2026. Final model trained on set 1 Libre + Libre Nov 18-19 + all fingerstick pairs sets 1-3 (sticks weight 5). Days with food log + watch only.
**Outputs:** glucose at any time up to 4 h (5-min steps), in blood mg/dL (Libre scale via a, b), 80% range (32 members + out-of-bag error model), what-if curves for two plans.

**Fair walk-forward (each day predicted by a model trained only on earlier days, real meals/insulin in window known). 2 h RMSE:**
| Test | Twin | Twin+v14 blend | v14 | v10 live | no change |
|---|---|---|---|---|---|
| Set 1 Libre (14 days) | 26.0 | **25.4** | 27.1 | 27.6* | 39.3 |
| Libre Nov 18-19 | 36.9 | **35.9** | 38.8 | 43.9 | 51.2 |
| Set 2 sticks (milk week) | 42.5 | 39.3 | 38.8 | **37.0** | 51.3 |
| Set 3 sticks (new foods) | 45.4 | 42.6 | **41.3** | 48.7 | 60.8 |
1 h: set1 19.4 (blend 19.1) vs v14 20.2/v10 20.2; Nov Libre 27.2 vs 28.4/31.0; set2 33.2 vs 28.6/29.2; set3 45.0 vs 50.9/49.4. *same-data v10.
80% range coverage: set1 84%, Nov Libre 73-76%, sticks 68-91%.
**Tried, no gain:** GBM trees on twin residuals (helps set 2 only), recent-error feedback (3 h/6 h), insulin sensitivity by time of day (worse), Libre lag 5-15 min, trend window 15-60 min, prior strength x0.3-x3, loss type (all within 0.3).
**Learned (median, 32 members, blood units):** insulin peak 69 min, 28 mg/dL per unit; per g carb: fast 2.8, milk 4.1, starch 2.9, legume 4.2; absorption peak: fast 14 min, milk 19, starch 20, legume 54; FPU 23 per FPU, peak 101 min; 30 g fat ~doubles absorption time.
**Pros:** best on all Libre tests; interpretable body numbers; full curve + what-if + honest ranges; blend with v14 is best overall on Libre. **Cons:** worse on the milk week; Libre-to-blood scale weakly identified (a ~1.0 on all data vs 0.72 level ratio); stick tests small; drift term (-10/h) is a data average, not physiology. Not dosing advice.
Files: twin.py (model), tdata.py (data), tfit.py (fit), wf.py (fair test), final_twin.py (final + ensemble), sim.js + what_if_simulator.html (browser twin). Page: https://claude.ai/artifact/1wdkDZ6SKuEMrKSx9ATAnU
