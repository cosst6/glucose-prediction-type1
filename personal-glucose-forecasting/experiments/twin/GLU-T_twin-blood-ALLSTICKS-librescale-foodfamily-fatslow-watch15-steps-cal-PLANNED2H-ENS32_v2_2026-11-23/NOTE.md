# GLU-T v2 twin trained on EVERY data point (2026-11-23): research only, NOT live

**Problem -> solution, why:** v1 learned mostly from Libre (≥30% MARD) and only used fingersticks that sat ~1 h or ~2 h after another stick. -> v2 uses every Libre window plus every fingerstick as truth: every stick -> every later stick 30 min-4 h, and every Libre reading -> every later stick 30 min-4 h (166 sticks, 125 usable as targets). Why: sticks are the accurate truth, and the Libre->stick pairs teach the twin how your Libre maps to blood.
**Inputs:** same as v1 (last reading + previous, planned meals/insulin for the full window, HR 15-min avg, raw steps, raw calories). **Date:** 23 Nov 2026, all sets 1-3. **Outputs:** blood glucose path to 4 h with 80% range; what-if.
**Fair walk-forward on every fingerstick (each day trained only on data whose outcome was before that day). RMSE vs blood:**
| Forecast | checks | v2 (all sticks) | v1 training | Libre as is |
|---|---|---|---|---|
| Libre now -> stick ~1 h | 232 | 41.3 | 47.4 | 64.7 |
| Libre now -> stick ~2 h | 205 | 46.5 | 53.1 | 75.5 |
| Libre now -> stick 3-4 h | 302 | 48.0 | 55.2 | 79.6 |
Stick -> stick, same pairs for all models (v10/v14 also trained with every stick pair): ~1 h (97) twin 36.3, blend 35.2, v14 36.2, v10 live 35.3, no-change 42.1. ~2 h (85) twin 54.4, blend 49.7, v14 47.8, v10 live 48.2, no-change 57.6 (set 1 has one huge miss; set 2 46.8 vs v14 43.9; set 3 49.3 vs 47.4).
**Weights tried:** sticks x2/x5/x20, Libre x0.3/x1, Libre->stick x0.2/0.5/2, own map for new sensor, recency half-life 7/14 days. Best: sticks x5, Libre x1, Libre->stick x0.1 each (~x1.4 total per stick). Heavier sticks or less Libre = worse (only 125 stick targets).
**Pros:** much better at turning a Libre reading into a blood forecast (2 h 46.5 vs 53.1 before, 75.5 raw Libre). **Cons:** stick-to-stick at 2 h still behind v10/v14; few sticks; not dosing advice.
