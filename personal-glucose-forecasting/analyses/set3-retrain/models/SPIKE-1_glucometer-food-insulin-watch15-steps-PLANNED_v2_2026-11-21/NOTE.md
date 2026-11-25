# SPIKE-1 v2: spike alarm retrained with set 3 (2026-11-21)

**Problem -> solution, why:** SPIKE-1 v1 learned from 92 Libre + 28 glucometer meals. -> Added 13 set-3 meals (biscuits, rice, roti, rajma, chole, puri). Why: new foods. Inputs are the same: carbs not covered by insulin (carbs - 8 g x units), insulin on board, pre-meal stick, HR 15-min avg, raw steps 60 min.
**Date:** 21 Nov 2026, 133 meals. **Outputs:** no spike / +30-60 / +60 or more.

**Fair test on set 3 meals (13 meals, 9 spikes; v2 = leave one day out):**
| | band right | spikes caught | false alarms |
|---|---|---|---|
| v1 | 31% | 6/9 | 3/4 |
| v2 | 46% | 8/9 | 3/4 |
| your rule "<1 u per 8 g carbs" | - | 9/9 | 1/4 |

**Verdict:** v2 beats v1 on set 3, but your simple rule beat both this week. With only 13 meals this can flip.
**Pros:** better on new foods. **Cons:** tiny test, and the band (size) is still weak. Research only.
**Files:** joblib, build13.py (meal table), spike13.py, spike13_results.csv.
