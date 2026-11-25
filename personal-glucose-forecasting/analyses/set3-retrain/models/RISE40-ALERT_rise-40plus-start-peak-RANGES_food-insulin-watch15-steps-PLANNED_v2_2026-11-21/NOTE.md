# RISE40-ALERT v2: rise of 40+ alert retrained with set 3 (2026-11-21)

**Problem -> solution, why:** v1 had not seen set 3. -> Re-fitted the yes/no part with the set 3 meals added (133 meals). Why: new foods. The timing ranges are unchanged from v1, because set 3 has only sticks (no Libre curve to time a rise on).
**Inputs:** carbs not covered by insulin, insulin on board, raw steps 60 min, HR 15-min avg, optional pre-meal reading.
**Date:** 21 Nov 2026. **Outputs:** chance of a 40+ rise, alert over 50%, start / +40 / peak time ranges.

**Fair test on set 3 (13 meals, 7 rises of 40+):**
| | right | rises warned | false alarms |
|---|---|---|---|
| v1 with pre reading | 46% | 4/7 | 4/6 |
| v2 with pre reading | 54% | 5/7 | 4/6 |
| v1 no reading | 62% | 7/7 | 5/6 |
| v2 no reading | 69% | 7/7 | 4/6 |

**Verdict:** small gain. Use v2. Most set 3 warnings came true, but false alarms are high.
**Cons:** 13 test meals, and the timing ranges were not re-learned. Research only, never for dosing.
**Files:** joblib (same layout as v1, p40 models replaced), spike13.py, spike13_results.csv.
