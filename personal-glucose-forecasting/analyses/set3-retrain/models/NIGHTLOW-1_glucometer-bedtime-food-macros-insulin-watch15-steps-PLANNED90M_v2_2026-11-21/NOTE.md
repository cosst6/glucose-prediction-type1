# NIGHTLOW-1 v2: overnight low model with blood nights added (2026-11-21)

**Problem -> solution, why:** v1 used 18 Libre nights only. -> Added 7 glucometer nights (5 from set 2, 2 from set 3). Both set-3 nights ended in a wake-up low (52 and 64). Why: real blood nights with real lows. Same sign-limited hourly model.
**Inputs (at bedtime):** glucometer reading, NovoRapid on board (incl. doses planned up to 90 min after bed), carbs + fat/protein in the last 4 h, bedtime milk, HR 15-min avg, raw steps 3 h.
**Date:** 21 Nov 2026. **Outputs:** hourly path for 9 h, low chance, wake-up value.

**Fair test on the 7 blood nights (each night left out):**
| | error vs stick | lows caught | false-alarm nights |
|---|---|---|---|
| v1 | 61.8 | 2/2 | 3/5 |
| v2 | 56.6 | 2/2 | 2/5 |
| "stays at bedtime value" | 89.8 | - | - |
- Both set-3 morning lows (Nov 18: 52, Nov 19: 64) were warned by v1 and v2 (chance 0.70-0.85).
- It still predicts impossible values after a big bedtime correction (Nov 6: -2).

**Verdict:** v2 is slightly better. Use v2. Still very few nights.
**Files:** json + joblib (coefs per hour), night13.py, nfinal.py, night13_results.csv, blood_nights.csv.
