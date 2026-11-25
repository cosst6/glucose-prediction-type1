# SPIKE-1: early spike alarm with rise size band (2026-11-14)

**Problem -> solution, why:** Forecasts miss spikes, and you only see them after they start. -> At each meal log the alarm says "no spike / +30-60 / +60 or more" for the meal window (until next meal, max 5 h), and asks for a fingerstick only when it needs one. Why: the rise is mostly set by carbs not covered by insulin (carbs - 8 g x units), and a reading 45 min in shows which band you are heading for.

**Bands (picked from 120 meals, blood scale):** under +30 = 43%, +30-60 = 22%, +60 or more = 35%. Over +120 was only 9 meals (8%), too few to learn, so it is inside "+60 or more".

**Inputs:** available carbs (sugar + starch; fiber left out), NovoRapid with the meal, earlier insulin still acting, HR 15-min mean, raw steps 60 min. No watch calories (user asked). Steps: pre-meal fingerstick (most meals already have one); 45-min fingerstick only when step 1 is borderline (~45% of meals).
**Date:** trained 14 Nov 2026 on 92 Libre meals (27 Aug-17 Sep, rises / 0.72 to blood scale) + 28 glucometer meals (6-12 Nov).
**Outputs:** band, P(rise >= 30), P(rise >= 60), and a message: "Check your blood glucose now" when needed.

**Fair test (leave one day out):**
| Setup | Data | Band right | Spikes caught | False alarms | Sticks/day |
|---|---|---|---|---|---|
| Your rule (<1u per 8 g) | Glucometer 28 meals | yes/no only | 8/12 | 7/16 | 0 |
| No stick | Glucometer | 36% | 10/12 | 8/16 | 0 |
| Pre-meal stick | Glucometer | 54% | 8/12 | 4/16 | ~4.7 |
| Pre-meal stick | Libre 92 meals | 55% | 48/57 | 15/35 | ~4.6 |
| + 45-min stick if borderline | Libre | 65% | 50/57 | 7/35 | ~6.6 |
- Warning time: alarm at meal log comes a median 40 min before the rise passes +30 and ~1 h 45 min before the peak. The 45-min check still comes ~1 h before the peak.
- Big spikes (+60 or more): pre-stick catches 21/33 on Libre but only 2/9 on glucometer. Weakest part.
- Food split tested: separate sugar, starch, fiber, fat/protein did not beat "carbs - 8 x units" with 120 meals (52% vs 55%). Fat/protein and fiber added nothing yet.
- Watch (HR, steps) adds almost nothing yet; kept because of the project rule.
- Later corrections inside the window are NOT given to the model: they are reactions to the spike and would leak the answer.

**Pros:** warns before the rise; uses the reading you already take before meals; asks for a 2nd stick only when unsure; tiny model (logistic).
**Cons:** only 28 blood-scored meals (6 days), so blood numbers can swing a lot; 45-min step tested only on Libre (no blood readings at 45 min yet); big-spike band weak on blood; thresholds for "borderline" picked on the same days (slightly optimistic). Research only: never use for dosing.
**Next:** 2-3 weeks of logs with a reading before each meal and one ~45 min after would let the 45-min step be tested on blood.
**Files:** joblib (step0/1/2 models), card (coefficients), spike_alarm.py (call alarm(...)), spike_alarm_results.csv, code (build.py, exp.py, exp2.py, final.py, train_final.py).
