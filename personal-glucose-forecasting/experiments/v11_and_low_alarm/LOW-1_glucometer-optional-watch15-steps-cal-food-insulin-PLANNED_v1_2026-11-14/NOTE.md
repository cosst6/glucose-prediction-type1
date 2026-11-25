# LOW-1: low (<70) alarm with planned meals, fingerstick optional (2026-11-14)

**Problem -> solution, why:** Forecasts miss lows (2h error ~48 on lows). -> A yes/no model: "will a glucometer reading be under 70 at time X?", asked up to 6 h ahead, using planned food + insulin up to time X. Two versions: no glucose at all, and one fingerstick (up to 8 h old). Trained on real blood lows only (Libre-era "lows" did not transfer: AUC ~0.5).

**Inputs:** insulin acting before X, insulin still on board, carbs and fat/protein acting before X, time until X, HR 15 min, raw steps 60 min, raw calories 60 min; one-stick version adds the last reading and its age.
**Date:** trained 14 Nov 2026 on Nov 6-12 glucometer log (76 readings, 10 lows under 70).
**Outputs:** probability of a low at time X; alarm if > 0.5.

**Fair test (leave one day out, each day predicted by a model trained on the other days):**
| Warning time | No glucose: lows caught / false alarms | 1 stick (<=8 h old) | 2-step (ask a stick only when risky) |
|---|---|---|---|
| 1 h | 5/10, 29% | 7/10, 35% | 4/10, 21%, stick asked 32% |
| 2 h | 8/10, 25% | 7/10, 37% | 7/10, 22%, stick asked 33% |
| 3 h | 6/10, 29% | 6/10, 37% | 4/10, 26% |
| 4 h | 5/9, 24% | 6/9, 35% | 3/9, 21% |
- Best warning time is ~2 h: no fingerstick needed, 8 of 10 lows caught, 1 in 4 non-low checks wrongly flagged.
- Old Libre-trained version: AUC 0.4-0.6 (useless). Simple rule "reading - 18 x units + 2 x carbs < 90": 1/10 at 2 h.
- Watch features add nothing yet (kept because of project rule).

**What drives it:** insulin in the window with little fat/protein (eggs/paneer) -> low; fat/protein protects. Matches earlier finding: lows follow corrections and big doses for low-carb food.
**Pros:** first model aimed at real lows; works without a fingerstick at 2 h; uses planned meals.
**Cons:** only 10 lows from 5 days, so results can swing a lot; carbs get a "more lows" weight, likely because fruit/milk days had lows (not causal); not tested at night after Lantus (few night readings). Research only: never use for dosing; confirm lows with the glucometer.
**Next:** 2-3 more weeks of logs with a reading before each dose and at 3 am a few nights would roughly triple the lows.
**Files:** joblib (dict: no_glucose / one_stick models + columns), card (coefficients), low_alarm_results.csv, code (lowfeat.py, lowexp.py, lowblood.py, lowfinal.py), log_nov6-12.csv.
