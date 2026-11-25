# RISE40-ALERT: any rise of 40+ after a meal, with start and peak time RANGES (2026-11-15)

**Problem -> solution, why:** You want to know, at meal time and with no reading after the meal, whether glucose will climb 40 or more (fast or slow) and roughly when. -> One yes/no answer ("rise of 40+ likely") plus three time ranges: starts rising (+20), passes +40, highest. Ranges instead of one exact time because exact times were often wrong; each range held the real time 7 of 10 times on unseen days.

**Inputs:** sugar, starch, fiber, fat/protein, milk flag, NovoRapid with the meal (planned), insulin still acting, night flag, HR 15-min mean, raw steps 60 min; optional pre-meal reading. No calories. No reading after the meal.
**Date:** 15 Nov 2026. Yes/no: 120 meals (92 Libre /0.72 + 28 glucometer). Timing: 48 Libre rises of 40+.
**Outputs:** chance of 40+, alert (over 50%), three ranges (minutes after the meal -> clock times, rounded to 5 min).

**Fair test (leave one day out):**
| | Libre days (92 meals, 48 rises) | Fingerstick days (28 meals, 11 rises) |
|---|---|---|
| Rises warned (with pre-meal reading) | 40/48 | 5/11 |
| Warnings that were real | 40/51 | 5/9 |
| Meals called right | 79% | 64% |
| No reading at all: warned / false | 37/48, 11 false | 7/11, 8 false |
| Range | Width | Real time inside | Just using "your usual" range |
|---|---|---|---|
| Starts rising (+20) | ~30 min | 7/10 | ~30 min |
| Passes +40 | ~60-70 min | 7/10 | ~55 min |
| Highest | ~3 h | 7/10 | ~3 h |
- Plain truth: the ranges are about as wide as "your usual timing". Food and insulin barely move them. Start time is easy (most rises start 25-55 min after eating); the highest point is hard to time.
- **No insulin-timing window.** Asked for, not built: the timing is not better than your usual pattern and 1 in 5 warnings is false, so a "take insulin now" window could push extra insulin into a rise that isn't coming. Your past lows mostly followed extra/stacked doses. Talk pre-bolus timing over with your diabetes doctor.
**Pros:** no reading needed after eating; catches 8 of 10 rises of 40+ on Libre days; start range is short.
**Cons:** weak on fingerstick days (5 of 11 rises warned; only 6 days); peak range ~3 h; timing learned only from 19 Libre days. Research only: never use for dosing.
**Files:** joblib (none / pre modes), code ranges.py, export2.py, golden2.json; phone: Glucose Alarms app v1.1 (TimingModels.kt, TimingParams.kt).
