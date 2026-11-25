# SPIKE-2: rise size + timing notification at meal log, no glucose after the meal (2026-11-14)

**Problem -> solution, why:** You want to know a rise is coming, roughly how big and when, without checking glucose during the rise. -> At meal log, one notification: "Rise of about +30 to 60 expected, starting around 2:30 PM, highest around 3:35 PM", plus "may start late, after 2 h" when the meal looks like a late riser. Why: size depends mostly on carbs not covered by insulin; timing on food speed (sugar share, starch, fiber, fat/protein, milk) and insulin.

**Inputs:** sugar, starch, fiber, fat/protein, milk flag, NovoRapid with the meal (planned), insulin still acting from earlier, HR 15-min mean, raw steps 60 min, night flag. Optional: pre-meal glucose only. No calories, no reading after the meal.
**Date:** 14 Nov 2026. Size: 120 meals (92 Libre /0.72 + 28 glucometer). Timing: 57 Libre spikes (fingerstick days are too sparse to time a rise).
**Outputs:** band (<+30 / +30-60 / +60+), rise start time (+30 above baseline), peak time, late-rise risk, message text.

**Fair test (leave one day out):**
| | No glucose | Pre-meal baseline |
|---|---|---|
| Band right, glucometer (28) | 36% | 54% |
| Spikes caught / false alarms, glucometer | 10/12, 8/16 | 8/12, 4/16 |
| Band right, Libre (92) | 50% | 55% |
| Spikes caught / false alarms, Libre | 46/57, 13/35 | 48/57, 15/35 |
| Rise start error (median) | 13 min | 13 min |
| Same, just guessing "40 min after eating" | 11 min | 14 min |
| Peak time error (median) | 54 min | 49 min |
| Warning before the rise (median) | 38 min | 40 min |
| Late rises (start 2 h+) flagged | 3/5 (4/41 false) | 4/7 (4/41 false) |
| Start error on late rises | ~90 min | ~90 min |

**Plain verdict on timing:** most rises start 30-60 min after eating (median 40) and the model can't beat that typical guess. Peak time is off by ~50 min. It can flag about half of the late rises (paneer/egg/dal meals with insulin), but it can't say when they will start. Timing is not reliable yet.
**Pros:** no glucose needed after eating; warns ~40 min before a typical rise; "may rise late" flag catches about half the late ones.
**Cons:** size band right only about half the time; big rises often called smaller; 18 of 57 peaks were cut by the next meal or the 5 h limit, so peak times are rough; timing tested on Libre only; tiny data (19 Libre days). Research only: never use for dosing.
**Next:** a live sensor (or many more days) is the only way to learn timing: each late rise needs a full curve.
**Files:** joblib (none / pre modes: size, start, peak, late models), spike2_notify.py (notify(...)), spike2_results.csv, code spike2.py, timing.py, build.py.
