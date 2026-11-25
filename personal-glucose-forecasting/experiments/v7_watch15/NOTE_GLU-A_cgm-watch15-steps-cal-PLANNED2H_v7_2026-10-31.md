# GLU-A v7: watch rules (HR 15 min, raw steps, raw calories), planned meals 2h (2026-10-31)

**Problem -> solution, why:** Watch data was used loosely (HR averaged over 30 min, no calories). -> HR averaged over 15 min, steps and calories as raw totals over the last 15/60/180 min, planned food + insulin known for the full 2h. -> A shorter HR window and calorie totals react faster to activity.

**Inputs:** Libre CGM (now + 15/30/60 min change), food log (fast sugar, milk sugar, GL, slow carbs, fat/protein, night food), NovoRapid, Lantus, clock blocks, planned food/insulin for the next 2h, Noise watch: HR 15-min avg, HR above resting (15 min), steps total 15/60/180 min, calories total 15/60/180 min, watch on.
**Date:** trained 31 Oct 2026 on 27 Aug-17 Sep 2026 (logged days only, 1751 rows).
**Outputs:** glucose in 60 and 120 min, 80% range (scale 1.75x / 2.0x), spike risk.
**Test (walk-forward 30 Aug-17 Sep, meals known 2h ahead):** 1h 21.2 (v5 21.3), 2h 28.8 (v5 29.0). No watch at all: 21.9 / 30.6. Adding 6-24h exercise totals: 21.8 / 30.1 (worse, not used).
**Verdict:** no real change vs v5 (about 1% better, within noise). The watch itself does help: removing it makes 2h about 6% worse.
**Pros:** follows the new watch rules; slightly better than v5 on 10-11 of 19 days; uses calories.
**Cons:** gain is tiny; still misses big swings (2h error 41.7 when glucose moves 40+); only 19 days of data; long-term exercise signal hurts for now.
