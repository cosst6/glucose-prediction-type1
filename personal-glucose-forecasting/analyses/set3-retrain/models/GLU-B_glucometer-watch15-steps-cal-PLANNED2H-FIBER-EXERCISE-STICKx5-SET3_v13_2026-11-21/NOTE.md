# GLU-B v13: v11 + set 3 fingersticks (2026-11-21)

**Problem -> solution, why:** v10 and v11 never saw set 3 (Nov 15-19: rice, roti, biscuits, rajma, chole, puri, many lows). -> Retrained the v10 body model + trees on Libre set 1 plus all fingerstick pairs from sets 1-3 (146 pairs, weight 5). Why: more real blood moves and new foods.

**Inputs:** one glucose reading, planned food + insulin for the next 2 h, fiber split, walk features, HR 15-min avg, raw steps, raw calories. The food db has 8 new foods (rajma, chole, puri, Hide & Seek, Good Day, bhujiya, chocolate, walnut). Fix: no crash when no Lantus is logged.
**Date:** trained 21 Nov 2026.
**Outputs:** glucose in 60 and 120 min (blood scale), with a range.

**Fair test (set 3 days, each day predicted by a model trained only on earlier days; truth = glucometer, 48 pairs):**
| | 1 h | 2 h |
|---|---|---|
| v13 (new) | 53.2 | 49.0 |
| v11 | 52.9 | 48.1 |
| v10 (live) | 49.4 | 48.7 |
| no change | 54.6 | 60.8 |
Weight 10 instead of 5: 52.2 / 51.4 (no better).
- Set 3 was a hard week. Nov 18 alone (lows of 45-67, treated with chocolate + milk + bhujiya) gave 2 h errors near 58.
- Body numbers changed: slow carbs now count +0.58 per g (was 0, the old main flaw), insulin -22 per unit, fat/protein +15 per FPU.

**Verdict:** no gain over v10 on unseen days. Keep v10 live. Retry after 2-3 more weeks of sticks.
**Pros:** learns from more blood readings and new foods. The slow-carb channel now works.
**Cons:** not better yet. Only 48 test pairs. Lows and big treatments still miss by about 50.
**Files:** joblib + card + code (exp13.py = test, final13.py = train), res13.csv (each test forecast), log_set3.csv (set 3 transcribed).
