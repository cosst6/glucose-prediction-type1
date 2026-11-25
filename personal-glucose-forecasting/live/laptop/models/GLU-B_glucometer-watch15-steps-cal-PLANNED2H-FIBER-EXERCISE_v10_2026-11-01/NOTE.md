# GLU-B v10: v8 + fiber + exercise (2026-11-01)

**Problem -> solution, why:** You see glucose rise ~40 min after starting evening walks; v8 only saw step/calorie totals, not "a walk started X min ago". -> Walks are detected from the watch (>=350 steps per 5 min for >=15 min). The model gets minutes since the walk started, in-walk flag, walk minutes and steps so far, HR above resting during the walk, and a rise bump (peak 45 min after start, gone by 2h) x walk intensity. Plus the v9 fiber split. -> Lets the model learn a rise (or drop) tied to the start of exercise.

**What the data shows (38 walks, 35 with CGM):** no consistent rise. Glucose 60 min after the start: median -4, range -33 to +52. Rises of 15+ in 23% of walks, drops of 15+ in 34%.
- No rapid insulin in the 2h before: +9 at 60 min (11 walks). With insulin on board: -5 (24 walks).
- Harder walks (HR 40+ above rest): +7; easier: -4. Evening walks: +3; daytime: -15.
- v8 is not biased after a walk start: mean miss +0 to +5 mg/dL. So v8 already handles the average; the problem is that some walks rise and some fall.

**Inputs:** v9 inputs + exercise features above. Watch rules unchanged (HR 15-min avg, raw steps, raw calories).
**Date:** trained 1 Nov 2026 on 27 Aug-17 Sep 2026 (logged days with watch data only).
**Outputs:** glucose in 60 and 120 min, 80% range, spike risk.

**Test (walk-forward 30 Aug-17 Sep, meals + insulin in the window known, same rows), RMSE mg/dL 1h / 2h:**
- v10 (fiber + exercise): 20.5 / 28.3
- exercise only (no fiber): 20.6 / 28.3
- v9 (fiber): 20.7 / 28.5
- v8: 20.7 / 28.6
- In the 2h after a walk starts (203 rows): v10 23.9 / 29.6 vs v8 24.6 / 30.5.
- Day-bootstrap 90% range of the gain vs v8: 1h -0.6 to +0.3, 2h -0.7 to +0.2 (could be noise).

**Verdict:** best so far by ~1%, mostly in the 2h after a walk (-3%), but within noise. The rise you see is real on some evenings (mostly when no insulin is on board and the walk is hard), not every walk.
**Pros:** knows when a walk started and how hard it is; small gain after walks.
**Cons:** only 35 walks; cannot know a walk is coming until 15 min in; gain not proven yet.
**Files:** joblib + code (glucoforecast9.py with FIB=split, EXF=1 defaults; ex_detect.py, ex_analysis.py, ex_resid.py; ex_bouts.csv = every walk with glucose change). Project files glucose-model/v9_v10_fiber_exercise/<this name>/.
