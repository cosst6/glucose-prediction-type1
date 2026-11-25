# GLU-A planned-meal v5 (2026-10-30)

**Problem -> solution -> why:** The forecast could not see a meal eaten after the forecast time, so the biggest misses came just before meals (1h miss 27 vs 17 when fasting). -> Let the user enter the food + insulin they are about to have in the next 60 min. -> The food maths already projects absorption forward, so a planned meal is treated exactly like a logged one.

**Inputs:** Libre CGM (now + last 15/30/60 min change), Noise watch (HR, steps), food log split into fast sugar, milk sugar (new fast channel), GL, slow carbs, fat/protein (peak moved to 3.5 h), night food, NovoRapid, Lantus (incl. "old Lantus" >22 h), 3-hour clock blocks, dawn, below-70/above-140, PLUS planned food/insulin in the next 60 min.
**Outputs:** glucose in 60 and 120 min, 80% likely range (scaled 1.75x / 2.0x), spike risk.
**Test (walk-forward 30 Aug-16 Sep):** 1h miss 21.0 (v4 21.5), 2h 29.0 (v4 31.4). When a meal is coming in the next hour: 1h 23.8 (v4 26.5), 2h 34.7 (v4 41.2).
**Pros:** best 2h model so far; biggest gain right around meals; honest range.
**Cons:** only helps if you log the meal before eating; only 60/120 min horizons; trained on 17 days; time-of-day/night/low-rebound features don't help yet (too little data) - without a planned meal v5 is slightly worse than v4, so keep v4 for normal use.
**Body numbers learned:** insulin -23 per unit, milk sugar +3.6 per g (fastest of all), free sugar +2.5/g, fat/protein +16.5 per FPU; strongest upward drift 03-06 h (dawn).
