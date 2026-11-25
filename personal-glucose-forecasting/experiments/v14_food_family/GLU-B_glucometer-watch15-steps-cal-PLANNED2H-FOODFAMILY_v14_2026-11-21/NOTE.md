# GLU-B v14: food-family model (2026-11-21)

**Problem -> solution, why:** v10 treats most carbs alike, so slow foods (roti, rice, dal, chole, rajma) were under-forecast and a new food learned nothing from similar ones. -> Carbs now go into 5 food families, each with its own rise and timing. Why: a new food borrows what its family already taught (chole learns from dal).

**Families:** fast (fruit, biscuits, chocolate, sweets; ~30 min peak), milk (milk, curd; ~40 min), starch (rice, roti, puri, poha, idli, potato; ~1 h, lasts 3 h), legume (dal, moong, chana/chole, rajma, peas, besan, bhujiya; ~2.5 h, lasts 5 h), veg. Fat/protein (paneer, egg, CIIS oil) stays one late channel.
**Inputs:** one glucose reading, planned food + insulin for next 2 h, family carbs, fat/protein, fiber split, walk features, HR 15-min avg, raw steps, raw calories.
**Date:** trained 21 Nov 2026 on set 1 Libre + new Libre Nov 18-19 + all fingerstick pairs sets 1-3 (weight 5).
**Outputs:** glucose in 60 and 120 min, with a range.

**Fair test (each day predicted by a model trained only on earlier days; meals that happened are known). RMSE mg/dL, 1 h / 2 h:**
| Test | v14 family | v10 same data | v10 live | no change |
|---|---|---|---|---|
| Set 1 Libre (14 days) | 20.2 / 27.1 | 20.2 / 27.6 | - | 27.0 / 39.3 |
| Libre Nov 18-19 (new sensor) | 28.4 / 38.8 | 31.8 / 43.7 | 31.0 / 43.9 | 36.8 / 51.2 |
| Glucometer set 2 (milk/fruit week) | 28.6 / 38.8 | 28.8 / 35.5 | 29.2 / 37.0 | 33.9 / 51.3 |
| Glucometer set 3 (new foods) | 50.9 / 41.3 | 53.2 / 49.0 | 49.4 / 48.7 | 54.6 / 60.8 |

**2 h by family (glucometer sets 2+3 / Libre Nov):** starch 40.6 vs v10 49.3 / 46.1 vs 51.1; legume 27.7 vs 38.3 / 21.5 vs 18.8; milk 39.6 vs 36.9 / 43.7 vs 65.1; fast 49.3 vs 48.2.
**Body numbers:** insulin -22.7/u; per g carb: fast 1.7, milk 2.3, starch 1.4, legume 1.9, veg 0; fat/protein +15.5 per FPU.

**Verdict:** better on new-food days (roti, rice, dal, chole): set 3 2 h 48.7 -> 41.3, new Libre 43.9 -> 38.8. Slightly worse on the milk/fruit week (37.0 -> 38.8). Small test (5 + 2 days), so not proven yet.
**Pros:** slow foods finally rise late; new foods learn from their family; works with one test meal.
**Cons:** milk/fruit 2 h a bit worse; lows and big treatments still miss ~50; only 7 new test days.
**Files:** joblib + card + code (fam11.py = model, exp14.py = test, final14.py = train), test_results.csv.
