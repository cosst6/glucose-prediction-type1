# NIGHTLOW-1: overnight low + wake-up glucose at bedtime (2026-11-15)

**Problem -> solution, why:** LOW-1 only looks up to 6 h ahead in the day; nothing answered "will I go low while I sleep 8-9 h fasted, and what will I wake up at?" -> At bedtime (Lantus time) the model predicts the glucose path hour by hour for 9 h, then gives the chance of a low (<70), a 1-3 h time range, and the wake-up value (8-9 h). Simple sign-limited model (insulin can only lower, food/milk only raise) because 18 nights is too few for anything bigger.

**Inputs (at bedtime):** glucometer reading; NovoRapid still on board (incl. doses planned up to 90 min after bed); carbs, fat+protein (FPU) eaten in last 4 h + bedtime snack; milk at bedtime (ml); HR mean last 15 min; raw steps last 3 h. Lantus is always ~28 u so it is in the baseline, not a separate input. No calories (as asked).
**Date:** 15 Nov 2026. Trained on 18 Libre nights (27 Aug - 16 Sep), Libre divided by 0.72 to blood scale. Checked on 5 fingerstick nights (6-10 Nov).
**Outputs:** low chance (alarm if > 40%), time range, wake-up value with an 80% band (about +/-65).

**Fair test (each night predicted by a model trained on the other nights):**
- Real lows (blood <70): only 2 in 18 nights. Caught 1 of 2. False alarm on 3 of 16 nights (~2 per 10 nights). Time range missed the one it caught (said too early).
- Near-lows (blood <97, = Libre under 70): caught 5 of 9, no false alarms; time range right 3 of 5, about 2 h wide.
- Low on waking: no real wake-up low in the data. Wake under 97: caught 3 of 4, no false alarm. Wake value error ~52 (staying-at-bedtime guess: 77).
- November blood nights (no lows happened): wake value within ~19 on 4 of 5 nights. 1 false alarm: the night with a 4 u bedtime correction (8 u on board).

**What drives it:** milk at bedtime keeps you up (~+90-110 by 3-6 h per 250 ml); each NovoRapid unit on board ~ -20 overnight; fat/protein adds slowly (~+11 per FPU); with no milk and no snack glucose drifts down overnight.
**Pros:** answers the exact bedtime question; uses planned bedtime snack/dose; easy to port (a few numbers per hour).
**Cons:** far too few nights (2 real lows) to trust; straight-line model can predict impossible values (<40) after large doses; big bedtime corrections give false alarms; Libre night data converted with one ratio. Research only: never use for dosing; check with the glucometer.
**Next:** 3 am + wake-up fingersticks on 15-20 more nights (note bedtime milk / dose) would make this testable.
**Files:** <name>.json / .joblib (coefficients per hour, sd, rules), golden.json (app test cases), nights_libre.csv, loo_nights.csv, nov_check.csv, score_final.txt, code (feats.py, build.py, phys.py, final.py, score_final.py, nov_phys.py).
