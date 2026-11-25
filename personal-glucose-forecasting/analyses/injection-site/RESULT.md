# Does NovoRapid injection site help forecasts? (Nov 21 2026)

**Verdict: No, not with current data.**

## Counts (NovoRapid doses, all sets)
- Site logged: 26 doses, all in set 2 (Nov 7-12): left stomach 15, right stomach 11.
- No site: set 1 (~120 doses), set 2 (13), set 3 (~43).
- Leg/arm NovoRapid: 0. (Leg entries are Lantus.)

## Fair test (set 2 glucometer, unseen-day, meals/insulin known)
Site feature = units given in left vs right stomach in the 3h before + forecast window, learned leave-one-day-out on top of each model.

| Model | 1h without | 1h with site | 2h without | 2h with site |
|---|---|---|---|---|
| v10 (live) | 29.2 | 30.4 | 37.0 | 39.2 |
| v14 | 28.6 | 30.3 | 38.8 | 40.6 |

Worse with site (overfits; only 43-44 test points).

## Simple check: glucose change 90-180 min after a sited dose
- Left: 6 doses, mean +2. Right: 9 doses, mean -34. Mostly meals/corrections mixed, so not a fair comparison.
- Corrections with no food: right -25/u (4), left -34/u (1). Too few.

## How many needed
- Left vs right stomach: no body reason to differ; likely never helps.
- Stomach vs leg (or arm): ~40 clean corrections per site (no food, in range 150-250, no other dose for 3h) to see a ~15 mg/dL difference; with normal meals ~140 per site.

Code: injection-site/code/
