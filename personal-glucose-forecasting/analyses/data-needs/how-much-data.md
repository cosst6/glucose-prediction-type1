# How much data do the models need? (15 Nov 2026)

## 1. Were my numbers right? Checked against sources

| My claim | Verdict | Evidence |
|---|---|---|
| Simple models need ~10–20 events per input | **Holds as a ballpark.** It's a rule of thumb, not a law. | Peduzzi 1996 simulation: under ~10 events per input, logistic models become biased and unreliable ([PubMed](https://pubmed.ncbi.nlm.nih.gov/8970487/)). van Smeden 2016: "evidence supporting EPV rules … is weak". It depends on more than one number ([paper](https://researchinformation.umcutrecht.nl/en/publications/no-rationale-for-1-variable-per-10-events-criterion-for-binary-lo/)). Riley 2020 BMJ replaced the rule with a calculation ([BMJ](https://dspace.library.uu.nl/handle/1874/457642), [pmsampsize](https://rdrr.io/pkg/pmsampsize/man/pmsampsize.html)). |
| For rare events, count events, not readings | **Holds.** | Events-per-variable counts the rarer outcome (Peduzzi). Riley's formulas depend on how common the outcome is. |
| ~50 lows for the low alarm | **Too low. Correct to ~100+.** | Riley's calculation for LOW-1 (8 inputs, 14% of sticks are lows, standard settings) needs about **115 lows** (~800 checks). |
| ~300 meals for spike/rise | **Too low. Correct to ~450.** | Riley's calculation for SPIKE-1/RISE40 (6 inputs) needs about **450–470 meals** (~160–220 rises). |
| ~60 nights | **My estimate. Probably too low.** | No source covers this directly. Riley's continuous-outcome formula (Part I, [Stat Med 2019](https://dspace.library.uu.nl/handle/1874/386405)) with 9 inputs gives **110–220 nights**. The physics limits built into NIGHTLOW-1 should lower that, but nobody has measured by how much. |
| 2h forecast goes from 28 to the mid-20s by 2–3 months | **Holds** (vs Libre). See section 2. | Published papers give no clean "error vs days" curve for one person. Karim 2021 found about 2 weeks of diary was enough for their meal model, and cutting it to ~26 meals made the 180-min error worse ([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8307794)). |

Riley numbers use shrinkage 0.9 and the "15% of max R²" default from Riley 2020. They are calculations, not measurements on your data.

## 2. Your own learning curves

Method: train on 25/50/75/100% of training days and score on days the model never saw. Repeated 30–40 times with random splits (LOW-1: each day held out in turn, 15 repeats; NIGHTLOW-1: each night held out in turn, 8 repeats). Planned meals are known. Data used: Libre Aug 27–Sep 17, fingersticks Nov 6–12 and Nov 15 (Nov 15 only adds to LOW-1, because it has no known rise or night outcome yet).

![curves](learning-curves.png)

| Model | 25% | 50% | 75% | 100% | Where it's heading |
|---|---|---|---|---|---|
| v10 2h error (lower better) | 32.6 (4 d) | 29.2 (8 d) | 28.0 (13 d) | 27.2 (17 d) | ~26 at 30 d, ~25 at 60 d, ~24 at 90 d; floor ~22 |
| v10 1h error | 22.7 | 20.9 | 19.9 | 19.4 | ~18.7 at 30 d, ~18 at 60 d |
| SPIKE-1 AUC (higher better) | 0.73 (23 meals) | 0.79 (46) | 0.80 (70) | 0.80 (92) | ~0.82 at 30 d, ~0.84 at 60 d |
| RISE40 AUC | 0.77 | 0.83 | 0.83 | 0.84 | ~0.86 at 30 d, ~0.88 at 60 d |
| LOW-1 AUC | 0.56 (2 d, 3 lows) | 0.58 (3 lows+) | 0.64 (6 lows) | 0.74 (9 lows) | Still rising steeply. Rough guess ~0.84 by 30 days, but this rests on only 4 points |
| NIGHTLOW-1 wake error | 162 (4 nights) | 147 (8) | 62 (13) | 52 (17) | Still falling fast; too early to say where it stops. Guessing "no change" scores 77 |

**What it means:**
- **v10 has learned most of what this kind of data can teach it.** Each doubling of days now saves only ~1–2 mg/dL. Bigger gains need better inputs (sensor calibration, clean food tests), not just more of the same days.
- **The meal alarms are flattening out** at about 20 days. More meals help slowly.
- **LOW-1 and NIGHTLOW-1 are data-starved.** Their curves are still steep, so every new low and every night with a 3am + wake stick helps a lot.

Caveats: the extrapolation is a simple a + b/√days fit and only a rough direction. The errors are vs Libre (vs blood they are much larger, see the Libre accuracy report). The test days are few, so each point moves ±3 mg/dL or ±0.05 AUC between splits.

## 3. Does variety help more than volume? (follow-up)

**Test on your data (v10, 2h, vs Libre).** The faulty sensor-1 days were left out. For each of 30 random splits: 3 test days, 5 random base days, then add either 4 *ordinary* days (fewest big swings, meaning ±50 in 2h) or 4 *varied* days (most big swings).

| Training set | All windows | Big-swing windows | Calm windows |
|---|---|---|---|
| 5 base days | 32.3 | 42.5 | 28.5 |
| + 4 ordinary days | 30.7 | **42.9** (no gain) | 26.0 |
| + 4 varied days | **28.4** | **37.1** (−14%) | 25.0 |

Varied days won on big swings in 26 of 30 splits and on all windows in 22 of 30. Code: code/variety.py, results: variety.csv.

**Sources**
- Error usually shrinks as a power law of data size, so each doubling buys less ([Kaplan 2020](https://arxiv.org/abs/2001.08361)).
- Choosing informative, non-redundant examples can beat that curve ([Sorscher 2022, NeurIPS](https://arxiv.org/abs/2206.14486)).
- Models do much worse on situations missing from their training data ([Koh 2021, WILDS](https://arxiv.org/abs/2012.07421)).
- There is a ceiling: the same meal eaten twice by the same person gives responses that correlate only ~0.7–0.8 ([Zeevi 2015, Cell](https://doi.org/10.1016/j.cell.2015.11.001)). On top of that, your Libre itself is ~30% off. Neither of these goes away with more data.
- Your own Nov 2 food test: new foods changed typical-meal error by less than 0.5 mg/dL, but roughly halved the error for that food type (rice/roti ±6→±3).
