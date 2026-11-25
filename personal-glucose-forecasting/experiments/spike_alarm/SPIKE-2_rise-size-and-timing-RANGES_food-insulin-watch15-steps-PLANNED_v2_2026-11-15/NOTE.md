# SPIKE-2 v2: rise size + start/peak time RANGES at meal time (2026-11-15)

**Problem -> solution, why:** v1 gave one exact clock time ("2:57 PM"), which was often wrong. -> Same size answer, but times are now ranges that held the real time 7 of 10 times on unseen days. Why: timing varies a lot from meal to meal, so an honest range beats a precise wrong time.

**Inputs:** as v1 (food split, NovoRapid with the meal, insulin still acting, HR 15 min, raw steps 60 min, night, optional pre-meal reading). No calories, no reading after the meal.
**Date:** 15 Nov 2026, same data as v1 (120 meals; timing from 57 Libre rises of 30+).
**Outputs:** band (<+30 / +30-60 / +60+), "starts rising" range (+30), "highest" range, late-rise flag.

**Fair test:** size as v1 (pre-meal reading: 54% band right on fingersticks, 55% Libre). Ranges (7/10 inside): start ~70-85 min wide, highest ~3 h wide. "Your usual" ranges: start ~105 min, highest ~3.2 h, so the model narrows the start range a little.
- No insulin-timing window (see RISE40 note: risk of extra insulin into a rise that isn't coming).
**Cons:** same as v1; peak range is wide. Research only: never use for dosing.
**Files:** joblib, ranges.py, export2.py; phone: Glucose Alarms app v1.1.
