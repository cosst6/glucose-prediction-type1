# LOW-1 v2: low alarm retrained with set 3 (2026-11-21)

**Problem -> solution, why:** LOW-1 v1 was trained on only 10 lows (Nov 6-12). -> Same logistic model (warns of glucose under 70 up to 6 h ahead), now trained on sets 2 + 3 (18 low readings, 963 rows). Why: more lows.

**Inputs:** planned NovoRapid + carbs + fat/protein in the window, insulin on board, last stick and its age (optional), HR 15-min avg, raw steps 60 min, raw calories 60 min.
**Date:** 21 Nov 2026. **Outputs:** chance of a low, alarm if over 50%.

**Fair test on set 3 (8 lows, 5 days):**
| Warn time | v1: caught / false alarms | v2 (trained on earlier days only) |
|---|---|---|
| 1 h | 4/8, 59% | 4/8, 62% |
| 2 h | 4/8, 72% | 4/8, 64% |
| 3 h | 5/8, 76% | 4/8, 54% |
AUC was 0.3-0.45 for both, which is worse than chance on set 3. The set 3 lows came after correction or stacked doses, with long gaps between sticks. Set 2 did not show that pattern.

**Verdict:** LOW-1 does NOT work on new data (v1 or v2). Do not trust its alarms. Plain "last stick + body-model change" did a bit better (AUC 0.62-0.67 at 1-2 h).
**Pros:** more lows in training. **Cons:** worse than chance on the unseen week, and 18 lows are far too few. Research only, never for dosing.
**Files:** joblib, low13.py, lowfeat.py, low13_results.csv.
