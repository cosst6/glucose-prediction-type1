# GLU-T v3 (2026-11-24): sticks set the level, Libre gives the shape. Research only, NOT live

**Problem -> solution, why:** Libre reads ~28% low with ≥30% error, but it is good for trends and nights; fingersticks are accurate but sparse. -> (1) Denoise: every Libre reading is turned into blood with a calibration (blood = 25.6 + 1.07 x Libre) plus a correction from your recent sticks (causal). (2) Fill: a continuous blood curve (sticks = level, calibrated Libre = shape, twin fills stick-to-stick gaps with no Libre, e.g. set 2). (3) Self-train: the twin learns from that curve plus every real stick pair (sticks x5, Libre windows x0.3). (4) Fusion tested with Chronos-2 (Amazon's pretrained forecaster, meals/insulin as known future inputs) and v10/v14.
**Inputs:** last reading (stick, or Libre corrected by recent sticks), planned meals/insulin for the full window, HR 15-min avg, raw steps, raw calories. **Date:** 24 Nov 2026, all sets 1-3 (166 sticks). **Outputs:** blood path to 4 h, 80% range, what-if.
**Fair walk-forward vs fingersticks (RMSE mg/dL), each day trained only on earlier days:**
| Forecast | checks | v3 | v2 | last reading as is |
|---|---|---|---|---|
| Libre -> stick ~1 h | 232 | 38.2 | 41.3 | 64.7 |
| Libre -> stick ~2 h | 205 | 43.1 | 46.5 | 75.5 |
| Libre -> stick 3-4 h | 302 | 44.3 | 48.0 | 79.6 |
| stick -> stick ~1 h | 97 | 34.3 | 36.3 | 42.1 |
| stick -> stick ~2 h | 86 | 51.9 | 55.0 | 57.7 |
| stick -> stick 3-4 h | 120 | 59.4 | 64.0 | 66.1 |
Stick -> stick vs others (same pairs): 1 h v14 36.2, v10 35.3, average twin+v14+v10 **33.7**; 2 h v14 47.8, v10 48.2, average **47.1**.
Denoising alone: Libre as blood error 49.5 -> 29.3 (fair). Denoise without self-training: 39.8/45.6/46.8 and 35.4/56.9/67.2, so most of the gain comes from the blood curve.
**Chronos-2 zero-shot:** Sept Libre 2 h 36.5 (Bolt 40.9) vs twin 26.0, v14 27.1; Libre->stick 2 h 57.0 vs twin 42.7. Blending it in (equal or learned weights) made results worse. Learned blend weights (twin/v14/v10) worse than plain average (too few days).
**Weights tried:** curve Libre part x0.25/0.5/1, curve sticks x2/x4, bridge x1/x2, Libre windows x0/0.3/1. Best: Libre x0.3, curve x1, curve sticks x2, sticks x5.
**Pros:** best twin vs blood so far; uses every data point; follows "sticks for accuracy, Libre for trend". **Cons:** stick->stick 2 h alone still behind v10/v14 (use the 3-model average); curve labels are partly estimated; few sticks. Not dosing advice.
Page: https://claude.ai/artifact/1wdkDZ6SKuEMrKSx9ATAnU (runs v3)
