# Experiments

One folder per model version, in the order they were built. Each has its training and evaluation code; most have a `NOTE.md` written in the same format: **problem -> solution, why**, inputs, date, outputs, test scores, pros, cons, verdict.

| Folder | Idea | Verdict |
|---|---|---|
| `v2`, `v3` | Ridge regression + engineered meal features (A: CGM, B: meter) | First baseline, ~15% better than persistence |
| `C` | Meter + food + insulin only, with a JS port for a browser forecaster | Simple baseline |
| `v4` | Glycemic index/load, absorption kernels, quantile ranges, spike classifier | Kept for honest 80% ranges |
| `public_pretrain` | Pretrain on public T1D data (Shanghai, HUPA-UCM, UCH), fine-tune | No gain yet |
| `v5` | Planned meals/insulin for the full horizon; `ctx48/` = 48 h look-back | Kept; look-back worse |
| `v6_moe` | Mixture of experts (spike vs stable) | Dropped |
| `v7_watch15` | 15-min heart rate, raw steps and calories | Within noise |
| `v8_B_planned_watch15` | Meter model + planned meals + watch | Big step |
| `v9_v10_fiber_exercise` | Fiber split + walk detection features (v10 = live model) | Live |
| `v11_and_low_alarm` | Finger-pricks weighted 5x; LOW-1 low alarm | Within noise / research alarm |
| `spike_alarm`, `spikes` | SPIKE-1, SPIKE-2, RISE40 alarms; spike analysis | Research alarms |
| `nightlow` | NIGHTLOW-1 overnight path and low risk | Too few nights |
| `ctx3d_nov16` | 0-3 days of past context | Dropped |
| `v14_food_family` | Carbs split into 5 food families | Best on new foods |
| `v15_glycogen_tank` | Liver/muscle glycogen state | No gain |
| `gru_blood` | GRU on CGM mapped to blood with gap filling (PyTorch) | No gain |
| `twin` | Physiological ODE digital twin, 32 bootstrap members (v1-v3) | Best on CGM, research |
| `miss_causes` | Classifies big forecast misses by cause | Analysis |

`evaluate.py` and `pipeline.py` hold the shared walk-forward evaluation and data loading.
