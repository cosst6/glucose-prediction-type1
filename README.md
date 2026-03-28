# GluPred — Long-Horizon Blood Glucose Prediction for Type 1 Diabetes

A deep learning project for predicting blood glucose levels up to **2–6 hours ahead** (72 steps at 5-minute intervals) in Type 1 Diabetes patients, using a Masked Reactor GRU architecture trained on multi-source CGM data.

---

## Project Overview

Accurate long-horizon glucose forecasting is one of the harder open problems in diabetes management. Standard CGM alarms react too late — by the time a hypoglycemic event is detected, the patient has limited time to respond. This project targets the 1–2 hour window with a focus on step-specific accuracy at later horizons, where predictions matter most clinically.

---

## Data Pipeline

### Sources

Data was collected and merged from three publicly available clinical CGM datasets:

| Dataset | Population | Approx. Patients |
|---------|-----------|-----------------|
| HUPA | Type 1 Diabetes, hospital-monitored | ~20 |
| Shanghai T1D | Chinese T1D cohort | ~12 |
| UCH | University Clinical Hospital cohort | ~19 |

**Total: ~51 patients, ~145,000 rows of CGM readings.**

### Cleaning & Combination

Raw CSV files from each source were cleaned and unified into a single canonical dataset. This involved:

- **Standardizing column names and units** across all three sources (glucose values normalized to mg/dL)
- **Removing physiologically impossible glucose readings** (values outside the clinically valid range were filtered out)
- **Handling missing values** — particularly `time_since_carb_meal`, which had ~53.6% NaN in HUPA due to incomplete meal logging; these were imputed or masked appropriately
- **Aligning timestamps** to a consistent 5-minute interval grid across all patients and datasets
- **Adding derived features** at the row level: Insulin-on-Board (IOB), Carbs-on-Board (COB), time-since-meal, and time-of-day encoding
- **Patient-level train/val/test splits** to prevent data leakage across subjects

The result is a single clean, merged dataframe in a canonical format used across all training phases.

---

## Model Architecture — Phase 8: Masked Reactor GRU

The current model is a **dual-encoder GRU** with binary input masking:

- **8 input features**: CGM reading, IOB, COB, time-since-carb-meal, time-since-insulin, time-of-day (sin/cos encoded), binary meal mask
- **Binary masking layer**: Handles missing physiological inputs (e.g. absent meal logs) without imputation artifacts
- **Reactor mechanism**: Weighted recurrent connections that amplify signal at physiologically active timesteps
- **Output**: 72-step forecast sequence (6 hours at 5-min resolution)

### Loss Function

A **weighted step-specific MAE loss** that assigns higher penalty to predictions at later horizons (60–120 min), reflecting their greater clinical importance.

---

## Results Summary

| Horizon | MAE (mg/dL) |
|---------|------------|
| 30 min (6 steps) | Strong |
| 60 min (12 steps) | Good |
| 120 min (24 steps) | Moderate — active area of improvement |

Phase 8 showed strong 1-hour performance with degradation at the 2-hour mark — the focus of ongoing Phase 9 development.

---

## Repository Structure

```
glu_pred_project/
│
├── data/
│   ├── raw/                  # Original CSVs from HUPA, Shanghai, UCH
│   └── processed/            # Cleaned, merged canonical dataset
│
├── notebooks/
│   └── glu_pred_phase8.ipynb # Main Colab training notebook
│
├── checkpoints/
│   ├── phase8_masked/        # Best Phase 8 model weights
│   └── earlier_phases/       # Phase 6, 7 checkpoints for reference
│
├── src/
│   ├── data_pipeline.py      # Cleaning, merging, feature engineering
│   ├── model.py              # Masked Reactor GRU definition
│   ├── train.py              # Training loop with weighted loss
│   └── evaluate.py           # Step-specific MAE evaluation
│
└── README.md
```

---

## Requirements

```
torch>=2.0
numpy
pandas
scikit-learn
matplotlib
```

---

## How to Run

1. Clone the repo and place raw CSVs in `data/raw/`
2. Run `src/data_pipeline.py` to generate the cleaned merged dataset
3. Open `notebooks/glu_pred_phase8.ipynb` in Google Colab
4. Mount your Drive, point to the processed data path, and run all cells

---

## Roadmap

- [x] Phase 1–7: Baseline GRU, Transformer, BigBrain, Reactor variants
- [x] Phase 8: Masked Reactor GRU — best performing architecture
- [ ] Phase 9: Add glucose history as 9th encoder feature; dual-output head for simultaneous 1h + 2h prediction
- [ ] MVP deployment: Simple web interface for real-time glucose forecast

---

## Author

**Chirag Sinha**  
[GitHub](https://github.com/) · [Email](mailto:cas06052005@gmail.com)

---

*This project is for research purposes. Not intended as a medical device or clinical decision support tool.*
