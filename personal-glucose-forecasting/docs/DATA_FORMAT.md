# Data format

No personal health data is in this repo. To run the code, put your own files in a `data/` folder (git-ignored):

| File | Columns | Source |
|---|---|---|
| `cgm_export_*.csv` | LibreView export (timestamp, historic glucose mg/dL, scan glucose) | FreeStyle Libre CGM |
| `noise hr logger/NoiseHR_Session_*.csv` | time, heart rate, steps, distance, calories (every 30 s) | `apps/watch-logger` |
| `log.csv` | time, rapid_u, lantus_u, bg (meter), cgm_noted, food (free text, items split by `;`) | hand log |

`food_db.csv` (in each model folder) holds carbs, sugar, fiber, fat, protein, glycemic index and load per serving, from published tables. `parse_meal.py` turns text like `2 roti; 1 cup dal cooked in indian style` into nutrients.

Trained model files (`.joblib`, `.pt`) are not included because they are fit to one person's data. Each model folder has a `*_card.json` (inputs, coefficients, test scores) and a `NOTE.md`.
