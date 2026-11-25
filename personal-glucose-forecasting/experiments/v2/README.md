# Personal glucose forecaster (v2)

Data folder layout: `<data>/cgm_export*.csv` (LibreView export), `<data>/noise hr logger/*.csv`, `<data>/log.csv`.

- `log.csv`: one row per entry: time, rapid_u, lantus_u, bg (glucometer), cgm_noted, food (free text, items separated by `;`).
- `food_db.csv` + `parse_meal.py`: turn food text into carbs/protein/fat/fiber. Add rows for new foods.
- `python train.py <data>`: fit both models and save `models.joblib`.
- `python predict.py <data> --mode cgm --now "YYYY-MM-DD HH:MM"`: Model A.
- `python predict.py <data> --mode fingerstick --bg 145 --now ... --food "..." --insulin 4`: Model B.
- `python evaluate.py <data>`: walk-forward evaluation (results.csv). `fingerstick_check.py`: glucometer-to-glucometer test.
