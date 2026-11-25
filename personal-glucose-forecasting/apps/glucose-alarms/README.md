# Glucose Alarms v1.1 (separate app, models only)

RESEARCH ONLY. Never use for dosing. Always confirm with your glucometer.

The Noise HR Logger app stays the data collector and is unchanged. This app runs four models, fully offline:
- **LOW-1**: chance of a low (<70) in 1/2/3 h, checked every 15 min. Alarm if the 2 h chance is > 50% and >= 0.5 u NovoRapid is on board (switchable).
- **SPIKE-1**: rise band after a meal (<30 / 30-60 / >=60). Asks for a reading before eating, and again 45 min after if borderline.
- **SPIKE-2** (new, 15 Nov): rise band + "starts rising" and "highest" time RANGES. No reading after the meal.
- **RISE40** (new, 15 Nov): "rise of 40+ likely?" (fast or slow) + ranges for starts rising / passes +40 / highest. No reading after the meal.

## Pick models (new)
"Models to show" chips at the top: tap to show/hide each model's card. Hidden models still run and are logged
(for scoring) but send no notifications. "Ask the models (not saved)" at the bottom: type a meal, units and
(optionally) a reading; every picked meal model answers side by side, nothing is logged.

Ranges held the real time 7 of 10 times on days the models never saw. Peak ranges are wide (~3 h): peak time is hard to predict.
No insulin-timing advice: timing is not better than your usual pattern and about 1 in 5 warnings is false.

## Inputs (no cloud, no internet)
- Food, NovoRapid, corrections, low treats, fingersticks: typed into this app.
- Watch HR (15 min mean), raw steps (60 min): read on the phone from the Noise app (read-only share). Calories are used by LOW-1 only.

## Log for scoring
Android/data/com.example.glucosealarms/files/Documents/GlucoseAlarms/ (or "Share log (CSV)"):
alarms.csv (every alarm incl. spike2 / rise40 rows + outcome from your sticks), low_checks.csv, events.csv.

## Checked
AlarmCoreTest: phone maths = Python models (meal parser, absorption features, LOW-1, SPIKE-1, and the new SPIKE-2 / RISE40 numbers).
