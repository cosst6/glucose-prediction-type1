# Glucose Alarms v1.3 (separate app, models only)

RESEARCH ONLY. Never use for dosing. Always confirm with your glucometer.

Four models, fully offline: **LOW-1** (low <70 in ~2 h, needs ≥0.5 u NovoRapid on board), **SPIKE-1** (rise band, asks for readings),
**SPIKE-2** (rise band + start/peak time ranges), **RISE40** (rise of 40+, fast or slow, with ranges), **NIGHTLOW-1** (new in 1.3: night low + wake-up value).

## New in 1.3: NIGHTLOW-1
On the Now tab, tap **🌙 Going to sleep**, set your bedtime fingerstick with the buttons, tap Save. It answers two lines, e.g.
"Night low: likely 2:00 am – 4:00 am (55%)" and "Wake-up: about 110 (50–170)". Values below 40 show as "under 40".
Uses: bedtime reading, NovoRapid still acting (incl. doses planned up to 90 min after bed), carbs and fat/protein of the last 4 h + bedtime snack,
bedtime milk, watch HR (15 min) and steps (3 h). No calories. Log bedtime milk/snack/dose first.
Weak: only 18 nights (2 real lows), caught 1 of 2, about 2 false alarms per 10 nights, wake value often off by ~50. Research only.
Scored next morning from your night/wake-up fingersticks (History tab).

## New in 1.2
- **No typing anywhere.** 4 tabs at the bottom:
  - **Now**: reading prompts (with a big +/− stepper), one card per model switched on, watch status.
  - **Log**: Meal / Insulin / Reading / Low treat. Food is a list with +/− buttons (common foods first, "More foods…" for the rest);
    NovoRapid in 0.5 u steps; Lantus starts at 28 u; reading stepper ±1 / ±10; "When" chips (−30m … +2h for planned).
  - **History**: last 12 h + planned (Delete), scores, recent alarms, Share log (CSV).
  - **Models**: on/off switch per model, notification switches, the ≥0.5 u gate, watch access, Check now.
  - "Ask the models" box removed (it needed typing). Log a planned meal (+30m) instead; delete it if you change your mind.
- **Watch data from the Noise HR Logger.** The updated Noise app also copies every CSV to `Documents/NoiseHR/` (it still saves where it did before).
  Glucose Alarms reads all past and live files there, **read only** (it never writes, renames or deletes them). It needs "All files access" for that.
  "Ignore steps" in the Noise app writes 0 steps, so Glucose Alarms never sees or uses steps while it is on.
- **Short watch gaps filled.** Gaps of 10 min or less in HR/steps are filled with a straight line between the two readings; longer gaps stay empty.
- **Late pre-meal reading counts.** A fingerstick from 45 min before up to 20 min after you start eating counts as the pre-meal reading
  (e.g. eat 8:26, stick 8:41 = pre-meal). The one closest to the meal start is used.

## Log for scoring
Android/data/com.example.glucosealarms/files/Documents/GlucoseAlarms/ (or Share log): alarms.csv, low_checks.csv, events.csv.
Meals logged with the buttons are saved with the same words as before (e.g. "250 ml milk; 2 roti"), so the models see the same numbers.

## Checked
AlarmCoreTest (10 tests pass, incl. NIGHTLOW-1 = Python on 3 golden cases): phone maths = Python models, every food button is understood by the meal parser,
gap filling (2-min gap filled, 28-min gap not), late stick at +15 min counts and +24 min does not.
The screens could not be built here (no Android build tools in this workspace): build in Android Studio as before.
