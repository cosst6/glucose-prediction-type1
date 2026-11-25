"""SPIKE-2 notification at meal log (research only, never for dosing). No glucose after the meal is used.
notify(meal_time, sugar_g, starch_g, fiber_g, fat_g, protein_g, milk, dose_u, iob_u, hr15, steps60, pre_bg=None)
  milk: 1 if the meal has milk; dose_u: NovoRapid with the meal (planned); iob_u: earlier insulin still acting; pre_bg optional."""
import joblib, glob, os, numpy as np, pandas as pd
M = joblib.load(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "*.joblib"))[0])
def notify(meal_time, sugar_g, starch_g, fiber_g, fat_g, protein_g, milk, dose_u, iob_u=0.0, hr15=80.0, steps60=0.0, pre_bg=None):
    t0 = pd.Timestamp(meal_time); avail = sugar_g + starch_g
    x = dict(uncov=avail - M["carbs_per_unit"] * dose_u, iob=iob_u, hr15=hr15, steps60=steps60, sug_share=sugar_g / (avail + 1), starch=starch_g,
             fiber=fiber_g, fpu=(9 * fat_g + 4 * protein_g) / 100, milk=float(milk), dose=dose_u, night=float(t0.hour >= 21 or t0.hour < 6))
    mode = "none"
    if pre_bg is not None: mode = "pre"; x.update(prec=float(np.clip(pre_bg, 40, 250)), low=float(pre_bg < 70), pre=float(pre_bg))
    m = M[mode]; X = pd.DataFrame([x])
    p30 = m["size"]["y30"].predict_proba(X[M["SIZE"][mode]])[0, 1]; p60 = min(m["size"]["y60"].predict_proba(X[M["SIZE"][mode]])[0, 1], p30)
    band = int(p30 > .5) + int(p60 > .5)
    st = float(np.exp(m["start"].predict(X[M["TIME"][mode]])[0])); pk = float(np.exp(m["peak"].predict(X[M["TIME"][mode]])[0]))
    late = float(m["late"].predict_proba(X[M["LATE"]])[0, 1])
    f = lambda mins: (t0 + pd.Timedelta(minutes=mins)).strftime("%-I:%M %p")
    if band == 0:
        msg = "No big rise expected after this meal."
        if late > .5: msg += f" It may still creep up late, after {f(120)}."
    else:
        size = "+30 to 60" if band == 1 else "+60 or more"
        msg = f"Rise of about {size} expected, starting around {f(st)}, highest around {f(max(pk, st + 15))} (give or take ~50 min)."
        if late > .5: msg += " It may start late, after 2 h."
    return dict(message=msg, p_rise30=round(float(p30), 2), p_rise60=round(float(p60), 2), start_min=round(st), peak_min=round(pk), late_risk=round(late, 2), used=mode)
if __name__ == "__main__":
    print(notify("2026-11-14 23:30", 12, 0, 0, 15, 8, 1, 0))                    # bedtime milk, no insulin
    print(notify("2026-11-14 13:00", 14, 20, 6, 30, 30, 1, 4, pre_bg=110))       # roti/paneer/milk lunch
    print(notify("2026-11-14 21:30", 3, 3, 3, 35, 36, 0, 3, pre_bg=120))         # paneer + veg dinner
