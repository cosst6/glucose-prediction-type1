"""SPIKE-1: will glucose rise after this meal, and by how much? (research only, never for dosing)
Call at meal log time with the meal's food split + insulin. Inputs:
  avail_carbs_g (sugar + starch, fiber excluded), dose_u (NovoRapid with the meal, -50..+30 min), iob_u (earlier insulin still acting),
  hr15 (mean HR last 15 min), steps60 (raw steps last 60 min), pre_bg (fingerstick within 45 min before, optional),
  d45 (fingerstick 45 min after meal minus pre_bg, optional).
"""
import joblib, glob, os, numpy as np, pandas as pd
M = joblib.load(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "*.joblib"))[0])
def alarm(avail_carbs_g, dose_u, iob_u=0.0, hr15=80.0, steps60=0.0, pre_bg=None, d45=None):
    x = dict(uncov=avail_carbs_g - M["carbs_per_unit"] * dose_u, iob=iob_u, hr15=hr15, steps60=steps60)
    step = "step0_no_stick"
    if pre_bg is not None:
        x.update(prec=float(np.clip(pre_bg, 40, 250)), low=float(pre_bg < 70)); step = "step1_pre_stick"
        if d45 is not None: x["d45"] = d45; step = "step2_45min_stick"
    s = M[step]; X = pd.DataFrame([x])[s["cols"]]
    p30 = s["m"]["y30"].predict_proba(X)[0, 1]; p60 = min(s["m"]["y60"].predict_proba(X)[0, 1], p30)
    band = int(p30 > .5) + int(p60 > .5); w = M["ask_step2_if_within"]
    msg = None
    if step == "step0_no_stick": msg = "Check your blood glucose now (before eating) for a better spike forecast."
    elif step == "step1_pre_stick" and (abs(p30 - .5) < w or abs(p60 - .5) < w): msg = "Borderline. Check your blood glucose 45 min after starting the meal."
    return dict(band=M["bands"][band], p_rise30=round(float(p30), 2), p_rise60=round(float(p60), 2), used=step, ask=msg)
if __name__ == "__main__":
    print(alarm(12, 1, pre_bg=110))           # 250 ml milk + 1u
    print(alarm(12, 2, pre_bg=110))           # 250 ml milk + 2u
    print(alarm(5, 3, pre_bg=150, iob_u=1))  # eggs + veg
    print(alarm(12, 1, pre_bg=110, d45=35))  # milk, +35 at 45 min
