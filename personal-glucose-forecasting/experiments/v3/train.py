"""Fit Model A and Model B on all available data and save them to models.joblib."""
import sys, joblib
from glucoforecast import *
DATA = sys.argv[1]
cgm = load_cgm(glob.glob(f"{DATA}/*glucose*.csv")[0]); cgm = cgm[cgm.index >= "2026-08-01"]
log = load_log(f"{DATA}/log.csv"); watch = load_watch(f"{DATA}/noise hr logger")
X = restrict_to_logged_days(build_frame(cgm, log, watch, sorted(set(HORIZONS_A) | set(HORIZONS_B))), log)
models = {"A": {h: fit(X, "A", h) for h in HORIZONS_A}, "B": {h: fit(X, "B", h) for h in HORIZONS_B}}
joblib.dump(models, "models.joblib")
# what model B learned, in plain units: change in glucose at 120 min per unit of each input acting in that window
m = models["B"][120]; sc, rg = m[0], m[1]
coef = dict(zip(m.cols, rg.coef_ / sc.scale_))
print(f"trained on {len(X.dropna(subset=['y60']))} CGM points")
print(f"insulin  : {coef['ins_next120']:+.1f} mg/dL per unit acting in the next 2 h")
print(f"net carbs: {coef['carb_next120']:+.2f} mg/dL per gram absorbed in the next 2 h")
print(f"fat+prot : {coef['fpu_next120']:+.1f} mg/dL per fat-protein unit absorbed in the next 2 h")
