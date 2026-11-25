"""Put the 3 public datasets and the user's data on one 15-minute grid with the same columns."""
import numpy as np, pandas as pd, glob
from glucoforecast import load_cgm, load_watch
from glucoforecast4 import load_log4
P = "/tmp/claude-0/-home-user/81e1b90b-b01b-535d-92ac-9a948ce80057/scratchpad/pub/"
COLS = ["g", "carbs", "bolus", "basal", "hr", "steps", "hr_mask", "steps_mask", "basal_mask"]

def parse_ts(s):
    iso = s.str.match(r"^\d{4}-")
    t = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    t[iso] = pd.to_datetime(s[iso], errors="coerce")
    t[~iso] = pd.to_datetime(s[~iso], format="%d-%m-%Y %H:%M", errors="coerce")
    return t

def public():
    out = []
    for src in ("hupa", "shanghai", "uch"):
        d = pd.read_csv(P + f"{src}.csv"); d["t"] = parse_ts(d.timestamp.astype(str)); d = d.dropna(subset=["t"])
        d.loc[d["glucose (mg/dL)"] <= 20, "glucose (mg/dL)"] = np.nan  # 0 = missing in UCH
        for pid, x in d.groupby("person_id"):
            x = x.sort_values("t").drop_duplicates("t").set_index("t")
            r = x.resample("15min")
            f = pd.DataFrame({"g": r["glucose (mg/dL)"].mean(), "carbs": r["carbs(g)"].sum().clip(upper=200),
                              "bolus": r["insulin_bolus (U)"].sum(),
                              "basal": r["insulin_basal (U/5min)"].sum() * (3 if src == "shanghai" else 1),  # U per 15 min (Shanghai rows are 15 min apart)
                              "hr": r["heart_rate (bpm)"].mean(), "steps": r["steps"].sum()})
            has_hr = src != "shanghai"; f["hr_mask"] = float(has_hr) * (f.hr > 0); f["steps_mask"] = float(has_hr); f["basal_mask"] = 1.0
            f.loc[f.hr_mask == 0, "hr"] = 0
            f["g"] = f["g"].interpolate(limit=2, limit_area="inside")
            f["pid"] = f"{src}_{pid}"; f["src"] = src
            out.append(f)
    return pd.concat(out)

def user(D):
    cgm = load_cgm(glob.glob(f"{D}/*glucose*.csv")[0]); cgm = cgm[cgm.index >= "2026-08-01"]
    L = load_log4(f"{D}/log.csv"); w = load_watch(f"{D}/noise hr logger")
    idx = pd.date_range(cgm.index.min(), cgm.index.max(), freq="15min")
    f = pd.DataFrame(index=idx); f["g"] = cgm["g"].reindex(idx)
    t = L.time.dt.floor("15min")
    f["carbs"] = L.groupby(t).avail.sum().reindex(idx).fillna(0)
    f["bolus"] = L.groupby(t).rapid_u.sum().reindex(idx).fillna(0)
    lan = L.groupby(t).lantus_u.sum().reindex(idx).fillna(0)
    f["basal"] = lan.rolling(96, min_periods=1).sum() / 96  # Lantus spread evenly over 24 h
    f["hr"] = w.hr.resample("15min").mean().reindex(idx); f["steps"] = w.steps.resample("15min").sum().reindex(idx)
    f["hr_mask"] = f.hr.notna().astype(float); f["steps_mask"] = f["hr_mask"]; f["basal_mask"] = 1.0
    f[["hr", "steps"]] = f[["hr", "steps"]].fillna(0)
    f["pid"] = "self"; f["src"] = "self"
    days = set(L.time.dt.normalize()); f = f[f.index.normalize().isin(days)]
    return f

if __name__ == "__main__":
    pub = public(); pub.to_pickle("pub15.pkl")
    me = user("../data3"); me.to_pickle("me15.pkl")
    for n, x in (("public", pub), ("self", me)):
        print(n, "people", x.pid.nunique(), "rows", len(x), "CGM rows", int(x.g.notna().sum()), "days≈", round(x.g.notna().sum() / 96))
    print(pub.groupby("src").agg(people=("pid", "nunique"), cgm_rows=("g", lambda s: s.notna().sum()), carbs_per_day=("carbs", lambda s: s.sum()), bolus=("bolus", "sum")).to_string())
    print(pub.describe().T[["mean", "max"]].round(2).to_string()); print(me.describe().T[["mean", "max"]].round(2).to_string())
