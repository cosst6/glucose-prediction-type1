import pandas as pd, html
R = pd.read_csv("all_windows.csv", parse_dates=["meal_t", "peak_t"])
S = R[R.spike].copy()
lab = {"covered, peak 1-3h (UNEXPLAINED)": "Covered meal, peak 1–3h (your 'unexplained')",
       "covered, late peak >3h": "Covered meal, late peak >3h", "covered, fast peak <1h": "Covered meal, fast peak <1h",
       "food, no insulin": "Food with no insulin", "low treated (no insulin) -> rebound": "Rebound after a low",
       "meal started low -> rebound + meal": "Rebound after a low", "under-dosed (1u per 15g rule)": "Under 1u/15g even by your rule"}
S["c"] = S.cls.map(lab)
S = S.sort_values(["c", "rise"], ascending=[True, False])
rows = ""
order = list(dict.fromkeys(lab.values()))
for c in order:
    for _, r in S[S.c == c].iterrows():
        t = r.meal_t.strftime("%b %d %H:%M")
        sz = "s1" if r.rise < 60 else "s2" if r.rise < 120 else "s3"
        rows += f"<tr class='{'hl' if 'unexplained' in c else ''}'><td>{c}</td><td>{t}</td><td>{r.sensor}</td><td>{html.escape(r.food)}</td><td>{r.dose:g}u</td><td>{r.carbs:g}g</td><td>{r.pre}</td><td class='{sz}'>+{r.rise}</td><td>{r.peak_min/60:.1f}h</td><td>{'' if pd.isna(r.steps_3h) else int(r.steps_3h)}</td></tr>\n"
doc = open("report_template.html").read().replace("{{ROWS}}", rows)
open("spike-report.html", "w").write(doc)
print(len(S))
