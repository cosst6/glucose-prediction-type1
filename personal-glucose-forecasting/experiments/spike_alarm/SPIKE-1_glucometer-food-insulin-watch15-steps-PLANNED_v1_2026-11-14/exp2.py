exec(open('exp.py').read().split('rows = []')[0])
T["ratio"] = T.avail / (T.dose + 0.5); T["sug_share"] = T.sugar / (T.avail + 1); T["fp"] = (9 * T.fat + 4 * T.protein) / 100
T["lratio"] = np.log1p(T.ratio)
SETS = {
 "uncov,iob,steps,hr": ["uncov", "iob", "steps60", "hr15"],
 "lratio,iob,steps,hr": ["lratio", "iob", "steps60", "hr15"],
 "uncov,sugshare,iob,steps,hr": ["uncov", "sug_share", "iob", "steps60", "hr15"],
 "uncov,sugar,starch,fp,iob,steps,hr": ["uncov", "sugar", "starch", "fp", "iob", "steps60", "hr15"],
 "uncov,milk,night,iob,steps,hr": ["uncov", "milk", "night", "iob", "steps60", "hr15"],
 "lratio,sugar,starch,fiber,fp,iob,steps,hr": ["lratio", "sugar", "starch", "fiber", "fp", "iob", "steps60", "hr15"],
}
rows = []
for name, cols in SETS.items():
    for st in (False, True):
        for C in (0.1, 0.3, 1.0):
            for gw in (1, 3):
                P = lodo(cols + (["prec", "low"] if st else []), C, gw)
                r = dict(set=name, stick=st, C=C, gw=gw)
                for src in ("Libre", "Glucometer"):
                    s = score(P, (T.src == src).values); r.update({f"{src[0]}_{k}": v for k, v in s.items() if k != "n"})
                s = score(P); r.update(all_acc=s["band_acc"], all_spk=s["spike_caught"], all_fa=s["false_alarm"], all_big=s["big_caught"])
                rows.append(r)
R = pd.DataFrame(rows); pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30); print(R.sort_values("all_acc", ascending=False).head(30).to_string(index=False))
