import sys, glob, numpy as np, pandas as pd
from glucoforecast import *
D=sys.argv[1]
cgm=load_cgm(glob.glob(f"{D}/*glucose*.csv")[0]); cgm=cgm[cgm.index>="2026-08-01"]; log=load_log(f"{D}/log.csv")
X=build_frame(cgm,log,None,HORIZONS_C); X=X[X.index.normalize().isin(list(set(log.time.dt.normalize())))]
days=[d for d in sorted(X.index.normalize().unique()) if d>=pd.Timestamp("2026-08-30")]
for h in HORIZONS_C:
    cols=feature_list("C",h); y=f"y{h}"; out=[]
    for day in days:
        te=X[X.index.normalize()==day].dropna(subset=cols+[y]); tr=X[X.index<day-pd.Timedelta(minutes=h)].dropna(subset=cols+[y])
        if len(te)==0: continue
        m=make_pipeline(StandardScaler(),Ridge(alpha=30)).fit(tr[cols],tr[y])
        out.append(pd.DataFrame({"true":te.g0+te[y],"pred":te.g0+m.predict(te[cols]),"persist":te.g0},index=te.index))
    pd.concat(out).to_csv(f"preds_C_{h}.csv")
