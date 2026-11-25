import sys, re, numpy as np, pandas as pd
sys.path.insert(0,'glucose-model/v4')
from glucoforecast import load_cgm, load_watch, load_log
G='glucose-model/'
cgm=load_cgm(G+'v3/cgm_export_29-10-2026.csv')
cgm=cgm[cgm.index>='2026-08-20']
raw=pd.read_csv(G+'v3/cgm_export_29-10-2026.csv',skiprows=1)
raw['t']=pd.to_datetime(raw['Device Timestamp'],format='%d-%m-%Y %I:%M %p')
hist=raw[raw['Record Type']==0].set_index('t')['Historic Glucose mg/dL'].sort_index(); hist=hist[hist.index>='2026-08-20']
log=load_log(G+'v4/log.csv')
w=load_watch('/tmp/fms/FMS/noise hr logger')
# predictions
R=pd.read_csv(G+'report_pred_vs_actual/predicted_vs_actual_all.csv',parse_dates=['forecast_made_at','for_time'])
P=[]
for m,lab in [('A v5 · Libre + planned meal','A v5'),('A v4 · Libre + watch + food','A v4')]:
    for h in (60,120):
        d=R[(R.model==m)&(R.horizon_min==h)]
        P.append(pd.DataFrame({'model':lab,'h':h,'t0':d.forecast_made_at,'t':d.for_time,'g0':d.glucose_at_forecast,'pred':d.predicted,'act':d.actual_libre}))
for f,lab in [('moe_A_p0','MoE v6')]:
    d=pd.read_csv(G+'v6_moe/'+f+'.csv',parse_dates=['t'])
    for h in (60,120):
        x=d[d.h==h]
        P.append(pd.DataFrame({'model':lab,'h':h,'t0':x.t,'t':x.t+pd.Timedelta(minutes=h),'g0':x.g0,'pred':x.moe_soft,'act':x['true']}))
P=pd.concat(P,ignore_index=True).dropna(subset=['pred','act'])
P['err']=P.pred-P.act
print(P.groupby(['model','h']).err.agg(lambda e: np.sqrt((e**2).mean())).round(1))
print('share |err|>40:',P.groupby(['model','h']).err.apply(lambda e:(e.abs()>40).mean().round(3)))
P.to_pickle('/tmp/an/P.pkl'); log.to_pickle('/tmp/an/log.pkl'); w.to_pickle('/tmp/an/w.pkl'); hist.to_pickle('/tmp/an/hist.pkl'); cgm.to_pickle('/tmp/an/cgm.pkl')
print(log.time.min(),log.time.max(),w.index.min(),w.index.max(),len(w))
