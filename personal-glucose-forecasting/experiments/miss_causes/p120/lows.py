import pandas as pd, numpy as np
exec(open('/tmp/an/ep.py').read().split('TH=40')[0])
TD=pd.Timedelta; g=cgm.g[cgm.index>='2026-08-28']
low=g<70; st=g.index[low & ~low.shift(1,fill_value=False)]
rows=[]
for t in st:
    pre=log[(log.time>t-TD('4h'))&(log.time<=t)]
    ins=pre.rapid_u.fillna(0).sum(); nd=(pre.rapid_u>0).sum()
    carbs=pre.carbs.sum()
    corr=[x for x in pre[pre.rapid_u>0].itertuples() if not ((food.time-x.time).abs()<=TD('30min')).any()]
    lan=log[(log.lantus_u>0)&(log.time>t-TD('8h'))&(log.time<=t)]
    ww=w[(w.index>t-TD('2h'))&(w.index<=t)]
    mn=g[(g.index>=t)&(g.index<=t+TD('2h'))].min()
    rows.append(dict(t=t,hr=t.hour,min=mn,ins4h=ins,doses=nd,corr=len(corr),carbs4h=round(carbs),lantus8h=len(lan)>0,steps2h=ww.steps.sum() if len(ww)>20 else np.nan,
        log='; '.join(f"{r.time:%H:%M} {str(int(r.rapid_u))+'u ' if r.rapid_u>0 else ''}{r.food[:40] if isinstance(r.food,str) else ''}" for r in pre.itertuples())))
L=pd.DataFrame(rows); pd.set_option('display.width',300); pd.set_option('display.max_colwidth',150)
print(L.to_string()); print(len(L))
L['type']=np.select([L.lantus8h&(L.doses==0),L['corr']>0,L.doses>=2,L.ins4h>0],['night, Lantus only','after correction dose','2+ doses stacked','meal dose'],'no insulin 4h')
print(L.type.value_counts()); print('active>=1500 steps 2h',(L.steps2h>=1500).mean())
L.to_csv('glucose-model/miss_causes/lows_and_triggers.csv',index=False)
