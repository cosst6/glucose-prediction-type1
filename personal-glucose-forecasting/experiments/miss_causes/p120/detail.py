import pandas as pd, numpy as np
exec(open('/tmp/an/ep.py').read().split('TH=40')[0])
TD=pd.Timedelta
E=pd.read_pickle('/tmp/an/p120/E2.pkl'); P=pd.read_pickle('/tmp/an/p120/Ptag.pkl')
pd.set_option('display.width',300); pd.set_option('display.max_colwidth',200)
def foods(a,b):
    f=log[(log.time>a)&(log.time<=b)]
    return '; '.join(f"{r.time:%H:%M} {(str(int(r.rapid_u))+'u ') if r.rapid_u>0 else ''}{r.food if isinstance(r.food,str) else ''}{' Lantus' if r.lantus_u>0 else ''}" for r in f.itertuples())
for m in E.main.unique():
    print('=====',m)
    for r in E[E.main==m].sort_values('peak').itertuples():
        pk=P[(P.h==r.h)&(P.t==r.peak)].iloc[0]; t0=pk.t0
        g=cgm.g
        ww=w[(w.index>t0-TD('1h'))&(w.index<=r.peak)]
        lo=g[(g.index>=t0-TD('3h'))&(g.index<=r.peak)]
        print(f"{r.peak:%d %b %H:%M} h{r.h} fc@{t0:%H:%M} g0 {r.g0:.0f} pred {r.pred:.0f} act {r.act:.0f} | min3h {lo.min():.0f}@{lo.idxmin():%H:%M} | steps {ww.steps.sum():.0f} | log(-4h..peak): {foods(t0-TD('4h'),r.peak)}")
