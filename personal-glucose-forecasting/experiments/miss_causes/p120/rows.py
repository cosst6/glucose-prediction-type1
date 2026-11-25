import re, numpy as np, pandas as pd
exec(open('/tmp/an/ep.py').read().split('TH=40')[0])
TD=pd.Timedelta
ft=food.time.values; fcls=list(food.cls)
rap=log[log.rapid_u>0]; corr_t=[x.time for x in rap.itertuples() if not ((food.time-x.time).abs()<=TD('30min')).any()]
corr_t=pd.Series(corr_t); lan=log[log.lantus_u>0].time
hs=hist.index.to_series(); wst=w.steps.fillna(0).cumsum(); 
def tag(r):
    t0,t=r.t0,r.t; T={}
    fa=food[(food.time>t0)&(food.time<=t)]; fb=food[(food.time>t0-TD('3h'))&(food.time<=t0)]
    T['food after forecast']=len(fa)>0; T['food in 3h before']=len(fb)>0
    both=pd.concat([fa,fb]); cl={c for l in both.cls for c in l}
    for c,_ in FOOD: T['food: '+c]=c in cl
    T['milk after forecast']=any('milk' in l for l in fa.cls)
    T['no food in 4h']=not ((food.time>t0-TD('4h'))&(food.time<=t)).any()
    T['correction dose']=((corr_t>t0-TD('3h'))&(corr_t<=t)).any()
    T['Lantus within 4h']=((lan>t0-TD('4h'))&(lan<=t)).any()
    lo=cgm.g[(cgm.index>=t0-TD('3h'))&(cgm.index<=t0)]
    T['low (<70) in 3h before']=len(lo)>0 and lo.min()<70
    h=t.hour
    T['tod']='night 22-3' if (h>=22 or h<3) else 'dawn 3-8' if h<8 else 'morning 8-12' if h<12 else 'afternoon 12-15' if h<15 else 'evening 15-22'
    hw=hs[(hs.index>=t0-TD('1h'))&(hs.index<=t)]
    T['sensor gap']=len(hw)<2 or hw.diff().max()>TD('25min')
    ww=w[(w.index>t0-TD('1h'))&(w.index<=t)]
    T['watch off']=len(ww)<20
    T['active (>=1500 steps)']=len(ww)>=20 and ww.steps.sum()>=1500
    T['rising fast at forecast']=False
    g=cgm.g; prev=g.get(t0-TD('30min'))
    if prev is not None and not np.isnan(prev): T['rising fast at forecast']=(r.g0-prev)>=20; T['falling fast at forecast']=(r.g0-prev)<=-20
    else: T['falling fast at forecast']=False
    return T
P=pd.read_pickle('/tmp/an/P_p120.pkl')
TT=pd.DataFrame([tag(r) for r in P.itertuples()])
P=pd.concat([P,TT],axis=1); P.to_pickle('/tmp/an/p120/Ptag.pkl'); print('ok',len(P))
