import re, numpy as np, pandas as pd
P=pd.read_pickle('/tmp/an/P_p120.pkl'); log=pd.read_pickle('/tmp/an/log.pkl'); w=pd.read_pickle('/tmp/an/w.pkl'); hist=pd.read_pickle('/tmp/an/hist.pkl'); cgm=pd.read_pickle('/tmp/an/cgm.pkl')
fs=pd.read_csv('glucose-model/data/fingerstick.csv',parse_dates=['time'])
FOOD=[('milk',r'milk|dahi|curd'),('dal/moong',r'moong|dal|besan|soy|chana|peas'),('fruit',r'banana|apple|guava|pomegranate|grape|rape'),('roti/rice',r'roti|rice|bread|poha'),('sweet',r'candy|sweet|sugar'),('egg/paneer/veg',r'egg|paneer|veg|gourd|okra|bhindi|cabbage|sauce|almond|cucumber|pumpkin|spinach')]
def classes(f):
    if not isinstance(f,str): return []
    f=f.lower(); return [k for k,p in FOOD if re.search(p,f)]
log['cls']=log.food.apply(classes)
food=log[log.food.notna()]
TH=40
M=P[P.err.abs()>TH].copy()
M['sign']=np.sign(M.err)
M=M.sort_values('t')
eps=[];cur=None
for r in M.itertuples():
    if cur is not None and r.sign==cur['sign'] and r.t-cur['end']<=pd.Timedelta('45min'):
        cur['rows'].append(r); cur['end']=r.t
    else:
        if cur: eps.append(cur)
        cur={'sign':r.sign,'end':r.t,'rows':[r]}
eps.append(cur)
def win(df,col,a,b): return df[(df[col]>a)&(df[col]<=b)]
out=[]
for e in eps:
    R=pd.DataFrame(e['rows']); pk=R.loc[R.err.abs().idxmax()]
    t0=R.t0.min(); t1=R.t.max(); tp=pk.t
    tags=[]; detail=[]
    under=e['sign']<0   # actual above forecast (missed rise / missed high)
    fa=win(food,'time',t0,t1)            # food eaten after forecast made
    fb=win(food,'time',t0-pd.Timedelta('4h'),t0)  # food still digesting
    fcls=sorted({c for l in pd.concat([fa,fb]).cls for c in l})
    if len(fa): tags.append('food after forecast'); detail.append('ate '+' + '.join(sorted({c for l in fa.cls for c in l}))+' at '+', '.join(fa.time.dt.strftime('%H:%M')))
    if len(fb): tags.append('food before forecast'); detail.append('ate '+' + '.join(sorted({c for l in fb.cls for c in l}))+' at '+', '.join(fb.time.dt.strftime('%H:%M')))
    for c in fcls: tags.append('food: '+c)
    if under and not len(fa) and not len(fb): tags.append('rise, no food logged')
    ra=win(log[log.rapid_u>0],'time',t0-pd.Timedelta('3h'),t1)
    if len(ra):
        corr=[x for x in ra.itertuples() if not ((food.time-x.time).abs()<=pd.Timedelta('30min')).any()]
        tags.append('NovoRapid in window')
        if corr: tags.append('correction dose (no food)'); detail.append('correction '+', '.join(f'{x.rapid_u:g}u {x.time:%H:%M}' for x in corr))
    la=win(log[log.lantus_u>0],'time',t0-pd.Timedelta('4h'),t1)
    if len(la): tags.append('Lantus nearby'); detail.append('Lantus '+', '.join(la.time.dt.strftime('%H:%M')))
    h=tp.hour+tp.minute/60
    tod='night 22-3' if (h>=22 or h<3) else 'dawn 3-8' if h<8 else 'morning 8-12' if h<12 else 'afternoon 12-15' if h<15 else 'evening 15-22'
    lo=cgm.g[(cgm.index>=t0-pd.Timedelta('3h'))&(cgm.index<=t0+pd.Timedelta('15min'))]
    if under and len(lo) and lo.min()<70: tags.append('low rebound'); detail.append(f'low {lo.min():.0f} before')
    if not under and pk.act<80: tags.append('missed low')
    # sensor
    hw=hist[(hist.index>=t0-pd.Timedelta('1h'))&(hist.index<=t1)]
    gaps=hw.index.to_series().diff().max() if len(hw)>1 else pd.Timedelta('9h')
    sen=[]
    if gaps>pd.Timedelta('25min'): sen.append(f'gap {gaps}')
    if t0<pd.Timestamp('2026-08-28 17:00'): sen.append('sensor day 1')
    g=cgm.g[(cgm.index>=t0-pd.Timedelta('30min'))&(cgm.index<=t1+pd.Timedelta('1h'))]
    if (h>=22 or h<8) and len(g)>4:
        i=g.values; j=np.argmin(i)
        if i[j]<80 and j>0 and j<len(i)-1 and i[:j].max()-i[j]>=30 and i[j:].max()-i[j]>=30 and (g.index[j:][np.argmax(i[j:])]-g.index[j])<=pd.Timedelta('75min'): sen.append(f'V-dip to {i[j]:.0f} (possible compression)')
    if sen: tags.append('sensor issue'); detail+=sen
    ww=w[(w.index>t0-pd.Timedelta('1h'))&(w.index<=t1)]
    act=None
    if len(ww)<20: tags.append('watch off')
    else:
        st=ww.steps.sum(); hrm=ww.hr.mean(); act=st
        if st>=1500 or ww.hr.rolling(20).mean().max()>110: tags.append('activity'); detail.append(f'{st:.0f} steps, max HR {ww.hr.max():.0f}')
    out.append(dict(start=t0,peak=tp,end=t1,dir='actual HIGHER than forecast' if under else 'actual LOWER than forecast',
        peak_err=pk.err,model=pk.model,h=pk.h,pred=pk.pred,act=pk.act,g0=pk.g0,n=len(R),
        models=','.join(sorted({f'{m} {h//60}h' for m,h in zip(R.model,R.h)})),tod=tod,tags=tags,detail='; '.join(detail)))
E=pd.DataFrame(out); E['abs']=E.peak_err.abs()
# main cause
def main(r):
    T=r.tags
    if 'sensor issue' in T: return 'sensor issue'
    if 'low rebound' in T: return 'low rebound'
    if 'missed low' in T and 'correction dose (no food)' in T: return 'insulin correction -> low'
    if 'food after forecast' in T: return 'food after forecast'
    if 'rise, no food logged' in T: return 'rise, no food logged'
    if 'correction dose (no food)' in T: return 'insulin correction'
    if 'food before forecast' in T: return 'food still digesting'
    if 'missed low' in T: return 'missed low (other)'
    return 'other'
E['main']=E.apply(main,axis=1)
E.to_pickle('/tmp/an/p120/E.pkl')
print(len(E),'episodes'); print(E.dir.value_counts())
print(E.main.value_counts())
from collections import Counter
c=Counter(t for l in E.tags for t in l); print(pd.Series(c).sort_values(ascending=False)/len(E))
print(E.tod.value_counts())
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',90)
print(E.sort_values('abs',ascending=False).head(25)[['peak','peak_err','models','g0','pred','act','tod','main','detail']])
