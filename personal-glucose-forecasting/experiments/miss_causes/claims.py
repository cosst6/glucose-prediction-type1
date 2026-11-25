import pandas as pd, numpy as np
exec(open('/tmp/an/ep.py').read().split('TH=40')[0])
TD=pd.Timedelta; g=cgm.g
lan=log[log.lantus_u>0].time
def curve(t,hrs=5):
    x=g[(g.index>=t-TD('15min'))&(g.index<=t+TD(f'{hrs}h'))]
    if len(x)<8: return None
    b=x[x.index<=t+TD('15min')]
    if len(b)==0: return None
    b=b.iloc[0]; pk=x.idxmax(); return b, x.max()-b, (pk-t).total_seconds()/3600, x.min()-b
print('--- milk-only events (no other carbs logged within 2h)')
rows=[]
for r in food.itertuples():
    if r.cls!=['milk']: continue
    other=food[(food.time-r.time).abs().between(TD('1min'),TD('2h'))]
    if len(other): continue
    c=curve(r.time,4)
    if c is None: continue
    nearL=((lan-r.time).abs()<=TD('90min')).any()
    rows.append((r.time,r.food[:30],r.rapid_u,r.carbs,c[0],round(c[1]),round(c[2],2),nearL))
D=pd.DataFrame(rows,columns=['t','food','u','carbs','start','rise','peak_h','nearLantus'])
D['night']=D.t.dt.hour.isin([22,23,0,1,2,3,4])
print(D.to_string()); print(D.groupby('nearLantus')[['rise','peak_h']].median())
print('--- dal events')
for r in food.itertuples():
    if 'dal/moong' in r.cls:
        c=curve(r.time,6); print(r.time,r.food[:60],r.rapid_u,c and (c[0],round(c[1]),round(c[2],2)))
print('--- all meals peak timing by class (meals with carbs>=10)')
rows=[]
for r in food.itertuples():
    c=curve(r.time,5)
    if c is None: continue
    rows.append(('+'.join(r.cls),r.carbs,r.rapid_u,round(c[1]),round(c[2],2)))
X=pd.DataFrame(rows,columns=['cls','carbs','u','rise','peak_h'])
print(X.groupby('cls').agg(n=('rise','size'),rise=('rise','median'),peak_h=('peak_h','median')).sort_values('n',ascending=False).head(15))
print('--- Lantus nights: rise in 4h after, with vs without food within 90 min')
for t in lan:
    f=food[(food.time-t).abs()<=TD('90min')]
    c=curve(t,4); print(t, '+'.join(sorted({x for l in f.cls for x in l})) or 'NO FOOD', c and (c[0],round(c[1]),round(c[2],1)))
