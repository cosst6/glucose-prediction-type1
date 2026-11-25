import os;os.environ.update(WSET='new',FIB='split',EXF='1')
from glucoforecast9 import *
from glucoforecast import insulin_cum
from parse_meal import parse_item
from scipy.optimize import lsq_linear
import numpy as np, pandas as pd, re, itertools
X=pd.read_pickle('X10.pkl'); L=pd.read_csv('/home/user/work/D/log.csv',parse_dates=['time']).sort_values('time')
FAM={'milk':('milk','paneer_from_milk'),'fruit':('apple','banana','guava','pomegranate','grape','melody','explicit_carbs'),
     'starch':('rice','roti'),'legume':('moong','dal','peas','besan','soy')}
rows=[]
for _,r in L.iterrows():
    d={'time':r.time,'milk':0,'fruit':0,'starch':0,'legume':0,'veg':0}
    if isinstance(r.food,str):
        for p in re.split(r'[;\n]',r.food):
            it=parse_item(p)
            if not it: continue
            av=max(it['carbs']-it['fiber'],0); f='veg'
            for k,v in FAM.items():
                if any(it['food'].startswith(x) for x in v): f=k
            if it['food'] in ('paneer','curd','egg','almond','fish_oil'): f='veg'
            d[f]+=av
    rows.append(d)
F=pd.DataFrame(rows)
tt=X.index.values.astype('datetime64[m]').astype(np.int64)[:,None].astype(float)
et=F.time.values.astype('datetime64[m]').astype(np.int64)[None,:].astype(float); m=tt-et; known=m>=-120
def nxt(col,cum,h):
    a=F[col].values[None,:]*known; return (a*(cum(m+h)-cum(m))).sum(1)
SHAPES={'milk':[(120,1/3),(150,1/3),(180,.4)],'fruit':[(90,1/3),(120,1/3),(150,1/3)],'starch':[(150,1/3),(180,1/3),(240,.4)],
        'legume':[(240,.4),(300,.5),(360,.5)],'veg':[(180,1/3),(240,.4),(300,.5)]}
def build(shapes,fams):
    Ds,ys,days=[],[],[]
    for h in (60,120):
        ok=X[f'y{h}'].notna().values; Z=X[ok]
        cols=[Z[f'ins_next{h}'].values]+[nxt(f,lambda q,d=shapes[f][0],p=shapes[f][1]:tri_cum(q,d,p),h)[ok] for f in fams]+[Z[f'fpu_next{h}'].values,
              np.full(len(Z),h/60),h/60*Z.tod_s.values,h/60*Z.tod_c.values,h/60*(Z.g0.values-110)/100]
        Ds.append(np.column_stack(cols)); ys.append(Z[f'y{h}'].values); days.append(Z.index.normalize().values)
    return np.vstack(Ds),np.concatenate(ys),np.concatenate(days)
def cv(D,y,days,nf):
    lo=[-np.inf]+[0]*(nf+1)+[-np.inf]*3+[-np.inf]; hi=[0]+[np.inf]*(nf+1)+[np.inf]*3+[0]
    e=[]
    for d in np.unique(days):
        tr=days!=d; b=lsq_linear(D[tr],y[tr],bounds=(lo,hi)).x; e.append(y[~tr]-D[~tr]@b)
    e=np.concatenate(e); return np.sqrt((e**2).mean()), lsq_linear(D,y,bounds=(lo,hi)).x
# baseline current body model, same CV
def base():
    Ds,ys,days=[],[],[]
    for h in (60,120):
        Z=X.dropna(subset=[f'y{h}']); Ds.append(np.column_stack([Z[f'{e}_next{h}'].values for e in PHYS_COLS]+[np.full(len(Z),h/60),h/60*Z.tod_s.values,h/60*Z.tod_c.values,h/60*(Z.g0.values-110)/100])); ys.append(Z[f'y{h}'].values); days.append(Z.index.normalize().values)
    D,y,dd=np.vstack(Ds),np.concatenate(ys),np.concatenate(days)
    lo=[-np.inf,0,0,0,0,-np.inf,-np.inf,-np.inf,-np.inf]; hi=[0,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,0]
    e=[]
    for d in np.unique(dd):
        tr=dd!=d; b=lsq_linear(D[tr],y[tr],bounds=(lo,hi)).x; e.append(y[~tr]-D[~tr]@b)
    return np.sqrt((np.concatenate(e)**2).mean())
print('current 5-channel body model, leave-one-day-out RMSE (60+120 pooled):',round(base(),2))
fams=['milk','fruit','starch','legume','veg']
lit={'milk':(120,1/3),'fruit':(90,1/3),'starch':(180,1/3),'legume':(300,.5),'veg':(240,.4)}
D,y,dd=build(lit,fams); r,b=cv(D,y,dd,len(fams)); print('food-family curves, literature timing:',round(r,2),dict(zip(['ins']+fams+['fpu'],np.round(b[:7],2))))
best=(9e9,None)
for f in fams:  # coordinate search per family (in-sample choice -> optimistic)
    for s in SHAPES[f]:
        sh=dict(lit); sh[f]=s; D,y,dd=build(sh,fams); r,_=cv(D,y,dd,len(fams))
        if r<best[0]: best=(r,sh)
print('best single tweak:',round(best[0],2),best[1])
print('carb grams logged by family:',F[fams].sum().round(0).to_dict(), ' meals with >5g:',(F[fams]>5).sum().to_dict())

# ---- per horizon CV & coefficient SE for family model
lo=[-np.inf]+[0]*6+[-np.inf]*4; hi=[0]+[np.inf]*6+[np.inf]*3+[0]
for h in (60,120):
    Dh,yh,dh=build(lit,fams); n=len(yh)//2  # rows ordered 60 then 120 (equal? not exactly) -> recompute
D,y,dd=build(lit,fams)
n60=X['y60'].notna().sum(); hz=np.r_[np.full(n60,60),np.full(len(y)-n60,120)]
for h in (60,120):
    e=[];eb=[]
    for d in np.unique(dd):
        tr=dd!=d; te=(~tr)&(hz==h); b=lsq_linear(D[tr],y[tr],bounds=(lo,hi)).x; e.append(y[te]-D[te]@b)
    print('family model CV RMSE h',h,round(np.sqrt((np.concatenate(e)**2).mean()),1))
b,*_=np.linalg.lstsq(D,y,rcond=None); s=(y-D@b).std(); INF=8
cov=np.linalg.inv(D.T@D)*s**2*INF; names=['ins']+fams+['fpu']
print('OLS',dict(zip(names,np.round(b[:7],2)))); print('SE ',dict(zip(names,np.round(np.sqrt(np.diag(cov))[:7],2))))
# candidate tests: (family carbs dict, fpu, units)
KER={'ins':insulin_cum,**{f:(lambda q,d=lit[f][0],p=lit[f][1]:tri_cum(q,d,p)) for f in fams},'fpu':lambda q:tri_cum(q,300,.5)}
def rows(comp,fpu,u,hour=13):
    R=[]
    for t in range(-120,241,15):
        for h in (60,120):
            mm=np.array([t]); v=[u*(KER['ins'](mm+h)-KER['ins'](mm))[0]]+[comp.get(f,0)*(KER[f](mm+h)-KER[f](mm))[0] for f in fams]+[fpu*(KER['fpu'](mm+h)-KER['fpu'](mm))[0]]
            a=2*np.pi*(hour+t/60)/24; R.append(v+[h/60,h/60*np.sin(a),h/60*np.cos(a),0])
    return np.array(R)
TESTS={'Rice/roti/poha/idli 50 g carbs':({'starch':50},1.0,5),'Dal/chana/rajma 40 g carbs':({'legume':40},2.0,4),
       'Milk alone 250 ml':({'milk':12},1.7,1),'Fruit alone 20 g carbs':({'fruit':20},0,2),'Paneer/eggs only':({},4,0)}
for name,(c,f,u) in TESTS.items():
    T=rows(c,f,u)
    fu=lambda C: np.sqrt(np.mean(np.einsum('ij,jk,ik->i',T[:,:7],C[:7,:7],T[:,:7])))
    res=[name,'now ±%.1f'%fu(cov)]
    for k in (2,4,8):
        c2=np.linalg.inv(D.T@D/INF+np.vstack([T]*k).T@np.vstack([T]*k)/2)*s**2; res.append(f'{k} tests ±%.1f'%fu(c2))
    se=np.sqrt(np.diag(np.linalg.inv(D.T@D/INF+np.vstack([T]*4).T@np.vstack([T]*4)/2)*s**2))[:7]
    print(res,'SE after 4:',dict(zip(names,np.round(se,2))))
