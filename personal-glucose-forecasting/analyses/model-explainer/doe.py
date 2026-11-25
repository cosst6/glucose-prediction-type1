import os;os.environ.update(WSET='new',FIB='split',EXF='1')
from glucoforecast9 import *
from glucoforecast import insulin_cum
import numpy as np, pandas as pd
X=pd.read_pickle('X10.pkl')
def design(Z,h): return np.column_stack([Z[f"{e}_next{h}"].values for e in PHYS_COLS]+[np.full(len(Z),h/60),h/60*Z.tod_s.values,h/60*Z.tod_c.values,h/60*(Z.g0.values-110)/100])
Ds,ys=[],[]
for h in (60,120):
    Z=X.dropna(subset=[f'y{h}']); Ds.append(design(Z,h)); ys.append(Z[f'y{h}'].values)
D=np.vstack(Ds); y=np.concatenate(ys)
b,*_=np.linalg.lstsq(D,y,rcond=None); r=y-D@b; s=r.std()
names=['insulin/u','free sugar/g','GL/g','slow/g','FPU']
# effective sample size: rows overlap heavily (15-min steps, 2 horizons) -> inflate by ~8
INF=8
cov=np.linalg.inv(D.T@D)*s**2*INF
print('resid sd',round(s,1),'n rows',len(y))
print('OLS coef', dict(zip(names,np.round(b[:5],2))))
print('SE now  ', dict(zip(names,np.round(np.sqrt(np.diag(cov))[:5],2))))
KER={'ins':insulin_cum,'fsug':lambda m:tri_cum(m,90),'gl':lambda m:tri_cum(m,150),'slow':lambda m:tri_cum(m,240,.4),'fpu':lambda m:tri_cum(m,300,.5)}
def test_rows(comp,units,hour=13):
    rows=[]
    for t in range(-120,241,15):
        for h in (60,120):
            m=np.array([t]); vals=[]
            for k,amt in zip(['ins','fsug','gl','slow','fpu'],[units]+comp):
                vals.append(amt*(KER[k](m+h)-KER[k](m))[0])
            ang=2*np.pi*(hour+t/60)/24
            rows.append(vals+[h/60,h/60*np.sin(ang),h/60*np.cos(ang),0])
    return np.array(rows)
# comps: free sugar, GL, slow, FPU (per 100kcal)
C={
 'Glucose tabs 15 g (pure fast, no fat)':([15,0,0,0],1),
 'Oats/barley/chana 40 g carbs, low fat (pure slow)':([0,8,32,1.5],4),
 'Rajma/chana 1.5 cup (slow + protein, high fiber)':([1,10,30,2.5],4),
 'White bread / poha / idli 40 g carbs (fast starch, low fat)':([1,28,11,1.0],4),
 'Potato/rice big plate 70 g carbs (big fast load)':([1,48,21,1.5],7),
 'Pizza-style: 60 g carbs + 35 g fat (fat delays)':([3,30,27,4.5],6),
 'Whey/soy chunks 40 g protein, no carbs/fat':([0,0,2,1.7],0),
 'Ghee/nuts 30 g fat, no carbs':([0,0,2,2.9],0),
 'Same meal at 2 sizes (dal-roti half and double)':([5,22,38,2.6],5),
}
# typical meals to score: user's real log rows as 'targets' -> use average of current design rows
base=np.trace(D[:,:5]@cov[:5,:5]@D[:,:5].T)/len(D)
print('coef-uncertainty part of forecast variance now: sd',round(np.sqrt(base),2),'mg/dL (vs total resid sd',round(s,1),')')
out=[]
for name,(comp,u) in C.items():
    T=test_rows(comp,u); Tm=np.vstack([T]*4)  # 4 tests
    c2=np.linalg.inv(D.T@D/INF + Tm.T@Tm/2)*s**2   # tests: less overlap? keep inflation 2
    se=np.sqrt(np.diag(c2))[:5]
    # uncertainty in forecasting this new food itself, before vs after
    def fu(c): return np.sqrt(np.mean(np.einsum('ij,jk,ik->i',T[:,:5],c[:5,:5],T[:,:5])))
    out.append({'food':name,**{f'SE {n}':round(v,2) for n,v in zip(names,se)},'own-forecast sd before':round(fu(cov),1),'after':round(fu(c2),1),
                'typical-meal sd after':round(np.sqrt(np.trace(D[:,:5]@c2[:5,:5]@D[:,:5].T)/len(D)),2)})
pd.set_option('display.width',250); print(pd.DataFrame(out).to_string())
