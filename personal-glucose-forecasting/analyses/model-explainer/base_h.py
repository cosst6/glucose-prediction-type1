import os;os.environ.update(WSET='new',FIB='split',EXF='1')
from glucoforecast9 import *
from scipy.optimize import lsq_linear
X=pd.read_pickle('X10.pkl'); Ds,ys,dd,hz=[],[],[],[]
for h in (60,120):
    Z=X.dropna(subset=[f'y{h}']); Ds.append(np.column_stack([Z[f'{e}_next{h}'].values for e in PHYS_COLS]+[np.full(len(Z),h/60),h/60*Z.tod_s.values,h/60*Z.tod_c.values,h/60*(Z.g0.values-110)/100])); ys.append(Z[f'y{h}'].values); dd.append(Z.index.normalize().values); hz.append(np.full(len(Z),h))
D,y,dd,hz=map(np.concatenate,(Ds,ys,dd,hz)) if False else (np.vstack(Ds),np.concatenate(ys),np.concatenate(dd),np.concatenate(hz))
lo=[-np.inf,0,0,0,0,-np.inf,-np.inf,-np.inf,-np.inf]; hi=[0,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,0]
for h in (60,120):
    e=[]
    for d in np.unique(dd):
        tr=dd!=d; te=(~tr)&(hz==h); b=lsq_linear(D[tr],y[tr],bounds=(lo,hi)).x; e.append(y[te]-D[te]@b)
    print(h,round(np.sqrt((np.concatenate(e)**2).mean()),1))
