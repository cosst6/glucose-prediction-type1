import pandas as pd, numpy as np
P=pd.read_pickle('/tmp/an/p120/Ptag.pkl'); P['big']=P.err.abs()>40; P['up']=P.err<-40; P['dn']=P.err>40
cols=[c for c in P.columns if P[c].dtype==bool and c not in('big','up','dn')]
for (m,h),d in P.groupby(['model','h']):
  if m!='A v5': continue
  base=d.big.mean(); rows=[]
  for c in cols:
    on=d[d[c]]; off=d[~d[c]]
    if len(on)<20: continue
    rows.append((c, round(d[c].mean()*100), round(on.big.mean()*100,1), round(off.big.mean()*100,1), round(on.up.mean()*100,1), round(on.dn.mean()*100,1), round(on.err.mean(),1), round(np.sqrt((on.err**2).mean()),1)))
  print(m,h,'base miss %',round(base*100,1))
  print(pd.DataFrame(rows,columns=['tag','%time','miss% on','miss% off','under% on','over% on','bias on','rmse on']).sort_values('miss% on',ascending=False).to_string(index=False))
  print(d.groupby('tod').agg(n=('big','size'),miss=('big','mean'),rmse=('err',lambda e: np.sqrt((e**2).mean())),bias=('err','mean')).round(2))
  d2=d.copy(); d2['hr']=d2.t.dt.hour
  print(d2.groupby('hr').agg(miss=('big','mean'),rmse=('err',lambda e: np.sqrt((e**2).mean()))).round(2).T.to_string())
