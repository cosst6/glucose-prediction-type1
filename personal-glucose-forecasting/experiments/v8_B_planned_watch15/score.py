import pandas as pd,numpy as np
F={'v8_full(new watch)':'r_new/preds7_new_B_p120.csv','v5style(old watch)':'r_old/preds7_old_B_p120.csv','v5style no watch':'r_none/preds7_none_B_p120.csv',
 'lean+new watch':'r_lean/preds7_lean_B_p120.csv','v4 planned (old watch)':'v4p/preds4_B.csv','v4 planned + new watch':'v4p7/preds4_B.csv'}
for h in (60,120):
  D={k:pd.read_csv(v,index_col=0,parse_dates=True).query('h==@h') for k,v in F.items()}
  idx=set.intersection(*[set(d.index) for d in D.values()]); idx=sorted(idx)
  for k,d in D.items():
    d=d.loc[~d.index.duplicated()].loc[idx]; e=d.hybrid-d.true; big=(d.true-d.persist).abs()>=40
    days=d.index.normalize()
    print(h,f'{k:24s}',len(d),'rmse',np.sqrt((e**2).mean()).round(1),'big',np.sqrt((e[big]**2).mean()).round(1),'persist',np.sqrt(((d.persist-d.true)**2).mean()).round(1))
