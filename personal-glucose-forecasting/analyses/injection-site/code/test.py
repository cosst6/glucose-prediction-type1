import pandas as pd,numpy as np
D=pd.read_csv('/tmp/claude-0/s/doses_set2.csv',parse_dates=['time'])
D.loc[D.txt.str.contains('right side'),'site']='right'
R=pd.read_csv('glucose-model/v14_food_family/GLU-B_glucometer-watch15-steps-cal-PLANNED2H-FOODFAMILY_v14_2026-11-21/test_results.csv',index_col=0,parse_dates=True)
R=R[R.test=='set2 glucometer'].copy()
for s in ('left','right','none'):
    R[s]=[D[(D.site==s)&(D.time>t-pd.Timedelta('3h'))&(D.time<=t+pd.Timedelta(minutes=h))].units.sum() for t,h in zip(R.index,R.h)]
tot=R[['left','right','none']].sum(1)
R['grp']=np.where(tot==0,'no insulin',np.where(R.none>0,'some unlogged',np.where(R.right==0,'left only',np.where(R.left==0,'right only','both'))))
r=lambda e: round(float(np.sqrt(np.mean(np.square(e)))),1)
for m in ('v10live','fam'):
    R['e_'+m]=R[m]-R.true
print(R.groupby(['h','grp']).agg(n=('true','size'),bias_v10=('e_v10live','mean'),bias_v14=('e_fam','mean'),rmse_v10=('e_v10live',r),rmse_v14=('e_fam',r)).round(1))
# fair LOO-day: residual ~ a*left + b*right (units in window), learned on other days, per horizon
R['day']=R.index.normalize()
for m in ('v10live','fam'):
  for h in (60,120):
    G=R[R.h==h]; adj=[]
    for d in G.day.unique():
        tr=G[G.day!=d]; te=G[G.day==d]
        X=tr[['left','right']].values; y=tr['e_'+m].values
        lam=1.0; w=np.linalg.solve(X.T@X+lam*np.eye(2),X.T@y)
        adj.append(pd.Series(te['e_'+m].values-te[['left','right']].values@w,index=te.index))
    adj=pd.concat(adj)
    print(m,h,'n',len(G),'without',r(G['e_'+m]),'with site',r(adj))
# per-unit drop by site: residual per unit
for h in (60,120):
    G=R[(R.h==h)&(R.none==0)&((R.left+R.right)>0)]
    print(h,'left-only n',(G.grp=='left only').sum(),'right-only n',(G.grp=='right only').sum())
