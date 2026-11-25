import pandas as pd,numpy as np
D=pd.read_csv('/tmp/claude-0/s/doses_set2.csv',parse_dates=['time'])
D.loc[D.txt.str.contains('right side'),'site']='right'
X=pd.read_excel('set3-retrain/training_numbers_all_sets.xlsx',sheet_name='Set2_Glucometer_Nov6-12',usecols=['time','glucometer_mgdl','carbs_g'],parse_dates=['time'])
B=X.dropna(subset=['glucometer_mgdl'])
out=[]
for _,d in D[D.site!='none'].iterrows():
    pre=B[(B.time>=d.time-pd.Timedelta('30min'))&(B.time<=d.time+pd.Timedelta('10min'))]
    post=B[(B.time>=d.time+pd.Timedelta('90min'))&(B.time<=d.time+pd.Timedelta('180min'))]
    other=D[(D.time>d.time)&(D.time<d.time+pd.Timedelta('150min'))]
    carbs=X[(X.time>=d.time-pd.Timedelta('30min'))&(X.time<=d.time+pd.Timedelta('60min'))].carbs_g.sum()
    if len(pre) and len(post):
        out.append(dict(site=d.site,u=d.units,carbs=carbs,chg=post.glucometer_mgdl.iloc[0]-pre.glucometer_mgdl.iloc[-1],clean=len(other)==0))
O=pd.DataFrame(out);print(O.to_string())
O['chg_per_u_nocarb']=np.where(O.carbs<5,O.chg/O.u,np.nan)
print(O.groupby('site').agg(n=('chg','size'),mean_chg=('chg','mean'),clean=('clean','sum'),nocarb_n=('chg_per_u_nocarb','count'),per_u=('chg_per_u_nocarb','mean')).round(1))
