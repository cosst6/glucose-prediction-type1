import pandas as pd, numpy as np
E=pd.read_pickle('/tmp/an/p120/E.pkl'); P=pd.read_pickle('/tmp/an/p120/Ptag.pkl')
key=P.set_index(['model','h','t'])
def main(r):
    T=key.loc[(r.model,r.h,r.peak)]
    if isinstance(T,pd.DataFrame): T=T.iloc[0]
    up=r.peak_err<0
    if T['sensor gap']: m='Sensor gap'
    elif up and T['low (<70) in 3h before']: m='Bounce after a low'
    elif (not up) and T['correction dose']: m='Correction insulin dropped faster'
    elif T['food after forecast']: m='Meal inside forecast window'
    elif T['food in 3h before']: m='Earlier meal peaked late'
    elif not up: m='Drop, no clear cause'
    else: m='Rise, no clear cause'
    foods=[c.split(': ')[1] for c in T.index if c.startswith('food: ') and T[c]]
    return pd.Series({'main':m,'foods':', '.join(foods),'milk':T['food: milk'],'lantus':T['Lantus within 4h'],'active':T['active (>=1500 steps)'],'corr':T['correction dose'],'watchoff':T['watch off'],'tod2':T['tod']})
E=pd.concat([E.drop(columns=['main']),E.apply(main,axis=1)],axis=1)
print(E.main.value_counts(), (E.main.value_counts(normalize=True)*100).round())
print(E.tod2.value_counts(normalize=True).round(2))
print('share with milk',E.milk.mean().round(2),'active',E.active.mean().round(2),'corr',E['corr'].mean().round(2),'lantus',E.lantus.mean().round(2))
print(pd.crosstab(E.main,E.dir))
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',120)
T=E.sort_values('abs',ascending=False).head(15)
print(T[['peak','models','g0','pred','act','peak_err','main','foods','detail']].to_string())
E.to_pickle('/tmp/an/p120/E2.pkl')
