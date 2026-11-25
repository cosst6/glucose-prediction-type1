import sys, json, joblib, numpy as np, pandas as pd; from nightmodel import load; from phys import *
NAME,OUT=sys.argv[1],sys.argv[2]; D=load()
# honest per-hour spread from leave-one-night-out errors
E={h:[] for h in H}
for i in D.index:
    te=D.loc[i]; M=fit(D.drop(i)); p,_=predict(M,te,70)
    for h in H:
        if pd.notna(te[f"g{h}"]): E[h].append(p[h-1]-te[f"g{h}"])
sd={h:float(np.sqrt(np.mean(np.square(E[h])))) for h in H}
M=fit(D); M={h:(M[h][0],max(sd[h],20.0)) for h in H}
model={"name":NAME,"scale":"glucometer (blood) mg/dL; Libre training data divided by 0.72","hours":list(H),"features":COLS,
 "feature_rules":{"gdev":"bedtime glucometer reading - 120","iob":"NovoRapid units not yet absorbed at bedtime incl. doses planned up to 90 min after bedtime (same insulin curve as other models: peak 75 min, 300 min), capped 5.5",
  "carbs":"net carbs (carbs-fiber) g eaten in last 4 h + planned up to 90 min after bedtime, cap 80","milk":"ml milk from 60 min before to 90 min after bedtime / 250, cap 2",
  "fpu":"(9*fat+4*protein)/100 g over last 4 h + planned 90 min, cap 8","hr":"(HR mean last 15 min - 70)/10, clipped -1.5..2.5; 0 if no watch","steps":"ln(1 + raw steps total last 3 h)"},
 "coef":{h:[round(float(v),4) for v in M[h][0]] for h in H},"sd":{h:round(M[h][1],1) for h in H},
 "low_rule":"P_h = normalCDF((70 - pred_h)/sd_h); night low if max P_h > 0.3; range = first hour P>0.3 minus 30 min to first hour P>0.7 (1-3 h wide)",
 "wake":"mean of hours 8 and 9; 80% band = +/-1.28*sd"}
json.dump(model,open(f"{OUT}/{NAME}.json","w"),indent=1); joblib.dump(model,f"{OUT}/{NAME}.joblib")
print(model["sd"]); print({h:model["coef"][h] for h in (3,6,8)})
# golden cases for app port
G=[]
for case in [dict(g_bed=110,iob=3,carbs4h=10,fpu=4,milk_bed=0,hr15=65,steps3h=200),dict(g_bed=110,iob=1,carbs4h=10,fpu=4,milk_bed=250,hr15=65,steps3h=200),dict(g_bed=180,iob=2,carbs4h=30,fpu=6,milk_bed=0,hr15=70,steps3h=1500)]:
    s=pd.Series(case,dtype=float); s["steps3h"]=np.log1p(s.steps3h)
    p,P=predict(M,s,70); r=low_range(P); G.append(dict(input=case,pred=[round(float(x),1) for x in p],p_low=[round(float(x),3) for x in P],range_h=[float(x) for x in r] if r else None,wake=round(float((p[7]+p[8])/2),1)))
json.dump(G,open(f"{OUT}/golden.json","w"),indent=1); print(json.dumps(G)[:900])
