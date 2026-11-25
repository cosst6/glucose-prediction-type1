import pandas as pd, numpy as np
lc = pd.read_csv("lc.csv"); imp = pd.read_csv("imp.csv")
# ---- learning curve svg (2h)
W,H,L,R,T,B = 640,300,52,20,20,44
xs = lambda d: L + (np.log(d)-np.log(2))/(np.log(90)-np.log(2))*(W-L-R)
ys = lambda v: T + (36-v)/(36-20)*(H-T-B)
s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="2-hour error versus days of training data">']
for v in (20,24,28,32,36):
    s.append(f'<line x1="{L}" x2="{W-R}" y1="{ys(v):.1f}" y2="{ys(v):.1f}" class="grid"/><text x="{L-8}" y="{ys(v)+4:.1f}" class="tick" text-anchor="end">{v}</text>')
for d in (2,4,8,13,30,60,90):
    s.append(f'<text x="{xs(d):.1f}" y="{H-B+18}" class="tick" text-anchor="middle">{d}</text>')
s.append(f'<text x="{(W+L)/2}" y="{H-6}" class="axl" text-anchor="middle">days of your data used for training (log scale)</text>')
s.append(f'<rect x="{xs(13):.1f}" y="{T}" width="{W-R-xs(13):.1f}" height="{H-T-B}" class="future"/><text x="{xs(13)+8:.1f}" y="{T+16}" class="tick">projected, not measured</text>')
d2 = lc[lc.h==120]
def path(xv, yv): return " ".join(("M" if i==0 else "L")+f"{xs(a):.1f},{ys(b):.1f}" for i,(a,b) in enumerate(zip(xv,yv)))
s.append(f'<path d="{path(d2.days,[35.2]*len(d2))}" class="l-per"/><text x="{xs(13)-4:.1f}" y="{ys(35.2)-8:.1f}" class="lab" text-anchor="end">"no change" guess 35.2</text>')
s.append(f'<path d="{path(d2.days,d2.body)}" class="l-body"/>')
s.append(f'<path d="{path(d2.days,d2.hyb)}" class="l-hyb"/>')
for a,b in zip(d2.days,d2.hyb): s.append(f'<circle cx="{xs(a):.1f}" cy="{ys(b):.1f}" r="4" class="d-hyb"/>')
f = lambda n: 21.6 + (25.0-21.6)*np.sqrt(13/n)
pr = [13,20,30,45,60,90]; s.append(f'<path d="{path(pr,[f(n) for n in pr])}" class="l-proj"/>')
s.append(f'<text x="{xs(90):.1f}" y="{ys(f(90))-10:.1f}" class="lab" text-anchor="end">~23 at 90 days</text>')
s.append(f'<text x="{xs(2)+6:.1f}" y="{ys(30.4)-8:.1f}" class="lab">30.4</text><text x="{xs(13):.1f}" y="{ys(25.2)+20:.1f}" class="lab" text-anchor="middle">25.2</text>')
s.append('</svg>'); lc_svg = "\n".join(s)
# ---- importance bars (2h)
d = imp[imp.h==120].sort_values("worse_by", ascending=False)
W2,rh,LL = 640,30,190; mx = 22
s = [f'<svg viewBox="0 0 {W2} {len(d)*rh+10}" role="img" aria-label="How much worse the 2-hour forecast gets when each input is scrambled">']
for i,(_,r) in enumerate(d.iterrows()):
    y = 4+i*rh; w = max(r.worse_by,0)/mx*(W2-LL-70)
    s.append(f'<text x="{LL-10}" y="{y+18}" class="lab" text-anchor="end">{r.group}</text><rect x="{LL}" y="{y+5}" width="{max(w,1.5):.1f}" height="18" rx="2" class="bar{" weak" if r.worse_by<1 else ""}"/><text x="{LL+max(w,1.5)+6:.1f}" y="{y+18}" class="num">+{max(r.worse_by,0):.1f}</text>')
s.append('</svg>'); imp_svg = "\n".join(s)
# ---- error share
E = [("Meal in the window", 43, 30.0), ("Started low (Libre under 80)", 23, 27.7), ("Correction dose working", 23, 28.5), ("Night, no food", 8, 28.0), ("Everything else", 3, 17.7)]
s = [f'<svg viewBox="0 0 640 {len(E)*30+10}" role="img" aria-label="Share of 2-hour forecast error by situation">']
for i,(n,p,r) in enumerate(E):
    y = 4+i*30; w = p/45*(640-230-110)
    s.append(f'<text x="220" y="{y+18}" class="lab" text-anchor="end">{n}</text><rect x="230" y="{y+5}" width="{w:.1f}" height="18" rx="2" class="bar2"/><text x="{236+w:.1f}" y="{y+18}" class="num">{p}% of error</text>')
s.append('</svg>'); err_svg = "\n".join(s)
html = open("tmpl.html").read().replace("%%LC%%", lc_svg).replace("%%IMP%%", imp_svg).replace("%%ERR%%", err_svg)
open("model-explainer/index.html","w").write(html)
