import pandas as pd, numpy as np, html, json
E=pd.read_pickle('/tmp/an/E2.pkl'); P=pd.read_pickle('/tmp/an/Ptag.pkl')
O='glucose-model/miss_causes/'
E.drop(columns=['tags']).assign(tags=E.tags.apply('; '.join)).to_csv(O+'episodes.csv',index=False)
d=P[(P.model=='A v5')&(P.h==120)].copy(); d['hr']=d.t.dt.hour; d['big']=d.err.abs()>40
hr=d.groupby('hr').big.mean().mul(100).round()
cs=E.main.value_counts(); n=len(E)
expl={'Meal inside forecast window':'Food eaten after the forecast was made rose more (or less) than the model expected.',
'Bounce after a low':'Glucose was under 70 shortly before. Low treatment plus rebound rose far more than forecast.',
'Correction insulin dropped faster':'A correction dose (no food) brought glucose down faster than forecast.',
'Earlier meal peaked late':'Food from 1 to 3 hours earlier kept rising (dal, egg and paneer meals).',
'Rise, no clear cause':'Rise with nothing logged nearby that explains it.','Drop, no clear cause':'Drop with nothing logged nearby.'}
def hbar(items,maxv,unit):
    rows=''; 
    for i,(k,v,tip) in enumerate(items):
        w=v/maxv*100
        rows+=f'<div class="row" title="{html.escape(tip)}"><div class="lab">{html.escape(k)}</div><div class="track"><div class="bar" style="width:{w:.1f}%"></div></div><div class="val">{v:.0f}{unit}</div></div>'
    return rows
causes=hbar([(k,v/n*100,f'{v} of {n} episodes. {expl[k]}') for k,v in cs.items()],60,'%')
# hour chart svg
W,H=720,220; bw=W/24
bars=''
for h in range(24):
    v=hr.get(h,0); bh=v/40*(H-40)
    bars+=f'<g><title>{h:02d}:00 – {v:.0f}% of 2h forecasts missed by >40</title><rect x="{h*bw+2:.1f}" y="{H-20-bh:.1f}" width="{bw-4:.1f}" height="{max(bh,0.5):.1f}" rx="3" fill="#2a78d6"/><rect x="{h*bw:.1f}" y="0" width="{bw:.1f}" height="{H}" fill="transparent"/></g>'
    if h%3==0: bars+=f'<text x="{h*bw+bw/2:.1f}" y="{H-4}" class="ax" text-anchor="middle">{h:02d}</text>'
grid=''.join(f'<line x1="0" x2="{W}" y1="{H-20-g/40*(H-40):.1f}" y2="{H-20-g/40*(H-40):.1f}" class="gl"/><text x="2" y="{H-23-g/40*(H-40):.1f}" class="ax">{g}%</text>' for g in (10,20,30))
svg=f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Share of 2 hour forecasts missing by more than 40, by hour">{grid}{bars}</svg>'
T=E.sort_values('abs',ascending=False).head(12)
trs=''.join(f"<tr><td>{r.peak:%d %b %H:%M}</td><td>{'higher' if r.peak_err<0 else 'lower'}</td><td class=n>{r.pred:.0f}</td><td class=n>{r.act:.0f}</td><td class=n>{abs(r.peak_err):.0f}</td><td>{r.main}</td><td class=sm>{html.escape(r.detail)}</td></tr>" for r in T.itertuples())
page=f'''<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Why big forecast misses happen</title>
<style>
:root{{--s:#fcfcfb;--p:#0b0b0b;--t2:#52514e;--g:#e6e5e0;--b:#2a78d6;--pl:#f9f9f7}}
@media(prefers-color-scheme:dark){{:root{{--s:#1a1a19;--p:#f0efec;--t2:#c3c2b7;--g:#383835;--b:#3987e5;--pl:#0d0d0d}}}}
body{{font:15px/1.5 system-ui,sans-serif;background:var(--pl);color:var(--p);margin:0;padding:20px}}
main{{max-width:860px;margin:auto}} .card{{background:var(--s);border:1px solid var(--g);border-radius:10px;padding:16px 18px;margin:14px 0}}
h1{{font-size:22px;margin:4px 0}} h2{{font-size:16px;margin:0 0 8px}} .sub{{color:var(--t2)}}
.row{{display:grid;grid-template-columns:230px 1fr 44px;gap:8px;align-items:center;margin:6px 0}}
.track{{height:14px}} .bar{{height:14px;background:var(--b);border-radius:0 4px 4px 0}} .val{{text-align:right;font-variant-numeric:tabular-nums}}
svg{{width:100%;height:auto}} .ax{{font-size:11px;fill:var(--t2)}} .gl{{stroke:var(--g)}}
svg g:hover rect:first-child{{opacity:.75}}
table{{border-collapse:collapse;width:100%;font-size:13px}} td,th{{border-bottom:1px solid var(--g);padding:5px 6px;text-align:left;vertical-align:top}} .n{{text-align:right;font-variant-numeric:tabular-nums}} .sm{{color:var(--t2);font-size:12px}}
ul{{margin:4px 0;padding-left:20px}} li{{margin:4px 0}}
</style></head><body><main>
<h1>Why the big forecast misses happen</h1>
<div class="sub">A v5 and MoE v6, 1h and 2h forecasts vs Libre, 30 Aug to 17 Sep 2026. A "miss" is more than 40 mg/dL off. Misses close together were grouped: <b>{n} episodes</b> ({(E.peak_err<0).sum()} where real glucose was higher, {(E.peak_err>0).sum()} lower).</div>
<div class="card"><h2>Short answer</h2><ul>
<li><b>Mostly food and insulin, not sleep or dawn.</b> About half the episodes are meals (in the window or still digesting), a quarter are bounces after a low, and a sixth are correction doses dropping faster than expected.</li>
<li><b>Dawn (3 to 8am) is the most accurate time</b>: only 5% of 2h forecasts miss, against 14% overall.</li>
<li><b>The bad hours are 11am to 1pm and 4 to 10pm</b>, when you eat and correct. 37% of episodes fall 3 to 10pm.</li>
<li><b>Sensor problems explain almost nothing</b> here (no episode traced to a gap; 1 possible compression dip).</li>
</ul></div>
<div class="card"><h2>Main cause of each episode</h2><div class="sub" style="margin-bottom:6px">Share of {n} episodes. Hover a bar for the meaning.</div>{causes}</div>
<div class="card"><h2>When 2h forecasts miss by more than 40</h2><div class="sub">A v5, share of forecasts per hour of the day (target time).</div>{svg}</div>
<div class="card"><h2>Old findings, checked</h2><ul>
<li><b>Night milk near Lantus: true, but it is the milk.</b> On Lantus nights with milk, Libre rose a median of +60 (range 0 to +116) over 4h. On Lantus nights with no food it rose only about +11. "+100" was the worst nights, not the usual.</li>
<li><b>Milk "acts fast": partly.</b> Daytime milk alone peaks at about 1.6h, but at night about 3h. The bigger problem is size: 250 ml (12g carbs) rises about +40 to +60, more than its carbs suggest. Milk in the window raises the 2h miss rate from 10% to 18%.</li>
<li><b>Dal peaks late: mostly true.</b> Dal or moong meals peaked at 2.7 to 5.9h in 7 of 10 meals (2 with roti peaked under 1h). When dal is in the window, 2h forecasts run 19 too low on average, the worst of any food. Only 10 meals, so treat as a hint.</li>
<li><b>Errors highest 3 to 9pm: close.</b> It is 4 to 10pm, plus a second peak 11am to 1pm.</li>
<li><b>Activity: small effect.</b> With 1500+ steps in the window, the 2h miss rate is 19% vs 11%, but walking happens mostly in the evening, so this overlaps with meals.</li>
<li><b>Unlogged food is not the issue.</b> With no food logged in the last 4h, only 3% of 2h forecasts miss.</li>
</ul></div>
<div class="card"><h2>Biggest 12 episodes</h2><table><tr><th>Time</th><th>Real was</th><th class=n>Forecast</th><th class=n>Libre</th><th class=n>Off by</th><th>Main cause</th><th>What was logged</th></tr>{trs}</table>
<div class="sub" style="margin-top:8px">All {n} episodes are in glucose-model/miss_causes/episodes.csv. Only 17 days of data, so small groups (dal, roti/rice) are rough.</div></div>
<div class="card"><h2>What would help the model</h2><ul>
<li>Treat milk as a bigger hit (about +50 per 250 ml), and a slower one at night.</li>
<li>Add a "just had a low and treated it" input; the model now keeps forecasting the fall (e.g. forecast 26, real 153 on 11 Sep).</li>
<li>Give correction doses (no food) a stronger, faster drop.</li>
</ul></div>
</main></body></html>'''
open(O+'why-forecasts-miss.html','w').write(page); print(cs, hr.to_dict())
