import re,glob,pandas as pd
rows=[]
for f in sorted(glob.glob('/tmp/claude-0/x/459bbb26*/*/insulin food bg/nov*.txt')):
    day=int(re.search(r'nov (\d+)',f).group(1))
    txt=open(f,errors='ignore').read()
    blocks=re.split(r'\n\s*\n',txt); last=None; pm=False
    for b in blocks:
        tm=None
        for l in b.splitlines():
            m=re.match(r'\s*(\d{1,2})\s*:\s*(\d{2})\s*(am|pm)?',l,re.I)
            if m and 'iu' not in l.lower(): tm=m;break
        if tm:
            h=int(tm.group(1));mi=int(tm.group(2));ap=(tm.group(3) or '').lower()
            if ap=='pm' and h<12:h+=12
            if ap=='am' and h==12:h=0
            last=(h,mi)
        doses=[]
        for l in b.splitlines():
            ll=l.lower()
            if 'lantus' in ll and 'novo' not in ll: continue
            if re.search(r'\b28\s*iu',ll): continue
            m=re.findall(r'(\d+(?:\.\d+)?)\s*iu',ll)
            if not m or 'worth' in ll: continue
            if '=' in ll: m=[ll.split('=')[1].strip().split('iu')[0]]
            if 'lantus' in ll: m=['1']
            doses.append(sum(float(x) for x in m))
        if not doses or last is None: continue
        s=re.findall(r'(left|right)\s*stomach|\b(ls)\b',b.lower().replace('lantus 28 iu left leg',''))
        site='none'
        for a,c in s: site='left' if (a=='left' or c) else 'right'
        rows.append(dict(time=pd.Timestamp(2026,11,day,*last),units=sum(doses),site=site,txt=b.strip().replace('\n',' | ')[:70]))
D=pd.DataFrame(rows).drop_duplicates(['time','units','site']).sort_values('time')
D.to_csv('/tmp/claude-0/s/doses_set2.csv',index=False)
pd.set_option('display.width',200);print(D.to_string());print(D.site.value_counts())
