import re,glob,sys,collections
files=glob.glob('/tmp/claude-0/x/510ee5bb*/FMS/food logger/*.txt')+glob.glob('/tmp/claude-0/x/459bbb26*/*/insulin food bg/nov*.txt')+glob.glob('/tmp/claude-0/x/851994c5*/*/food and insulin,bg/*.txt')
pat=re.compile(r'(left|right|l|r)?\s*(stomach|leg|thigh|arm|ls|rs)\b',re.I)
cnt=collections.Counter(); per=collections.defaultdict(collections.Counter)
rows=[]
for f in files:
    day=f.split('/')[-1][:-4]; st='set1' if '510ee5bb' in f else ('set2' if '459bbb26' in f else 'set3')
    blocks=re.split(r'\n\s*\n',open(f,errors='ignore').read())
    for b in blocks:
        low=b.lower()
        doses=[l for l in b.splitlines() if re.search(r'\d+(\.\d+)?\s*iu',l,re.I) and 'lantus' not in l.lower() and 'worth' not in l.lower() and not re.search(r'no (novo|insulin)',l,re.I)]
        if not doses: continue
        # sites in block excluding lantus lines
        s=[]
        for l in b.splitlines():
            if 'lantus' in l.lower() and 'novo' not in l.lower(): continue
            for m in re.finditer(r'(left|right)\s*(stomach|leg|thigh|arm)|\b(ls|rs)\b',l,re.I):
                s.append((m.group(0).lower().replace('ls','left stomach').replace('rs','right stomach')))
        site=s[-1] if s else 'none'
        if '28 iu lantus and 1 iu novorapid' in low: site='left stomach'
        cnt[site]+=len(doses); per[st][site]+=len(doses)
        rows.append((st,day,site,len(doses),doses[0].strip()[:40]))
print(cnt);[print(k,dict(v)) for k,v in per.items()]
import csv;csv.writer(open('/tmp/claude-0/s/site_rows.csv','w')).writerows(rows)
