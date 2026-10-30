import json,sys,glob,collections
th=collections.Counter(); tools=collections.Counter(); n=0
rows=[]
for f in glob.glob(sys.argv[1]+'/*.jsonl'):
    ntu=nth=nthtext=nas=0; chars=0; models=set()
    for l in open(f):
        try:d=json.loads(l)
        except: continue
        m=d.get('message')
        if d.get('type')!='assistant' or not isinstance(m,dict): continue
        models.add(m.get('model')); nas+=1
        for b in m.get('content') or []:
            if b.get('type')=='tool_use': ntu+=1; tools[b['name']]+=1
            if b.get('type')=='thinking':
                nth+=1; nthtext+= len(b.get('thinking',''))>0
            chars+=len(json.dumps(b))
    rows.append((f.split('/')[-1][:8],nas,ntu,nth,nthtext,chars,tuple(models)))
print(len(rows)); print(tools.most_common(15))
print('sessions with tool_use',sum(1 for r in rows if r[2]),'with thinking text',sum(1 for r in rows if r[4]))
print(collections.Counter(m for r in rows for m in r[6]))
rows.sort(key=lambda r:r[5]); print(rows[len(rows)//2], sum(r[4] for r in rows), sum(r[1] for r in rows))
