import sys,json,random,glob,collections,statistics
sys.path.insert(0,'.')  # run from data/v2/build
from lib import *
random.seed(2024)
OUT="../"
pool=[]; seen=set(); drop=collections.Counter()
for f in ["dsf","armand0e","alin","cursor","ag"]:
    for l in open(f"{OUT}raw/ex_{f}.jsonl"):
        e=json.loads(l); m=e["msgs"]
        u=next(x["content"] for x in m if x["role"]=="user"); k=hash(u[:400]+m[-1].get("content","")[:100])
        if k in seen: drop["dup"]+=1; continue
        seen.add(k)
        if any("<think>" in (x.get("content") or "") or "</think>" in (x.get("content") or "") for x in m if x["role"]=="assistant"): drop["think_tag_in_content"]+=1; continue
        pool.append(e)
print(len(pool),dict(drop))
bysrc=collections.defaultdict(list)
for e in pool: bysrc[e["src"].rsplit("-",1)[0] if e["src"].startswith("fable5") else e["src"]].append(e)
val=[];train=[]
for s,es in bysrc.items():
    random.shuffle(es); nv=round(150*len(es)/len(pool)); val+=es[:nv]; train+=es[nv:]
random.shuffle(val); train+=val[150:]; val=val[:150]
def emit(L,path):
    random.shuffle(L); st=collections.defaultdict(list)
    with open(path,"w") as o:
        for e in L:
            r=finish(e["src"].replace("-prefix","").replace("-full","") if False else e["src"],e["msgs"],None if e.get("notools") else TOOLS)
            o.write(json.dumps(r)+"\n"); st[e["src"]].append((len(r["ids"]),sum(r["mask"])))
    return st
vs=emit(val,OUT+"val.jsonl"); ts=emit(train,OUT+"train.jsonl")
json.dump({"train":{k:v for k,v in ts.items()},"val":{k:v for k,v in vs.items()}},open(OUT+"build/stats.json","w"))
for nm,st in(("train",ts),("val",vs)):
    print(nm)
    tot=n=tm=0
    for k,v in sorted(st.items()):
        t=[a for a,_ in v]; print(f"  {k:32s} n={len(v):5d} tok={sum(t):9d} mean={statistics.mean(t):6.0f} med={statistics.median(t):6.0f} max={max(t)} trained={sum(b for _,b in v)}")
        tot+=sum(t);n+=len(v);tm+=sum(b for _,b in v)
    print("  TOTAL",n,tot,tm)
