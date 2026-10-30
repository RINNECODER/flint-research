import sys,json,pickle,random,statistics,collections,os
sys.path.insert(0,__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from render import *
random.seed(99)
OUT=os.environ.get("OUT","out"); os.makedirs(OUT,exist_ok=True)  # final train/val jsonl
P_AGENT=float(sys.argv[1]) if len(sys.argv)>1 else 0.33
BUDGET=4_500_000; NVAL=150
ex=pickle.load(open("ex.pkl","rb"))
items=[]
for e in ex:
    if e.get("plain_len",9999)>3072 and e.get("agent_len",9999)>3072: continue
    ag="agent" in e and e["agent"] is not None and e.get("agent_len",9999)<=3072
    kind="agent" if ag and random.random()<P_AGENT else "plain"
    if e.get(kind+"_len",9999)>3072: kind="plain" if kind=="agent" else "agent"
    if e.get(kind+"_len",9999)>3072 or e[kind] is None: continue
    items.append((e[kind+"_len"],kind,e))
items.sort(key=lambda t:t[0])
sel=[];tot=0
for L,k,e in items:
    if tot+L>BUDGET*(1+NVAL/3000) or len(sel)>=5000+NVAL: break   # val tokens counted loosely, trimmed below
    sel.append((L,k,e)); tot+=L
random.shuffle(sel)
val=sel[:NVAL]; train=sel[NVAL:]
# trim longest train items until train tokens <= budget
train.sort(key=lambda t:t[0])
while sum(t[0] for t in train)>BUDGET: train.pop()
random.shuffle(train)
def src(x,k): return ("codeforces-cots" if x["file"]!="ocr" else "opencodereasoning")+("-agentic" if k=="agent" else "")
def dump(lst,path):
    with open(path,"w") as f:
        for L,k,e in lst:
            msgs,tools=e[k]; ids,mask=build(msgs,tools); assert len(ids)==L==len(mask)
            f.write(json.dumps({"src":src(e["x"],k),"ids":ids,"mask":mask})+"\n")
dump(train,OUT+"/train.jsonl"); dump(val,OUT+"/val.jsonl")
# sample messages
with open(OUT+"/messages_sample.jsonl","w") as f:
    ag=[t for t in train if t[1]=="agent"][:7]; pl=[t for t in train if t[1]=="plain"][:13]
    for L,k,e in ag+pl:
        m,t=e[k]; f.write(json.dumps({"src":src(e["x"],k),"id":e["x"]["id"],"tools":t,"messages":m},ensure_ascii=False)+"\n")
# mask check log + stats
lg=open("maskcheck.log","w")
rows=[json.loads(l) for l in open(OUT+"/train.jsonl")]
for r in random.sample(rows,3):
    on=[i for i,m in zip(r["ids"],r["mask"]) if m]; off=[i for i,m in zip(r["ids"],r["mask"]) if not m]
    lg.write("=== %s len=%d trained=%d\n--- MASKED-IN (first 500 / last 300 chars)\n%s\n...\n%s\n--- MASKED-OUT head\n%s\n"%(r["src"],len(r["ids"]),len(on),tok.decode(on,skip_special_tokens=False)[:500],tok.decode(on,skip_special_tokens=False)[-300:],tok.decode(off,skip_special_tokens=False)[:400]))
st={}
for name in ("train","val"):
    rs=rows if name=="train" else [json.loads(l) for l in open(OUT+"/val.jsonl")]
    d=collections.defaultdict(list)
    for r in rs: d[r["src"]].append(len(r["ids"]))
    L=[len(r["ids"]) for r in rs]
    st[name]={"n":len(rs),"tokens":sum(L),"mean":statistics.mean(L),"median":statistics.median(L),"max":max(L),"trained_tokens":sum(sum(r["mask"]) for r in rs),
      "agentic_frac":sum(r["src"].endswith("agentic") for r in rs)/len(rs),
      "by_src":{k:{"n":len(v),"tokens":sum(v),"mean":statistics.mean(v),"median":statistics.median(v),"max":max(v)} for k,v in d.items()}}
json.dump(st,open("stats.json","w"),indent=1)
print(json.dumps({k:{a:b for a,b in v.items() if a!="by_src"} for k,v in st.items()}))
for k,v in st["train"]["by_src"].items(): print(k,v)
