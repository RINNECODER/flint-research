import sys,json,re,glob,random,hashlib,collections,statistics
sys.path.insert(0,__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from render import *
import pyarrow.parquet as pq
random.seed(1234)
rows=[json.loads(l) for f in ["stage1.jsonl"]+sorted(glob.glob("stage1b_*.jsonl")+glob.glob("old/stage1b_*.jsonl")) for l in open(f)]
stats=collections.OrderedDict(); stats["verified_in"]=len(rows)
# ---- decontamination
W=lambda s:re.findall(r"\w+",s.lower())
grams=lambda w,n:{" ".join(w[i:i+n]) for i in range(len(w)-n+1)}
bench=[];names=[]
for r in pq.read_table(glob.glob("ep/humanevalplus/data/*.parquet")[0]).to_pylist():
    bench.append(r["prompt"]+"\n"+r["canonical_solution"]); names.append((r["entry_point"],r["prompt"]))
for r in pq.read_table(glob.glob("ep/mbppplus/data/*.parquet")[0]).to_pylist():
    bench.append(r["prompt"]+"\n"+r["code"])
B13=set()
for b in bench: B13|=grams(W(b),13)
B8=[(n,grams(W(p),8)) for n,p in names]
def contaminated(t):
    w=W(t)
    if grams(w,13)&B13: return "13gram"
    g8=grams(w,8)
    for n,s in B8:
        if re.search(r"def\s+"+re.escape(n)+r"\s*\(",t) and g8&s: return "fn+doc"
    return None
def clean_reason(x):
    t=x["think"]; 
    if len(t)<200: return "short_think"
    if len(re.findall(r"[^\x00-\x7f]",t))/len(t)>0.05: return "garbled"
    ls=[l.strip() for l in t.splitlines() if len(l.strip())>20]
    if ls and collections.Counter(ls).most_common(1)[0][1]>=6: return "repetition"
    if "```" not in x["answer"] or len(x["code"])<20: return "no_code"
    u=x["user"]; letters=[c for c in u if c.isalpha()]
    if not letters or sum(c.isascii() for c in letters)/len(letters)<0.97: return "non_english"
    return None
drop=collections.Counter(); keep=[]
for x in rows:
    d=clean_reason(x)
    if d: drop[d]+=1; continue
    c=contaminated(x["user"]+"\n"+x["answer"]+"\n"+x["think"])
    if c: drop["decontam_"+c]+=1; continue
    keep.append(x)
# ---- dedup: exact + near (word 5-gram jaccard >=0.7 on problem text)
seen=set(); uniq=[]; inv=collections.defaultdict(list); sh=[]
for x in keep:
    k=hashlib.md5(" ".join(W(x["user"])).encode()).hexdigest()
    if k in seen: drop["dup_exact"]+=1; continue
    s={hash(g) for g in grams(W(x["user"]),5)}
    cand=collections.Counter(j for h in s if h%4==0 for j in inv[h])
    if any(len(s&sh[j])/max(1,len(s|sh[j]))>=0.7 for j in cand): drop["dup_near"]+=1; continue
    seen.add(k); idx=len(sh); sh.append(s); uniq.append(x)
    for h in s:
        if h%4==0: inv[h].append(idx)
print("verified",len(rows),"after filters",len(uniq),dict(drop))
# ---- build examples
SYS=[None,None,"You are a helpful coding assistant.","You are an expert competitive programmer. Think carefully, then give a complete, correct solution."]
ASYS="You are a coding assistant with access to a Python sandbox via the run_python tool. Test your code on the sample input before giving the final answer."
TOOLS=[{"type":"function","function":{"name":"run_python","description":"Run a Python 3 program in a sandbox and return its stdout and exit status.","parameters":{"type":"object","properties":{"code":{"type":"string","description":"Python source code to run"},"stdin":{"type":"string","description":"Text fed to the program's standard input"}},"required":["code"]}}}]
def user_text(x): return x["user"] if x["file"]!="ocr" else "Solve the following problem with a Python program that reads from standard input and writes to standard output.\n\n"+x["user"].strip()
def plain(x):
    m=[]; s=random.choice(SYS)
    if s: m.append({"role":"system","content":s})
    m+= [{"role":"user","content":user_text(x)},{"role":"assistant","reasoning_content":x["think"],"content":x["answer"]}]
    return m,None
def agentic(x):
    runs=x["pub_runs"][:2]
    if not runs or any(len(r["input"])>600 or len(r["stdout"])>400 for r in runs): return None
    tcs=[{"type":"function","function":{"name":"run_python","arguments":{"code":x["code"],"stdin":r["input"]}}} for r in runs]
    m=[{"role":"system","content":ASYS},{"role":"user","content":user_text(x)},
       {"role":"assistant","reasoning_content":x["think"],"content":"Let me run the solution on the sample input.","tool_calls":tcs}]
    for r in runs: m.append({"role":"tool","content":"exit code: 0\nstdout:\n"+r["stdout"]})
    m.append({"role":"assistant","content":"The program's output matches the expected sample output.\n\n"+x["answer"]})
    return m,TOOLS
ex=[]
for x in uniq:
    p=plain(x); a=agentic(x)
    ex.append({"x":x,"plain":p,"agent":a})
random.shuffle(ex)
out=[]
for e in ex:
    for kind in ("plain","agent"):
        if e[kind] is None: continue
        msgs,tools=e[kind]; ids,mask=build(msgs,tools)
        e[kind+"_len"]=len(ids)
pickle=__import__("pickle"); pickle.dump(ex,open("ex.pkl","wb"))
for k in ("plain","agent"):
    L=[e[k+"_len"] for e in ex if k+"_len" in e]; ok=[l for l in L if l<=3072]
    print(k,len(L),"<=3072:",len(ok),"mean",statistics.mean(ok),"median",statistics.median(ok))
