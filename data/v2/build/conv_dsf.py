# DSFFGFG456/fable-5-coding-and-debugging-traces -> examples (coding trajectories only)
import json,sys,collections
sys.path.insert(0,'.')  # run from data/v2/build
from lib import *
out=open("../raw/ex_dsf.jsonl","w"); st=collections.Counter()
CODE={"bash","write","edit"}
for l in open("../raw/dsf_full.jsonl"):
    r=json.loads(l); m=r["messages"]; msgs=[{"role":"system","content":SYSTEM}]; idmap={}; ok=True; used=set()
    for x in m:
        if x["role"]=="system": continue
        if x["role"]=="user": msgs.append({"role":"user","content":x["content"]})
        elif x["role"]=="assistant":
            calls=[]
            try:
                for t in x.get("tool_calls",[]):
                    a=json.loads(t["function"]["arguments"]); n,a2=norm_call(t["function"]["name"],a); calls.append((n,a2)); used.add(n)
            except (Skip,ValueError,KeyError) as e:
                st["cut:"+str(e)[:30]]+=1; break
            msgs.append(asst(x.get("reasoning_content",""),x.get("content",""),calls))
        elif x["role"]=="tool":
            msgs.append({"role":"tool","content":x.get("content","")})
    # a cut can leave dangling tool calls whose results are missing; window() ends at an assistant msg anyway
    if not (used&CODE): st["not_code"]+=1; continue
    # tool-result count must match calls of previous assistant (parallel order) -> drop otherwise
    bad=False
    for i,x in enumerate(msgs):
        if x["role"]=="assistant" and x.get("tool_calls"):
            k=len(x["tool_calls"]); nxt=msgs[i+1:i+1+k]
            if i+1<len(msgs) and not all(y["role"]=="tool" for y in nxt): bad=True
    if bad: st["mismatch"]+=1; continue
    w=window(msgs,min_asst=1)
    if not w: st["too_short_or_long"]+=1; continue
    pm,full=w
    if contaminated(pm): st["contam"]+=1; continue
    st["full" if full else "prefix"]+=1
    out.write(json.dumps({"src":"fable5-dsf-"+("full" if full else "prefix"),"msgs":pm,"lang":r["lang"],"cat":r["category"]},ensure_ascii=False)+"\n")
print(dict(st))
