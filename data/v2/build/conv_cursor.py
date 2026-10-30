# TeichAI/Fable-5-Cursor-Traces (apache-2.0) -> prefix windows
import json,sys,collections
sys.path.insert(0,'.')  # run from data/v2/build
from lib import *
st=collections.Counter(); out=open("../raw/ex_cursor.jsonl","w")
for l in open("../raw/s_cursor/train.jsonl"):
    d=json.loads(l); m=d["messages"]; msgs=[{"role":"system","content":SYSTEM}]; used=set(); i=1
    if m[1]["role"]!="user": st["nouser"]+=1; continue
    msgs.append({"role":"user","content":m[1]["content"]}); i=2
    while i<len(m):
        x=m[i]
        if x["role"]!="assistant": st["cut:order"]+=1; break
        calls=[];ids=[]
        try:
            for t in x.get("tool_calls") or []:
                if t["function"]["name"]=="todo_write": continue   # planning-ui call: dropped together with its result
                a=t["function"]["arguments"]
                if isinstance(a,str): a=json.loads(a)
                n,a2=norm_call(t["function"]["name"],a); calls.append((n,a2)); ids.append(t["id"]); used.add(n)
        except (Skip,ValueError) as e: st["cut:"+str(e)[:18]]+=1; break
        res={}; j=i+1
        while j<len(m) and m[j]["role"]=="tool": res[m[j]["tool_call_id"]]=m[j]["content"]; j+=1
        if any(k not in res for k in ids): st["cut:noresult"]+=1; break
        if calls or x.get("content") or x.get("reasoning_content"): msgs.append(asst(x.get("reasoning_content",""),x.get("content",""),calls))
        for k in ids: msgs.append({"role":"tool","content":res[k] if isinstance(res[k],str) else json.dumps(res[k])})
        i=j
    if not(used&{"bash","write","edit"}): st["not_code"]+=1; continue
    w=window(msgs,min_asst=2)
    if not w: st["short_or_long"]+=1; continue
    pm,full=w
    if contaminated(pm): st["contam"]+=1; continue
    st["full" if full else "prefix"]+=1
    out.write(json.dumps({"src":"fable5-cursor-"+("full" if full else "prefix"),"msgs":pm},ensure_ascii=False)+"\n")
print(dict(st))
