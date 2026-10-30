# raw Claude Code session jsonl (armand0e/claude-fable-5-claude-code MIT; AlinCiocan/fable-5-claude-code-traces CC-BY-4.0) -> examples
import json,sys,glob,re,collections
sys.path.insert(0,'.')  # run from data/v2/build
from lib import *
st=collections.Counter()
SR=re.compile(r"<system-reminder>.*?</system-reminder>",re.S)
def txt(c):
    if isinstance(c,str): return c
    o=[]
    for b in c or []:
        if isinstance(b,dict):
            if b.get("type")=="text": o.append(b.get("text",""))
            elif b.get("type")=="image": o.append("[image omitted]")
    return "\n".join(o)
def sessions(files,tag):
    out=open(f"../raw/ex_{tag}.jsonl","w")
    for f in files:
        evs=[]
        for l in open(f):
            try: d=json.loads(l)
            except: continue
            if d.get("isSidechain") or d.get("type") not in("user","assistant"): continue
            evs.append(d)
        # build segments
        segs=[]; cur=None; i=0
        while i<len(evs):
            d=evs[i]; m=d["message"]
            if d["type"]=="user":
                c=m.get("content")
                if isinstance(c,list) and any(b.get("type")=="tool_result" for b in c if isinstance(b,dict)):
                    if cur: cur["results"].update({b["tool_use_id"]:txt(b.get("content")) for b in c if b.get("type")=="tool_result"}); cur["seq"].append(("res",[b["tool_use_id"] for b in c if b.get("type")=="tool_result"]))
                else:
                    t=SR.sub("",txt(c)).strip()
                    if d.get("isMeta") or not t or t.startswith(("<task-notification","<scheduled","<system","<command-","<local-command","[Request interrupted","<bash-","Caveat:")): i+=1; continue
                    cur={"prompt":t,"seq":[],"results":{},"asst":{}}; segs.append(cur)
            else:
                if cur is None: i+=1; continue
                mid=m.get("id") or d.get("uuid")
                if mid not in cur["asst"]:
                    cur["asst"][mid]={"model":m.get("model"),"blocks":[]}; cur["seq"].append(("asst",mid))
                cur["asst"][mid]["blocks"]+=m.get("content") or []
            i+=1
        for si,s in enumerate(segs):
            if len(s["prompt"])<15 or s["prompt"].lower().strip(" .!") in("continue","go on","yes","ok"): st["vague_prompt"]+=1; continue   # no context to resume from
            msgs=[{"role":"system","content":SYSTEM},{"role":"user","content":s["prompt"]}]; used=set(); stop=None
            for kind,v in s["seq"]:
                if kind=="res": continue
                a=s["asst"][v]
                if a["model"]!="claude-fable-5": stop="model"; break
                text="\n\n".join(b.get("text","") for b in a["blocks"] if b.get("type")=="text").strip()
                calls=[];ids=[]
                try:
                    for b in a["blocks"]:
                        if b.get("type")=="tool_use": n,a2=norm_call(b["name"],b["input"]); calls.append((n,a2)); ids.append(b["id"]); used.add(n)
                except Skip as e: stop=str(e); break
                if any(x not in s["results"] for x in ids): stop="noresult"; break
                msgs.append(asst("",text,calls))
                for x in ids: msgs.append({"role":"tool","content":s["results"][x]})
            if stop: st["cut:"+stop[:20]]+=1
            if not(used&{"bash","write","edit"}): st["seg_not_code"]+=1; continue
            w=window(msgs,min_asst=2)
            if not w: st["seg_short_or_long"]+=1; continue
            pm,full=w
            if contaminated(pm): st["contam"]+=1; continue
            st["full" if full else "prefix"]+=1
            out.write(json.dumps({"src":f"fable5-cc-{tag}-"+("full" if full else "prefix"),"msgs":pm},ensure_ascii=False)+"\n")
sessions(sorted(glob.glob("../raw/s_fable_cc/*.jsonl")),"armand0e")
print("armand0e",dict(st)); st.clear()
sessions(sorted(glob.glob("../raw/s_alin/*.jsonl")),"alin")
print("alin",dict(st))
