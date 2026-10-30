# angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k code_train.jsonl, category=coding (apache-2.0). Single/multi-turn chat, no tools.
import json,sys,re,collections
sys.path.insert(0,'.')  # run from data/v2/build
from lib import *
st=collections.Counter(); out=open("../raw/ex_ag.jsonl","w")
T=re.compile(r"^<think>\n?(.*?)\n?</think>\s*(.*)$",re.S)
for l in open("../raw/s_ag/code_train.jsonl"):
    r=json.loads(l)
    if r["category"]!="coding": continue
    msgs=[]; ok=True
    for x in r["messages"]:
        if x["role"]=="assistant":
            mm=T.match(x["content"])
            if not mm or len(mm.group(1))<80 or not mm.group(2).strip(): ok=False;break
            msgs.append({"role":"assistant","reasoning_content":mm.group(1).strip(),"content":mm.group(2).strip()})
        else: msgs.append({"role":x["role"],"content":x["content"]})
    if not ok or msgs[0]["role"]!="system": st["format"]+=1; continue
    if "```" not in msgs[-1]["content"]: st["no_code_in_answer"]+=1; continue   # keep only answers containing code
    if ntok(msgs,None)>MAXTOK: st["long"]+=1; continue
    if contaminated(msgs): st["contam"]+=1; continue
    st["ok"]+=1
    out.write(json.dumps({"src":"opus46-47-chat-coding","msgs":msgs,"notools":True},ensure_ascii=False)+"\n")
print(dict(st))
