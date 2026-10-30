import sys,json
sys.path.insert(0,'.')  # run from data/v2/build
from render import tok,EOS
O='../'
want={"fable5-dsf-full":1,"opus46-47-chat-coding":1}
bad=0
for l in open(O+'train.jsonl'):
    r=json.loads(l); ids,mask=r['ids'],r['mask']
    assert len(ids)==len(mask) and ids[-1]==EOS and mask[-1]==1 and len(ids)<=6152
    # mask contiguous runs must each end with EOS
    runs=[];s=None
    for i,m in enumerate(mask+[0]):
        if m and s is None: s=i
        if not m and s is not None: runs.append((s,i)); s=None
    if any(ids[e-1]!=EOS for s,e in runs): bad+=1
    if r['src'] in want and want[r['src']]:
        want[r['src']]=0
        print("="*20,r['src'],"len",len(ids),"trained",sum(mask),"assistant runs",len(runs))
        for k,(s,e) in enumerate(runs[:3]):
            print(f"--- trained run {k} [{s}:{e}] ---"); print(tok.decode(ids[s:e],skip_special_tokens=False)[:500].replace("\n","\\n"))
        print("--- untrained head (first 60 tokens) ---"); print(tok.decode(ids[:60],skip_special_tokens=False).replace("\n","\\n"))
        print("--- context just before run 1 (mask=0) ---"); print(tok.decode(ids[runs[1][0]-40:runs[1][0]],skip_special_tokens=False).replace("\n","\\n") if len(runs)>1 else "")
print("examples whose masked run does not end in EOS:",bad)
