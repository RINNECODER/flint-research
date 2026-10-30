# nvidia/OpenCodeReasoning split_0 (R1 traces) joined to deepmind/code_contests tests; verify by execution.
import sys,json,re
sys.path.insert(0,__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from stage1 import work,CODE,tok,MAXGEN
from multiprocessing import Pool
from huggingface_hub import HfFileSystem
import pyarrow.parquet as pq
key=lambda s:re.sub(r"[^a-z0-9]","",s.lower())[:400]
def main():
    done=set(); import glob
    for fn in glob.glob("old/stage1b_*.jsonl"):
        for l in open(fn): done.add(re.sub(r"[^a-z0-9]","",json.loads(l)["user"].lower())[:400])
    idx={}
    for l in open("ccindex.jsonl"):
        d=json.loads(l); idx.setdefault(d["k"],d)
    print("index",len(idx),flush=True)
    shards=range(int(sys.argv[1]),int(sys.argv[2]))
    fs=HfFileSystem(); out=open(f"stage1b_{sys.argv[1]}.jsonl","w"); seen=set(); tot=ok=0
    for s in shards:
        p=f"datasets/nvidia/OpenCodeReasoning/split_0/train-{s:05d}-of-00030.parquet"
        with fs.open(p,"rb") as f:
            pf=pq.ParquetFile(f); cs=[]; nrow=0
            for g in range(pf.num_row_groups):
                for r in pf.read_row_group(g,columns=["id","input","output","source"]).to_pylist():
                    nrow+=1; gen=r["output"] or ""
                    if len(gen)>12000 or not gen.startswith("<think>") or gen.count("</think>")!=1: continue
                    think,ans=gen[7:].split("</think>"); think,ans=think.strip(),ans.strip()
                    m=CODE.findall(ans)
                    if len(think)<200 or not m or len(m)>2: continue
                    if len(tok.encode(gen).ids)>MAXGEN: continue
                    if key(r["input"]) in done: continue
                    t=idx.get(key(r["input"]))
                    if not t: continue
                    cs.append({"id":"ocr/"+r["id"],"title":"","user":r["input"],"think":think,"answer":ans,"code":m[-1].strip(),"pub":t["pub"],"priv":t["priv"],"file":"ocr"})
        n=0
        with Pool(3) as pool:
            for r in pool.imap_unordered(work,cs,chunksize=8):
                if r and r["id"] not in seen: seen.add(r["id"]); out.write(json.dumps(r)+"\n"); n+=1
        out.flush(); tot+=len(cs); ok+=n
        print("shard",s,"rows",nrow,"cands",len(cs),"verified",n,flush=True)
if __name__=='__main__': main()
