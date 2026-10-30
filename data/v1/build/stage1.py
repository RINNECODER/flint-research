# Filter open-r1/codeforces-cots python traces (short), verify code by execution on tests.
import glob, json, re, sys, os, subprocess, tempfile, resource
import pyarrow.parquet as pq
from multiprocessing import Pool
from tokenizers import Tokenizer
M=__import__("os").path.expanduser("~/.cache/huggingface/hub/models--dima6312--Flint-40B-A2.5B-abliterated-Rec-4bit/snapshots/c0d2388f441399e43a0bdeccccf14f66043fedad/tokenizer.json")
tok=Tokenizer.from_file(M)
MAXGEN=2700  # tokens of generation (think+answer)
CODE=re.compile(r"```(?:python|py|python3)?\n(.*?)```",re.S)
def lim():
    resource.setrlimit(resource.RLIMIT_CPU,(6,6))
def run(code,inp,d):
    try:
        r=subprocess.run(["/usr/bin/python3","-I","-c",code],input=inp,capture_output=True,text=True,timeout=5,cwd=d,preexec_fn=lim)
        return r.returncode==0, r.stdout
    except Exception: return False,""
norm=lambda s:[l.rstrip() for l in s.strip().splitlines()]
def work(c):
    with tempfile.TemporaryDirectory() as d:
        pub=[]
        for i,o in zip(c["pub"]["input"],c["pub"]["output"]):
            ok,out=run(c["code"],i,d)
            if not ok or norm(out)!=norm(o): return None
            pub.append({"input":i,"stdout":out})
        for i,o in zip(c["priv"]["input"][:6],c["priv"]["output"][:6]):
            ok,out=run(c["code"],i,d)
            if not ok or norm(out)!=norm(o): return None
    c=dict(c); c["pub_runs"]=pub; del c["pub"],c["priv"]; return c
def cands():
    for f in sorted(glob.glob("raw/cf/*/*.parquet")):
        pf=pq.ParquetFile(f)
        for g in range(pf.num_row_groups):
            for r in pf.read_row_group(g,columns=["id","title","prompt","generation","finish_reason","problem_type","public_tests","private_tests","messages"]).to_pylist():
                gen=r["generation"] or ""
                if r["finish_reason"]!="stop" or r["problem_type"]!="diff" or not gen.startswith("<think>") or gen.count("</think>")!=1: continue
                if not (r["public_tests"] or {}).get("input") or not (r["private_tests"] or {}).get("input"): continue
                think,ans=gen[7:].split("</think>")
                think,ans=think.strip(),ans.strip()
                m=CODE.findall(ans)
                if len(think)<200 or not m or len(m)>2: continue
                if len(tok.encode(gen).ids)>MAXGEN: continue
                user=r["messages"][0]["content"] if r["messages"] and r["messages"][0]["role"]=="user" else r["prompt"]
                yield {"id":r["id"],"title":r["title"],"user":user,"think":think,"answer":ans,"code":m[-1].strip(),"pub":r["public_tests"],"priv":r["private_tests"],"file":f.split("/")[-2]}
if __name__=="__main__":
    cs=list(cands()); print("candidates",len(cs),flush=True)
    seen=set(); out=open("stage1.jsonl","w"); n=0
    with Pool(6) as p:
        for r in p.imap_unordered(work,cs,chunksize=8):
            if r and r["id"] not in seen: seen.add(r["id"]); out.write(json.dumps(r)+"\n"); n+=1
    print("verified",n)
