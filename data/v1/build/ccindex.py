import json,re
from huggingface_hub import HfFileSystem
import pyarrow.parquet as pq
fs=HfFileSystem()
key=lambda s:re.sub(r"[^a-z0-9]","",s.lower())[:400]
files=sorted(p for p in fs.glob("datasets/deepmind/code_contests/data/train-*.parquet"))
out=open("ccindex.jsonl","w"); n=0
for p in files:
    with fs.open(p,"rb") as f:
        pf=pq.ParquetFile(f)
        for g in range(pf.num_row_groups):
            t=pf.read_row_group(g,columns=["description","public_tests","private_tests"]).to_pylist()
            for r in t:
                pub,priv=r["public_tests"],r["private_tests"]
                if not pub["input"] or not priv["input"]: continue
                out.write(json.dumps({"k":key(r["description"]),"pub":pub,"priv":{"input":priv["input"][:6],"output":priv["output"][:6]}})+"\n"); n+=1
    print(p[-40:],n,flush=True)
print("done")
