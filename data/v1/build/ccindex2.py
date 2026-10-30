import json,re
from huggingface_hub import HfFileSystem
import pyarrow.parquet as pq
fs=HfFileSystem()
key=lambda s:re.sub(r"[^a-z0-9]","",s.lower())[:400]
files=sorted(fs.glob("datasets/deepmind/code_contests/data/*.parquet"))
out=open("ccindex.jsonl","w"); n=0
for p in files:
    with fs.open(p,"rb") as f:
        pf=pq.ParquetFile(f)
        for g in range(pf.num_row_groups):
            for r in pf.read_row_group(g,columns=["description","public_tests","private_tests","generated_tests"]).to_pylist():
                pub=r["public_tests"]; priv=r["private_tests"]; gen=r["generated_tests"]
                if not pub["input"]: continue
                pi=(priv["input"]+gen["input"])[:6]; po=(priv["output"]+gen["output"])[:6]
                if not pi: continue
                out.write(json.dumps({"k":key(r["description"]),"pub":pub,"priv":{"input":pi,"output":po}})+"\n"); n+=1
    print(p[-40:],n,flush=True)
print("done")
