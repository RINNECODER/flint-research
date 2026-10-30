# keep the longest (final) row per trajectory task+sha
import glob,json,sys
import pyarrow.parquet as pq
best={}
for f in sorted(glob.glob(sys.argv[1]+'/0*.parquet')):
    for r in pq.read_table(f).to_pylist():
        k=r['source_trajectory_sha256']
        if k not in best or r['assistant_step']>best[k]['assistant_step']: best[k]=r
with open(sys.argv[2],'w') as o:
    for r in best.values():
        r['messages']=[{k:v for k,v in m.items() if v not in (None,'',[])} for m in r['messages']]
        o.write(json.dumps(r,ensure_ascii=False)+'\n')
print(len(best))
