from common import *
from timed import project_timed
from initall import initialize_all
out=root/'timeline';out.mkdir(exist_ok=True)
Count=torch.empty(1536,device='cuda',dtype=torch.int32)
records=[]
for split in [8,16,32]:
 bn=64 if split==8 else 128
 grid=1536//bn*split;T=torch.empty((grid,4),device='cuda',dtype=torch.int64)
 def run():return project_timed[(grid,)](X,XS,W,WS,P,NW,Acc,Count,Y,T,BN=bn,SPLIT=split,DIRECT=True,WAVES=4,BK=128,UNROLL=1,UPSCALE=True,REPLICATE=True,CACHE='.ca',SCALAR=True,PACKED=False,MODE=1,OUTLAY=1,NVEC=1,COUNTSTRIDE=128,NATIVE=0,**kw)
 initialize_all[(6,)](Acc,Count,**kw);obj=run();torch.cuda.synchronize()
 (out/(str(split)+'.amdgcn')).write_text(obj.asm['amdgcn'])
 for _ in range(20):run()
 cg=torch.cuda.CUDAGraph()
 with torch.cuda.graph(cg):run()
 for trial in range(30):
  cg.replay();t=T.cpu();base=t[:,0].min();rel=(t-base).double()/100.;last=t[:,3]!=0
  assert last.sum().item()==2
  row=dict(split=split,bn=bn,trial=trial,cta_timestamps_us=rel.tolist(),projection_us=((t[:,1]-t[:,0]).double()/100.).tolist(),completion_us=((t[:,2]-t[:,1]).double()/100.).tolist(),norm_and_reset_us=((t[last,3]-t[last,2]).double()/100.).tolist(),last_ctas=last.nonzero().flatten().tolist(),span_us=(t[last,3].max()-base).item()/100.)
  records.append(row)
 print(split,'median stage durations',*[statistics.median([v for r in records if r['split']==split for v in r[k]]) for k in ['projection_us','completion_us','norm_and_reset_us']],flush=True)
(out/'timestamps.json').write_text(json.dumps(records,indent=2));print('DONE',flush=True)
