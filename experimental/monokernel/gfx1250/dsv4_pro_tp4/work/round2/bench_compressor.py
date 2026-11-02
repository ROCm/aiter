import copy,json,statistics
from pathlib import Path
import torch
from aiter.tuned_gemm import tgemm
from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config
from triton.testing import do_bench_cudagraph
r=Path('/work/round2');rows=[];torch.manual_seed(1250)
for m in (64,128):
 for n in (512,1024,2048):
  k=7168;x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16);weights=[torch.randn(n,k,device='cuda',dtype=torch.bfloat16)*0.1 for _ in range(16)];w=weights[0]
  ref=tgemm.mm(x,w,otype=torch.bfloat16)
  config,_=get_gemm_config('GEMM-A16W16',m,n,k,backend='gluon')
  variants=[('default',copy.deepcopy(config))]
  for depth in (2,3,4,5):
   c=copy.deepcopy(config);c['NUM_BUFFERS']=depth;variants.append((f'lds{depth}',c))
  for name,c in variants:
   row=dict(m=m,n=n,k=k,name=name,config=c)
   try:
    got=gemm_a16w16(x,w,config=copy.deepcopy(c),backend='gluon');torch.cuda.synchronize()
    row['relative_rmse']=((got.float()-ref.float()).square().mean()/ref.float().square().mean()).sqrt().item();row['exact_fraction']=(got==ref).float().mean().item()
    torch.testing.assert_close(got,ref,rtol=.01,atol=.01)
    def base():
     for wi in weights:tgemm.mm(x,wi,otype=torch.bfloat16)
    def cand():
     for wi in weights:gemm_a16w16(x,wi,config=copy.deepcopy(c),backend='gluon')
    bs=[];cs=[]
    for i in range(3):
     for mode in ((False,True) if i%2==0 else (True,False)):(cs if mode else bs).append(do_bench_cudagraph(cand if mode else base,rep=80)*1000/len(weights))
    row.update(baseline_us=statistics.median(bs),us=statistics.median(cs))
   except Exception as e:row['error']=str(e)[-1200:]
   rows.append(row);(r/'compressor.json').write_text(json.dumps(rows,indent=2));print(row,flush=True)
