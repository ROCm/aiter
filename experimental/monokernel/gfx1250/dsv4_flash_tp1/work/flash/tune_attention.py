"""Experiment harness: tune existing public AITER projection wrappers."""
import argparse,copy,itertools,json,statistics,time
from pathlib import Path
import torch
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.quant import per_group_quant_hip
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_bpreshuffle
from aiter.ops.triton.gemm.basic.gemm_afp8wfp8 import gemm_afp8wfp8_preshuffle
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config
from aiter.ops.triton.utils.config_utils import load_config_json
from triton.testing import do_bench_cudagraph
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['screen','tune'],default='screen');a=p.parse_args()
r=Path(__file__).resolve().parent;grid=load_config_json(r/'tuning_grid.json');rows=[]
torch.manual_seed(1250)
for n,k in grid['shapes']:
 wf=torch.randn(n,k,device='cuda')*0.1
 wb=wf.reshape(n//128,128,k//128,128).permute(0,2,1,3)
 ws=torch.pow(2.,torch.ceil(torch.log2(wb.abs().amax((2,3)).clamp(min=1e-10)/448))).to(torch.float8_e8m0fnu)
 wq=shuffle_weight((wf/ws.float().repeat_interleave(128,0).repeat_interleave(128,1)).to(torch.float8_e4m3fn),layout=(16,16))
 for m in grid['rows']:
  x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
  xq,xs=per_group_quant_hip(x,quant_dtype=torch.float8_e4m3fn,group_size=128,transpose_scale=True,scale_type=torch.float8_e8m0fnu)
  def baseline():return gemm_a8w8_blockscale_bpreshuffle(xq,wq,xs,ws)
  ref=baseline();base_us=statistics.median(do_bench_cudagraph(baseline,rep=100)*1000 for _ in range(3))
  config,_=get_gemm_config('GEMM-AFP8WFP8_PRESHUFFLED',m,n,k,backend='gluon')
  variants=[('default',copy.deepcopy(config))]
  if a.phase=='tune':
   for bm,bn,bk,sk,nw in grid['tiles']:
    c=copy.deepcopy(config);c.update(BLOCK_SIZE_M=bm,BLOCK_SIZE_N=bn,BLOCK_SIZE_K=bk,NUM_KSPLIT=sk,num_warps=nw,kernel_type='bandwidth_bound',CTAS_M=1,CTAS_N=1)
    variants.append((f'tile_{bm}_{bn}_{bk}_{sk}_{nw}',c))
   for nb in grid['buffer_depths']:
    c=copy.deepcopy(config);c['NUM_BUFFERS']=nb;variants.append((f'lds_{nb}',c))
   for cm,cn in grid['clusters']:
    c=copy.deepcopy(config);c.update(CTAS_M=cm,CTAS_N=cn,kernel_type='bandwidth_bound');variants.append((f'cluster_{cm}_{cn}',c))
   for pf in grid['prefetch_distances']:
    c=copy.deepcopy(config);c.update(L2_PREFETCH_DISTANCE=pf,kernel_type='bandwidth_bound');variants.append((f'prefetch_{pf}',c))
  seen=set()
  for name,c in variants:
   key=json.dumps(c,sort_keys=True)
   if key in seen:continue
   seen.add(key)
   row=dict(m=m,n=n,k=k,name=name,config=c,baseline_us=base_us)
   try:
    def fn():return gemm_afp8wfp8_preshuffle(xq,wq,xs.view(torch.uint8),ws.view(torch.uint8),backend='gluon',is_x_scale_transposed=True,config=copy.deepcopy(c))
    got=fn();torch.cuda.synchronize();torch.testing.assert_close(got,ref,rtol=0,atol=0)
    err=got.float()-ref.float();row.update(relative_rmse=(err.square().mean()/ref.float().square().mean()).sqrt().item(),exact_fraction=(got==ref).float().mean().item())
    assert row['relative_rmse']<0.001
    row['us']=statistics.median(do_bench_cudagraph(fn,rep=60)*1000 for _ in range(3))
   except Exception as e:row['error']=str(e)[-1500:]
   rows.append(row);(r/f'attention_{a.phase}.json').write_text(json.dumps(rows,indent=2));print({kk:vv for kk,vv in row.items() if kk!='config'},flush=True)
 del wf,wb,ws,wq
