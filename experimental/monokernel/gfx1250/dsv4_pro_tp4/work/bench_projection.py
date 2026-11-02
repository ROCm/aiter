import json,statistics
from pathlib import Path
import torch,aiter
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.quant import per_group_quant_hip
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_bpreshuffle
from aiter.ops.triton.gemm.basic.gemm_afp8wfp8 import gemm_afp8wfp8_preshuffle
from triton.testing import do_bench_cudagraph
rows=[]
torch.manual_seed(123)
k,n=7168,2048
wf=torch.randn(n,k,device='cuda')*0.1
# Quantize each 128x128 block with a power-of-two scale.
wblocks=wf.reshape(n//128,128,k//128,128).permute(0,2,1,3)
amax=wblocks.abs().amax((2,3)).clamp(min=1e-10)
ws=torch.pow(2.,torch.ceil(torch.log2(amax/448))).to(torch.float8_e8m0fnu)
w=(wf/ws.float().repeat_interleave(128,0).repeat_interleave(128,1)).to(torch.float8_e4m3fn)
wq=shuffle_weight(w,layout=(16,16))
for m in (1,8,16,32,64,128):
 x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
 xq,xs=per_group_quant_hip(x,quant_dtype=torch.float8_e4m3fn,group_size=128,transpose_scale=True,scale_type=torch.float8_e8m0fnu)
 def baseline():return gemm_a8w8_blockscale_bpreshuffle(xq,wq,xs,ws)
 def gluon():return gemm_afp8wfp8_preshuffle(xq,wq,xs.view(torch.uint8),ws.view(torch.uint8),backend='gluon',is_x_scale_transposed=True)
 ref=baseline();got=gluon();torch.cuda.synchronize()
 torch.testing.assert_close(got,ref,rtol=0.01,atol=0.001)
 err=(got.float()-ref.float());relrmse=(err.square().mean()/ref.float().square().mean()).sqrt().item()
 assert relrmse<0.001,relrmse
 bs=[];gs=[]
 for i in range(5):
  for name,fn in ([('b',baseline),('g',gluon)] if i%2==0 else [('g',gluon),('b',baseline)]):
   val=do_bench_cudagraph(fn,rep=100)*1000;(bs if name=='b' else gs).append(val)
 rows.append(dict(m=m,baseline_us=statistics.median(bs),gluon_us=statistics.median(gs),relative_rmse=relrmse,max_abs_error=err.abs().max().item(),exact_fraction=(got==ref).float().mean().item()))
 print(rows[-1],flush=True)
Path('/work/results/projection_microbench.json').write_text(json.dumps(rows,indent=2))
