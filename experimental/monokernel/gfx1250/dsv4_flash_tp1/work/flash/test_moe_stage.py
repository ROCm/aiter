import json
from pathlib import Path
import torch
from triton.testing import do_bench_cudagraph
from aiter.ops.shuffle import moe_shuffle_weight,moe_shuffle_scale
from aiter.ops.triton.utils.shuffle import moe_weight_decode_view
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp
from aiter.ops.triton.moe.moe_routing.routing import RoutingData,ExptData
from aiter.ops.triton.moe.moe_op_gemm_a8w4 import moe_gemm_a8w4
from moe_stage_prototype import fused,stage
r=Path(__file__).parent
torch.manual_seed(1250)
x=torch.randn(1,4096,device='cuda',dtype=torch.bfloat16);xq,xs=downcast_to_mxfp(x,torch.float8_e4m3fn,axis=-1)
def weight(n,k):
 w=torch.randint(0,256,(6,n,k//2),device='cuda',dtype=torch.uint8)
 s=torch.full((6,n,k//32),121,device='cuda',dtype=torch.uint8)
 return moe_weight_decode_view(moe_shuffle_weight(w)),moe_shuffle_scale(s).transpose(-1,-2)
w1,s1=weight(4096,4096);w2,s2=weight(4096,2048)
h=torch.ones(6,device='cuda',dtype=torch.int32);o=torch.arange(7,device='cuda',dtype=torch.int32);p=o.clone();mp=torch.arange(6,device='cuda',dtype=torch.int32);gather=torch.arange(6,device='cuda',dtype=torch.int32);gamma=torch.rand(6,device='cuda')
rd=RoutingData(16,gamma,h,6,6,ExptData(h,o,p,mp))
ri,rs=moe_gemm_a8w4(xq,w1,xs,s1,routing_data=rd,gather_indx=gather,swizzle_mx_scale='GFX1250_SCALE',apply_swiglu=True,alpha=1.,limit=10.,swiglu_add_residual=False,out_mx_quant=True,preshuffled=True)
ry=moe_gemm_a8w4(ri,w2,rs,s2,routing_data=rd,gammas=gamma,swizzle_mx_scale='GFX1250_SCALE',preshuffled=True)
i=torch.empty_like(ri);is_=torch.empty_like(rs);y=torch.empty_like(ry);counter=torch.zeros(6,device='cuda',dtype=torch.uint32);health=torch.zeros(1,device='cuda',dtype=torch.int32)
a,b,c,d=[t.transpose(1,2) for t in (w1,s1,w2,s2)]
args=(xq,xs,a,b,c,d,i,is_,y,gather,h,o,p[-1:],mp,gamma,counter,health,a.stride(),b.stride(),c.stride(),d.stride(),10.)
rows=[]
for buffers in (2,3):
 def run():return fused[(192,)](*args,BUFFERS=buffers,num_warps=4)
 k=run();torch.cuda.synchronize()
 row=dict(buffers=buffers,health=health.item(),lds=k.metadata.shared,registers=k.n_regs,spills=k.n_spills)
 print(row,flush=True)
 if health.item():raise RuntimeError('bounded wait failed')
 torch.testing.assert_close(i.view(torch.uint8),ri.view(torch.uint8),rtol=0,atol=0);torch.testing.assert_close(is_,rs,rtol=0,atol=0);torch.testing.assert_close(y,ry,rtol=0,atol=0)
 for rep in range(2):run()
 torch.cuda.synchronize();torch.testing.assert_close(y,ry,rtol=0,atol=0)
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph):
  xq.view(torch.uint8).bitwise_xor_(128)
  run()
 for rep in range(1000):graph.replay()
 torch.cuda.synchronize();torch.testing.assert_close(y,ry,rtol=0,atol=0);assert health.item()==0
 counter.fill_(4294967264)
 run();torch.cuda.synchronize();assert (counter==0).all().item();assert health.item()==0
 torch.testing.assert_close(y,ry,rtol=0,atol=0)
 def baseline():
  qi,qs=moe_gemm_a8w4(xq,w1,xs,s1,routing_data=rd,gather_indx=gather,swizzle_mx_scale='GFX1250_SCALE',apply_swiglu=True,alpha=1.,limit=10.,swiglu_add_residual=False,out_mx_quant=True,preshuffled=True)
  return moe_gemm_a8w4(qi,w2,qs,s2,routing_data=rd,gammas=gamma,swizzle_mx_scale='GFX1250_SCALE',preshuffled=True)
 row['baseline_us']=do_bench_cudagraph(baseline,rep=100)*1000
 row['changing_replays']=1000;row['counter_wrap_passed']=True
 row['us']=do_bench_cudagraph(run,rep=100)*1000
 (r/f'moe-stage-b{buffers}.amdgcn').write_text(k.asm['amdgcn']);rows.append(row);print(row,flush=True)
 (r/'moe_stage_results.json').write_text(json.dumps(rows,indent=2))
