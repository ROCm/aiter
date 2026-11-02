import json,statistics
from pathlib import Path
import torch
from triton.testing import do_bench_cudagraph
from aiter.ops.shuffle import moe_shuffle_weight,moe_shuffle_scale
from aiter.ops.triton.utils.shuffle import moe_weight_decode_view
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp
from aiter.ops.triton.moe.moe_routing.routing import routing
from aiter.ops.triton.moe.moe_op_gemm_a8w4 import moe_gemm_a8w4
from moe_multitoken_prototype import fused
r=Path(__file__).parent;torch.manual_seed(1250);rows=[]
def weight(n,k):
 w=torch.randint(0,256,(256,n,k//2),device='cuda',dtype=torch.uint8)
 s=torch.full((256,n,k//32),121,device='cuda',dtype=torch.uint8)
 return moe_weight_decode_view(moe_shuffle_weight(w)),moe_shuffle_scale(s).transpose(-1,-2)
w1,s1=weight(4096,4096);w2,s2=weight(4096,2048)
a,b,c,d=[t.transpose(1,2) for t in (w1,s1,w2,s2)]
for m in (8,16):
 for pattern in ('distinct','hot'):
  x=torch.randn(m,4096,device='cuda',dtype=torch.bfloat16);xq,xs=downcast_to_mxfp(x,torch.float8_e4m3fn,axis=-1)
  logits=torch.randn(m,256,device='cuda')
  if pattern=='hot':logits[:,6:]-=100
  rd,gather,scatter=routing(logits,6)
  assert rd.block_m==16
  def baseline():
   ri,rs=moe_gemm_a8w4(xq,w1,xs,s1,routing_data=rd,gather_indx=gather,swizzle_mx_scale='GFX1250_SCALE',apply_swiglu=True,alpha=1.,limit=10.,swiglu_add_residual=False,out_mx_quant=True,preshuffled=True)
   return moe_gemm_a8w4(ri,w2,rs,s2,routing_data=rd,gammas=rd.gate_scal,scatter_indx=scatter,swizzle_mx_scale='GFX1250_SCALE',preshuffled=True)
  ri_ref,rs_ref=moe_gemm_a8w4(xq,w1,xs,s1,routing_data=rd,gather_indx=gather,swizzle_mx_scale='GFX1250_SCALE',apply_swiglu=True,alpha=1.,limit=10.,swiglu_add_residual=False,out_mx_quant=True,preshuffled=True)
  ry_ref=moe_gemm_a8w4(ri_ref,w2,rs_ref,s2,routing_data=rd,gammas=rd.gate_scal,swizzle_mx_scale='GFX1250_SCALE',preshuffled=True)
  ref=baseline();i=torch.empty((m*6,2048),device='cuda',dtype=xq.dtype);is_=torch.empty((m*6,64),device='cuda',dtype=torch.uint8);y=torch.empty((m*6,4096),device='cuda',dtype=torch.bfloat16);z=torch.empty_like(ref)
  count=torch.zeros(2,device='cuda',dtype=torch.uint32);health=torch.zeros(1,device='cuda',dtype=torch.int32);e=rd.expt_data
  args=(xq,xs,a,b,c,d,i,is_,y,gather,e.hist,e.token_offs_raw,e.token_offs_pad[-1:],e.block_pid_map,rd.gate_scal,count,health,z,scatter,a.stride(),b.stride(),c.stride(),d.stride(),10.,m)
  for grid,buf in ((128,3),(256,3),(256,5)):
   count.zero_();health.zero_()
   row=dict(m=m,pattern=pattern,grid=grid,buffers=buf)
   try:
    def run():return fused[(grid,)](*args,GRID=grid,BUFFERS=buf,num_warps=4)
    k=run();torch.cuda.synchronize()
    row['health']=health.item();row['counter']=count.tolist();row['x_scale_shape']=list(xs.shape);row['x_scale_stride']=list(xs.stride())
    row['intermediate_mismatches']=int((i.view(torch.uint8)!=ri_ref.view(torch.uint8)).sum().item())
    row['scale_mismatches']=int((is_!=rs_ref).sum().item())
    row['down_mismatches']=int((y!=ry_ref).sum().item())
    assert health.item()==0,'bounded wait failed'

    torch.testing.assert_close(z,ref,rtol=0,atol=0)
    torch.testing.assert_close(i.view(torch.uint8),ri_ref.view(torch.uint8),rtol=0,atol=0)
    torch.testing.assert_close(is_,rs_ref,rtol=0,atol=0)
    torch.testing.assert_close(y,ry_ref,rtol=0,atol=0)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
     xq.view(torch.uint8).bitwise_xor_(128);run()
    for rep in range(1000):graph.replay()
    torch.cuda.synchronize();assert health.item()==0;torch.testing.assert_close(z,ref,rtol=0,atol=0)
    count.fill_(4294967296-grid);run();torch.cuda.synchronize();assert (count==0).all().item();torch.testing.assert_close(z,ref,rtol=0,atol=0)
    bs=[];cs=[]
    for rep in range(5):
     for cand in ((False,True) if rep%2==0 else (True,False)):
      (cs if cand else bs).append(do_bench_cudagraph(run if cand else baseline,rep=80)*1000)
    row.update(exact=True,changing_replays=1000,wrap=True,baseline_us=statistics.median(bs),candidate_us=statistics.median(cs),baseline_samples=bs,candidate_samples=cs,lds=k.metadata.shared,vgprs=k.n_regs,spills=k.n_spills)
    (r/f'moe-m{m}-{pattern}-g{grid}-b{buf}.amdgcn').write_text(k.asm['amdgcn'])
   except Exception as ex:row['error']=str(ex)
   rows.append(row);print(row,flush=True);(r/'moe_multitoken_results.json').write_text(json.dumps(rows,indent=2))
