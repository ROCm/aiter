import json,statistics,traceback
from pathlib import Path
import torch,triton,aiter
from triton.testing import do_bench_cudagraph
import importlib
public=importlib.import_module("aiter.ops.triton.fusions.mhc")
from aiter.ops.mhc import mhc_res_shuffle_enabled, MHC_RES_KS_GLUON
from mhc_stage_prototype import _mhc_one_launch_kernel,create_layouts,create_reduce_apply_layouts
root=Path('/work/flash'); records=[]

def prepare(m,last,fuse_qkv=False):
 c=4096; n=4; torch.manual_seed(1250+m)
 x=torch.randn(m,c,device='cuda',dtype=torch.bfloat16)
 r=aiter.mhc_res_shuffle(torch.randn(m,4,c,device='cuda',dtype=torch.bfloat16))
 p=torch.sigmoid(torch.randn(m,4,device='cuda'))*2
 co=torch.softmax(torch.randn(m,4,4,device='cuda'),-1)
 fn=aiter.mhc_shuffle_fn(torch.randn(24,4*c,device='cuda')*.01)
 a=torch.tensor([.25,.25,.25],device='cuda');b=torch.randn(24,device='cuda')*.2
 norm=(1+torch.randn(c,device='cuda')*.05).bfloat16()
 args=(x,r,p,co,fn,a,b,norm)
 cfg=public.get_mhc_post_pre_gluon_gfx1250_config(m)
 shuffled=mhc_res_shuffle_enabled(m)
 ks=MHC_RES_KS_GLUON
 split=public._mhc_post_pre_gluon_split_k(m,c,shuffled,ks,cfg,256)
 bm=16; kl=c//(split*ks)
 stages=public._gluon_split_num_stages(cfg,triton.cdiv(m,bm)*split,kl,True,256)
 layouts=create_layouts(4,bm,ks,32,stages,True,4)
 chunk,apply,r4,r16,sk,sm,rs=create_reduce_apply_layouts(4,1,4,split)
 assert chunk==1024
 partial=torch.empty(split,m,32,device='cuda');sq=torch.empty(split,m,device='cuda')
 post=torch.empty(m,4,1,device='cuda');comb=torch.empty(m,4,4,device='cuda')
 li=torch.empty_like(x);res=torch.empty_like(r)
 q=torch.empty_like(x,dtype=torch.float8_e4m3fn);qs=torch.empty(m,c//128,device='cuda',dtype=torch.float8_e8m0fnu)
 counter=torch.zeros(2,device='cuda',dtype=torch.uint32);health=torch.zeros(1,device='cuda',dtype=torch.int32)
 grid=(triton.cdiv(m,bm),split)
 from aiter.ops.shuffle import shuffle_weight
 from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_bpreshuffle
 from aiter.ops.triton.gemm.basic.gemm_afp8wfp8 import gemm_afp8wfp8_preshuffle
 qw=shuffle_weight(torch.randn(1536,c,device='cuda').to(torch.float8_e4m3fn),layout=(16,16))
 qws=torch.ones(12,c//128,device='cuda').to(torch.float8_e8m0fnu)
 qo=torch.empty(m,1536,device='cuda',dtype=torch.bfloat16)
 qcfg=dict(BLOCK_SIZE_M=16,BLOCK_SIZE_N=64,BLOCK_SIZE_K=256,NUM_KSPLIT=1,
           GROUP_SIZE_M=1,CTAS_M=1,CTAS_N=1,kernel_type='bandwidth_bound',
           num_warps=4,cache_modifier='.ca',NUM_BUFFERS=3,LOOP_UNROLL_FACTOR=1,
           B_SCALE_TDM=True,waves_per_eu=0)
 def candidate():
  kern=_mhc_one_launch_kernel[grid](partial,sq,res,x,r,fn.view(torch.bfloat16),p,co,m,c,32,
    c,4,24,32,bm,ks,kl,split,stages,shuffled,True,*layouts,
    post,comb,li,a,b,norm,q,qs.view(torch.uint8),counter,health,
    apply,r4,r16,sk,sm,rs,grid[0]*grid[1],20,last,qw,qws.view(torch.uint8),qo,fuse_qkv,num_warps=4,enable_fp_fusion=True)
  return (post,comb,li,res,q,qs)+((qo,) if fuse_qkv else ()),kern
 def baseline():
  outputs=public.mhc_post_pre_quant_gfx1250(*args,res_preshuffle=True)
  if fuse_qkv:
   output=gemm_afp8wfp8_preshuffle(outputs[4],qw,outputs[5].view(torch.uint8),qws.view(torch.uint8),backend='gluon',is_x_scale_transposed=True,config=dict(qcfg))
   return (*outputs,output)
  return outputs
 return args,counter,health,candidate,baseline,grid

# Validate the existing mHC+quant hybrid on all Flash buckets against the
# unfused baseline, including original HIP quantization and graph replay.
from aiter.ops.quant import per_group_quant_hip
preflight=[]
for pm in (1,2,7,8,16,32,64,128):
 pargs,pc,ph,pcand,pbase,pg=prepare(pm,False)
 expected=aiter.mhc_fused_post_pre(*pargs[:7],norm_weight=pargs[7],
        res_preshuffle=True,w_preshuffle_bf16=True,hc_post_mult_value=2.,backend='gluon')
 pq,pqs=per_group_quant_hip(expected[2],quant_dtype=torch.float8_e4m3fn,
        group_size=128,transpose_scale=True,scale_type=torch.float8_e8m0fnu)
 expected=(*expected,pq,pqs)
 for _ in range(2):
  actual=pbase()
  for got,ref in zip(actual,expected):torch.testing.assert_close(got.view(torch.uint8),ref.view(torch.uint8),rtol=0,atol=0)
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph):actual=pbase()
 for _ in range(3):graph.replay()
 torch.cuda.synchronize()
 for got,ref in zip(actual,expected):torch.testing.assert_close(got.view(torch.uint8),ref.view(torch.uint8),rtol=0,atol=0)
 preflight.append(dict(m=pm,exact=True));print('MHC hybrid preflight',pm,'exact',flush=True)
 (root/'mhc_hybrid_validation.json').write_text(json.dumps(preflight,indent=2))

for m,last,fuse in [(m,last,False) for m in (1,8,16,32) for last in (False,True)]+[(m,False,True) for m in (1,8,16)]:
 try:
  args,counter,health,cand,base,grid=prepare(m,last,fuse)
  for _ in range(2):
   expected=base();actual,k=cand();torch.cuda.synchronize()
   assert health.item()==0,'bounded barrier timed out'
   for i,(a,b) in enumerate(zip(actual,expected)):
    torch.testing.assert_close(a.view(torch.uint8),b.view(torch.uint8),rtol=0,atol=0,msg=lambda msg:f'output {i}: {msg}')
  counter.fill_(2**32-grid[0]*grid[1]*2)
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):
   args[0].mul_(-1)
   actual,k=cand()
  for _ in range(1000):graph.replay()
  torch.cuda.synchronize();assert health.item()==0
  expected=base()
  for a,b in zip(actual,expected):torch.testing.assert_close(a.view(torch.uint8),b.view(torch.uint8),rtol=0,atol=0)
  bs=[];cs=[]
  for rep in range(5):
   for which in ((0,1) if rep%2==0 else (1,0)):
    (cs if which else bs).append(do_bench_cudagraph(cand if which else base,rep=80)*1000)
  (root/f'mhc_mono_m{m}_last{int(last)}_qkv{int(fuse)}.amdgcn').write_text(k.asm['amdgcn'])
  row=dict(m=m,last_producer=last,fuse_qkv=fuse,baseline_us=statistics.median(bs),one_launch_us=statistics.median(cs),shared=k.metadata.shared,n_regs=k.n_regs,n_spills=k.n_spills,grid=grid,exact=True)
 except Exception as e:
  row=dict(m=m,last_producer=last,fuse_qkv=fuse,error=str(e));traceback.print_exc()
 records.append(row);print(json.dumps(row),flush=True)
 (root/'mhc_stage_results.json').write_text(json.dumps(records,indent=2))
