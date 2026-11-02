import os,sys,json,statistics,time
os.environ['ATOM_USE_TRITON_MOE']='1';os.environ['ATOM_USE_TRITON_GEMM']='1';os.environ['AITER_TRITON_USE_HERD']='1';os.environ['ATOM_DSV4_FLASH_MOE']='0'
sys.path[:0]=['/work/batch-full','/work/modelwide','/work/full-attention-native/prefix','/work/ffn-c1/moe','/app/ATOM']
import torch
from safetensors import safe_open
from aiter.ops.shuffle import moe_shuffle_weight,moe_shuffle_scale
from aiter.ops.triton.utils.shuffle import moe_weight_decode_view
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp
from aiter.ops.triton.moe.moe_routing.routing import routing
from atom.model_ops.fused_moe_triton import _fused_experts_silu_gugu
from aiter.ops.triton.moe.moe_op_gemm_a8w4 import moe_gemm_a8w4,get_kernel_config_gluon
from moe_multitoken_prototype import fused
from graphbatch import FlatGraphBatch
M=1;torch.manual_seed(1250)
index=json.load(open('/data/DeepSeek-V4-Flash/model.safetensors.index.json'))['weight_map'];handles={}
def load(k):
 f=index[k]
 if f not in handles:handles[f]=safe_open('/data/DeepSeek-V4-Flash/'+f,framework='pt')
 return handles[f].get_tensor(k).view(torch.uint8).cuda()
# Exact checkpoint gate/up interleave, then native gfx1250 WMMA shuffle.
w13=torch.empty((256,4096,2048),device='cuda',dtype=torch.uint8);sc13=torch.empty((256,4096,128),device='cuda',dtype=torch.uint8);wd=torch.empty((256,4096,1024),device='cuda',dtype=torch.uint8);sd=torch.empty((256,4096,64),device='cuda',dtype=torch.uint8)
for e in range(256):
 p=f'layers.0.ffn.experts.{e}.'
 w13[e,0::2]=load(p+'w1.weight');w13[e,1::2]=load(p+'w3.weight');sc13[e,0::2]=load(p+'w1.scale');sc13[e,1::2]=load(p+'w3.scale');wd[e]=load(p+'w2.weight');sd[e]=load(p+'w2.scale')
 if e%64==0:print('loaded',e,flush=True)
w1=moe_weight_decode_view(moe_shuffle_weight(w13));s1=moe_shuffle_scale(sc13).transpose(-1,-2);w2=moe_weight_decode_view(moe_shuffle_weight(wd));s2=moe_shuffle_scale(sd).transpose(-1,-2)
del w13,sc13,wd,sd;torch.cuda.empty_cache()
a,b,c,d=[t.transpose(1,2) for t in (w1,s1,w2,s2)]

from routing_batch import route_body
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from combined import sync
@g.jit
def run(X,XS,W1,S1,W2,S2,I,IS,Y,Gather,Hist,Offs,Pad,Map,Gamma,Count,Health,Z,Scatter,LG,CH,WE,CNT,
 W1S:gl.constexpr,S1S:gl.constexpr,W2S:gl.constexpr,S2S:gl.constexpr,M:gl.constexpr):
 route_body(LG,Hist,Offs,Pad,Map,Gather,Scatter,Gamma,None,False,CH,WE,CNT,Health,M)
 sync(CNT,Health,256,3)
 fused(X,XS,W1,S1,W2,S2,I,IS,Y,Gather,Hist,Offs,Pad,Map,Gamma,Count,Health,Z,Scatter,W1S,S1S,W2S,S2S,10.,M,256,3)
def alloc(m):
 def e(shape,dt=torch.int32):return torch.zeros(shape,device='cuda',dtype=dt)
 return [e((m*6,2048),torch.float8_e4m3fn),e((m*6,64),torch.uint8),e((m*6,4096),torch.bfloat16),e(m*6),e(256),e(256),e(1),e(m*6),e(m*6,torch.bfloat16),e(2),e(1),e((m,4096),torch.bfloat16),e(m*6),e((m,256)),e((m,256),torch.bfloat16),e(1)]
results=[]
for m in [1,2,4,8]:
 for collision in [False,True]:
  torch.manual_seed(6764);x=torch.randn((m,4096),device='cuda',dtype=torch.bfloat16);logits=torch.randn((m,256),device='cuda',dtype=torch.bfloat16)
  if collision:logits[:,:6]+=10
  xq,xs=downcast_to_mxfp(x,torch.float8_e4m3fn,axis=-1);out=alloc(m)
  obj=run[(256,)](xq,xs,a,b,c,d,*out[:13],logits,*out[13:],a.stride(),b.stride(),c.stride(),d.stride(),m,num_warps=4,enable_fp_fusion=False)
  refs=[]
  for row in range(m):
   t=alloc(1);run[(256,)](xq[row],xs[row],a,b,c,d,*t[:13],logits[row],*t[13:],a.stride(),b.stride(),c.stride(),d.stride(),1,num_warps=4,enable_fp_fusion=False);refs.append(t)
  torch.cuda.synchronize();ref=torch.cat([t[11] for t in refs]);value=out[11]
  check=dict(m=m,collision=collision,equal=torch.equal(value,ref),relative_rms=((value.float()-ref.float()).square().mean()/ref.float().square().mean()).sqrt().item(),health=int(out[10]),vgpr=obj.n_regs,lds=obj.metadata.shared,spills=obj.n_spills)
  print(check,flush=True);results.append(check);assert check['equal'] and check['health']==0
open('/work/batch-full/moe-checks.json','w').write(json.dumps(results,indent=2))
