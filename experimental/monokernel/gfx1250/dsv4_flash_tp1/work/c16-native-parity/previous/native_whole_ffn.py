from previous.common import *
import os,sys,json,statistics,time
os.environ['ATOM_USE_TRITON_MOE']='1';os.environ['ATOM_USE_TRITON_GEMM']='1';os.environ['AITER_TRITON_USE_HERD']='1';os.environ['ATOM_DSV4_FLASH_MOE']='0'
sys.path.insert(0,'/app/ATOM')
import torch
from safetensors import safe_open
from aiter.ops.shuffle import moe_shuffle_weight,moe_shuffle_scale
from aiter.ops.triton.utils.shuffle import moe_weight_decode_view
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp
from aiter.ops.triton.moe.moe_routing.routing import routing
from atom.model_ops.fused_moe_triton import _fused_experts_silu_gugu
from aiter.ops.triton.moe.moe_op_gemm_a8w4 import moe_gemm_a8w4,get_kernel_config_gluon
from previous.optimized_moe import fused
from graphbatch import FlatGraphBatch
M=int('16');torch.manual_seed(1250)
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
torch.manual_seed(6764);x=torch.randn((M,4096),device='cuda',dtype=torch.bfloat16)
rf=torch.load('/work/ffn-c1/shared-router/fixture.pt',weights_only=False);rw=rf['layers.0.ffn.gate.weight'].cuda();logits=torch.nn.functional.linear(x,rw);None
rd,gather,scatter=routing(logits,6,score_mode='sqrtsoftplus',renorm=True,routed_scaling_factor=1.5)
xq,xs=downcast_to_mxfp(x,torch.float8_e4m3fn,axis=-1)
print('routing',rd.block_m,rd.n_expts_act,rd.expt_data.hist.tolist(),flush=True)
def native():return _fused_experts_silu_gugu(x,w1,w2,rd,gather,scatter,s1,s2,'GFX1250_SCALE','GFX1250_SCALE',swiglu_limit=10.,preshuffled=True)
i=torch.empty((M*6,2048),device='cuda',dtype=xq.dtype);is_=torch.empty((M*6,64),device='cuda',dtype=torch.uint8);y=torch.empty((M*6,4096),device='cuda',dtype=torch.bfloat16);z=torch.empty((M,4096),device='cuda',dtype=torch.bfloat16);count=torch.zeros(2,device='cuda',dtype=torch.uint32);health=torch.zeros(1,device='cuda',dtype=torch.int32);e=rd.expt_data
args=(a,b,c,d,i,is_,y,gather,e.hist,e.token_offs_raw,e.token_offs_pad[-1:],e.block_pid_map,rd.gate_scal,count,health,z,scatter,a.stride(),b.stride(),c.stride(),d.stride(),10.,M)
obj=None
from previous.isolated import moe_region
ct=torch.zeros(1,device='cuda',dtype=torch.int32)
def candidate():
 global obj
 if obj is None:
  obj=moe_region.warmup(x,xq,xs,args,ct,health,grid=(256,),**kw);obj._init_handles()
  print('resources',obj.n_regs,obj.n_spills,obj.metadata.shared,flush=True)
  assert obj.n_spills==0,'Refusing spilling persistent kernel'
 obj=moe_region[(256,)](x,xq,xs,args,ct,health,**kw)
 return z

print('native configs',get_kernel_config_gluon(M*6,4096,4096,rd,True),get_kernel_config_gluon(M*6,4096,2048,rd,False),flush=True)
ref=native().clone();candidate();torch.cuda.synchronize()
bench('routed_moe',{'native':native,'fused':candidate},dict(**err(z,ref),health=int(health),note='routing metadata prepared outside both timings; input MXFP8 quantization included in both'))

# Matched whole FFN boundary: learned router + shared/routed experts + sum + post.
import importlib.util
sp=importlib.util.spec_from_file_location('shared_reference','/work/ffn-c1/shared-router/native.py');nm=importlib.util.module_from_spec(sp);sp.loader.exec_module(nm)
ns=nm.NativeSharedRouter()
from previous.scratch_batch import BatchState
from previous.isolated import whole_ffn
st=BatchState(M,128);F=st.f
res=torch.randn((M,4,4096),device='cuda',dtype=torch.bfloat16)*.05;res_s=aiter.mhc_res_shuffle(res)
post=torch.rand((M,4,1),device='cuda');comb=torch.rand((M,4,4),device='cuda');mix=torch.zeros((M,24),device='cuda');mix[:,4:8]=post[:,:,0];mix[:,8:]=comb.reshape(M,16)
F[0]=res;F[1]=st.output;F[6]=mix;F[9]=x;F[19]=ns.learned_gate.weight
F[21]=ns.shared.gate_up_proj.weight;F[22]=ns.shared.gate_up_proj.weight_scale.view(torch.uint8);F[23]=ns.shared.w2.weight;F[24]=ns.shared.w2.weight_scale.view(torch.uint8)
F[25]=a;F[26]=b;F[27]=c;F[28]=d
fargs=tuple(F);refout=torch.empty_like(res)
def native_ffn():
 logit=ns.learned_gate(x)
 rd1,g1,s1_=routing(logit,6,score_mode='sqrtsoftplus',renorm=True,routed_scaling_factor=1.5)
 so=ns.shared(x)
 mo=_fused_experts_silu_gugu(x,w1,w2,rd1,g1,s1_,s1,s2,'GFX1250_SCALE','GFX1250_SCALE',swiglu_limit=10.,preshuffled=True)
 aiter.mhc_post(refout,so+mo,res_s,post,comb,res_preshuffle=True)
 return refout
from frozen_before.isolated import whole_ffn as previous_ffn
def fused_ffn(op=whole_ffn):
 op[(256,)](fargs,a.stride(),b.stride(),c.stride(),d.stride(),M,st.chosen,st.weights,**kw)
 return st.output
rff=native_ffn().clone();fused_ffn();torch.cuda.synchronize()
plain=st.output.reshape(64,4,M,64).permute(2,1,0,3).reshape(M,4,4096)
bench('whole_ffn',{'native':native_ffn,'previous':lambda:fused_ffn(previous_ffn),'fused':fused_ffn},dict(final=err(plain,rff),health=int(F[-1]),note='same checkpoint experts and inputs; learned router/HERD at C16, final layout differs natively'))
