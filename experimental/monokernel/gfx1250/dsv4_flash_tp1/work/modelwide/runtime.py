"""Experimental model-wide Flash C1 adapter, restricted to <=2048 context."""
import os,sys,logging
sys.path[:0]=['/work/modelwide','/work/full-attention-native','/work/full-attention-native/prefix','/work/full-attention-native/post-kernel','/work/full-attention-native/output-kernel','/work/ffn-c1/moe']
from scratch import LayerState
sys.path.insert(0,'/work/modelwide')
from binding import LayerWeights,compressor_args
if os.getenv("ATOM_DSV4_PRECISE_MHC")=="1":
 from model_layer_precise import model_layer
elif os.getenv("ATOM_DSV4_MATCHED_ATTN")=="1":
 from model_layer_matched import model_layer
elif os.getenv("ATOM_DSV4_SCALED_WMMA")=="1":
 from model_layer_scaled import model_layer
else:
 from model_layer import model_layer
import torch,aiter
_BATCH_SERIAL=os.getenv('ATOM_DSV4_BATCH_SERIAL')=='1'
if _BATCH_SERIAL:
 raise RuntimeError('Serialized batched full-layer prototype rejected: VGPR spilling and barrier stalls; use the batched region diagnostics instead.')
_MAX_BATCH=8 if _BATCH_SERIAL else 1
if _BATCH_SERIAL:
 sys.path.insert(0,'/work/concurrency-rollout')
 from model_layer_serial import batched_layer
log=logging.getLogger(__name__)
_registry={};_scratch=None;_prev_fp32=None

@torch.library.custom_op('atom::dsv4_modelwide_c1',mutates_args=('cache','states'))
def full_op(residual:torch.Tensor,positions:torch.Tensor,cache:torch.Tensor,states:list[torch.Tensor],prefix:str)->torch.Tensor:
 from atom.utils.forward_context import get_forward_context,AttnState
 from atom.models.deepseek_v4 import HCState
 obj,block=_registry[prefix];fc=get_forward_context();md=fc.attn_metadata
 if (not fc.context.is_dummy_run and md is not None and md.state is AttnState.DECODE and residual.shape==(1,4,4096) and positions.numel()==1):
  if os.getenv("ATOM_DSV4_MODELWIDE_VALIDATE")=="1":
   from shadow import shadow
   return shadow(obj,residual,positions,fc)
  return obj.execute(residual,positions,md)
 saved=block._experimental_full_layer;block._experimental_full_layer=None
 try:
  out=block.forward(HCState(residual,None,None,None,True),positions)
  plain=block.hc_post(out.x_prev,out.residual,out.post_mix,out.comb_mix,res_preshuffle=out.res_preshuffle)
  return aiter.mhc_res_shuffle(plain)
 finally:block._experimental_full_layer=saved

@full_op.register_fake
def fake(residual,positions,cache,states,prefix):return torch.empty_like(residual)

@torch.library.custom_op('atom::dsv4_modelwide_state_c1',mutates_args=('effect',))
def state_op(residual:torch.Tensor,positions:torch.Tensor,post:torch.Tensor|None,comb:torch.Tensor|None,previous:torch.Tensor|None,effect:torch.Tensor,prefix:str)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
 from atom.utils.forward_context import get_forward_context,AttnState
 from atom.models.deepseek_v4 import HCState
 obj,block=_registry[prefix];fc=get_forward_context();md=fc.attn_metadata
 if os.getenv("ATOM_DSV4_NATIVE_STATE")!="1" and not fc.context.is_dummy_run and md is not None and md.state is AttnState.DECODE and residual.ndim==3 and residual.shape[1:]==(4,4096) and residual.shape[0]<=_MAX_BATCH:
  out=obj.execute(residual,positions,md)
  return out,obj.post[:residual.shape[0]],obj.comb[:residual.shape[0]],obj.zero[:residual.shape[0]]
 saved=block._experimental_full_layer;block._experimental_full_layer=None
 try:
  out=block.forward(HCState(residual,post,comb,previous,True),positions)
  return out.residual.clone(),out.post_mix.clone(),out.comb_mix.clone(),out.x_prev.clone()
 finally:block._experimental_full_layer=saved

@state_op.register_fake
def state_fake(residual,positions,post,comb,previous,effect,prefix):
 m=residual.shape[0]
 return (torch.empty_like(residual),torch.empty((m,4),device=residual.device,dtype=torch.float32),torch.empty((m,4,4),device=residual.device,dtype=torch.float32),torch.empty((m,4096),device=residual.device,dtype=residual.dtype))

# Avoid Inductor's implicit-fallback diagnostic recursively formatting the
# four-output HCState DAG (exponential expansion across 43 layers).
# Inductor eagerly formats implicit-fallback diagnostics even with info logging
# disabled. Multi-output layer DAGs expand exponentially in IR __str__.
# Bound diagnostics only; preserve target and argument types, not semantics.
from torch._inductor.exc import OperatorIssue
OperatorIssue.operator_str=staticmethod(lambda target,args,kwargs: f"target: {target}; argument types: {[type(a).__name__ for a in args]}; keyword names: {list(kwargs)}")
from torch._inductor.lowering import make_fallback
make_fallback(torch.ops.atom.dsv4_modelwide_state_c1.default, warn=False)

class ServingLayer:
 def __init__(self):
  global _scratch
  if os.getenv('ATOM_DSV4_MODELWIDE_MAX_CONTEXT')!='2048':raise RuntimeError('Model-wide experimental kernel requires explicit 2048 context cap')
  if _scratch is None:
   _scratch=LayerState();a=_scratch.a.s.attn
   a.cache=torch.empty((640,512),device='cuda',dtype=torch.bfloat16)
   a.indices=torch.empty(640,device='cuda',dtype=torch.int32)
   if os.getenv('ATOM_DSV4_MATCHED_ATTN')=='1':
    a.m=torch.empty((64,64),device='cuda',dtype=torch.float32);a.l=torch.empty_like(a.m);a.a=torch.empty((64,64,512),device='cuda',dtype=torch.float32)
  self.state=_scratch;self.calls=0;self.dummy=torch.empty(0,device='cuda',dtype=torch.bfloat16)
  self.zero=torch.zeros((_MAX_BATCH,4096),device='cuda',dtype=torch.bfloat16)
  self.post=torch.zeros((_MAX_BATCH,4),device='cuda',dtype=torch.float32);self.comb=torch.eye(4,device='cuda',dtype=torch.float32)[None].repeat(_MAX_BATCH,1,1)
 def bind(self,block):
  global _prev_fp32
  self.block=block;self.weights=None;_registry[block.prefix]=(self,block)
  self.main_plan=torch.empty(8,device='cuda',dtype=torch.int32);self.index_plan=torch.empty_like(self.main_plan)
  self.row_input=torch.empty(16384,device='cuda',dtype=torch.bfloat16);self.row_output=torch.empty_like(self.row_input)
  self.extra={}
  if os.getenv('ATOM_DSV4_PRECISE_MHC')=='1':
   self.mid32=torch.empty((_MAX_BATCH,4,4096),device='cuda',dtype=torch.float32)
   self.out32=torch.empty_like(self.mid32)
   self.extra=dict(FP32IN=_prev_fp32,FP32MID=self.mid32,FP32OUT=self.out32)
   _prev_fp32=self.out32
 def execute(self,residual,positions,md):
  if self.weights is None:
   self.weights=LayerWeights(self.block)
   if _BATCH_SERIAL:self.weights.out=torch.empty((_MAX_BATCH,4,4096),device='cuda',dtype=torch.bfloat16)
  b=self.block;w=self.weights;r=w.ratio;a=b.attn
  A,F=w.args(self.state,residual);cos,sin=a.rotary_emb.cos_sin_2d()
  if r:
   MC,MM=compressor_args(a.compressor,md,r);MW=w.compressor_weights['main']
  else:MC=();MM=();MW=self.dummy
  if r==4:
   IC,IM=compressor_args(a.indexer.compressor,md,4);IW=w.compressor_weights['indexer']
  else:IC=();IM=();IW=self.dummy
  ptr=getattr(md,'kv_indptr_'+({0:'swa',4:'csa',128:'hca'}[r]))
  idx=getattr(md,'kv_indices_'+({0:'swa',4:'csa',128:'hca'}[r]))
  fn=batched_layer if _BATCH_SERIAL else model_layer
  batch_args=(self.main_plan,self.index_plan,self.row_input,self.row_output) if _BATCH_SERIAL else ()
  batch_meta=dict(M=residual.shape[0],BLOCK_STRIDE=md.block_tables.stride(0)) if _BATCH_SERIAL else {}
  obj=fn[(256,)](A,F,MC,IC,MW,IW,w.my,w.iy,positions,cos,sin,a.unified_kv,idx,ptr,md.swa_dest_rows[r],w.top,md.block_tables,w.bias if w.bias is not None else self.dummy,*batch_args,MM,IM,r,md.envelope_rows,**w.meta,**self.extra,**batch_meta,HAS_BIAS=w.bias is not None,T={0:128,4:640,128:144}[r],num_warps=4,enable_fp_fusion=False)
  self.calls+=1
  if self.calls==1:
   log.warning('MODELWIDE MONOKERNEL engaged layer=%d ratio=%d VGPR=%d LDS=%d spills=%d; accuracy not qualified',b.layer_id,r,obj.n_regs,obj.metadata.shared,obj.n_spills)
  return w.out[:residual.shape[0]]
 def __call__(self,block,hc,positions):
  from atom.models.deepseek_v4 import HCState
  if os.getenv('ATOM_DSV4_STATE_DISPATCH')=='1':
   r,p,c,x=state_op(hc.residual,positions,hc.post_mix,hc.comb_mix,hc.x_prev,self.dummy,block.prefix)
   return HCState(r,p,c,x,True)
  # Native ATOM attention also resolves persistent cache/state inside its opaque op.
  # A stable effect token keeps this boundary stateful before cache allocation.
  out=full_op(hc.residual,positions,self.dummy,[],block.prefix)
  if block.layer_id==42:
   m=out.shape[0]
   return HCState(out,self.post.expand(m,4).contiguous(),self.comb.expand(m,4,4).contiguous(),self.zero.expand(m,4096).contiguous(),True)
  return HCState(out,None,None,None,True)
