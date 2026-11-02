"""Default-off, experimental Flash layer-0 C1 adapter; not accuracy-qualified."""
import logging
import torch
import aiter
from layer_setup import LayerState
from serving_kernel import serving_full_layer
log=logging.getLogger(__name__)
_registry={}

@torch.library.custom_op('atom::dsv4_experimental_full_layer',mutates_args=('cache',))
def full_layer_op(residual:torch.Tensor,positions:torch.Tensor,cache:torch.Tensor,prefix:str)->torch.Tensor:
 from atom.utils.forward_context import get_forward_context
 from atom.models.deepseek_v4 import HCState
 obj,block=_registry[prefix];fc=get_forward_context();hc=HCState(residual,None,None,None,True)
 if obj.eligible(block,hc,positions,fc):return obj.execute(block,hc,positions,fc)
 # Execute the actual native block for prefill/dummy runs, then materialize its
 # pending post so both branches have the same residual output contract.
 saved=block._experimental_full_layer;block._experimental_full_layer=None
 try:
  out=block.forward(hc,positions)
  plain=block.hc_post(out.x_prev,out.residual,out.post_mix,out.comb_mix,res_preshuffle=out.res_preshuffle)
  return aiter.mhc_res_shuffle(plain)
 finally:block._experimental_full_layer=saved

@full_layer_op.register_fake
def _fake(residual,positions,cache,prefix):return torch.empty_like(residual)

class ServingLayer:
 def __init__(self):
  self.state=LayerState();self.calls=0
  self.zero=torch.zeros((1,4096),device='cuda',dtype=torch.bfloat16)
  self.post=torch.zeros((1,4),device='cuda',dtype=torch.float32)
  self.comb=torch.eye(4,device='cuda',dtype=torch.float32).view(1,4,4)
  self.dummy_cache=torch.empty(0,device='cuda',dtype=torch.bfloat16)
 def bind(self,block):_registry[block.prefix]=(self,block)
 def eligible(self,block,hc,positions,fc):
  from atom.utils.forward_context import AttnState
  md=fc.attn_metadata
  return (not fc.context.is_dummy_run and md is not None and md.state is AttnState.DECODE
    and hc.residual.shape==(1,4,4096) and positions.numel()==1 and block.layer_id==0
    and block.attn.compress_ratio==0 and block.attn.unified_kv.dtype==torch.bfloat16
    and md.swa_dest_rows is not None)
 def execute(self,block,hc,positions,fc):
  s=self.state;md=fc.attn_metadata;s.r=hc.residual;cache=block.attn.unified_kv
  assert cache.is_contiguous() and cache.shape[-1]==512
  cos,sin=block.attn.rotary_emb.cos_sin_2d()
  if self.calls==0 and not torch.cuda.is_current_stream_capturing():
   import json
   from pathlib import Path
   meta={'position':positions.tolist(),'cache_shape':list(cache.shape),'cache_stride':list(cache.stride()),'cache_ptr':cache.data_ptr(),'ptr':md.kv_indptr_swa.tolist(),'indices':md.kv_indices_swa[:256].tolist(),'dest':md.swa_dest_rows[0].tolist(),'cos_shape':list(cos.shape),'cos_dtype':str(cos.dtype),'res_shape':list(hc.residual.shape),'res_stride':list(hc.residual.stride())}
   Path('/work/full-layer-native/serving-metadata.json').write_text(json.dumps(meta,indent=2))
   torch.save(hc.residual.cpu(),'/work/full-layer-native/serving-capture-residual.pt')
  s.obj=serving_full_layer[(256,)](s.attn_args(),s.f.args(),positions,cos,sin,cache,md.kv_indices_swa,md.kv_indptr_swa,md.swa_dest_rows[0],**s.f.meta(),REALPAGES=cache.numel()//512,num_warps=4,enable_fp_fusion=False)
  self.calls+=1
  if self.calls==1:log.warning('EXPERIMENTAL full-layer monokernel engaged: layer=0 C1, not accuracy-qualified, VGPR=%d LDS=%d spills=%d',s.obj.n_regs,s.obj.metadata.shared,s.obj.n_spills)
  return s.f.out
 def __call__(self,block,hc,positions):
  from atom.models.deepseek_v4 import HCState
  cache=getattr(block.attn,'unified_kv',None)
  out=full_layer_op(hc.residual,positions,self.dummy_cache if cache is None else cache,block.prefix)
  # Pending identity post preserves ATOM's fused post/pre path in the next layer.
  m=hc.residual.shape[0]
  return HCState(out,self.post.expand(m,4).contiguous(),self.comb.expand(m,4,4).contiguous(),self.zero.expand(m,4096).contiguous(),True)
