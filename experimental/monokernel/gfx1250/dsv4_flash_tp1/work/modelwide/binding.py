"""Bind each model layer's weights; scratch is shared only by serial C1 launches."""
import json
from pathlib import Path
import torch
from safetensors import safe_open

class LayerWeights:
 def __init__(self,block,model='/data/DeepSeek-V4-Flash'):
  self.block=block;self.layer=block.layer_id;self.ratio=block.attn.compress_ratio
  root=Path(model);index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
  def load(name):
   key=f'layers.{self.layer}.'+name
   with safe_open(str(root/index[key]),framework='pt') as f:return f.get_tensor(key).cuda().contiguous()
  self.a={i:load(name) for i,name in [(3,'hc_attn_fn'),(4,'hc_attn_scale'),(5,'hc_attn_base'),(6,'attn_norm.weight'),(24,'attn.wq_b.weight'),(34,'attn.attn_sink'),(40,'attn.wo_a.weight'),(42,'attn.wo_b.weight')]}
  self.a[12]=torch.cat([load('attn.wq_a.weight').view(torch.uint8),load('attn.wkv.weight').view(torch.uint8)]).view(torch.float8_e4m3fn)
  self.a[13]=torch.cat([load('attn.wq_a.scale').view(torch.uint8),load('attn.wkv.scale').view(torch.uint8)])
  self.a[15]=torch.cat([load('attn.q_norm.weight').float(),load('attn.kv_norm.weight').float()])
  for i,name in [(25,'attn.wq_b.scale'),(41,'attn.wo_a.scale'),(43,'attn.wo_b.scale')]:self.a[i]=load(name).view(torch.uint8)
  self.f={i:load(name) for i,name in [(2,'hc_ffn_fn'),(3,'hc_ffn_scale'),(4,'hc_ffn_base'),(5,'ffn_norm.weight')]}
  sh=block.ffn.shared_experts;ex=block.ffn.experts
  # Serving's ATOM_USE_TRITON_MOE_DECODE=1 keeps FlyDSL physical storage.
  views=ex.quant_method._triton_views_of_flydsl_weights(ex)
  w1,w2,s1,s2=views[:4]
  self.f.update({19:block.ffn.gate.weight,21:sh.gate_up_proj.weight,22:sh.gate_up_proj.weight_scale.view(torch.uint8),23:sh.w2.weight,24:sh.w2.weight_scale.view(torch.uint8),25:w1.transpose(1,2),26:s1.transpose(1,2),27:w2.transpose(1,2),28:s2.transpose(1,2)})
  self.meta=dict(zip(['W1S','S1S','W2S','S2S'],[self.f[i].stride() for i in [25,26,27,28]]))
  self.bias=getattr(block.ffn.gate,'e_score_correction_bias',None)
  self.compressor_weights={}
  for name,prefix in [('main','attn.compressor.'),('indexer','attn.indexer.compressor.')]:
   if (name=='main' and self.ratio) or (name=='indexer' and self.ratio==4):
    self.compressor_weights[name]=torch.cat([load(prefix+'wkv.weight'),load(prefix+'wgate.weight')]).contiguous()
  self.out=torch.empty((1,4,4096),device='cuda',dtype=torch.bfloat16)
  self.my=torch.empty((1,2048 if self.ratio==4 else 1024),device='cuda',dtype=torch.bfloat16)
  self.iy=torch.empty((1,512),device='cuda',dtype=torch.bfloat16)
  self.top=torch.empty(512,device='cuda',dtype=torch.int32)
 def args(self,state,residual):
  a=list(state.attn_args());f=list(state.f.args());a[0]=residual
  for i,w in self.a.items():a[i]=w
  for i,w in self.f.items():f[i]=w
  f[1]=self.out
  return tuple(a),tuple(f)

def compressor_args(comp,md,ratio):
 plan=md.compress_plans[ratio];cos,sin=comp.rotary_emb.cos_sin_2d()
 cache=comp.kv_cache;sc=comp.cache_scale
 if sc is None:sc=comp.norm.weight
 C=(comp.ape,comp.norm.weight,cos,sin,plan.compress_plan_gpu,plan.write_plan_gpu,md.state_slot_in,md.state_slot_out,comp.kv_state,comp.score_state,cache,sc,md.block_tables)
 M=(comp.head_dim,ratio,ratio==4,comp.kv_state.shape[1],comp.kv_state.stride(0),comp.kv_state.stride(1),comp.score_state.stride(0),comp.score_state.stride(1),cache.stride(0),cache.stride(1),sc.stride(0),comp.quant_mode=='per_row_fp8')
 return C,M
