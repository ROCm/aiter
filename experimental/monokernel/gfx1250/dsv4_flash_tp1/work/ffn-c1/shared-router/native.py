import os,sys,json,copy
os.environ.setdefault('ATOM_USE_TRITON_GEMM','1');os.environ.setdefault('ATOM_FP8_BLOCKSCALE_USE_E8M0_SCALE','1')
sys.path.insert(0,'/app/aiter');sys.path.insert(0,'/app/ATOM')
from pathlib import Path
import torch
from types import SimpleNamespace
from transformers import AutoConfig
from safetensors import safe_open
from atom.config import QuantizationConfig
from atom.model_ops import linear
from atom.models.deepseek_v4 import Expert
from atom.model_ops.triton_hash_topk import hash_topk_triton
from atom.model_ops.moe import FusedMoE
from aiter.ops.triton.gemm.basic.gemm_afp8wfp8 import gemm_afp8wfp8_preshuffle
from atom.models.deepseek_v4 import aiter_silu_and_mul

class NativeSharedRouter:
 def __init__(self):
  path=Path(__file__).parent/'fixture.pt';model=Path('/data/DeepSeek-V4-Flash')
  if path.exists():f=torch.load(path,weights_only=False)
  else:
   index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];f={}
   keys=['layers.0.ffn.shared_experts.'+w+s for w in ['w1','w3','w2'] for s in ['.weight','.scale']]+['layers.0.ffn.gate.weight','layers.0.ffn.gate.tid2eid','layers.3.ffn.gate.weight','layers.3.ffn.gate.bias']
   for key in keys:
    with safe_open(model/index[key],framework='pt',device='cpu') as sf:f[key]=sf.get_tensor(key)
   torch.save(f,path)
  self.fixture=f;hf=AutoConfig.from_pretrained(model,trust_remote_code=True);qc=QuantizationConfig(hf)
  old=linear.get_tp_group;linear.get_tp_group=lambda:SimpleNamespace(rank_in_group=0,world_size=1)
  try:
   with torch.device('cuda'):
    self.shared=Expert(4096,2048,10.,qc,False,'layers.0.ffn.shared_experts')
    self.hash_gate=linear.LinearBase(4096,256,prefix='layers.0.ffn.gate')
    self.learned_gate=linear.LinearBase(4096,256,prefix='layers.3.ffn.gate')
  finally:linear.get_tp_group=old
  def merged(s):return torch.cat([f['layers.0.ffn.shared_experts.'+w+s].view(torch.uint8) for w in ['w1','w3']],dim=0)
  with torch.no_grad():
   self.shared.gate_up_proj.weight.copy_(merged('.weight').cuda().view(torch.float8_e4m3fn));self.shared.gate_up_proj.weight_scale.copy_(merged('.scale').cuda().view(torch.float8_e8m0fnu))
   self.shared.w2.weight.copy_(f['layers.0.ffn.shared_experts.w2.weight'].cuda());self.shared.w2.weight_scale.copy_(f['layers.0.ffn.shared_experts.w2.scale'].cuda())
   self.hash_gate.weight.copy_(f['layers.0.ffn.gate.weight'].cuda());self.learned_gate.weight.copy_(f['layers.3.ffn.gate.weight'].cuda())
  self.shared.gate_up_proj.process_weights_after_loading();self.shared.w2.process_weights_after_loading()
  self.bias=f['layers.3.ffn.gate.bias'].cuda();self.tid=f['layers.0.ffn.gate.tid2eid'].cuda();self.ids=torch.arange(1,device='cuda',dtype=torch.int64)
  torch.manual_seed(6764);self.x=torch.randn(1,4096,device='cuda',dtype=torch.bfloat16)
  self.configs=json.loads(Path(__file__).with_name('selected_shared.json').read_text())
 def hash(self,x=None):
  x=self.x if x is None else x;l=self.hash_gate(x);ids=torch.empty((1,6),device='cuda',dtype=torch.int32);ws=torch.empty((1,6),device='cuda');hash_topk_triton(self.ids,l,self.tid,True,1.5,ids,ws);return ws,ids
 def learned(self,x=None):
  x=self.x if x is None else x;l=self.learned_gate(x)
  return FusedMoE.select_experts(x,l,6,False,True,scoring_func='sqrtsoftplus',e_score_correction_bias=self.bias,routed_scaling_factor=1.5)
 def gemm_candidate(self,x,xs,layer,key):return gemm_afp8wfp8_preshuffle(x,layer.weight,xs.view(torch.uint8),layer.weight_scale.view(torch.uint8),backend='gluon',is_x_scale_transposed=True,config=copy.deepcopy(self.configs[key]['config']))
 def candidate(self,x=None):
  x=self.x if x is None else x
  q,s=linear.per_1x128_e8m0_quant(x,torch.float8_e4m3fn,True)
  gu=self.gemm_candidate(q,s,self.shared.gate_up_proj,'1,4096,4096')
  act=torch.empty((1,2048),device=x.device,dtype=x.dtype);aiter_silu_and_mul(act,gu,10.)
  q,s=linear.per_1x128_e8m0_quant(act,torch.float8_e4m3fn,True)
  return self.gemm_candidate(q,s,self.shared.w2,'1,4096,2048')
