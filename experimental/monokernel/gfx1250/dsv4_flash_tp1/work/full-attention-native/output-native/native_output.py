"""Actual installed ATOM attention output methods, isolated TP1/layer0 harness.
Native post alone is extracted boundary; serving usually fuses post with FFN-pre.
"""
import os,sys,json
os.environ.setdefault('ATOM_USE_TRITON_GEMM','1')
os.environ.setdefault('ATOM_FP8_BLOCKSCALE_USE_E8M0_SCALE','1')
sys.path.insert(0,'/app/aiter');sys.path.insert(0,'/app/ATOM')
import torch,aiter
from types import SimpleNamespace
from pathlib import Path
from transformers import AutoConfig
from atom.config import QuantizationConfig
from atom.model_ops import linear
from atom.models.deepseek_v4 import DeepseekV4Attention,Block,_V4RoPE

class NativeOutput:
 _wo_a_grouped_lora=DeepseekV4Attention._wo_a_grouped_lora
 def __init__(self,fixture=None,position=1024):
  f=torch.load(fixture or Path(__file__).with_name('fixture.pt'),weights_only=False);self.fixture=f
  hf=AutoConfig.from_pretrained('/data/DeepSeek-V4-Flash',trust_remote_code=True);qc=QuantizationConfig(hf)
  old=linear.get_tp_group;linear.get_tp_group=lambda:SimpleNamespace(rank_in_group=0,world_size=1)
  try:
   with torch.device('cuda'):
    self.wo_a=linear.LinearBase(4096,8192,quant_config=qc,prefix='layers.0.attn.wo_a')
    self.wo_b=linear.LinearBase(8192,4096,quant_config=qc,prefix='layers.0.attn.wo_b')
  finally:linear.get_tp_group=old
  with torch.no_grad():
   self.wo_a.weight.copy_(f['OA'].cuda());self.wo_a.weight_scale.copy_(f['OAS'].cuda())
   self.wo_b.weight.copy_(f['OB'].cuda());self.wo_b.weight_scale.copy_(f['OBS'].cuda())
  self.n_local_heads=64;self.head_dim=512;self.rope_head_dim=64;self.n_local_groups=8;self.o_lora_rank=1024
  self._is_gfx1250=True;self._is_gfx950=False;self._is_preshuffle=True;self._wo_a_mxscale=False;self.layer_id=0;self.layer_name='layers.0.attn'
  DeepseekV4Attention.process_weights_after_loading(self)
  self.wo_b.process_weights_after_loading()
  rs=hf.rope_scaling or {}
  with torch.device('cuda'):
   self.rotary_emb=_V4RoPE(64,2048,hf.rope_theta,rs.get('factor',1.),0,rs.get('beta_fast',32),rs.get('beta_slow',1),torch.bfloat16)
  self.positions=torch.tensor([position],dtype=torch.int64,device='cuda')
  self._mhc_post=aiter.mhc_post
 def __call__(self,o):return DeepseekV4Attention._attn_post(self,o,self.positions)
 def oa(self,o):return self._wo_a_grouped_lora(o,self.positions,prefix='layers.0.attn.wo_a')
 def post(self,x,residual,post,comb,res_preshuffle=False):return Block.hc_post(self,x,residual,post,comb,res_preshuffle=res_preshuffle)
 def full(self,o,residual,post,comb,res_preshuffle=False):return self.post(self(o),residual,post,comb,res_preshuffle)
 def metadata(self):return {'OA_fp8':self._wo_a_mxscale,'OA_preshuffle':self._is_preshuffle,'OAshape':list(self._wo_a_w_fp8.shape),'OAscale':list(self._wo_a_w_scale.shape),'OBshape':list(self.wo_b.weight.shape),'OBscale':list(self.wo_b.weight_scale.shape),'OBmxscale_shuffle':self.wo_b.use_mxscale_shuffle,'OBpreshuffle':linear.envs.ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE}
if __name__=='__main__':
 n=NativeOutput();x=torch.randn(1,64,512,device='cuda',dtype=torch.bfloat16);y=n(x);torch.cuda.synchronize();print(n.metadata(),y.shape,flush=True)
