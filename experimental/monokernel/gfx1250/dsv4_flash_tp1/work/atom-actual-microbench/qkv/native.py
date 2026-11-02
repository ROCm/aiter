import os,sys,json
os.environ.setdefault('ATOM_USE_TRITON_GEMM','1')
os.environ.setdefault('ATOM_FP8_BLOCKSCALE_USE_E8M0_SCALE','1')
sys.path.insert(0,'/app/aiter');sys.path.insert(0,'/app/ATOM')
import torch
from types import SimpleNamespace
from transformers import AutoConfig
from atom.config import QuantizationConfig
from atom.model_ops import linear,layernorm
from atom.model_ops.v4_kernels.qk_norm_rope_maybe_quant import qk_norm_rope_maybe_quant
from aiter import QuantType

class NativeQKV:
 def __init__(self, fixture='/work/mhc-optimized/fixture.pt', positions=1024, cache=None, dest=0):
  self.f=torch.load(fixture,weights_only=False); f=self.f
  hf=AutoConfig.from_pretrained('/data/DeepSeek-V4-Flash',trust_remote_code=True)
  qc=QuantizationConfig(hf)
  # Singleton harness has no distributed process group; linear arithmetic and loading unchanged.
  old=linear.get_tp_group;linear.get_tp_group=lambda:SimpleNamespace(rank_in_group=0,world_size=1)
  oldn=layernorm.get_tensor_model_parallel_world_size;layernorm.get_tensor_model_parallel_world_size=lambda:1
  try:
   with torch.device('cuda'):
    self.qa=linear.LinearBase(4096,[1024,512],quant_config=qc,prefix='layers.0.attn.wqkv_a')
    self.qb=linear.LinearBase(1024,32768,quant_config=qc,prefix='layers.0.attn.wq_b')
    self.norm=layernorm.RMSNorm(1024,1e-6,fused_quant=True,quant_config=qc,prefix='layers.0.attn.q_norm')
  finally:linear.get_tp_group=old;layernorm.get_tensor_model_parallel_world_size=oldn
  with torch.no_grad():
   for obj,w,s in [(self.qa,'W_bytes','WS'),(self.qb,'QB_bytes','QBS')]:
    obj.weight.copy_(f[w].cuda().view(torch.float8_e4m3fn));obj.weight_scale.copy_(f[s].cuda().view(torch.float8_e8m0fnu));obj.process_weights_after_loading()
   self.norm.weight.data=f['NW'][:1024].cuda().bfloat16()
  self.norm.share_mxscale_shuffle([self.qb]);self.nw=f['NW'][1024:].cuda().bfloat16().contiguous()
  self.pos=torch.tensor([positions],device='cuda',dtype=torch.int64)
  from atom.models.deepseek_v4 import _build_cos_sin_cache
  rs=hf.rope_scaling or {}
  self.cos,self.sin=_build_cos_sin_cache(64,2048,hf.rope_theta,rs.get('factor',1.),0,rs.get('beta_fast',32),rs.get('beta_slow',1),torch.bfloat16,torch.device('cuda'))
  self.cache=cache if cache is not None else torch.zeros((128,512),device='cuda',dtype=torch.bfloat16)
  self.dest=torch.tensor([dest],device='cuda',dtype=torch.int32);self.batch=torch.zeros(1,device='cuda',dtype=torch.int32)
  self.x=f['live'].cuda().reshape(1,4096)
 def __call__(self,x=None,scatter=True):
  x=self.x if x is None else x.reshape(1,4096)
  a=self.qa(x);qr,kv=a.split([1024,512],dim=-1);v,s=self.norm(qr);q=self.qb(v,x_scale=s)
  self.qaout=a;self.qr=v;self.qrs=s;self.qbout=q
  self.out=qk_norm_rope_maybe_quant(q,kv,self.nw,self.cos,self.sin,self.pos,64,512,64,1e-6,
   swa_kv=self.cache if scatter else None,swa_dest_rows=self.dest if scatter else None,batch_id_per_q_token=self.batch if scatter else None)
  return self.out.q_sa,self.out.kv
 def metadata(self):
  return {'qa_weight':list(self.qa.weight.shape),'qb_weight':list(self.qb.weight.shape),'qa_mxshuffle':self.qa.use_mxscale_shuffle,'qb_mxshuffle':self.qb.use_mxscale_shuffle,'weight_preshuffle':linear.envs.ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE,'norm_quant':str(self.norm.quant_type),'eps':self.norm.eps}
if __name__=='__main__':
 n=NativeQKV();print(n.metadata());q,k=n();torch.cuda.synchronize();print(q.shape,k.shape)
