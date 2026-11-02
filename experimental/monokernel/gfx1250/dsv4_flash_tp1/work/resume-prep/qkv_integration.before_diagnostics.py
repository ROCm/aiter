"""Opaque full-QKV region with runtime dispatch and native prefill fallback."""
import sys,json,os
from pathlib import Path
sys.path[:0]=['/work/concurrency-rollout','/work/modelwide','/work/full-attention-native/prefix']
import torch
from safetensors import safe_open
from qkv_batch import qkv_batch
_registry={}

def attach(b):
 assert not b.attn.kv_fp8
 b.attn._batch_qkv_enabled=True
 _registry[b.attn.layer_name]=[b,None]

@torch.library.custom_op('atom::dsv4_batch_qkv',mutates_args=())
def group(x:torch.Tensor,positions:torch.Tensor,prefix:str)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
 from atom.utils.forward_context import get_forward_context,AttnState
 b,state=_registry[prefix];a=b.attn;fc=get_forward_context()
 if state is None:
  if torch.cuda.is_current_stream_capturing():raise RuntimeError('QKV state initialization attempted during graph capture')
  state=State(b);_registry[prefix][1]=state
 if os.getenv('ATOM_DSV4_BATCH_QKV_NATIVE_CONTROL')!='1' and not fc.context.is_dummy_run and fc.attn_metadata.state is AttnState.DECODE and x.shape[0]<=8:
  return state.run(x,positions,fc.attn_metadata)
 raw=a.wqkv_a(x);qr,scale=a.q_norm(raw[:,:1024]);q=a.wq_b(qr,x_scale=scale)
 out=a._qk_norm_rope_native(q,raw[:,1024:],positions)
 if b.layer_id==0:
  print('BATCH_QKV_PREFILL',tuple(x.shape),tuple(qr.shape),tuple(scale.shape),scale.stride(),scale.dtype,flush=True)
 return out.q_sa,out.kv,qr,scale

@group.register_fake
def fake(x,positions,prefix):
 m=x.shape[0]
 return (torch.empty((m,64,512),device=x.device,dtype=torch.bfloat16),torch.empty((m,512),device=x.device,dtype=torch.bfloat16),torch.empty((m,1024),device=x.device,dtype=torch.float8_e4m3fn),torch.empty((m,8),device=x.device,dtype=torch.float8_e8m0fnu))

class State:
 def __init__(self,b):
  self.b=b;root=Path('/data/DeepSeek-V4-Flash');idx=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
  def load(s):
   key=f'layers.{b.layer_id}.attn.'+s
   with safe_open(str(root/idx[key]),framework='pt') as f:return f.get_tensor(key).cuda().contiguous()
  self.w=torch.cat([load('wq_a.weight').view(torch.uint8),load('wkv.weight').view(torch.uint8)]).view(torch.float8_e4m3fn)
  self.ws=torch.cat([load('wq_a.scale').view(torch.uint8),load('wkv.scale').view(torch.uint8)])
  self.nw=torch.cat([load('q_norm.weight').float(),load('kv_norm.weight').float()])
  self.qbw=load('wq_b.weight');self.qbs=load('wq_b.scale').view(torch.uint8);self.buffers={}
 def run(self,x,positions,md):
  m=x.shape[0]
  if m not in self.buffers:
   def e(shape,dtype=torch.float32):return torch.empty(shape,device='cuda',dtype=dtype)
   self.buffers[m]=(e((m,4096),torch.float8_e4m3fn),e((m,32),torch.uint8),e((16,m,1536)),e((m,1536),torch.bfloat16),e((m,1024),torch.float8_e4m3fn),e((m,8),torch.uint8),e((m,32),torch.bfloat16),e((m,32),torch.bfloat16),e((m,512),torch.bfloat16),torch.zeros(1,device='cuda',dtype=torch.int32),torch.zeros(1,device='cuda',dtype=torch.int32),e((m,32768),torch.bfloat16),e((m,64,512),torch.bfloat16))
  xq,xs,p,raw,qr,scale,c,s,kv,count,health,qbr,qo=self.buffers[m]
  a=self.b.attn;ct,st=a.rotary_emb.cos_sin_2d();c.copy_(ct[positions].reshape(m,32));s.copy_(st[positions].reshape(m,32))
  qkv_batch[(256,)](x,xq,xs,self.w,self.ws,p,self.nw,raw,qr,scale,c,s,kv,count,health,self.qbw,self.qbs,qbr,qo,m,Cache=None if os.getenv('ATOM_DSV4_BATCH_QKV_NATIVE_TAIL')=='1' else a.swa_plane,Dest=md.swa_dest_rows[a.compress_ratio],BatchId=md.batch_id_per_q_token,num_warps=4,enable_fp_fusion=False)
  if os.getenv('ATOM_DSV4_BATCH_QKV_NATIVE_TAIL')=='1':
   out=a._qk_norm_rope_native(qbr,raw[:,1024:],positions)
   return out.q_sa,out.kv,qr,scale.view(torch.float8_e8m0fnu)
  return qo,kv,qr,scale.view(torch.float8_e8m0fnu)
