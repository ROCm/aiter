"""Opaque full-QKV region with runtime dispatch and native prefill fallback."""
import sys,json,os
from pathlib import Path
sys.path[:0]=['/work/resume-prep','/work/concurrency-rollout','/work/modelwide','/work/full-attention-native/prefix']
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
 b,state=_registry[prefix];a=b.attn;fc=get_forward_context();shadow_outputs=None
 if state is None:
  if torch.cuda.is_current_stream_capturing():raise RuntimeError('QKV state initialization attempted during graph capture')
  state=State(b);_registry[prefix][1]=state
 if state.audit is not None and b.layer_id==0 and not fc.context.is_dummy_run and not torch.cuda.is_current_stream_capturing():
  path=Path(os.environ['ATOM_DSV4_BATCH_QKV_AUDIT'])
  rows={str(bl.layer_id):st.audit.cpu().tolist() for bl,st in _registry.values() if st is not None and st.audit is not None}
  path.write_text(json.dumps({'columns':['all_rows','active_bid_negative_position','active_bid_invalid_destination','inactive_bid'],'layers':rows},indent=2))
 if state.debug is not None and not torch.cuda.is_current_stream_capturing():
  bad=[(m,int(buf[10].item())) for m,buf in state.buffers.items() if int(buf[10].item())]
  if bad:raise RuntimeError(f'QKV diagnostic layer{b.layer_id} health{bad} records{state.debug.cpu().tolist()}')
 if os.getenv('ATOM_DSV4_BATCH_QKV_NATIVE_CONTROL')!='1' and not fc.context.is_dummy_run and fc.attn_metadata.state is AttnState.DECODE and x.shape[0]<=8:
  outputs=state.run(x,positions,fc.attn_metadata)
  if state.shadow_stats is not None:shadow_outputs=outputs
  elif os.getenv('ATOM_DSV4_BATCH_QKV_CLONE_OUTPUT')=='1':
   return tuple(t.clone() for t in outputs)
  if shadow_outputs is None:return outputs
 raw=a.wqkv_a(x);qr,scale=a.q_norm(raw[:,:1024]);q=a.wq_b(qr,x_scale=scale)
 out=a._qk_norm_rope_native(q,raw[:,1024:],positions)
 if b.layer_id==0:
  print('BATCH_QKV_PREFILL',tuple(x.shape),tuple(qr.shape),tuple(scale.shape),scale.stride(),scale.dtype,flush=True)
 if shadow_outputs is not None:
  from shadow_metrics import compare
  buf=state.buffers[x.shape[0]]
  for slot,(candidate,reference) in enumerate([(buf[3],raw),(buf[4],qr),(buf[11],q),(shadow_outputs[0],out.q_sa),(shadow_outputs[1],out.kv)]):
   compare[((reference.numel()+1023)//1024,)](candidate,reference,state.shadow_stats,reference.numel(),slot,num_warps=4)
 if state.shadow_stats is not None and b.layer_id==42 and not torch.cuda.is_current_stream_capturing():
  rows={str(bl.layer_id):{'metrics':st.shadow_stats.cpu().tolist(),'health':{str(m):int(buf[10].item()) for m,buf in st.buffers.items()}} for bl,st in _registry.values() if st is not None and st.shadow_stats is not None}
  Path(os.environ['ATOM_DSV4_BATCH_QKV_SHADOW']).write_text(json.dumps({'groups':['QA','QR','QB','Qnorm','KVnorm'],'columns':['nonfinite_count','max_abs_difference','max_abs_reference'],'layers':rows},indent=2))
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
  assert self.qbw.dtype==torch.float8_e4m3fn and tuple(self.qbw.shape)==(32768,1024)
  assert self.qbw.is_contiguous() and self.qbw.data_ptr()%256==0
  assert self.qbw.storage_offset()==0 and self.qbw.untyped_storage().nbytes()>=33554432
  self.shadow_stats=torch.zeros(15,device='cuda',dtype=torch.float32) if os.getenv('ATOM_DSV4_BATCH_QKV_SHADOW') else None
  self.audit=torch.zeros(4,device='cuda',dtype=torch.int64) if os.getenv('ATOM_DSV4_BATCH_QKV_AUDIT') else None
  self.debug=torch.full((8,3),-1,device='cuda',dtype=torch.int64) if os.getenv('ATOM_DSV4_BATCH_QKV_GUARD')=='1' else None
  if os.getenv("ATOM_DSV4_BATCH_QKV_PREALLOC")=="1":
   for m in [1,2,4,8]:self.allocate(m)
 def allocate(self,m):
  if m not in self.buffers:
   def e(shape,dtype=torch.float32):return torch.empty(shape,device='cuda',dtype=dtype)
   self.buffers[m]=(e((m,4096),torch.float8_e4m3fn),e((m,32),torch.uint8),e((16,m,1536)),e((m,1536),torch.bfloat16),e((m,1024),torch.float8_e4m3fn),e((m,8),torch.uint8),e((m,32),torch.bfloat16),e((m,32),torch.bfloat16),e((m,512),torch.bfloat16),torch.zeros(1,device='cuda',dtype=torch.int32),torch.zeros(1,device='cuda',dtype=torch.int32),e((m,32768),torch.bfloat16),e((m,64,512),torch.bfloat16))
 def run(self,x,positions,md):
  m=x.shape[0]
  assert x.dtype==torch.bfloat16 and x.stride()==(4096,1),(x.shape,x.stride(),x.dtype)
  assert positions.is_contiguous(),positions.stride()
  if m not in self.buffers:
   if os.getenv("ATOM_DSV4_BATCH_QKV_PREALLOC")=="1" and torch.cuda.is_current_stream_capturing():raise RuntimeError("QKV missing preallocated batch")
   self.allocate(m)
  xq,xs,p,raw,qr,scale,c,s,kv,count,health,qbr,qo=self.buffers[m]
  a=self.b.attn;ct,st=a.rotary_emb.cos_sin_2d();c.copy_(ct[positions].reshape(m,32));s.copy_(st[positions].reshape(m,32))
  if os.getenv('ATOM_DSV4_BATCH_QKV_INPUT_ONLY')=='1':
   from qa_batch import qa_batch
   qa_batch[(256,)](x,xq,xs,self.w,self.ws,p,self.nw,raw,qr,scale,c,s,kv,count,health,m,STAGE=1,num_warps=4,enable_fp_fusion=False)
   if os.getenv('ATOM_DSV4_BATCH_QKV_PREFETCH_ONLY')=='1':
    from qa_batch import prefetch_only
    prefetch_only[(256,)](self.qbw,num_warps=4)
   raw_native=a.wqkv_a(xq,x_scale=xs.view(torch.float8_e8m0fnu))
   qr_native,scale_native=a.q_norm(raw_native[:,:1024])
   qb_native=a.wq_b(qr_native,x_scale=scale_native)
   out=a._qk_norm_rope_native(qb_native,raw_native[:,1024:],positions)
   return out.q_sa,out.kv,qr_native,scale_native
  if os.getenv('ATOM_DSV4_BATCH_QKV_QA_ONLY')=='1':
   from qa_batch import qa_batch
   if os.getenv('ATOM_DSV4_BATCH_QKV_SPLIT_INPUT')=='1':
    qa_batch[(256,)](x,xq,xs,self.w,self.ws,p,self.nw,raw,qr,scale,c,s,kv,count,health,m,STAGE=1,num_warps=4,enable_fp_fusion=False)
   qa_batch[(256,)](x,xq,xs,self.w,self.ws,p,self.nw,raw,qr,scale,c,s,kv,count,health,m,QBW=None if os.getenv("ATOM_DSV4_BATCH_QKV_NO_PREFETCH")=="1" else self.qbw,STAGE=2 if os.getenv("ATOM_DSV4_BATCH_QKV_SPLIT_INPUT")=="1" else 0,SPLIT_NORM=os.getenv("ATOM_DSV4_BATCH_QKV_SPLIT_NORM")=="1",num_warps=4,enable_fp_fusion=False)
   if os.getenv('ATOM_DSV4_BATCH_QKV_SPLIT_NORM')=='1':
    from qa_batch import finish_batch
    finish_batch[(2*m,)](p,self.nw,raw,qr,scale,c,s,kv,m,num_warps=4,enable_fp_fusion=False)
   qb_native=a.wq_b(qr,x_scale=scale.view(torch.float8_e8m0fnu))
   out=a._qk_norm_rope_native(qb_native,raw[:,1024:],positions)
   return out.q_sa,out.kv,qr,scale.view(torch.float8_e8m0fnu)
  qkv_batch[(256,)](x,xq,xs,self.w,self.ws,p,self.nw,raw,qr,scale,c,s,kv,count,health,self.qbw,self.qbs,qbr,qo,m,EARLY=os.getenv('ATOM_DSV4_BATCH_QKV_LATE_TDM')!='1',Cache=None if self.shadow_stats is not None or os.getenv('ATOM_DSV4_BATCH_QKV_NATIVE_TAIL')=='1' or os.getenv('ATOM_DSV4_BATCH_QKV_NATIVE_SCATTER')=='1' else a.swa_plane,Dest=md.swa_dest_rows[a.compress_ratio],BatchId=md.batch_id_per_q_token,CacheRows=a.swa_plane.shape[0],Debug=self.debug,DIRECT_KV=os.getenv('ATOM_DSV4_BATCH_QKV_DIRECT_KV')=='1',Positions=positions,Audit=self.audit,PREFETCH=os.getenv("ATOM_DSV4_BATCH_QKV_NO_PREFETCH")!="1",SYNC_STYLE=int(os.getenv("ATOM_DSV4_BATCH_QKV_SYNC_STYLE","3")),PREFETCH_STYLE=int(os.getenv("ATOM_DSV4_BATCH_QKV_PREFETCH_STYLE","2")),QB_BYTES=self.qbw.numel()*self.qbw.element_size(),num_warps=4,enable_fp_fusion=False)
  if os.getenv('ATOM_DSV4_BATCH_QKV_NATIVE_TAIL')=='1' or os.getenv('ATOM_DSV4_BATCH_QKV_NATIVE_SCATTER')=='1':
   out=a._qk_norm_rope_native(qbr,raw[:,1024:],positions)
   if os.getenv('ATOM_DSV4_BATCH_QKV_NATIVE_SCATTER')!='1':
    return out.q_sa,out.kv,qr,scale.view(torch.float8_e8m0fnu)
  return qo,kv,qr,scale.view(torch.float8_e8m0fnu)
