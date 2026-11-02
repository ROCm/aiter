"""Native-led batched QA region integration, opt in, decode only."""
import sys,os,json
from pathlib import Path
sys.path[:0]=['/work/concurrency-rollout','/work/modelwide','/work/full-attention-native/prefix']
import torch
from safetensors import safe_open
from qa_batch import qa_batch

def attach(b):
 original=b.attn.forward_impl
 state=[None]
 def forward(x,positions,*args,**kw):
  if state[0] is None:state[0]=Region(b)
  obj=state[0];obj.positions=positions;obj.ready=False
  return original(x,positions,*args,**kw)
 b.attn.forward_impl=forward

class Region:
 def __init__(self,b):
  self.b=b;self.mode=os.environ['ATOM_DSV4_BATCH_QA'];self.ready=False
  root=Path('/data/DeepSeek-V4-Flash');idx=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
  def load(s):
   k=f'layers.{b.layer_id}.attn.'+s
   with safe_open(str(root/idx[k]),framework='pt') as f:return f.get_tensor(k).cuda().contiguous()
  self.w=torch.cat([load('wq_a.weight').view(torch.uint8),load('wkv.weight').view(torch.uint8)]).view(torch.float8_e4m3fn)
  self.ws=torch.cat([load('wq_a.scale').view(torch.uint8),load('wkv.scale').view(torch.uint8)])
  self.nw=torch.cat([load('q_norm.weight').float(),load('kv_norm.weight').float()]);self.qbw=b.attn.wq_b.weight
  self.buffers={}
  oldqa=b.attn.wqkv_a.forward;oldnorm=b.attn.q_norm.forward
  def qa(x,*args,**kw):
   if not self.eligible(x):return oldqa(x,*args,**kw)
   self.run(x)
   if self.mode=='shadow':
    out=oldqa(x,*args,**kw);self.record('qa',self.raw,out);return out
   return self.raw
  def norm(x,*args,**kw):
   if not self.ready:return oldnorm(x,*args,**kw)
   if self.mode=='shadow':
    out=oldnorm(x,*args,**kw);self.record('quant',self.q,out[0]);return out
   return self.q,self.scale.view(torch.float8_e8m0fnu)
  b.attn.wqkv_a.forward=qa;b.attn.q_norm.forward=norm
 def eligible(self,x):
  from atom.utils.forward_context import get_forward_context,AttnState
  fc=get_forward_context()
  return not fc.context.is_dummy_run and fc.attn_metadata.state is AttnState.DECODE and x.shape[0]<=8
 def run(self,x):
  m=x.shape[0]
  if m not in self.buffers:
   def e(shape,dtype=torch.float32):return torch.empty(shape,device='cuda',dtype=dtype)
   self.buffers[m]=(e((m,4096),torch.float8_e4m3fn),e((m,32),torch.uint8),e((16,m,1536)),e((m,1536),torch.bfloat16),e((m,1024),torch.float8_e4m3fn),e((m,8),torch.uint8),e((m,32),torch.bfloat16),e((m,32),torch.bfloat16),e((m,512),torch.bfloat16),torch.zeros(1,device='cuda',dtype=torch.int32),torch.zeros(1,device='cuda',dtype=torch.int32))
  xq,xs,p,self.raw,self.q,self.scale,c,s,kv,count,health=self.buffers[m]
  ct,st=self.b.attn.rotary_emb.cos_sin_2d();c.copy_(ct[self.positions].reshape(m,32));s.copy_(st[self.positions].reshape(m,32))
  qa_batch[(256,)](x,xq,xs,self.w,self.ws,p,self.nw,self.raw,self.q,self.scale,c,s,kv,count,health,m,QBW=self.qbw,num_warps=4,enable_fp_fusion=False)
  self.ready=True
 def record(self,stage,a,b):
  a=a.float();b=b.float();d=a-b
  row=dict(layer=self.b.layer_id,m=a.shape[0],stage=stage,neq=int((a!=b).sum()),rms=float((d.square().mean()/b.square().mean().clamp_min(1e-30)).sqrt()))
  with open('/work/concurrency-rollout/qa-shadow.jsonl','a') as f:f.write(json.dumps(row)+'\n')
