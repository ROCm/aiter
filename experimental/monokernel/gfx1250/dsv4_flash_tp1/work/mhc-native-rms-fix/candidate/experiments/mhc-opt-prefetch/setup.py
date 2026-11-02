from pathlib import Path
import torch,json,statistics
from safetensors import safe_open
from region import QKVAttentionRegion
from full_prefix import full_prefix
from mhc import mhc_stage,mhc_region
from qa import project_fused
from tail import quant,norm_only,norm_rope
from split_kernels import *
root=Path(__file__).resolve().parent
f=torch.load(root/'fixture.pt',weights_only=True)
W=f['W_bytes'].cuda().view(torch.float8_e4m3fn);WS=f['WS'].cuda();NW=f['NW'].cuda();BW=f['QB_bytes'].cuda().view(torch.float8_e4m3fn);BS=f['QBS'].cuda()
Cos=f['cos'].cuda();Sin=f['sin'].cuda();Sink=torch.load(root/'sink.pt',weights_only=True).cuda()
if not (root/'mhc-fixture.pt').exists():
 model=Path('/data/DeepSeek-V4-Flash');index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
 def load(k):
  with safe_open(str(model/index[k]),framework='pt',device='cpu') as sf:return sf.get_tensor(k)
 data={n:load('layers.0.'+k) for n,k in [('HAF','hc_attn_fn'),('HAS','hc_attn_scale'),('HAB','hc_attn_base'),('AN','attn_norm.weight')]}
 with safe_open(str(model/index['embed.weight']),framework='pt',device='cpu') as sf:
  data['inputs']=torch.stack([sf.get_slice('embed.weight')[t:t+1].repeat(4,1).contiguous() for t in [42,0,1,7,43,100,1024,8192,32000]])
 torch.save(data,root/'mhc-fixture.pt')
f2=torch.load(root/'mhc-fixture.pt',weights_only=True)
HAF,HAS,HAB,AN=[f2[k].cuda().contiguous() for k in ['HAF','HAS','HAB','AN']]
R=f2['inputs'][0].cuda().contiguous()
kw=dict(num_warps=4,enable_fp_fusion=False)
class State(QKVAttentionRegion):
 def __init__(self,t=128):
  gen=torch.Generator(device='cuda').manual_seed(777+t)
  cache=(torch.randn((t,512),device='cuda',generator=gen)*.2).bfloat16();idx=torch.arange(t,device='cuda',dtype=torch.int32)
  super().__init__(W,WS,NW,BW,BS,cache,idx,Sink,splits=4,fused=True,direct=True)
  def z(shape,dtype=torch.float32):return torch.zeros(shape,device='cuda',dtype=dtype)
  self.mix=z(24);self.temp=z((128,24));self.raw=z(4096,torch.bfloat16);self.x=z(4096,torch.bfloat16)
  self.xp=z(4096,torch.float8_e4m3fn);self.xs=z(32,torch.uint8);self.qaraw=z(1536);self.qn=z(32768)
 def mhc_split(self,r):
  for stage,grid in enumerate([256,24,1,16,1]):mhc_stage[(grid,)](r,HAF,HAS,HAB,AN,self.mix,self.temp,self.raw,self.x,STAGE=stage,**kw)
 def mhc_fused(self,r):
  self.mobj=mhc_region[(256,)](r,HAF,HAS,HAB,AN,self.mix,self.temp,self.raw,self.x,self.counter,self.health,**kw)
 def qkv_split(self):
  quant[(32,)](self.x,self.xp,self.xs,**kw)
  project_fused[(192,)](self.xp,self.xs,self.W,self.WS,self.p,self.NW,self.acc,self.count,self.y,self.lp,self.ls,Cos,Sin,self.kv,
    INLINE=False,TAIL=False,MODE=0,BN=128,SPLIT=16,DIRECT=True,WAVES=4,BK=128,UNROLL=1,UPSCALE=True,REPLICATE=True,CACHE='.ca',SCALAR=True,PACKED=False,OUTLAY=1,NVEC=1,COUNTSTRIDE=128,NATIVE=0,NORMCACHE='.ca',ALLTDM=True,PRODUCE_ONLY=True,**kw)
  qa_reduce[(2,)](self.p,self.qaraw,**kw);qa_norm[(2,)](self.qaraw,self.NW,self.y,**kw)
  quant[(8,)](self.y,self.lp,self.ls,**kw)
  qb_stage[(256,)](self.lp,self.ls,self.QB,self.QBS,self.qp,self.qc,Cos,Sin,self.q,**kw)
  norm_only[(64,)](self.qp,self.qn,**kw)
  norm_rope[(64,)](self.qn,Cos,Sin,self.q,DO_NORM=False,**kw)
  norm_rope[(1,)](self.y[1024:],Cos,Sin,self.kv,DO_NORM=False,**kw)
 def qkv_fused(self):
  from qkv_region import QKVRegion
  return QKVRegion.__call__(self,self.x,Cos,Sin)
 def attn_split(self):
  a=self.attn;cache_store[(1,)](self.kv,a.cache,SLOT=a.write_slot,**kw)
  attention_partials[(16,)](self.q,a.cache,a.indices,a.indptr,a.sink,a.m,a.l,a.a,a.out,PAGES=a.pages,SPLITS=4,**kw)
  attention_reduce[(4,)](a.m,a.l,a.a,a.sink,a.indptr,a.out,SPLITS=4,**kw)
 def attn_fused(self):return self.attn(self.q,self.kv)
 def run(self,r,mode):
  if mode=='unfused':self.mhc_split(r);self.qkv_split();self.attn_split()
  elif mode=='three_regions':self.mhc_fused(r);self.qkv_fused();self.attn_fused()
  else:
   a=self.attn
   self.obj=full_prefix[(256,)](r,HAF,HAS,HAB,AN,self.mix,self.temp,self.raw,self.x,self.ls,self.W,self.WS,self.p,self.NW,self.acc,self.count,self.y,self.lp,self.ls,Cos,Sin,self.kv,self.QB,self.QBS,self.qp,self.qc,self.q,self.counter,self.health,a.cache,a.indices,a.indptr,a.sink,a.m,a.l,a.a,a.out,a.counts,T=a.t,PAGES=a.pages,WRITE_SLOT=a.write_slot,SPLITS=4,BH=16,BK=32,DIRECT=True,EARLY_QA=mode in ('early','both'),L2_QA=mode in ('l2','both'),**kw)
  return self.attn.out

def err(a,b):return ((a.float()-b.float()).square().mean()/b.float().square().mean().clamp_min(1e-30)).sqrt().item()
def attn_ref(q,cache,idx,sink):
 k=cache[idx.long()].double();scores=q.double()@k.T/(512**.5);mx=torch.maximum(scores.max(-1).values,sink.double())
 p=(scores-mx[:,None]).exp();return (p@k/(p.sum(-1)+(sink.double()-mx).exp())[:,None]).bfloat16()
