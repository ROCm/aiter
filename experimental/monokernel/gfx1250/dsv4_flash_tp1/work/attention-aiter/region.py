"""Flash C1 BF16 attention adapted from AITER. Isolated region API, not ATOM serving.
All tiled reads use TDM; non-speculative cache prefetch retained. 64 heads x512.
For sparse layouts indices are physical row IDs, -1 denotes invalid. Indices
must be -1 or in [0,pages); cache and indices sizes fixed for captured graphs.
No overlapping calls sharing state. Check health after stream synchronization.
"""
import torch
from attention import attention_kernel,qkv_attention
from qkv_region import QKVRegion

class AttentionRegion:
 def __init__(self,cache,indices,sink,splits=None,write_slot=None,block_h=16,block_k=32,vector=False):
  assert cache.dtype==torch.bfloat16 and cache.ndim==2 and cache.shape[1]==512 and cache.is_contiguous()
  assert indices.dtype==torch.int32 and indices.ndim==1 and indices.is_contiguous() and indices.numel()>0
  assert sink.dtype==torch.float32 and sink.shape==(64,) and sink.is_contiguous()
  assert cache.device==indices.device==sink.device and cache.is_cuda
  if splits is None:splits=4 if indices.numel()>=1024 else 1
  assert splits in (1,2,4) and block_h in (16,32,64) and block_k in (32,64)
  self.t=indices.numel();self.pages=cache.shape[0];self.write_slot=self.pages-1 if write_slot is None else write_slot
  assert 0<=self.write_slot<self.pages
  tiles=(self.t+block_k-1)//block_k;per=(self.t+splits*block_k-1)//(splits*block_k)
  assert (splits-1)*per<tiles,'each configured split must own a tile; reduce splits'
  assert bool(((indices>=-1)&(indices<self.pages)).all())
  self.cache,self.indices,self.sink=cache,indices,sink
  self.splits,self.bh,self.bk,self.vector=splits,block_h,block_k,vector
  self.grid=256 if vector else 64//block_h*splits
  if vector:assert self.t==self.pages==128 and bool((indices==torch.arange(128,device=indices.device)).all())
  def z(shape,dtype=torch.float32):return torch.zeros(shape,device=cache.device,dtype=dtype)
  self.indptr=torch.tensor([0,self.t],device=cache.device,dtype=torch.int32)
  self.m=z((splits,64));self.l=z((splits,64));self.a=z((splits,64,512));self.out=z((64,512),torch.bfloat16)
  self.counts=z(64//block_h,torch.int32);self.bar=z(1,torch.int32);self.health=z(1,torch.int32)
  self.kw=dict(num_warps=4,enable_fp_fusion=False)
 def __call__(self,q,kv):
  assert q.shape==(64,512) and kv.numel()==512 and q.dtype==kv.dtype==torch.bfloat16
  assert q.is_contiguous() and kv.is_contiguous() and q.device==kv.device==self.cache.device
  self.obj=attention_kernel[(self.grid,)](q,kv,self.cache,self.indices,self.indptr,self.sink,self.m,self.l,self.a,self.out,self.counts,self.bar,self.health,T=self.t,PAGES=self.pages,WRITE_SLOT=self.write_slot,SPLITS=self.splits,GRID=self.grid,BH=self.bh,BK=self.bk,VECTOR=self.vector,**self.kw)
  return self.out
 def check(self):
  return self.health.item()==0 and not self.counts.any().item()

class QKVAttentionRegion(QKVRegion):
 def __init__(self,W,WS,NW,QB,QBS,cache,indices,sink,splits=None,write_slot=None,block_h=16,block_k=32,fused=True):
  super().__init__(W,WS,NW,QB,QBS)
  self.attn=AttentionRegion(cache,indices,sink,splits,write_slot,block_h,block_k)
  self.fused=fused
 def __call__(self,X,Cos,Sin):
  a=self.attn
  if not self.fused:
   q,kv=super().__call__(X,Cos,Sin);return a(q,kv)
  assert X.numel()==4096 and X.dtype==torch.bfloat16
  assert Cos.numel()==Sin.numel()==32 and Cos.dtype==Sin.dtype==torch.float32
  assert all(t.device==self.device and t.is_contiguous() for t in [X,Cos,Sin])
  self.obj=qkv_attention[(256,)](X,self.ls,self.W,self.WS,self.p,self.NW,self.acc,self.count,self.y,self.lp,self.ls,Cos,Sin,self.kv,self.QB,self.QBS,self.qp,self.qc,self.q,self.counter,self.health,a.cache,a.indices,a.indptr,a.sink,a.m,a.l,a.a,a.out,a.counts,T=a.t,PAGES=a.pages,WRITE_SLOT=a.write_slot,SPLITS=a.splits,BH=a.bh,BK=a.bk,**self.kw)
  return a.out
 def check(self):
  return self.health.item()==0 and not self.count.any().item() and not self.qc.any().item() and self.attn.check()
