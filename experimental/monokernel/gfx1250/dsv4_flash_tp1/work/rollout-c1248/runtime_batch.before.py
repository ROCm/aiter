"""Opt-in multirow full-layer serving adapter. Experimental; C1 adapter unchanged."""
import os,sys,logging
sys.path[:0]=['/work/integrated-c8','/work/batch-full','/work/modelwide','/work/concurrency-rollout','/work/full-attention-native','/work/full-attention-native/prefix','/work/full-attention-native/output-kernel','/work/ffn-c1/moe']
import runtime as base
import torch
from scratch_batch import BatchState
from model_layer_batch import model_layer
from binding import LayerWeights,compressor_args
base._MAX_BATCH=8
_shared={}
log=logging.getLogger(__name__)
class ServingLayer(base.ServingLayer):
 def execute(self,residual,positions,md):
  m=residual.shape[0]
  if not 1 <= m <= 8:raise RuntimeError(f'Unsupported full monokernel batch {m}')
  if self.weights is None:
   self.weights=LayerWeights(self.block)
   w=self.weights
   import aiter
   w.pre_packed=aiter.mhc_shuffle_fn(w.a[3].contiguous().float()).view(torch.bfloat16)
   w.mhc_packed=aiter.mhc_shuffle_fn(w.f[2].contiguous().float()).view(torch.bfloat16)
   w.out=torch.empty((8,4,4096),device='cuda',dtype=torch.bfloat16)
   w.my=torch.empty((8,2048 if w.ratio==4 else 1024),device='cuda',dtype=torch.bfloat16)
   w.iy=torch.empty((8,512),device='cuda',dtype=torch.bfloat16)
   w.top=torch.empty((8,512),device='cuda',dtype=torch.int32)
  b=self.block;w=self.weights;r=w.ratio;a=b.attn;t={0:128,4:640,128:144}[r]
  if (m,t) not in _shared:_shared[m,t]=BatchState(m,t)
  s=_shared[m,t];A=list(s.a);F=list(s.f);A[0]=residual
  for i,v in w.a.items():A[i]=v
  for i,v in w.f.items():F[i]=v
  cos,sin=a.rotary_emb.cos_sin_2d()
  if r:MC,MM=compressor_args(a.compressor,md,r);MW=w.compressor_weights['main']
  else:MC=();MM=();MW=self.dummy
  if r==4:IC,IM=compressor_args(a.indexer.compressor,md,4);IW=w.compressor_weights['indexer']
  else:IC=();IM=();IW=self.dummy
  name={0:'swa',4:'csa',128:'hca'}[r];ptr=getattr(md,'kv_indptr_'+name);idx=getattr(md,'kv_indices_'+name)
  extra=s.extra()[:-1]+(w.out,)
  kwargs=dict(MM=MM,IM=IM,RATIO=r,ENVELOPE=md.envelope_rows,**w.meta,**self.extra,HAS_BIAS=w.bias is not None,M=m,BLOCK_STRIDE=md.block_tables.stride(0),T=t,BatchIds=md.batch_id_per_q_token,num_warps=4,enable_fp_fusion=False)
  args=(tuple(A),tuple(F),MC,IC,MW,IW,w.my,w.iy,positions,cos,sin,a.unified_kv,idx,ptr,md.swa_dest_rows[r],w.top,md.block_tables,w.bias if w.bias is not None else self.dummy,*extra)
  from aiter.ops.triton.fusions.mhc import _gluon_post_pre_layouts,_gluon_reduce_apply_layouts
  ml=_gluon_post_pre_layouts(4,16,64,32,2,True,4);mv=_gluon_reduce_apply_layouts(4,1,4,64)
  kwargs.update(CacheRows=a.unified_kv.numel()//512,Work=s.pool,CompCounter=s.comp_counter,MHC=(w.mhc_packed,*s.mhc),ML=ml,MV=mv,OPT_MHC=os.getenv('ATOM_DSV4_OPT_MHC','1')=='1',OPT_COMP=os.getenv('ATOM_DSV4_OPT_COMP','1')=='1')
  kwargs.update(PRE_W=w.pre_packed,OPT_PRE=os.getenv('ATOM_DSV4_AITER_PRE','1')=='1')
  edge_mask=int(os.getenv('ATOM_DSV4_EDGE_MASK','31' if os.getenv('ATOM_DSV4_EDGE_SYNC','0')=='1' else '0'))
  tile_mask=int(os.getenv('ATOM_DSV4_TILE_MASK','0'))
  overlap=int(os.getenv('ATOM_DSV4_OVERLAP','0'))
  if tile_mask&8 and not edge_mask&8:raise ValueError('Shared-expert tile handoffs require EDGE_MASK bit 8')
  kwargs.update(Edges=s.edges if edge_mask or tile_mask else None,EDGE_MASK=edge_mask,Tiles=s.tiles if tile_mask else None,TILE_MASK=tile_mask,OVERLAP=overlap)
  key=(m,r,kwargs['OPT_MHC'],kwargs['OPT_COMP'],kwargs['OPT_PRE'],edge_mask,tile_mask,overlap)
  if key not in getattr(self,'compiled',set()):
   obj=model_layer.warmup(*args,grid=(256,),**kwargs);obj._init_handles()
   if obj.n_spills or obj.metadata.shared>327680:raise RuntimeError(f'Unsafe persistent resource budget: {obj.n_regs=} {obj.n_spills=} {obj.metadata.shared=}')
   if not hasattr(self,'compiled'):self.compiled=set()
   self.compiled.add(key)
  obj=model_layer[(256,)](*args,**kwargs)
  self.calls+=1
  if (r,m) not in getattr(self,'reported',set()):
   if not hasattr(self,'reported'):self.reported=set()
   self.reported.add((r,m))
   log.warning('OPTIMIZED FULL MONOKERNEL layer=%d batch=%d ratio=%d VGPR=%d LDS=%d spills=%d',b.layer_id,m,r,obj.n_regs,obj.metadata.shared,obj.n_spills)
  if os.getenv('ATOM_DSV4_BATCH_FULL_CHECK')=='1':
   torch.cuda.synchronize()
   if int(s.a[30]) or int(s.f[-1]):raise RuntimeError('Full monokernel barrier health failed')
  return w.out[:m]
