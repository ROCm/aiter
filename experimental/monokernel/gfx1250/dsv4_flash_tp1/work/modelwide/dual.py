"""Unfused ATOM is authoritative. Compare isolated fused regions on its inputs.
No candidate writes native KV/compressor state. Intended for --level 0 only.
"""
import sys,os,json
from pathlib import Path
sys.path[:0]=['/work/modelwide','/work/full-attention-native','/work/full-attention-native/prefix','/work/full-attention-native/post-kernel','/work/full-attention-native/output-kernel','/work/ffn-c1/moe']
import torch
from scratch import LayerState
from binding import LayerWeights
from region_kernels import qkv,output,mhc,ffn_body
from attention_matched import attention_kernel
_scratch=None

def err(a,b):
 a=a.float().flatten();b=b.float().flatten();d=a-b
 return dict(rms=(d.square().mean()/b.square().mean().clamp_min(1e-30)).sqrt().item(),max_abs=d.abs().max().item(),neq=int((a!=b).sum()),finite=torch.isfinite(a).all().item())

def attach(b):
 # Bind after the complete model weight-preparation traversal. Building MoE
 # decode views in the parent block hook caches views of unshuffled weights.
 original=b.forward
 def first_forward(*args,**kwargs):
  global _scratch
  b.forward=original
  if _scratch is None:_scratch=LayerState()
  d=Dual(b,_scratch);b._dual_regions=d
  return b.forward(*args,**kwargs)
 b.forward=first_forward

class Dual:
 def __init__(self,b,state):
  self.b=b;self.s=state;self.w=LayerWeights(b);self.mode=os.environ['ATOM_DSV4_DUAL_REGION'];self.position=None
  self.count=0;self.x=None;self.qkn=None;self.rows=[];self.compressor_weights={}
  self.r32=torch.empty((4,4096),device='cuda',dtype=torch.float32)
  self.mprev=torch.zeros(24,device='cuda',dtype=torch.float32)
  oldblock=b.forward
  def block(hc,positions):
   self.position=positions
   return oldblock(hc,positions)
  b.forward=block
  old=b.attn.forward_impl
  def attention(x,positions,*args,**kw):
   self.position=positions;self.x=x
   return old(x,positions,*args,**kw)
  b.attn.forward_impl=attention
  if self.mode=='compressor':
   pairs=[]
   if b.attn.compressor is not None:pairs.append(('main',b.attn.compressor))
   if b.attn.indexer is not None:pairs.append(('indexer',b.attn.indexer.compressor))
   for name,comp in pairs:
    from safetensors import safe_open
    root=Path('/data/DeepSeek-V4-Flash')
    index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
    prefix=f'layers.{b.layer_id}.attn.'+('indexer.' if name=='indexer' else '')+'compressor.'
    weights=[]
    for suffix in ['wkv.weight','wgate.weight']:
     key=prefix+suffix
     with safe_open(str(root/index[key]),framework='pt') as f:weights.append(f.get_tensor(key).cuda())
    self.compressor_weights[name]=torch.cat(weights).contiguous()
    original=comp.forward
    def compress(x,*args,_name=name,_comp=comp,_fn=original,**kw):
     if not self.eligible():return _fn(x,*args,**kw)
     ks=_comp.kv_state.clone();ss=_comp.score_state.clone()
     out=_fn(x,*args,**kw)
     self.check_compressor(_name,_comp,x,ks,ss)
     return out
    comp.forward=compress
  if self.mode=='mhc':
   oldhc=b.fuse_hc
   def hc(hc_state,hfn,hfs,hfb,norm=None,eps=1e-6):
    result=oldhc(hc_state,hfn,hfs,hfb,norm,eps)
    if self.eligible():self.check_mhc(hc_state,result,hfn is b.hc_attn_fn)
    return result
   b.fuse_hc=hc
  if self.mode=='qkv':
   b.attn.wqkv_a.register_forward_hook(lambda mod,args,out:setattr(self,'native_qa',out.clone()))
   b.attn.q_norm.register_forward_hook(lambda mod,args,out:setattr(self,'native_qnorm',(out[0].clone(),out[1].clone())))
   oldq=b.attn._qk_norm_rope
   def qnorm(q,kv,positions):
    out=oldq(q,kv,positions)
    if self.eligible():self.check_qkv(out,positions,q)
    return out
   b.attn._qk_norm_rope=qnorm
  if self.mode=='attention_output_ffn':
   olds=b.attn._sparse_attention
   def sparse(qkn,positions,*args,**kw):
    out=olds(qkn,positions,*args,**kw)
    if self.eligible():self.check_attention(qkn,out)
    return out
   b.attn._sparse_attention=sparse
   oldo=b.attn._attn_post
   def post(o,positions):
    out=oldo(o,positions)
    if self.eligible():self.check_output(o,positions,out)
    return out
   b.attn._attn_post=post
   oldf=b.ffn.forward
   def ffn(x,*args,**kw):
    out=oldf(x,*args,**kw)
    if self.eligible():self.check_ffn(x,out)
    return out
   b.ffn.forward=ffn
 def eligible(self):
  from atom.utils.forward_context import get_forward_context,AttnState
  fc=get_forward_context()
  return not fc.context.is_dummy_run and fc.attn_metadata is not None and fc.attn_metadata.state is AttnState.DECODE
 def record(self,group,metrics):
  from atom.utils.forward_context import get_forward_context
  row=dict(layer=self.b.layer_id,ratio=self.w.ratio,group=group,position=None if self.position is None else int(self.position[0]),**metrics)
  path=Path('/work/modelwide/dual-'+self.mode+'.jsonl')
  with path.open('a') as f:f.write(json.dumps(row)+'\n')
 def args(self):return self.w.args(self.s,self.s.av[0])
 def tables(self,A,pos):
  c,s=self.b.attn.rotary_emb.cos_sin_2d();A[21].copy_(c[pos[0]].flatten());A[22].copy_(s[pos[0]].flatten())
 def check_qkv(self,ref,pos,native_qb):
  A,_=self.args();A=list(A);A[10]=self.x.flatten();self.tables(A,pos)
  qkv[(256,)](tuple(A),num_warps=4,enable_fp_fusion=False)
  metrics=dict(query=err(A[28],ref.q_sa),kv=err(A[23],ref.kv),qa_projection=err(A[14].sum(0).bfloat16(),self.native_qa),qa_quant=err(A[19],self.native_qnorm[0]),qb_projection=err(A[26],native_qb))
  self.record('qkv',metrics)
  if metrics['qa_quant']['neq']:
   data={k:v.detach().cpu() for k,v in dict(x=self.x,qa_weight=A[12],qa_scale=A[13],norm=A[15],qb_weight=A[24],qb_scale=A[25],native_qa=self.native_qa,native_qquant=self.native_qnorm[0],native_qscale=self.native_qnorm[1],native_qb=native_qb,native_query=ref.q_sa,native_kv=ref.kv,partial_qa=A[14],our_normalized=A[18],our_qquant=A[19],our_qscale=A[20],our_qb=A[26],our_query=A[28],cos=A[21],sin=A[22]).items()}
   torch.save(data,f'/work/modelwide/qkv-layer{self.b.layer_id}-pos{int(pos[0])}.pt')
 def check_output(self,o,pos,ref):
  A,_=self.args();A=list(A);A[38]=o;self.tables(A,pos)
  output[(256,)](tuple(A),num_warps=4,enable_fp_fusion=False)
  self.record('output_projection',dict(output=err(A[51],ref)))
 def check_ffn(self,x,ref):
  _,F=self.args();F=list(F);F[9]=x
  ffn_body[(256,)](*F,**self.w.meta,DO_PRE=False,POST_SHUFFLED=True,Bias=self.w.bias,HAS_BIAS=self.w.bias is not None,num_warps=4,enable_fp_fusion=False)
  self.record('ffn',dict(output=err(F[41],ref)))
 def check_attention(self,qkn,ref):
  from atom.utils.forward_context import get_forward_context
  md=get_forward_context().attn_metadata;k={0:'swa',4:'csa',128:'hca'}[self.w.ratio]
  ptr=getattr(md,'kv_indptr_'+k);idx=getattr(md,'kv_indices_'+k);n=int(ptr[1]-ptr[0])
  # Private compact cache. The candidate cannot repair or corrupt native state.
  slots=idx[int(ptr[0]):int(ptr[1])].long();cache=torch.zeros((640,512),device='cuda',dtype=torch.bfloat16);cache[:n].copy_(self.b.attn.unified_kv[slots]);cache[639].copy_(qkn.kv.flatten())
  inds=torch.arange(640,device='cuda',dtype=torch.int32);inds[n-1]=639;ip=torch.tensor([0,n],device='cuda',dtype=torch.int32)
  m=torch.empty((64,64),device='cuda',dtype=torch.float32);l=torch.empty_like(m);a=torch.empty((64,64,512),device='cuda',dtype=torch.float32)
  o=torch.empty_like(ref);count=torch.zeros(4,device='cuda',dtype=torch.int32);counter=torch.zeros(1,device='cuda',dtype=torch.int32);health=torch.zeros_like(counter)
  attention_kernel[(256,)](qkn.q_sa,qkn.kv,cache,inds,ip,self.w.a[34],m,l,a,o,count,counter,health,T=640,PAGES=640,WRITE_SLOT=639,SPLITS=64,GRID=256,BH=16,BK=16,DIRECT=True,num_warps=4,enable_fp_fusion=False)
  self.record('attention',dict(output=err(o,ref),health=int(health[0]),length=n))
 def check_mhc(self,hc,ref,is_attn):
  A,F=self.args();w,s,b,n=[self.w.a[i] for i in [3,4,5,6]] if is_attn else [self.w.f[i] for i in [2,3,4,5]]
  if hc.x_prev is not None:
   self.mprev[4:8].copy_(hc.post_mix.flatten());self.mprev[8:].copy_(hc.comb_mix.flatten())
  mhc[(256,)](hc.residual,A[1],A[2],hc.x_prev if hc.x_prev is not None else A[51],self.mprev,self.r32,w,s,b,n,A[7],A[8],A[9],A[10],A[29],A[30],HAS_PREV=hc.x_prev is not None,num_warps=4,enable_fp_fusion=False)
  self.record('mhc_attn' if is_attn else 'mhc_ffn',dict(input=err(A[10],ref.x_prev),post=err(A[7][4:8],ref.post_mix),comb=err(A[7][8:],ref.comb_mix)))

 def check_compressor(self,name,comp,x,ks,ss):
  from atom.utils.forward_context import get_forward_context
  from binding import compressor_args
  from project_compress import project_compress
  md=get_forward_context().attn_metadata;r=self.w.ratio if name=='main' else 4
  C,M=compressor_args(comp,md,r);C=list(C);M=list(M);d=comp.head_dim;kb=256//r
  C[8]=ks;C[9]=ss
  M[4]=ks.stride(0);M[5]=ks.stride(1);M[6]=ss.stride(0);M[7]=ss.stride(1)
  cache=torch.zeros((1,kb,d),device='cuda',dtype=comp.kv_cache.dtype)
  scale=torch.zeros((1,kb),device='cuda',dtype=torch.float32)
  C[10]=cache;C[11]=scale;C[12]=torch.zeros_like(md.block_tables)
  M[8]=cache.stride(0);M[9]=cache.stride(1);M[10]=scale.stride(0)
  weight=self.compressor_weights[name];N=weight.shape[0];y=torch.empty((1,N),device='cuda',dtype=torch.bfloat16)
  count=torch.zeros(1,device='cuda',dtype=torch.int32);health=torch.zeros_like(count)
  project_compress[(256,)](x,weight,y,tuple(C),count,health,tuple(M),N,num_warps=4,enable_fp_fusion=False)
  slot=int(md.state_slot_out[0]);metrics=dict(kv_state=err(ks[slot],comp.kv_state[slot]),score_state_neq=int((ss[slot]!=comp.score_state[slot]).sum()),health=int(health[0]))
  pos=int(C[4].flatten()[2])
  if pos>=0:
   ci=pos//r;block=int(md.block_tables.flatten()[ci//kb]);tok=ci%kb
   lhs=cache[0,tok];rhs=comp.kv_cache[block,tok,:d]
   if comp.quant_mode=='per_row_fp8':rhs=comp.kv_cache[block].flatten()[tok*d:(tok+1)*d]
   if comp.quant_mode=='per_row_fp8' and lhs.dtype==torch.uint8:lhs=lhs.view(torch.float8_e4m3fn);rhs=rhs.view(torch.float8_e4m3fn)
   metrics['cache']=err(lhs,rhs)
   if comp.cache_scale is not None:metrics['scale']=err(scale[0,tok],comp.cache_scale[block,tok])
  self.record('compressor_'+name,metrics)
