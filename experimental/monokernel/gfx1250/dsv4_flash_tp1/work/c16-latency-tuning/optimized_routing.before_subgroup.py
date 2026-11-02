from edge_sync import edge,publish,consume,prefetch_bytes
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from combined_scaled import sync
@g.jit
def add(a,b):return a+b
@g.jit
def route_body(Logits,Hist,Offs,Pad,Map,Gather,Scatter,Gamma,Bias,HAS_BIAS:gl.constexpr,Chosen,Weights,Counter,Health,M:gl.constexpr,Edges=None,Epoch=0):
 p=gl.program_id(0);L:gl.constexpr=gl.BlockedLayout([1],[32],[4],[0]);i=gl.arange(0,256,layout=L)
 if p<M:
  x=gl.load(Logits+p*256+i).to(gl.float32)
  scores=gl.sqrt(gl.maximum(x,0.)+gl.log(1.+gl.exp(-gl.abs(x)))).to(gl.bfloat16).to(gl.float32)
  if HAS_BIAS:
   bias=gl.load(Bias+i);scores=(scores+bias.to(gl.bfloat16).to(gl.float32)).to(gl.bfloat16).to(gl.float32)
  remain=scores;chosen=gl.full((256,),0,gl.int32,L)
  for j in gl.static_range(7 if M>=16 else 6):
   best=gl.max(remain,0);idx=gl.max(gl.where(remain==best,i,-1),0)
   chosen=gl.where(i==idx,1,chosen);remain=gl.where(i==idx,float('-inf'),remain)
  if HAS_BIAS:scores=(scores-bias).to(gl.bfloat16).to(gl.float32)
  total=gl.sum(gl.where(chosen!=0,scores,0.),0)
  gl.store(Chosen+p*256+i,chosen);gl.store(Weights+p*256+i,(scores if M>=16 else scores/(total+1.e-20)*1.5).to(gl.bfloat16))
 if Edges is not None:edge(Edges,Health,M,p<M,p<M,Epoch=Epoch)
 else:sync(Counter,Health,256,3)
 # HERD turns on at native decode batch16: top7 -> drop least popular,
 # then smallest unbiassed score, then smallest expert index (native list order).
 if M>=16:
  if p<M:
   HL:gl.constexpr=gl.BlockedLayout([1,1],[1,32],[1,4],[1,0])
   hr=gl.arange(0,M,layout=gl.SliceLayout(1,HL));hc=gl.arange(0,256,layout=gl.SliceLayout(0,HL))
   top=gl.load(Chosen+hr[:,None]*256+hc[None,:])
   popularity=gl.convert_layout(gl.sum(top,0),L)
   own=gl.load(Chosen+p*256+i);value=gl.load(Weights+p*256+i).to(gl.float32)
   least=gl.min(gl.where(own!=0,popularity,2147483647),0)
   low=gl.min(gl.where((own!=0)&(popularity==least),value,float('inf')),0)
   drop=gl.min(gl.where((own!=0)&(popularity==least)&(value==low),i,2147483647),0)
   keep=gl.where(i==drop,0,own)
   total=gl.sum(gl.where(keep!=0,value,0.),0)
   # Keep top7 immutable until all readers have loaded it. A separate scratch
   # column stores each drop; overwrite is done only after the subgroup barrier.
   gl.store(Chosen+M*256+p,drop)
   gl.store(Weights+p*256+i,(value/(total+1.e-20)*1.5).to(gl.bfloat16))
  sync(Counter,Health,256,3)
  if p<M:
   drop=gl.load(Chosen+M*256+p,cache_modifier=".cg",volatile=True)
   gl.store(Chosen+p*256+i,gl.where(i==drop,0,gl.load(Chosen+p*256+i)))
  sync(Counter,Health,256,3)
 if p<M:
  LL:gl.constexpr=gl.BlockedLayout([1,1],[1,32],[1,4],[1,0])
  rr=gl.arange(0,M,layout=gl.SliceLayout(1,LL));cc=gl.arange(0,256,layout=gl.SliceLayout(0,LL))
  allch=gl.amd.gfx1250.buffer_load(Chosen,rr[:,None]*256+cc[None,:],cache=".cg")
  hist=gl.convert_layout(gl.sum(allch,0),L);used=gl.convert_layout(gl.sum(gl.where(rr[:,None]<p,allch,0),0),L)
  offs=gl.associative_scan(hist,0,add)-hist
  if p==0:
   active=(hist>0).to(gl.int32);rank=gl.associative_scan(active,0,add)-active;nactive=gl.sum(active,0)
   gl.store(Hist+i,hist);gl.store(Offs+i,offs);gl.store(Pad,nactive)
   gl.store(Map+i,-1,i<M*6);gl.barrier();gl.store(Map+rank,i,active!=0)
  ch=gl.amd.gfx1250.buffer_load(Chosen,p*256+i,cache=".cg");j=gl.associative_scan(ch,0,add)-ch;dst=offs+used
  gl.store(Gather+dst,p*6+j,ch!=0);gl.store(Scatter+p*6+j,dst,ch!=0)
  gl.store(Gamma+dst,gl.amd.gfx1250.buffer_load(Weights,p*256+i,cache=".cg"),ch!=0)
