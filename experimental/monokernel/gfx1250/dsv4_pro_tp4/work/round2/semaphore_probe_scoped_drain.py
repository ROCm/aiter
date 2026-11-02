"""Isolated last-producer protocol; no GEMM or serving integration claims."""
import json,statistics
from pathlib import Path
import torch,triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.testing import do_bench_cudagraph

@g.jit
def _publish(X,P,C,Y,T:gl.constexpr,S:gl.constexpr,B:gl.constexpr,SEM:gl.constexpr):
 t=gl.program_id(0);s=gl.program_id(1)
 layout:gl.constexpr=gl.BlockedLayout([1,4],[1,32],[1,4],[1,0])
 j=gl.arange(0,B,layout=gl.SliceLayout(0,layout))
 v=gl.load(X+(s*T+t)*B+j)
 if SEM:
  gl.inline_asm_elementwise("global_store_b32 $1, $2, off scope:SCOPE_DEV\n s_wait_storecnt 0\n v_mov_b32 $0, 0", constraints="=v,v,v,~{memory}", args=[(P+(s*T+t)*B+j).to(gl.int64),v], dtype=gl.int32, is_pure=False, pack=1)
 else:
  gl.store(P+(s*T+t)*B+j,v)
 if SEM:
  gl.barrier()
  ticket=gl.atomic_add(C+t,1,sem='relaxed',scope='gpu')
  if ticket % S == S-1:
   si=gl.arange(0,S,layout=gl.SliceLayout(1,layout))
   p=gl.inline_asm_elementwise("global_load_b32 $0, $1, off scope:SCOPE_DEV\n s_wait_loadcnt 0", constraints="=v,v,~{memory}", args=[(P+(si[:,None]*T+t)*B+j[None,:]).to(gl.int64)], dtype=gl.float32, is_pure=False, pack=1)
   out=gl.sum(p,axis=0)
   gl.store(Y+t*B+j,out)

@g.jit
def _reduce(P,Y,T:gl.constexpr,S:gl.constexpr,B:gl.constexpr):
 t=gl.program_id(0)
 layout:gl.constexpr=gl.BlockedLayout([1,4],[1,32],[1,4],[1,0])
 j=gl.arange(0,B,layout=gl.SliceLayout(0,layout));s=gl.arange(0,S,layout=gl.SliceLayout(1,layout))
 v=gl.load(P+(s[:,None]*T+t)*B+j[None,:]);gl.store(Y+t*B+j,gl.sum(v,axis=0))

def boundary(x,p,c,y,semaphore):
 s,t,b=x.shape
 _publish[(t,s)](x,p,c,y,t,s,b,semaphore,num_warps=4)
 if not semaphore:_reduce[(t,)](p,y,t,s,b,num_warps=4)
 return y

rows=[]
for s in (2,4,8):
 for t,b in ((32,1024),(128,1024),(512,1024),(32,8192),(128,8192)):
  x=torch.randn(s,t,b,device='cuda');p=torch.empty_like(x);c=torch.zeros(t,device='cuda',dtype=torch.uint32);y=torch.empty(t,b,device='cuda');ref=boundary(x,p,c,y,False).clone()
  for _ in range(2):torch.testing.assert_close(boundary(x,p,c,y,True),ref,rtol=0,atol=0)
  c.fill_(2**32-16)
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):
   x.add_(0.125)
   boundary(x,p,c,y,True)
  for _ in range(1000):graph.replay()
  torch.cuda.synchronize();actual=y.clone();ref=boundary(x,p,c,y,False)
  torch.testing.assert_close(actual,ref,rtol=0,atol=0)
  assert bool(torch.all(c.to(torch.int64) % s == 0))
  bs=[];ss=[]
  for i in range(5):
   for mode in ((False,True) if i%2==0 else (True,False)):
    (ss if mode else bs).append(do_bench_cudagraph(lambda:boundary(x,p,c,y,mode),rep=80)*1000)
  row=dict(splits=s,tiles=t,elements_per_tile=b,two_launch_us=statistics.median(bs),semaphore_us=statistics.median(ss));rows.append(row);print(row,flush=True)
Path('/work/round2/semaphore_probe_scoped_drain.json').write_text(json.dumps(rows,indent=2))
