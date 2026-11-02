import sys,torch,triton,json
sys.path[:0]=['/work/integrated-c8','/work/modelwide','/work/concurrency-rollout','/work/full-attention-native/prefix']
from edge_sync import edge,publish,consume
from combined_scaled import sync
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
@g.jit
def check(E,P,O,H,C,NP:gl.constexpr,NC:gl.constexpr):
 p=gl.program_id(0)
 if p==0:gl.store(E,0)
 sync(C,H,256,3)
 if p<NP:
  v=gl.load(P+p)+1
  gl.store(P+p,v)
 edge(E,H,NP,p<NP,p>=256-NC,1)
 if p>=256-NC:
  i=gl.arange(0,256,layout=gl.BlockedLayout([1],[32],[4],[0]))
  v=gl.load(P+i,i<NP,0)
  gl.store(O+p,gl.sum(v,0))
for np,nc in [(64,32),(192,16),(8,256),(128,8),(8,8)]:
 e=torch.zeros(320,device='cuda',dtype=torch.int32);p=torch.zeros(256,device='cuda',dtype=torch.int32);o=p.clone();h=torch.zeros(1,device='cuda',dtype=torch.int32)
 c=torch.zeros_like(h)
 fn=lambda:check[(256,)](e,p,o,h,c,np,nc,num_warps=4)
 fn();torch.cuda.synchronize()
 with torch.cuda.graph(graph:=torch.cuda.CUDAGraph()):
  for _ in range(100):fn()
 for _ in range(10):graph.replay()
 torch.cuda.synchronize()
 assert int(h)==0 and bool((o[256-nc:]==p[:np].sum()).all())
 assert int(e[0])==np
 print(dict(producers=np,consumers=nc,iterations=int(p[0]),health=int(h)),flush=True)
