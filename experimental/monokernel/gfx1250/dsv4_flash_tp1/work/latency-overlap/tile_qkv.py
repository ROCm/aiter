import sys,os
sys.path.insert(0,'/work/c8-tuning')
from common import *
sys.path.insert(0,'/work/integrated-c8')
from qkv_candidate import qkv_batch as candidate
sys.path.insert(0,'/work/atom-actual-microbench/qkv')
from native import NativeQKV
from qkv_full_batch import qkv_batch
from atom.model_ops.v4_kernels.qk_norm_rope_maybe_quant import qk_norm_rope_maybe_quant
n=NativeQKV();f=n.f;m=int(os.getenv('TEST_M','8'))
torch.manual_seed(1776);x=f['live'].cuda().reshape(1,4096).repeat(m,1)+torch.randn((m,4096),device='cuda',dtype=torch.bfloat16)*.01
w=f['W_bytes'].cuda().view(torch.float8_e4m3fn);ws=f['WS'].cuda().view(torch.uint8);nw=f['NW'].cuda();qb=f['QB_bytes'].cuda().view(torch.float8_e4m3fn);qbs=f['QBS'].cuda().view(torch.uint8)
xq=torch.empty((m,4096),device='cuda',dtype=torch.float8_e4m3fn);xs=torch.empty((m,32),device='cuda',dtype=torch.uint8)
p=torch.empty((16,m,1536),device='cuda');y=torch.empty((m,1536),device='cuda',dtype=torch.bfloat16);q=torch.empty((m,1024),device='cuda',dtype=torch.float8_e4m3fn);s=torch.empty((m,8),device='cuda',dtype=torch.uint8);kv=torch.empty((m,512),device='cuda',dtype=torch.bfloat16)
cos=n.cos[1024].reshape(1,32).repeat(m,1);sin=n.sin[1024].reshape(1,32).repeat(m,1);counter=torch.zeros(1,device='cuda',dtype=torch.int32);health=torch.zeros_like(counter)
qbr=torch.empty((m,32768),device='cuda');qo=torch.empty((m,32768),device='cuda',dtype=torch.bfloat16);pos=torch.full((m,),1024,device='cuda',dtype=torch.int64)

import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from combined_scaled import sync,poll_read
@g.jit
def run_qkv(X,XQ,XS,W,WS,P,NW,Y,Q,S,Cos,Sin,KV,C,H,QBW,QBS,QBR,QO,E,M:gl.constexpr,TILED:gl.constexpr):
 if gl.program_id(0)==0:gl.atomic_add(E,1,sem='release',scope='gpu')
 sync(C,H,256,3)
 epoch=poll_read(E,1)
 candidate(X,XQ,XS,W,WS,P,NW,Y,Q,S,Cos,Sin,KV,C,H,QBW,QBS,QBR,QO,M,PREFETCH_STYLE=1,QBTiles=E+64 if TILED else None,Epoch=epoch)

def snapshot():return [a.clone() for a in (qo,kv,y,q.view(torch.uint8),s)]
ca=(x,xq,xs,w,ws,p,nw,y,q,s,cos,sin,kv,counter,health,qb,qbs,qbr,qo)
states=[torch.zeros(65*64,device='cuda',dtype=torch.int32) for _ in range(2)]
fns={}
for tiled in [False,True]:
 e=states[int(tiled)]
 def run(tiled=tiled,e=e):run_qkv[(256,)](*ca,e,m,tiled,**kw)
 fns['tile' if tiled else 'grid']=run
 run();torch.cuda.synchronize()
 if not tiled:ref=snapshot()
 else:
  for a,b in zip(snapshot(),ref):assert torch.equal(a,b)
 assert int(health)==0
print('initial exact',m,flush=True)
result=bench('tile-release-c'+str(m),fns,dict(exact=True,producer_ctas=256,producers_per_head=4,heads=64))
assert int(health)==0
for fn in fns.values():
 fn();torch.cuda.synchronize()
 for a,b in zip(snapshot(),ref):assert torch.equal(a,b)
print('replay exact health=0',m,flush=True)
