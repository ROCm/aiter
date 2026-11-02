import sys,os
sys.path.insert(0,'/work/c8-tuning')
from common import *
sys.path.insert(0,'/work/c8-qkv-deps')
from qkv_candidate import qkv_batch as candidate
sys.path.insert(0,'/work/atom-actual-microbench/qkv')
from native import NativeQKV
from qkv_full_batch import qkv_batch
from atom.model_ops.v4_kernels.qk_norm_rope_maybe_quant import qk_norm_rope_maybe_quant
n=NativeQKV();f=n.f;m=8
torch.manual_seed(1776);x=f['live'].cuda().reshape(1,4096).repeat(m,1)+torch.randn((m,4096),device='cuda',dtype=torch.bfloat16)*.01
w=f['W_bytes'].cuda().view(torch.float8_e4m3fn);ws=f['WS'].cuda().view(torch.uint8);nw=f['NW'].cuda();qb=f['QB_bytes'].cuda().view(torch.float8_e4m3fn);qbs=f['QBS'].cuda().view(torch.uint8)
xq=torch.empty((m,4096),device='cuda',dtype=torch.float8_e4m3fn);xs=torch.empty((m,32),device='cuda',dtype=torch.uint8)
p=torch.empty((16,m,1536),device='cuda');y=torch.empty((m,1536),device='cuda',dtype=torch.bfloat16);q=torch.empty((m,1024),device='cuda',dtype=torch.float8_e4m3fn);s=torch.empty((m,8),device='cuda',dtype=torch.uint8);kv=torch.empty((m,512),device='cuda',dtype=torch.bfloat16)
cos=n.cos[1024].reshape(1,32).repeat(m,1);sin=n.sin[1024].reshape(1,32).repeat(m,1);counter=torch.zeros(1,device='cuda',dtype=torch.int32);health=torch.zeros_like(counter)
qbr=torch.empty((m,32768),device='cuda');qo=torch.empty((m,32768),device='cuda',dtype=torch.bfloat16);pos=torch.full((m,),1024,device='cuda',dtype=torch.int64)
def old():
 qkv_batch[(256,)](x,xq,xs,w,ws,p,nw,y,q,s,cos,sin,kv,counter,health,qb,qbs,qbr,qo,m,PREFETCH_STYLE=1,**kw)
 return qo
def native():
 a=n.qa(x);rq,rs=n.norm(a[:,:1024]);raw=n.qb(rq,x_scale=rs)
 return qk_norm_rope_maybe_quant(raw,a[:,1024:],n.nw,n.cos,n.sin,pos,64,512,64,1e-6).q_sa


import argparse
ap=argparse.ArgumentParser();ap.add_argument('--variant',default='sem');args=ap.parse_args()
opts={'sem':dict(SEM=True),'early':dict(SEM=False),'s8':dict(SPLITS=8,BN=64,SEM=True),'s4':dict(SPLITS=4,BN=32,SEM=True),'atomic':dict(SPLITS=8,BN=64,ATOMIC=True,SEM=True)}[args.variant]
ready=torch.zeros(257,device='cuda',dtype=torch.int32)
ca=(x,xq,xs,w,ws,p,nw,y,q,s,cos,sin,kv,counter,health,qb,qbs,qbr,qo,m)
ck=dict(PREFETCH_STYLE=1,Ready=ready,**opts,**kw)
c=candidate.warmup(*ca,grid=(256,),**ck);c._init_handles();assert c.n_spills==0
checks=[]
# No counter/epoch resets between changing-input invocations or captured replays.
def run():candidate[(256,)](*ca,**ck)
graph=FlatGraphBatch(run,32)
for seed in range(32):
 torch.manual_seed(8000+seed)
 if seed%4==0: xx=torch.zeros_like(x)
 elif seed%4==1:xx=torch.randn_like(x)*.2
 elif seed%4==2:xx=f['live'].cuda().reshape(1,4096).repeat(m,1)+torch.randn_like(x)*.03
 else:xx=torch.randn_like(x)*torch.pow(torch.tensor(2.,device='cuda'),torch.arange(m,device='cuda')[:,None]-4)
 x.copy_(xx)
 ref=native().clone();oq=old().clone();ok=kv.clone();oy=y.clone();of=q.clone();os=s.clone()
 for rep in range(2):
  # Poison outputs; producer scratch is overwritten/cleared by the kernel itself.
  y.fill_(float('nan'));qo.fill_(float('nan'));kv.fill_(float('nan'))
  graph.replay();torch.cuda.synchronize()
  d=dict(seed=seed,rep=rep,q_native=err(qo,ref),q_old=err(qo,oq),kv_old=err(kv,ok),raw_old=err(y,oy),fp8_equal=bool(torch.equal(q.view(torch.uint8),of.view(torch.uint8))),scales_equal=bool(torch.equal(s,os)),health=int(health.item()))
  checks.append(d)
  assert d['health']==0 and d['q_native']['finite'] and d['kv_old']['finite'],d
summary=dict(variant=args.variant,cases=len(checks),replays=len(checks)*32,checks=checks,max_q_rms=max(d['q_old']['relative_rms'] for d in checks),max_q_native_rms=max(d['q_native']['relative_rms'] for d in checks),max_kv_rms=max(d['kv_old']['relative_rms'] for d in checks),all_fp8_equal=all(d['fp8_equal'] and d['scales_equal'] for d in checks),all_q_exact=all(d['q_old']['max_abs']==0 for d in checks))
(ROOT/('stress-'+args.variant+'.json')).write_text(json.dumps(summary,indent=2));print({k:v for k,v in summary.items() if k!='checks'},flush=True)
