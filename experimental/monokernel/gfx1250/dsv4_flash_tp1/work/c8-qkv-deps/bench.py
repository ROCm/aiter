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
ap=argparse.ArgumentParser();ap.add_argument('--variants',required=True);args=ap.parse_args()
ready=torch.zeros(257,device='cuda',dtype=torch.int32)
variants={
 'control':dict(LOCAL=False,PRE=False,SEM=False),
 'local':dict(LOCAL=True,PRE=False,SEM=False),
 'early':dict(LOCAL=True,PRE=True,SEM=False),
 'sem':dict(LOCAL=True,PRE=True,SEM=True),
 's8n128':dict(SPLITS=8,BN=128),
 's8n64':dict(SPLITS=8,BN=64),
 's4n128':dict(SPLITS=4,BN=128),
 's4n32':dict(SPLITS=4,BN=32),
 'atomic16':dict(ATOMIC=True),
 'atomic8':dict(SPLITS=8,BN=64,ATOMIC=True),
 'atomic4':dict(SPLITS=4,BN=32,ATOMIC=True),
 'sem_late':dict(PRE=False),
 'early_only':dict(LOCAL=False,PRE=True,SEM=False),
 'sem_only':dict(LOCAL=False,PRE=False,SEM=True),
}
results={}
for name in args.variants.split(','):
 opts=dict(SPLITS=16,BN=128,LOCAL=True,PRE=True,SEM=True,ATOMIC=False);opts.update(variants[name])
 ca=(x,xq,xs,w,ws,p,nw,y,q,s,cos,sin,kv,counter,health,qb,qbs,qbr,qo,m)
 ck=dict(PREFETCH_STYLE=1,Ready=ready,**opts,**kw)
 try:
  compiled=candidate.warmup(*ca,grid=(256,),**ck)
  meta=compiled.metadata._asdict();(ROOT/(name+'-metadata.json')).write_text(json.dumps(meta,indent=2,default=str))
  (ROOT/(name+'.amdgcn')).write_text(compiled.asm['amdgcn'])
  (ROOT/(name+'.llir')).write_text(compiled.asm['llir'])
  compiled._init_handles()
  resources=dict(regs=compiled.n_regs,spills=compiled.n_spills,shared=meta['shared']);print(name,resources,flush=True)
  assert compiled.n_spills==0 and meta['shared']<=327680,resources
  def run():
   candidate[(256,)](*ca,**ck)
   return qo
  validation=[]
  for seed in [1776,23,129]:
   torch.manual_seed(seed)
   x.copy_(f['live'].cuda().reshape(1,4096).repeat(m,1)+torch.randn_like(x)*(.01 if seed==1776 else .2))
   ref=native().clone();oldq=old().clone();oldkv=kv.clone();oldraw=y.clone();oldqr=q.clone();oldqs=s.clone()
   for repeat in range(3):
    run();torch.cuda.synchronize()
    validation.append(dict(seed=seed,repeat=repeat,q_vs_native=err(qo,ref),q_vs_old=err(qo,oldq),kv_vs_old=err(kv,oldkv),qa_raw_vs_old=err(y,oldraw),qa_fp8_equal=bool(torch.equal(q.view(torch.uint8),oldqr.view(torch.uint8))),qa_scales_equal=bool(torch.equal(s,oldqs)),health=int(health.item())))
   assert health.item()==0
   assert validation[-1]['q_vs_native']['finite']
   assert validation[-1]['q_vs_old']['relative_rms']<0.001, validation[-1]
  torch.manual_seed(1776);x.copy_(f['live'].cuda().reshape(1,4096).repeat(m,1)+torch.randn_like(x)*.01)
  results[name]=bench(name,{'native':native,'old':old,'candidate':run},dict(resources=resources,checks=validation,options=opts))
 except Exception as e:
  import traceback
  traceback.print_exc();results[name]={'error':repr(e)}
 (ROOT/'worker-results.json').write_text(json.dumps(results,indent=2,default=str))
