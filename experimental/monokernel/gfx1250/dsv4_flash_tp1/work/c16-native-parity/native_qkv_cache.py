from common import *
sys.path.insert(0,'/work/atom-actual-microbench/qkv')
from native import NativeQKV
from qkv_candidate import qkv_batch
from atom.model_ops.v4_kernels.qk_norm_rope_maybe_quant import qk_norm_rope_maybe_quant
n=NativeQKV();f=n.f;m=M
torch.manual_seed(1776);x=f['live'].cuda().reshape(1,4096).repeat(m,1)+torch.randn((m,4096),device='cuda',dtype=torch.bfloat16)*.01
w=f['W_bytes'].cuda().view(torch.float8_e4m3fn);ws=f['WS'].cuda().view(torch.uint8);nw=f['NW'].cuda();qb=f['QB_bytes'].cuda().view(torch.float8_e4m3fn);qbs=f['QBS'].cuda().view(torch.uint8)
xq=torch.empty((m,4096),device='cuda',dtype=torch.float8_e4m3fn);xs=torch.empty((m,32),device='cuda',dtype=torch.uint8)
p=torch.empty((16,m,1536),device='cuda');y=torch.empty((m,1536),device='cuda',dtype=torch.bfloat16);q=torch.empty((m,1024),device='cuda',dtype=torch.float8_e4m3fn);s=torch.empty((m,8),device='cuda',dtype=torch.uint8);kv=torch.empty((m,512),device='cuda',dtype=torch.bfloat16)
cos=n.cos[1023].reshape(1,32).repeat(m,1);sin=n.sin[1023].reshape(1,32).repeat(m,1);counter=torch.zeros(2048*64,device='cuda',dtype=torch.int32);health=torch.zeros(1024 if os.getenv("DIAG") else 1,device="cuda",dtype=torch.int32)
qbr=torch.empty((m,32768),device='cuda');qo=torch.empty((m,32768),device='cuda',dtype=torch.bfloat16);pos=torch.full((m,),1023,device='cuda',dtype=torch.int64)
from isolated import qkv_cache
from packing import pack_weight
wp=pack_weight(w);qbp=pack_weight(qb)
cache=torch.zeros((m*128,512),device="cuda",dtype=torch.bfloat16);cache2=torch.zeros_like(cache);dest=torch.arange(m,device="cuda",dtype=torch.int32)*128+127;bid=torch.arange(m,device="cuda",dtype=torch.int32)
from previous.isolated import qkv_cache as previous_op
resources={}
def fused(op=qkv_cache):
 obj=op[(256,)]((x,xq,xs,w if op is previous_op or os.getenv("PREVIOUS_QA")=="1" else wp,ws,p,nw,y,q,s,cos,sin,kv,counter,health,qb if op is previous_op else qbp,qbs,qbr,qo),m,cache2,dest,m*128,**kw)
 if str(op) not in resources:
  resources[str(op)]=dict(regs=obj.n_regs,spills=obj.n_spills,lds=obj.metadata.shared)
  print('RESOURCES',str(op),resources[str(op)],flush=True)
  (ROOT/('asm_'+('previous' if op is previous_op else 'new')+'.amdgcn')).write_text(obj.asm['amdgcn'])
 return qo
def native():
 a=n.qa(x);rq,rs=n.norm(a[:,:1024]);raw=n.qb(rq,x_scale=rs)
 return qk_norm_rope_maybe_quant(raw,a[:,1024:],n.nw,n.cos,n.sin,pos,64,512,64,1e-6,swa_kv=cache,swa_dest_rows=dest,batch_id_per_q_token=bid).q_sa
rr=native();vv=fused();torch.cuda.synchronize();print("INITIAL_HEALTH",int(health[0]),"COUNTERS",counter[::64][:105].tolist(),flush=True);print("DIAG",counter.data_ptr(),health.tolist(),flush=True) if int(health[0]) else None;assert int(health[0])==0
bench('qkv_cache',{'native':native,'previous':lambda:fused(previous_op),'fused':fused},dict(q=err(vv,rr),cache=err(cache2,cache)))

(ROOT/"resources.json").write_text(json.dumps(resources,indent=2))
