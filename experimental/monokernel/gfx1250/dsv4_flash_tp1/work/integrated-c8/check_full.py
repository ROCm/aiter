import sys,torch,json,os
sys.path[:0]=['/work/integrated-c8','/work/batch-full','/work/modelwide','/work/concurrency-rollout','/work/full-attention-native','/work/full-attention-native/prefix','/work/full-attention-native/output-kernel','/work/ffn-c1/moe']
from scratch_batch import BatchState,A_WEIGHTS,F_WEIGHTS
from model_layer_batch import model_layer as batch
from model_layer_precise import model_layer as single
spec=json.load(open('/work/modelwide/scratch-spec.json'));torch.manual_seed(371)
def weight(k,i):
 s=spec[k][i];dt=getattr(torch,s['dtype'].split('.')[-1]);sh=s['shape']
 if dt==torch.uint8:return torch.full(sh,0x22 if k=='f' and i in [25,27] else 120,device='cuda',dtype=dt)
 v=torch.randn(sh,device='cuda')
 if i in ([3] if k=='a' else [2]):v*=.002
 if i in ([6,15] if k=='a' else [5]):v.fill_(1.)
 if i in ([4] if k=='a' else [3]):v.fill_(1.)
 return v.to(dt)
aw={i:weight('a',i) for i in A_WEIGHTS};fw={i:weight('f',i) for i in F_WEIGHTS}
# Physical scales are fully initialized, random values are finite.
strides=[fw[i].stride() for i in [25,26,27,28]]
def bind(s):
 for i,w in aw.items():s.a[i]=w
 for i,w in fw.items():s.f[i]=w
 return tuple(s.a),tuple(s.f)
results=[];t=128;d=torch.empty(0,device='cuda');cs=torch.randn((2048,32),device='cuda');sn=torch.randn_like(cs)
for m in [int(x) for x in os.getenv('TEST_BATCHES','1,2,4,8').split(',')]:
 s=BatchState(m,t);raw=torch.randn((m,4,4096),device='cuda',dtype=torch.bfloat16)
 inp=raw.reshape(m,4,64,64).permute(2,1,0,3).contiguous().reshape(m,4,4096)
 s.a[0]=inp;A,F=bind(s)
 pos=torch.arange(1023,1023+m,device='cuda',dtype=torch.int32);cache=torch.randn((m*t,512),device='cuda',dtype=torch.bfloat16)*.1
 idx=torch.arange(m*t,device='cuda',dtype=torch.int32);ptr=torch.arange(0,(m+1)*t,t,device='cuda',dtype=torch.int32);dest=ptr[1:]-1
 kw=dict(MM=(),IM=(),RATIO=0,ENVELOPE=640,W1S=strides[0],S1S=strides[1],W2S=strides[2],S2S=strides[3],HAS_BIAS=False,T=t,num_warps=4,enable_fp_fusion=False)
 import aiter
 from aiter.ops.triton.fusions.mhc import _gluon_post_pre_layouts,_gluon_reduce_apply_layouts
 kw.update(CacheRows=cache.numel()//512,Work=s.pool,CompCounter=s.comp_counter,MHC=(aiter.mhc_shuffle_fn(fw[2].contiguous().float()).view(torch.bfloat16),*s.mhc),ML=_gluon_post_pre_layouts(4,16,64,32,2,True,4),MV=_gluon_reduce_apply_layouts(4,1,4,64),OPT_MHC=os.getenv('OPT_MHC','1')=='1',OPT_COMP=os.getenv('OPT_COMP','1')=='1')
 args=(A,F,(),(),d,d,d,d,pos,cs,sn,cache,idx,ptr,dest,d,d,d,*s.extra())
 obj=batch.warmup(*args,M=m,BLOCK_STRIDE=32,FP32MID=s.mid32,FP32OUT=s.out32,grid=(256,),STOP=int(os.getenv("STOP","0")),PREFETCH=os.getenv("PREFETCH","1")=="1",**kw)
 obj._init_handles()
 from pathlib import Path
 Path('/work/integrated-c8/compiled-'+str(kw['RATIO'])+'-'+str(m)+'.amdgcn').write_text(obj.asm['amdgcn'])
 print(dict(compiled=m,vgpr=obj.n_regs,spills=obj.n_spills,lds=obj.metadata.shared),flush=True)
 if obj.n_spills:raise RuntimeError('Refusing to launch spilling persistent full layer')
 batch[(256,)](*args,M=m,BLOCK_STRIDE=32,FP32MID=s.mid32,FP32OUT=s.out32,STOP=int(os.getenv("STOP","0")),PREFETCH=os.getenv("PREFETCH","1")=="1",**kw)
 torch.cuda.synchronize();print("batch launch completed",flush=True)
 if os.getenv("STOP","0")!="0":continue
 refs=[]
 for row in range(m):
  r=BatchState(1,t);r.a[0]=raw[row].reshape(4,64,64).permute(1,0,2).contiguous();AR,FR=bind(r)
  # C1 final post writes the native shuffled layout into F[1].
  single[(256,)](AR,FR,(),(),d,d,d,d,pos[row:row+1],cs,sn,cache,idx,ptr[row:row+2],dest[row:row+1],d,d,d,FP32MID=r.mid32,FP32OUT=r.out32,**{k:v for k,v in kw.items() if k not in ['CacheRows','Work','CompCounter','MHC','ML','MV','OPT_MHC','OPT_COMP']});refs.append(r)
 torch.cuda.synchronize();out=s.output.reshape(64,4,m,64).permute(2,1,0,3).reshape(m,4,4096)
 ref=torch.cat([r.f[1].reshape(64,4,1,64).permute(2,1,0,3).reshape(1,4,4096) for r in refs])
 checks={}
 for name,k in [('norm',10),('q',28),('kv',23),('attention',38),('output',51),('mhc_post',2)]:
  got=s.a[k].reshape(m,-1);want=torch.cat([r.a[k].reshape(1,-1) for r in refs]);checks[name]=dict(exact=bool(torch.equal(got,want)),rms=float(((got.float()-want.float()).square().mean()/want.float().square().mean().clamp_min(1.e-20)).sqrt()),max_abs=float((got.float()-want.float()).abs().max()))
 checks['final']=dict(exact=torch.equal(out[:ref.shape[0]],ref),rms=float(((out[:ref.shape[0]].float()-ref.float()).square().mean()/ref.float().square().mean().clamp_min(1.e-20)).sqrt()),max_abs=float((out[:ref.shape[0]].float()-ref.float()).abs().max()));checks['finite']=bool(torch.isfinite(out).all())
 result=dict(m=m,checks=checks,health=[int(s.a[30]),int(s.f[-1])],vgpr=obj.n_regs,lds=obj.metadata.shared,spills=obj.n_spills)
 print(result,flush=True);results.append(result);assert checks['finite'] and result['health']==[0,0]
open('/work/integrated-c8/full-checks.json','w').write(json.dumps(results,indent=2))
