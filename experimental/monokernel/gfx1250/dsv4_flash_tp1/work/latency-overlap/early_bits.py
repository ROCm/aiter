import sys,torch,json,os,triton
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
 kw.update(PRE_W=aiter.mhc_shuffle_fn(aw[3].contiguous().float()).view(torch.bfloat16))
 saved=[(cache,cache.clone()),(idx,idx.clone())]
 for cc in [MC,IC] if kw['RATIO'] else []:
  if cc:
   saved.extend((x,x.clone()) for x in cc[8:12])
 outputs=[];metrics=[]
 for use in [0,1]:
  for x,backup in saved:x.copy_(backup)
  kw['OPT_PRE']=True
  kw['Edges']=None
  kw['OVERLAP']=int(os.getenv('EARLY_MASK','2')) if use else 0
  obj=batch.warmup(*args,M=m,BLOCK_STRIDE=32,grid=(256,),**kw);obj._init_handles()
  print(dict(compiled=m,ratio=kw['RATIO'],new=use,vgpr=obj.n_regs,spills=obj.n_spills,lds=obj.metadata.shared),flush=True)
  assert obj.n_spills==0
  batch[(256,)](*args,M=m,BLOCK_STRIDE=32,**kw)
  torch.cuda.synchronize()
  assert int(s.a[30])==0 and int(s.f[-1])==0
  outputs.append([s.output.clone(),s.a[10].clone(),s.a[28].clone()])
  assert all(torch.isfinite(x).all() for x in outputs[-1])
  fn=lambda:batch[(256,)](*args,M=m,BLOCK_STRIDE=32,**kw)
  times=[triton.testing.do_bench_cudagraph(fn,rep=1)*1000 for _ in range(2)]
  assert int(s.a[30])==0 and int(s.f[-1])==0
  metrics.append(dict(mode=use,vgpr=obj.n_regs,spills=obj.n_spills,lds=obj.metadata.shared,rounds_us=times,second_us=times[1]))
 checks={}
 for mode in [1]:
  checks[mode]={}
  for name,x,y in zip(['final','norm','q'],outputs[mode],outputs[0]):
   xf=x.float();yf=y.float();checks[mode][name]=dict(exact=bool(torch.equal(x,y)),max_abs=float((xf-yf).abs().max()),relative_rms=float(((xf-yf).square().mean()/yf.square().mean().clamp_min(1.e-20)).sqrt()))
  assert all(v['exact'] for v in checks[mode].values()), checks
 print(dict(m=m,ratio=kw['RATIO'],checks=checks,metrics=metrics),flush=True)
 results.append(dict(m=m,ratio=kw['RATIO'],checks=checks,resources=metrics))
open('/work/latency-overlap/early-'+os.getenv('EARLY_MASK','2')+'.json','w').write(json.dumps(results,indent=2))
