import sys,torch,json,os,triton,statistics
sys.path[:0]=['/work/c16-native-projections','/work/batch-full','/work/modelwide','/work/concurrency-rollout','/work/full-attention-native','/work/full-attention-native/prefix','/work/full-attention-native/output-kernel','/work/ffn-c1/moe']
from scratch_batch import BatchState,A_WEIGHTS,F_WEIGHTS
from model_layer_batch import model_layer as batch
from regions import region
from bench_regions import benchmark_regions
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
ratio=int(os.getenv('RATIO','4'));results=[];t={4:640,128:144}[ratio];d=torch.empty(0,device='cuda');cs=torch.randn((2048,32),device='cuda');sn=torch.randn_like(cs)
for m in [int(x) for x in os.getenv('TEST_BATCHES','8,16').split(',')]:
 s=BatchState(m,t);raw=torch.randn((m,4,4096),device='cuda',dtype=torch.bfloat16)
 inp=raw.reshape(m,4,64,64).permute(2,1,0,3).contiguous().reshape(m,4,4096)
 s.a[0]=inp;A,F=bind(s)
 pos=torch.full((m,),1023,device='cuda',dtype=torch.int32);cache=torch.randn((m*32*640,512),device='cuda',dtype=torch.bfloat16)*.1
 lengths=128+(pos+1)//ratio;ptr=torch.cat([torch.zeros(1,device='cuda',dtype=torch.int32),lengths.cumsum(0).int()]);idx=torch.cat([torch.arange(int(lengths[j]),device='cuda',dtype=torch.int32)+j*32*640 for j in range(m)]);dest=torch.stack([idx[ptr[j+1]-1] for j in range(m)])
 blocks=torch.arange(m*32,device='cuda',dtype=torch.int32).reshape(m,32);top=torch.empty((m,512),device='cuda',dtype=torch.int32)
 def comp(dim,quant):
  df=dim*(2 if ratio==4 else 1);state=ratio*(2 if ratio==4 else 1)
  ape=torch.randn((ratio,df),device='cuda');norm=torch.randn(dim,device='cuda');cp=torch.zeros((m,4),device='cuda',dtype=torch.int32);wp=cp.clone();cp[:,2]=torch.where((pos+1)%ratio==0,pos,-1);cp[:,3]=state-1;wp[:,2]=pos
  slots=torch.arange(m,device='cuda',dtype=torch.int32)
  ks=torch.randn((m,state,df),device='cuda');ss=torch.randn_like(ks);cc=torch.zeros((m*32,256//ratio,dim),device='cuda',dtype=torch.float8_e4m3fn if quant else torch.bfloat16);sc=torch.zeros((m*32,256//ratio),device='cuda')
  c=(ape,norm,cs,sn,cp,wp,slots,slots,ks,ss,cc,sc,blocks)
  meta=(dim,ratio,ratio==4,state,ks.stride(0),ks.stride(1),ss.stride(0),ss.stride(1),cc.stride(0),cc.stride(1),sc.stride(0),quant)
  ref=c[:8]+tuple(v.clone() for v in c[8:12])+c[12:]
  w=torch.randn((2*df,4096),device='cuda',dtype=torch.bfloat16)*.02;y=torch.empty((m,2*df),device='cuda',dtype=torch.bfloat16)
  return c,meta,ref,w,y
 MC,MM,MREF,MW,MY=comp(512,False)
 if ratio==4:IC,IM,IREF,IW,IY=comp(128,True)
 else:IC=();IM=();IREF=();IW=d;IY=d
 def rowcomp(c,row):return c[:4]+(c[4][row],c[5][row],c[6][row:row+1],c[7][row:row+1])+c[8:12]+(c[12][row],)

 kw=dict(MM=MM,IM=IM,RATIO=ratio,ENVELOPE=640,W1S=strides[0],S1S=strides[1],W2S=strides[2],S2S=strides[3],HAS_BIAS=False,T=t,num_warps=4,enable_fp_fusion=False)
 import aiter
 from aiter.ops.triton.fusions.mhc import _gluon_post_pre_layouts,_gluon_reduce_apply_layouts
 kw.update(CacheRows=cache.numel()//512,Work=s.pool,CompCounter=s.comp_counter,MHC=(aiter.mhc_shuffle_fn(fw[2].contiguous().float()).view(torch.bfloat16),*s.mhc),ML=_gluon_post_pre_layouts(4,16,64,32,2,True,4),MV=_gluon_reduce_apply_layouts(4,1,4,64),OPT_MHC=os.getenv('OPT_MHC','1')=='1',OPT_COMP=os.getenv('OPT_COMP','1')=='1')
 args=(A,F,MC,IC,MW,IW,MY,IY,pos,cs,sn,cache,idx,ptr,dest,top,blocks,d,*s.extra())
 kw.update(PRE_W=aiter.mhc_shuffle_fn(aw[3].contiguous().float()).view(torch.bfloat16))
 saved=[(cache,cache.clone()),(idx,idx.clone())]
 for cc in [MC,IC] if kw['RATIO'] else []:
  if cc:
   saved.extend((x,x.clone()) for x in cc[8:12])
 outputs=[];metrics=[];graphs=[];variant_kwargs=[]
 variants=[(0,0),(128,0) if m<=2 else (192,2)]
 for use in range(len(variants)):
  for x,backup in saved:x.copy_(backup)
  kw['OPT_PRE']=True
  kw['Edges']=s.edges if use else None
  kw['OVERLAP']=variants[use][1]
  kw['EDGE_MASK']=0
  kw['Tiles']=s.tiles if use else None
  kw['TILE_MASK']=variants[use][0]
  obj=batch.warmup(*args,M=m,BLOCK_STRIDE=32,grid=(256,),**kw);obj._init_handles()
  print(dict(compiled=m,ratio=kw['RATIO'],new=use,vgpr=obj.n_regs,spills=obj.n_spills,lds=obj.metadata.shared),flush=True)
  from pathlib import Path
  Path(f'/work/c16-native-projections/{os.getenv("EXPERIMENT","full")}-compiled-{m}-{kw["RATIO"]}-{use}.amdgcn').write_text(obj.asm['amdgcn'])
  assert obj.n_regs*32*obj.metadata.num_warps<=131072
  assert obj.metadata.shared<=327680
  assert obj.n_spills==0
  batch[(256,)](*args,M=m,BLOCK_STRIDE=32,**kw)
  torch.cuda.synchronize()
  print('HEALTH',use,int(s.a[30]),int(s.f[-1]),'edges',[(i,int(e[0]),[(j,int(v)) for j,v in enumerate(e[64:]) if int(v)]) for i,e in enumerate(s.edges[:20])],flush=True)
  assert int(s.a[30])==0 and int(s.f[-1])==0
  outputs.append([s.output.clone(),s.a[10].clone(),s.a[28].clone(),s.a[23].clone(),s.a[38].clone(),s.a[51].clone(),s.f[0].clone(),s.f[9].clone(),s.f[20].clone(),s.f[14].clone(),s.f[18].clone(),s.f[40].clone(),s.f[41].clone()])
  assert all(torch.isfinite(x).all() for x in outputs[-1])
  fn=lambda:batch[(256,)](*args,M=m,BLOCK_STRIDE=32,**kw)
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):
   for _ in range(16):fn()
  graphs.append(graph);variant_kwargs.append(dict(kw));times=[0.,0.]
  print('HEALTH',use,int(s.a[30]),int(s.f[-1]),'edges',[(i,int(e[0]),[(j,int(v)) for j,v in enumerate(e[64:]) if int(v)]) for i,e in enumerate(s.edges[:20])],flush=True)
  assert int(s.a[30])==0 and int(s.f[-1])==0
  metrics.append(dict(mode=use,tile_mask=variants[use][0],overlap=variants[use][1],vgpr=obj.n_regs,spills=obj.n_spills,lds=obj.metadata.shared,rounds_us=times,second_us=times[1]))
  for x,backup in saved:x.copy_(backup)
  fn();torch.cuda.synchronize()
  replay=[s.output,s.a[10],s.a[28],s.a[23],s.a[38],s.a[51],s.f[0],s.f[9],s.f[20],s.f[14],s.f[18],s.f[40],s.f[41]]
  replay_checks=[dict(index=j,exact=torch.equal(x,y),max_abs=float((x.float()-y.float()).abs().max())) for j,(x,y) in enumerate(zip(replay,outputs[-1]))]
  print('REPLAY_CHECKS',use,replay_checks,flush=True)
  assert all(q['exact'] for q in replay_checks), 'replay output mismatch'
  assert int(s.a[30])==0 and int(s.f[-1])==0
 timing_rounds=[]
 for rnd in [1,2]:
  samples={i:[] for i in range(len(graphs))}
  for _ in range(5):
   for graph in graphs:graph.replay()
  for it in range(40):
   for i in (list(range(len(graphs))) if it%2 else list(range(len(graphs)))[::-1]):
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start.record();graphs[i].replay();end.record();end.synchronize()
    samples[i].append(start.elapsed_time(end)*1000/16)
    assert not int(s.a[30]) and not int(s.f[-1]),('paired',rnd,it,i,int(s.a[30]),int(s.f[-1]))
  timing_rounds.append(samples)
  for i in range(len(graphs)):metrics[i]['rounds_us'][rnd-1]=statistics.median(samples[i])
 for i in range(len(graphs)):
  metrics[i]['second_us']=metrics[i]['rounds_us'][1]
  for x,backup in saved:x.copy_(backup)
  batch[(256,)](*args,M=m,BLOCK_STRIDE=32,**variant_kwargs[i]);torch.cuda.synchronize()
  assert torch.equal(s.output,outputs[i][0]),('post-paired-replay',m,i)
  assert not int(s.a[30]) and not int(s.f[-1])
 checks={}
 for mode in range(1,len(variants)):
  checks[mode]={}
  for name,x,y in zip(['final','norm','q','kv','attn','outproj','ffnres','ffnnorm','router','shared_up','shared_down','moe','sum'],outputs[mode],outputs[0]):
   xf=x.float();yf=y.float();checks[mode][name]=dict(exact=bool(torch.equal(x,y)),max_abs=float((xf-yf).abs().max()),relative_rms=float(((xf-yf).square().mean()/yf.square().mean().clamp_min(1.e-20)).sqrt()))
  print('CHECKS',checks,flush=True)
  assert all(v['exact'] for v in checks[mode].values()), checks
 print(dict(m=m,ratio=kw['RATIO'],checks=checks,metrics=metrics),flush=True)
 groups={}
 results.append(dict(m=m,ratio=kw['RATIO'],checks=checks,resources=metrics,groups=groups))
open('/work/c16-native-projections/'+os.getenv('EXPERIMENT','full')+'-final-micro-'+str(m)+'-'+str(kw['RATIO'])+'.json','w').write(json.dumps(results,indent=2))


torch.save([[v.cpu() for v in xs] for xs in outputs],f'/work/c16-native-projections/{os.getenv("EXPERIMENT","full")}-outputs-{m}-{kw["RATIO"]}.pt')
if os.getenv('REFERENCE'):
 ref=torch.load(f'/work/c16-native-projections/{os.environ["REFERENCE"]}-outputs-{m}-{kw["RATIO"]}.pt',weights_only=True)
 errors=[]
 for a,b in zip(outputs[1],ref[1]):
  a=a.cpu().float();b=b.float();errors.append(dict(rrms=float(((a-b).square().mean()/b.square().mean().clamp_min(1e-30)).sqrt()),max_abs=float((a-b).abs().max())))
 print('BEFORE_AFTER_ERRORS',errors,flush=True)
 Path(f'/work/c16-native-projections/{os.getenv("EXPERIMENT","full")}-errors-{m}-{kw["RATIO"]}.json').write_text(json.dumps(errors,indent=2))
