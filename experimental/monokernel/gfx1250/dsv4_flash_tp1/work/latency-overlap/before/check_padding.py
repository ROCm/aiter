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
ratio=int(os.getenv('RATIO','4'));results=[];t={4:640,128:144}[ratio];d=torch.empty(0,device='cuda');cs=torch.randn((2048,32),device='cuda');sn=torch.randn_like(cs)
for m in [int(x) for x in os.getenv('TEST_BATCHES','2,4,8').split(',')]:
 s=BatchState(m,t);raw=torch.randn((m,4,4096),device='cuda',dtype=torch.bfloat16)
 inp=raw.reshape(m,4,64,64).permute(2,1,0,3).contiguous().reshape(m,4,4096)
 s.a[0]=inp;A,F=bind(s)
 pos=torch.arange(1023,1023+m,device='cuda',dtype=torch.int32);cache=torch.randn((m*32*640,512),device='cuda',dtype=torch.bfloat16)*.1
 valid=m-1;pos[valid:]=0;batch_ids=torch.arange(m,device='cuda',dtype=torch.int32);batch_ids[valid:]=-1
 lengths=128+(pos+1)//ratio;lengths[valid:]=0;ptr=torch.cat([torch.zeros(1,device='cuda',dtype=torch.int32),lengths.cumsum(0).int()]);idx=torch.cat([torch.arange(int(lengths[j]),device='cuda',dtype=torch.int32)+j*32*640 for j in range(m)]);dest=torch.stack([idx[ptr[j+1]-1] if j<valid else torch.zeros((),device='cuda',dtype=torch.int32) for j in range(m)])
 blocks=torch.arange(m*32,device='cuda',dtype=torch.int32).reshape(m,32);top=torch.empty((m,512),device='cuda',dtype=torch.int32)
 def comp(dim,quant):
  df=dim*(2 if ratio==4 else 1);state=ratio*(2 if ratio==4 else 1)
  ape=torch.randn((ratio,df),device='cuda');norm=torch.randn(dim,device='cuda');cp=torch.zeros((m,4),device='cuda',dtype=torch.int32);wp=cp.clone();cp[:,2]=torch.where((pos+1)%ratio==0,pos,-1);cp[:,3]=state-1;wp[:,2]=pos
  slots=torch.arange(m,device='cuda',dtype=torch.int32);slots[valid:]=0
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
 obj=batch.warmup(*args,M=m,BLOCK_STRIDE=32,FP32MID=s.mid32,FP32OUT=s.out32,grid=(256,),BatchIds=batch_ids,STOP=int(os.getenv("STOP","0")),PREFETCH=os.getenv("PREFETCH","1")=="1",**kw)
 obj._init_handles()
 from pathlib import Path
 Path('/work/integrated-c8/compiled-'+str(kw['RATIO'])+'-'+str(m)+'.amdgcn').write_text(obj.asm['amdgcn'])
 print(dict(compiled=m,vgpr=obj.n_regs,spills=obj.n_spills,lds=obj.metadata.shared),flush=True)
 if obj.n_spills:raise RuntimeError('Refusing to launch spilling persistent full layer')
 batch[(256,)](*args,M=m,BLOCK_STRIDE=32,FP32MID=s.mid32,FP32OUT=s.out32,BatchIds=batch_ids,STOP=int(os.getenv("STOP","0")),PREFETCH=os.getenv("PREFETCH","1")=="1",**kw)
 torch.cuda.synchronize();print("batch launch completed",flush=True)
 if os.getenv("STOP","0")!="0":continue
 refs=[]
 for row in range(valid):
  r=BatchState(1,t);r.a[0]=raw[row].reshape(4,64,64).permute(1,0,2).contiguous();AR,FR=bind(r)
  # C1 final post writes the native shuffled layout into F[1].
  single[(256,)](AR,FR,rowcomp(MREF,row),rowcomp(IREF,row) if ratio==4 else (),MW,IW,torch.empty_like(MY[:1]),torch.empty_like(IY[:1]),pos[row:row+1],cs,sn,cache,idx,ptr[row:row+2],dest[row:row+1],top[row],blocks[row],d,FP32MID=r.mid32,FP32OUT=r.out32,**{k:v for k,v in kw.items() if k not in ['CacheRows','Work','CompCounter','MHC','ML','MV','OPT_MHC','OPT_COMP']});refs.append(r)
 torch.cuda.synchronize();out=s.output.reshape(64,4,m,64).permute(2,1,0,3).reshape(m,4,4096)
 ref=torch.cat([r.f[1].reshape(64,4,1,64).permute(2,1,0,3).reshape(1,4,4096) for r in refs])
 checks={}
 for name,k in [('norm',10),('q',28),('kv',23),('attention',38),('output',51),('mhc_post',2)]:
  got=s.a[k].reshape(m,-1)[:valid];want=torch.cat([r.a[k].reshape(1,-1) for r in refs]);checks[name]=dict(exact=bool(torch.equal(got,want)),rms=float(((got.float()-want.float()).square().mean()/want.float().square().mean().clamp_min(1.e-20)).sqrt()),max_abs=float((got.float()-want.float()).abs().max()))
 checks['compressor_state']=all(torch.equal(MC[k].view(torch.uint8),MREF[k].view(torch.uint8)) for k in range(8,12))
 if ratio==4:checks['indexer_state']=all(torch.equal(IC[k].view(torch.uint8),IREF[k].view(torch.uint8)) for k in range(8,12))
 checks['final']=dict(exact=torch.equal(out[:valid],ref),rms=float(((out[:valid].float()-ref.float()).square().mean()/ref.float().square().mean().clamp_min(1.e-20)).sqrt()));checks['finite']=bool(torch.isfinite(out).all())
 result=dict(m=m,checks=checks,health=[int(s.a[30]),int(s.f[-1])],vgpr=obj.n_regs,lds=obj.metadata.shared,spills=obj.n_spills)
 print(result,flush=True);results.append(result);assert checks['finite'] and result['health']==[0,0]
open(f'/work/integrated-c8/padding-{ratio}-checks.json','w').write(json.dumps(results,indent=2))
