from common import *
from fused import project_fused
from initall import initialize_all
import time
out=root/'final-validation';out.mkdir(exist_ok=True)
Count=torch.empty(1536,device='cuda',dtype=torch.int32)
P=torch.empty((32,1536),device='cuda',dtype=torch.float32)
# Nine original inputs plus zero, constant, and 256 reproducible random/perturbed inputs.
extra=[torch.zeros(4096,device='cuda'),torch.ones(4096,device='cuda'),-torch.ones(4096,device='cuda')]
for seed in range(256):
 gen=torch.Generator(device='cuda').manual_seed(10000+seed)
 x=torch.randn(4096,device='cuda',generator=gen)
 if seed%4==0:x=live.float()+x*.01
 else:x=x*(2.**((seed%17)-8))
 extra.append(x.bfloat16())
for x in extra:
 z=x.float().reshape(-1,128);amax=z.abs().amax(-1).clamp_min(1.e-10)/448.
 e=(((amax.view(torch.int32)&0x7fffffff)+0x7fffff)>>23).clamp_max(254)
 scale=(e<<23).view(torch.float32)
 bank.append(((z/scale[:,None]).to(torch.float8_e4m3fn).flatten(),e.to(torch.uint8)))
xf=torch.stack([x.double()*torch.pow(2.,s.double()-127).repeat_interleave(128) for x,s in bank])
z=(xf@wf.t()).bfloat16().float()
refs=torch.cat([z[:,:1024]*torch.rsqrt(z[:,:1024].square().mean(-1,keepdim=True)+1.e-6)*NW[:1024],z[:,1024:]*torch.rsqrt(z[:,1024:].square().mean(-1,keepdim=True)+1.e-6)*NW[1024:]],dim=-1)
def candidate(native,unroll):
 return project_fused[(192,)](X,XS,W,WS,P,NW,Acc,Count,Y,BN=128,SPLIT=16,DIRECT=True,WAVES=4,BK=128,UNROLL=unroll,UPSCALE=True,REPLICATE=True,CACHE='.ca',SCALAR=True,PACKED=False,MODE=1,OUTLAY=1,NVEC=1,COUNTSTRIDE=128,NATIVE=native,**kw)
def baseline():
 o=project[(384,)](X,XS,W,WS,NW,P,Acc,SS,Count,Y,MODE=0,**kw);reduce_norm[(2,)](P,Acc,NW,Y,ATOMIC=False,**kw);return o
from accepted import QAKVNorm
s16=QAKVNorm(split=16);s32=QAKVNorm(split=32)
runs=[('baseline_A',baseline),('accepted_split16',lambda:s16(X,XS,W,WS,NW,out=Y)),('accepted_split32',lambda:s32(X,XS,W,WS,NW,out=Y))]
report=[]
for name,run in runs:
 initialize_all[(6,)](Acc,Count,**kw);P.fill_(float('nan'));SS.fill_(float('nan'));Y.fill_(float('nan'))
 run();torch.cuda.synchronize()
 cg=torch.cuda.CUDAGraph()
 with torch.cuda.graph(cg):run()
 outputs=torch.empty((len(bank)*8,1536),device='cuda')
 # No reset between replays; check every output, using eight passes over changed inputs.
 for i in range(len(outputs)):
  xp,xs=bank[(i*37)%len(bank)];X.copy_(xp);XS.copy_(xs);cg.replay();outputs[i].copy_(Y)
 torch.cuda.synchronize()
 ref=refs[torch.tensor([(i*37)%len(bank) for i in range(len(outputs))],device='cuda')]
 d=outputs-ref;errs=(d.square().mean(-1)/ref.square().mean(-1).clamp_min(1e-30)).sqrt()
 q=(d[:,:1024].square().mean(-1)/ref[:,:1024].square().mean(-1).clamp_min(1e-30)).sqrt()
 k=(d[:,1024:].square().mean(-1)/ref[:,1024:].square().mean(-1).clamp_min(1e-30)).sqrt()
 row=dict(name=name,cases=len(bank),replays=len(outputs),checked_outputs=len(outputs),finite=torch.isfinite(outputs).all().item(),max_relative_rms=errs.max().item(),max_q_relative_rms=q.max().item(),max_kv_relative_rms=k.max().item(),max_abs=d.abs().max().item(),failing_outputs=(errs>=.001).sum().item(),counter_zero=torch.count_nonzero(Count).item()==0,acc_zero=torch.count_nonzero(Acc).item()==0,per_output_relative_rms=errs.tolist())
 state=s16 if name=='accepted_split16' else s32 if name=='accepted_split32' else None
 if state is not None:
  row['counter_zero']=torch.count_nonzero(state.count).item()==0;row['acc_zero']=torch.count_nonzero(state.acc).item()==0
 row['passed']=row['finite'] and row['failing_outputs']==0 and row['max_q_relative_rms']<.001 and row['max_kv_relative_rms']<.001
 report.append(row);print({k:v for k,v in row.items() if k!='per_output_relative_rms'},flush=True)
 (out/'results.json').write_text(json.dumps(report,indent=2))
print('DONE',flush=True)
