from harness import *
from variants import run,State,CONFIGS
out=root/'scratch-stress';out.mkdir(exist_ok=True)
xs=[];cs=[];ss=[];refs=[]
for i in range(32):
 gen=torch.Generator(device='cuda').manual_seed(8123+i)
 x=(torch.randn(4096,device='cuda',generator=gen)*2.**(i%17-8)).bfloat16()
 if i==0:
  gen=torch.Generator(device='cuda').manual_seed(32151)
  x=(torch.randn(4096,device='cuda',generator=gen)*2.**(151%17-8)).bfloat16()
 a=(i*1231)*10000.**(-torch.arange(32,device='cuda',dtype=torch.float32)/32)
 xs.append(x);cs.append(a.cos());ss.append(a.sin());Cos.copy_(cs[-1]);Sin.copy_(ss[-1]);refs.append(reference(x))
rq=torch.stack([r['q'] for r in refs]).float();rk=torch.stack([r['kv'] for r in refs]).float()
results=[]
for mode,bn in [(n,0) for n in CONFIGS]:
 s=State();run(s,mode);torch.cuda.synchronize()
 oq=torch.empty((32,32768),device='cuda',dtype=torch.bfloat16);ok=torch.empty((32,512),device='cuda',dtype=torch.bfloat16)
 cg=torch.cuda.CUDAGraph()
 with torch.cuda.graph(cg):
  for i in range(32):
   X.copy_(xs[i]);Cos.copy_(cs[i]);Sin.copy_(ss[i]);run(s,mode);oq[i].copy_(s.q);ok[i].copy_(s.kv)
 maxq=maxk=0.;fails=0
 for replay in range(320):
  cg.replay()
  eq=((oq.float()-rq).square().mean(-1)/rq.square().mean(-1).clamp_min(1e-30)).sqrt()
  ek=((ok.float()-rk).square().mean(-1)/rk.square().mean(-1).clamp_min(1e-30)).sqrt()
  maxq=max(maxq,eq.max().item());maxk=max(maxk,ek.max().item());fails+=int(((eq>=.001)|(ek>=.001)|~torch.isfinite(eq)|~torch.isfinite(ek)).sum().item())
  assert all(v==0 if k=='health' else v for k,v in s.check_state().items())
 row=dict(mode=mode,checked_invocations=320*32,failures=fails,max_q_relative_rms=maxq,max_kv_relative_rms=maxk,state=s.check_state());results.append(row);print(row,flush=True)
 (out/'results.json').write_text(json.dumps(results,indent=2))
