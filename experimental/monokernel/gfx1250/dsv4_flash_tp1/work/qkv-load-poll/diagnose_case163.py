from harness import *
from variants import CONFIGS,run,State
out=root/'case163';out.mkdir(exist_ok=True)
seed=151
gen=torch.Generator(device='cuda').manual_seed(32000+seed)
x=torch.randn(4096,device='cuda',generator=gen)*2.**(seed%17-8)
X.copy_(x.bfloat16());r=reference(X)
rows=[]
for name in CONFIGS:
 s=State();run(s,name);torch.cuda.synchronize();cg=torch.cuda.CUDAGraph()
 with torch.cuda.graph(cg):run(s,name)
 failures=[];maxerr=0.
 for i in range(500):
  cg.replay();err=metric(s.q,r['q'])['relative_rms'];maxerr=max(maxerr,err)
  if err>=.001:
   lp,ls,lf=tq(s.y[:1024])
   qp=(bwf@lf).bfloat16().float();qn=qp.reshape(64,512);qn=qn*torch.rsqrt(qn.square().mean(-1,keepdim=True)+1e-6)
   actualref=rotary(qn)
   failures.append(dict(replay=i,error=err,lp_matches_current_y=torch.equal(lp.view(torch.uint8),s.lp.view(torch.uint8)),ls_matches_current_y=torch.equal(ls,s.ls),lp_changed_vs_fp64=int((s.lp.view(torch.uint8)!=r['lp'].view(torch.uint8)).sum().item()),error_vs_current_y_reference=metric(s.q,actualref)['relative_rms'],state=s.check_state()))
 row=dict(name=name,max_error=maxerr,failures=failures);rows.append(row)
 print(name,'failures',len(failures),'max',maxerr, 'first', failures[:1],flush=True)
 (out/'results.json').write_text(json.dumps(rows,indent=2))
