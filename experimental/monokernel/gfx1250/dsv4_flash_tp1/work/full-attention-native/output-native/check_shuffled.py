from native_output import *
n=NativeOutput();torch.manual_seed(91);x=torch.randn(1,4096,device='cuda',dtype=torch.bfloat16);r=torch.randn(1,4,4096,device='cuda',dtype=torch.bfloat16);p=torch.rand(1,4,device='cuda');c=torch.rand(1,4,4,device='cuda');rs=aiter.mhc_res_shuffle(r)
a=n.post(x,r,p,c);b=n.post(x,rs,p,c,True);torch.cuda.synchronize()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:b=n.post(x,rs,p,c,True);torch.cuda.synchronize()
prof.export_chrome_trace('post-shuffled-trace.json');es=json.load(open('post-shuffled-trace.json'))['traceEvents'];d={'equal_plain_post':torch.equal(a,b),'output_layout':'plain [1,4,4096]','kernels':[e['name'] for e in es if e.get('cat')=='kernel']};json.dump(d,open('post-shuffled-checks.json','w'),indent=2);print(d,flush=True)
