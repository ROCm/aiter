from full_setup import *
sys.path.insert(0,'/work/atom-actual-microbench/attention')
from graphbatch import FlatGraphBatch
f=FullState();rsh=aiter.mhc_res_shuffle(ps.R.view(1,4,4096));rows=[]
with torch.inference_mode():
 f.fused(rsh);torch.cuda.synchronize();g=FlatGraphBatch(lambda:f.fused(rsh),128)
 for i in range(32):
  r=torch.randn((1,4,4096),device='cuda',generator=torch.Generator(device='cuda').manual_seed(921000+i)).bfloat16()
  rsh.copy_(aiter.mhc_res_shuffle(r));expected=f.staged(rsh).clone();torch.cuda.synchronize()
  f.rout.fill_(float('nan'));f.oy.fill_(float('nan'));g.replay();torch.cuda.synchronize()
  rows.append({'case':i,'exact_staged':torch.equal(f.rout,expected),'finite':bool(torch.isfinite(f.rout).all()),'health':f.check()})
  assert all(rows[-1][k] for k in ['exact_staged','finite','health']),rows[-1]
 (root/'stress.json').write_text(json.dumps({'invocations':4096,'changing_inputs':32,'graph_kernel_nodes':g.kernel_count,'rows':rows},indent=2))
print('4096 full-fused changing-input invocations passed',flush=True)
