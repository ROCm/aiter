exec(open('/work/ffn-c1/mhc/bench.py').read().split('rows=[]')[0])
hybrid();torch.cuda.synchronize()
from aiter.ops.triton._gluon_kernels.gfx1250.fusions import mhc
out=[]
for name in ['_mhc_post_pre_gemm_sqrsum_gfx1250_kernel','_mhc_pre_reduce_apply_gfx1250_kernel']:
 k=getattr(mhc,name)
 for dev,entry in k.device_caches.items():
  cache=entry[0]
  for key,obj in cache.items():
   print(name,type(obj),flush=True)
   if hasattr(obj,'metadata'):
    label=obj.metadata.name
    row={'name':label,'vgpr':obj.n_regs,'shared':obj.metadata.shared,'scratch':obj.metadata.global_scratch_size};out.append(row)
    (root/(label+'.amdgcn')).write_text(obj.asm['amdgcn'])
(root/'resources.json').write_text(json.dumps(out,indent=2));print(out,flush=True)
