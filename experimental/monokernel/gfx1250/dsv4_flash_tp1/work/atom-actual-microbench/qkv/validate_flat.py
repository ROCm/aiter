from native import *
import statistics,time,collections
sys.path.insert(0,'/work/mhc-optimized')
from qkv_region import QKVRegion
from split_kernels import cache_store
n=NativeQKV();f=n.f
r=QKVRegion(f['W_bytes'].cuda().view(torch.float8_e4m3fn),f['WS'].cuda(),f['NW'].cuda(),f['QB_bytes'].cuda().view(torch.float8_e4m3fn),f['QBS'].cuda())
c=n.cos[1024].flatten().float().contiguous();s=n.sin[1024].flatten().float().contiguous()
def fused():return r(n.x,c,s)
def fused_scatter():
 q,k=fused();cache_store[(1,)](k,n.cache,SLOT=0,num_warps=4,enable_fp_fusion=False);return q,k
def error(a,b):return ((a.float()-b.float()).square().mean()/b.float().square().mean().clamp_min(1e-30)).sqrt().item()
funcs={'native_with_scatter':n,'native_without_scatter':lambda:n(scatter=False),'fused':fused,'fused_plus_scatter':fused_scatter}
for fn in funcs.values():
 for i in range(3):fn()
torch.cuda.synchronize()
q,k=n();q=q.clone();k=k.clone();fq,fk=fused();torch.cuda.synchronize()
validation={'q_relative_rms':error(fq,q),'kv_relative_rms':error(fk,k),'q_max_abs':(fq.float()-q.float()).abs().max().item(),'kv_max_abs':(fk.float()-k.float()).abs().max().item(),'health':r.health.item()}
json.dump(validation,open('validation.json','w'),indent=2);print('validation',validation,flush=True)
sys.path.insert(0,'/work/atom-actual-microbench/attention')
from graphbatch import FlatGraphBatch as GraphBatch
graphs={name:GraphBatch(fn,128) for name,fn in funcs.items()}
checks={}
for name,g in graphs.items():
 expected=funcs[name]();expected=tuple(x.clone() for x in expected)
 g.replay();torch.cuda.synchronize()
 checks[name]={"kernel_count":g.kernel_count,"inner_node_types":g.node_types,"output_equal":[torch.equal(a,b) for a,b in zip(g.out,expected)],"health":r.health.item()}
json.dump(checks,open('fixed-graph-structure.json','w'),indent=2);print(checks,flush=True)
