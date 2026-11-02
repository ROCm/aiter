from common import *
sys.path.insert(0,'/work/ffn-c1/shared-router')
from native import NativeSharedRouter
from shared_defs import scratch
from shared_kernel import region
n=NativeSharedRouter();torch.manual_seed(42);x=torch.randn((8,4096),device='cuda',dtype=torch.bfloat16)*.05;a=scratch(8)
def fused():
 region[(256,)](x,a[0],a[1],n.learned_gate.weight,a[4],n.shared.gate_up_proj.weight,n.shared.gate_up_proj.weight_scale.view(torch.uint8),a[5],a[6],a[7],a[8],n.shared.w2.weight,n.shared.w2.weight_scale.view(torch.uint8),a[9],a[-2],a[-1],8,**kw)
 return a[9]
def native():
 n.learned_gate(x)
 return n.shared(x)
rr=native();vv=fused();torch.cuda.synchronize();bench('shared_router',{'native':native,'fused':fused},err(vv,rr))
