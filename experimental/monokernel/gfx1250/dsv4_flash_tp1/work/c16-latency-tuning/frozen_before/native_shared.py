from frozen_before.common import *
sys.path.insert(0,'/work/ffn-c1/shared-router')
from native import NativeSharedRouter
from frozen_before.isolated import shared
n=NativeSharedRouter();torch.manual_seed(42);x=torch.randn((M,4096),device='cuda',dtype=torch.bfloat16)*.05
e=lambda sh,dt=torch.bfloat16:torch.empty(sh,device='cuda',dtype=dt)
a=[e((M,4096),torch.float8_e4m3fn),e((32,max(8,M)),torch.uint8),e((M,256)),e((M,4096)),e((M,2048)),e((M,2048),torch.float8_e4m3fn),e((16,max(8,M)),torch.uint8),e((M,4096)),torch.zeros(1,device='cuda',dtype=torch.int32),torch.zeros(1,device='cuda',dtype=torch.int32)]
def fused():
 shared[(256,)](x,a[0],a[1],n.learned_gate.weight,a[2],n.shared.gate_up_proj.weight,n.shared.gate_up_proj.weight_scale.view(torch.uint8),a[3],a[4],a[5],a[6],n.shared.w2.weight,n.shared.w2.weight_scale.view(torch.uint8),a[7],a[8],a[9],M,**kw)
 return a[7]
def native():
 n.learned_gate(x)
 return n.shared(x)
rr=native();fused();torch.cuda.synchronize();assert int(a[-1])==0
bench('shared_router',{'native':native,'fused':fused},err(a[7],rr))
