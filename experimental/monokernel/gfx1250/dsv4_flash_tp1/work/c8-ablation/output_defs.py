import sys,torch,json
sys.path[:0]=['/work/batch-full','/work/modelwide','/work/full-attention-native/prefix','/work/full-attention-native/output-kernel']
from output_batch import output_body
from output_scaled import output_kernel
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
@g.jit
def batched(O,C,S,WA,SA,WB,SB,XQ,XS,AP,AY,AQ,AS,BP,Y,Count,Health,M:gl.constexpr):
 output_body(O,C,S,WA,SA,WB,SB,XQ,XS,AP,AY,AQ,AS,BP,Y,Count,Health,M)
torch.manual_seed(13);d='cuda'
wa=torch.randn((8192,4096),device=d).to(torch.float8_e4m3fn);wb=torch.randn((4096,8192),device=d).to(torch.float8_e4m3fn)
sa=torch.full((64,32),122,device=d,dtype=torch.uint8);sb=torch.full((32,64),122,device=d,dtype=torch.uint8)
def scratch(m):
 def e(shape,dt=torch.bfloat16):return torch.empty(shape,device=d,dtype=dt)
 return [e((m,32768),torch.float8_e4m3fn),e((m,256),torch.uint8),e((4,m,8192),torch.float32),e((m,8192)),e((m,8192),torch.float8_e4m3fn),e((m,64),torch.uint8),e((8,m,4096),torch.float32),e((m,4096)),torch.zeros(1,device=d,dtype=torch.int32),torch.zeros(1,device=d,dtype=torch.int32)]
