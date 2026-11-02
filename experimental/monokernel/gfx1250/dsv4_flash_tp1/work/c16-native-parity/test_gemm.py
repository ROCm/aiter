from common import *
from packing import pack_weight
from native_tiles import gemm
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
@g.jit
def run(X,XS,W,WS,Y,M:gl.constexpr,N:gl.constexpr,K:gl.constexpr,BM:gl.constexpr,BN:gl.constexpr,BK:gl.constexpr,NB:gl.constexpr):
 gemm(X,XS,W,WS,Y,M,N,K,BM,BN,BK,NB,gl.program_id(0))
torch.manual_seed(774)
case=sys.argv[1] if len(sys.argv)>1 else 'qa'
n,k,bm,bn,bk,nb={'qa':(1536,4096,16,32,1024,4),'qb':(32768,1024,16,128,512,2),'oa':(1024,4096,32,32,1024,4),'ob':(4096,8192,16,64,512,4)}[case]
x=(torch.randn((16,k),device='cuda')*.1).to(torch.float8_e4m3fn);w=(torch.randn((n,k),device='cuda')*.1).to(x.dtype)
xs=torch.full((16,k//128),127,device='cuda',dtype=torch.uint8);ws=torch.full((n//128,k//128),127,device='cuda',dtype=torch.uint8);wp=pack_weight(w);y=torch.empty((16,n),device='cuda',dtype=torch.bfloat16)
ref=(x.float()@w.float().T).bfloat16()
fn=lambda:run[(n//bn,)](x,xs,wp,ws,y,16,n,k,bm,bn,bk,nb,**kw)
z=fn();torch.cuda.synchronize();print('RESOURCE',z.n_regs,z.n_spills,z.metadata.shared,flush=True)
check=err(y,ref);print('CHECK',check,flush=True);assert check['finite'] and check['relative_rms']<.001
bench('gemm_'+case,{'gluon':fn},check)
