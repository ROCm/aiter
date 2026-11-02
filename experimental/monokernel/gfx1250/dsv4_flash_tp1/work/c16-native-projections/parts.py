from common import *
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from native_tiles import gemm
@g.jit
def entry(X,XS,W,WS,Y,M:gl.constexpr,N:gl.constexpr,K:gl.constexpr,BM:gl.constexpr,BN:gl.constexpr,BK:gl.constexpr,NB:gl.constexpr,GROUP:gl.constexpr=False):
 gemm(X,XS,W,WS,Y,M,N,K,BM,BN,BK,NB,gl.program_id(0),GROUPED=GROUP)
kind=sys.argv[1]
if kind=='q':
 exec(Path('native_qkv_cache.py').read_text().split('rr=native();')[0])
 native();fused();torch.cuda.synchronize()
 for name,lin,xx,ss,ww,sw,nn,kk,bn,bk,nb in [('qa',n.qa,xq,xs,wp,ws,1536,4096,32,1024,4),('qb',n.qb,q,s,qbp,qbs,32768,1024,128,512,3)]:
  yy=torch.empty((16,nn),device='cuda',dtype=torch.bfloat16)
  fn=lambda:entry[(nn//bn,)](xx,ss,ww,sw,yy,16,nn,kk,16,bn,bk,nb,**kw)
  ns=ss.T.contiguous().T.view(torch.float8_e8m0fnu)
  rf=lambda:lin(xx,x_scale=ns)
  rr=rf();fn();torch.cuda.synchronize()
  bench('part_'+name,{'native':rf,'gluon':fn},err(yy,rr))
else:
 exec(Path('native_output.py').read_text().split('rr=n(o);')[0])
 fo();torch.cuda.synchronize()
 from atom.models.deepseek_v4 import batched_gemm_a8w8_mxscale_bpreshuffle as bmm
 rf=lambda:bmm(a[0].reshape(16,8,4096),n._wo_a_w_fp8,a[1].reshape(16,8,32),n._wo_a_w_scale,dtype=torch.bfloat16)
 fn=lambda:entry[(256,)](a[0],a[1],wap,sa,a[3],16,8192,4096,32,32,1024,4,True,**kw)
 rr=rf();fn();torch.cuda.synchronize();bench('part_oa',{'native':rf,'gluon':fn},err(a[3],rr))
 ns=a[5].T.contiguous().T.view(torch.float8_e8m0fnu)
 rf=lambda:n.wo_b(a[4],x_scale=ns)
 fn=lambda:entry[(64,)](a[4],a[5],wbp,sb,a[7],16,4096,8192,16,64,512,4,**kw)
 rr=rf();fn();torch.cuda.synchronize();bench('part_ob',{'native':rf,'gluon':fn},err(a[7],rr))
