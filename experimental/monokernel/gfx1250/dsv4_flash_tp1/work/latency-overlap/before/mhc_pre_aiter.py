"""Pre-only specialization of AITER gfx1250 mHC producer/reduce-apply design."""
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from mhc_native_inline import _fn_read_layout, _load_fn_plane
from mhc_native_inline import _mhc_pre_reduce_apply_gfx1250_kernel as reduce_ws
from combined_scaled import sync

@g.jit
def pre_project(R,W,G,SQ,M:gl.constexpr,L:gl.constexpr,FP32=None):
 p=gl.program_id(0)
 WM:gl.constexpr=L[0]; OA:gl.constexpr=L[1]; OB:gl.constexpr=L[2]
 RS:gl.constexpr=L[4]; WS:gl.constexpr=L[5]
 rd=gl.amd.gfx1250.tdm.make_tensor_descriptor(R,[4,M,4096],[4096,16384,1],[4,16,64],RS)
 wd=gl.amd.gfx1250.tdm.make_tensor_descriptor(W,[4,24,8192],[8192,32768,1],[4,32,128],WS)
 rb=gl.allocate_shared_memory(gl.bfloat16,[4,16,64],RS)
 wb=gl.allocate_shared_memory(gl.bfloat16,[1,4,32,128],WS)
 wr=wb.reinterpret(gl.bfloat16,[1,4,32,4,32],_fn_read_layout(4,32,64))
 # Static weights issue before the activation transfer.
 gl.amd.gfx1250.tdm.async_load(wd,[0,0,p*128],wb.index(0))
 gl.amd.gfx1250.tdm.async_load(rd,[0,0,p*64],rb)
 if FP32 is not None:
  fd=gl.amd.gfx1250.tdm.make_tensor_descriptor(FP32,[4,M,4096],[4096,16384,1],[4,16,64],RS)
  fb=gl.allocate_shared_memory(gl.float32,[4,16,64],RS)
  gl.amd.gfx1250.tdm.async_load(fd,[0,0,p*64],fb)
 gl.amd.gfx1250.tdm.async_wait(0)
 gl.barrier()
 a=rb.load(OA)
 hi=_load_fn_plane(wr,0,0,4,32,64,OB);lo=_load_fn_plane(wr,0,1,4,32,64,OB)
 acc=gl.amd.gfx1250.wmma(a,hi,gl.full((4,16,32),0,gl.float32,WM))
 acc=gl.amd.gfx1250.wmma(a,lo,acc)
 if FP32 is not None:f=fb.load(OA)
 else:f=a.to(gl.float32)
 sq=gl.sum(f*f,2)
 red=gl.allocate_shared_memory(gl.float32,[4,16,32],L[7])
 sqb=gl.allocate_shared_memory(gl.float32,[4,16],L[8])
 red.store(acc);sqb.store(sq);gl.barrier()
 RL:gl.constexpr=gl.BlockedLayout([4,1,4],[1,4,8],[1,4,1],[2,1,0])
 out=gl.sum(red.load(RL),0)
 OL:gl.constexpr=gl.SliceLayout(0,RL)
 rows=gl.arange(0,16,layout=gl.SliceLayout(1,OL));cols=gl.arange(0,32,layout=gl.SliceLayout(0,OL))
 gl.store(G+p*M*32+rows[:,None]*32+cols[None,:],out,rows[:,None]<M)
 SL:gl.constexpr=gl.BlockedLayout([4,1],[1,32],[1,4],[1,0])
 sumsq=gl.sum(sqb.load(SL),0);rr=gl.arange(0,16,layout=gl.SliceLayout(0,SL))
 gl.store(SQ+p*M+rr,sumsq,rr<M)

@g.jit
def pre(R,Mix,Y,S,B,NW,Args,H,M:gl.constexpr,L:gl.constexpr,V:gl.constexpr,FP32=None):
 p=gl.program_id(0)
 W,G,SQ,PO,CO,CT=Args
 if p<64:
  pre_project(R,W,G,SQ,M,L,FP32)
  sync(CT,H,64,3)
 if p<M:
  reduce_ws(PO,CO,Y,G,SQ,S,B,R,NW,M,32,1.e-6,1.e-6,1.e-6,1.e-6,2.,4096,4,64,1,4096,V[0],4,20,False,True,V[1],V[2],V[3],V[4],V[5],V[6],RES_KS=64)
  i=gl.arange(0,32,layout=gl.BlockedLayout([1],[32],[4],[0]))
  comb=gl.load(CO+p*16+i-8,(i>=8)&(i<24),0.)
  post=gl.load(PO+p*4+i-4,(i>=4)&(i<8),0.)
  gl.store(Mix+p*24+i,gl.where(i<8,post,comb),(i>=4)&(i<24))
