"""One-launch batched input quant -> QA/KV -> norms -> QB -> Q norm/RoPE.
Experimental; validate before integrating. Native MoE remains separate.
"""
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from qa_batch import qa_batch
from combined_scaled import sync

@g.jit
def qkv_batch(X,XQ,XS,W,WS,P,NW,Raw,QR,QS,Cos,Sin,KV,Counter,Health,
              QBW,QBS,QBRaw,QOut,M:gl.constexpr,EARLY:gl.constexpr=True,Cache=None,Dest=None,BatchId=None):
 pid=gl.program_id(0)
 SW:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,16]],[64,128],[1,0])
 wb=gl.allocate_shared_memory(gl.float8e4nv,[8,64,128],SW)
 wd=gl.amd.gfx1250.tdm.make_tensor_descriptor(QBW+pid*128*1024,[64,1024],[1024,1],[64,128],SW)
 if EARLY:
  for j in gl.static_range(8):gl.amd.gfx1250.tdm.async_load(wd,[0,j*128],wb.index(j))
 qa_batch(X,XQ,XS,W,WS,P,NW,Raw,QR,QS,Cos,Sin,KV,Counter,Health,M,QBW)
 sync(Counter,Health,256,3)
 SA:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,16]],[16,128],[1,0])
 ab=gl.allocate_shared_memory(gl.float8e4nv,[8,16,128],SA)
 ad=gl.amd.gfx1250.tdm.make_tensor_descriptor(QR,[M,1024],[1024,1],[16,128],SA)
 for j in gl.static_range(8):gl.amd.gfx1250.tdm.async_load(ad,[0,j*128],ab.index(j))
 ML:gl.constexpr=gl.amd.AMDWMMALayout(version=3,transposed=True,warp_bases=[[0,1],[0,2]],instr_shape=[16,16,128])
 DA:gl.constexpr=gl.DotOperandLayout(0,ML,16);DB:gl.constexpr=gl.DotOperandLayout(1,ML,16)
 AL:gl.constexpr=gl.to_linear_layout(DA,[16,128]);BL:gl.constexpr=gl.to_linear_layout(DB,[128,64])
 ASL:gl.constexpr=gl.amd.gfx1250.get_wmma_scale_layout(DA,[16,4],scale_factor=32)
 BSL:gl.constexpr=gl.amd.gfx1250.get_wmma_scale_layout(DB,[64,4],scale_factor=32)
 sr=gl.arange(0,16,layout=gl.SliceLayout(1,ASL));sk=gl.arange(0,4,layout=gl.SliceLayout(0,ASL))
 OL:gl.constexpr=gl.to_linear_layout(ML,[16,64])
 r=gl.arange(0,16,layout=gl.SliceLayout(1,OL));c=gl.arange(0,64,layout=gl.SliceLayout(0,OL))
 for half in gl.static_range(2):
  if half==1 or not EARLY:
   hd=gl.amd.gfx1250.tdm.make_tensor_descriptor(QBW+(pid*128+half*64)*1024,[64,1024],[1024,1],[64,128],SW)
   for j in gl.static_range(8):gl.amd.gfx1250.tdm.async_load(hd,[0,j*128],wb.index(j))
  gl.amd.gfx1250.tdm.async_wait(0)
  acc=gl.full((16,64),0,gl.float32,ML)
  for j in gl.static_range(8):
   aa=gl.convert_layout(ab.index(j).load(AL),DA);bb=gl.convert_layout(wb.index(j).permute([1,0]).load(BL),DB)
   asc=gl.load(QS+sr[:,None]*8+j+sk[None,:]*0,sr[:,None]<M,0)
   bsc=gl.load(QBS+pid*8+j)+gl.full((64,4),0,gl.uint8,BSL)
   acc=gl.amd.gfx1250.wmma_scaled(aa,asc,'e4m3',bb,bsc,'e4m3',acc)
  gl.store(QBRaw+r[:,None]*32768+pid*128+half*64+c[None,:],gl.convert_layout(acc,OL),r[:,None]<M)
  gl.barrier()
 sync(Counter,Health,256,3)
 L:gl.constexpr=gl.BlockedLayout([1],[32],[4],[0]);i=gl.arange(0,512,layout=L)
 for rr in gl.static_range(gl.cdiv(M,4)):
  row=pid//64+rr*4;head=pid%64
  if row<M:
   v=gl.load(QBRaw+row*32768+head*512+i).to(gl.float32)
   y=v*gl.rsqrt(gl.sum(v*v,0)/512+1.e-6)
   other=gl.gather(y,i^1,0)
   cv=gl.load(Cos+row*32+(i-448)//2,i>=448,1.);sv=gl.load(Sin+row*32+(i-448)//2,i>=448,0.)
   gl.store(QOut+row*32768+head*512+i,y*cv+gl.where(i%2==0,-1.,1.)*other*sv)
 if Cache is not None:
  if pid<M:
   bid=gl.load(BatchId+pid)
   dest=gl.load(Dest+pid,bid>=0,-1).to(gl.int64)
   SK:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,8]],[1,512],[1,0])
   kb=gl.allocate_shared_memory(gl.bfloat16,[1,512],SK)
   kd=gl.amd.gfx1250.tdm.make_tensor_descriptor(KV+pid*512,[1,512],[512,1],[1,512],SK)
   gl.amd.gfx1250.tdm.async_load(kd,[0,0],kb);gl.amd.gfx1250.tdm.async_wait(0)
   KL:gl.constexpr=gl.BlockedLayout([1,1],[1,32],[1,4],[1,0])
   ki=gl.arange(0,512,layout=gl.SliceLayout(0,KL))
   gl.store(Cache+dest*512+ki[None,:],kb.load(KL),dest>=0)
