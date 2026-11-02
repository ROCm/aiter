from triton.experimental.gluon import language as gl
import os
OUTPUT_COEFF = gl.constexpr(int(os.getenv("OUTPUT_COEFF","0")))
OUTPUT_EARLIER = gl.constexpr(int(os.getenv("OUTPUT_EARLIER","0")))
OUTPUT_PACKED = gl.constexpr(int(os.getenv("OUTPUT_PACKED","0")))
OUTPUT_READY = gl.constexpr(int(os.getenv("OUTPUT_READY","0")))
OUTPUT_TUNE = gl.constexpr(int(os.getenv('OUTPUT_TUNE','0')))
OA_BN = gl.constexpr(int(os.getenv('OA_BN','32')))
OA_BK = gl.constexpr(int(os.getenv('OA_BK','1024')))
OA_NB = gl.constexpr(int(os.getenv('OA_NB','4')))
OA_SPLIT = gl.constexpr(int(os.getenv('OA_SPLIT','1')))
OB_BN = gl.constexpr(int(os.getenv('OB_BN','64')))
OB_BK = gl.constexpr(int(os.getenv('OB_BK','512')))
OB_NB = gl.constexpr(int(os.getenv('OB_NB','4')))
OB_SPLIT = gl.constexpr(int(os.getenv('OB_SPLIT','4')))
from edge_sync import edge,publish,consume,prefetch_bytes
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from device_scope_helpers import dev_load,dev_store
from combined_scaled import sync
from tail import quant_scale
@g.jit
def split_project_old(X,XS,W,WS,P,N:gl.constexpr,K:gl.constexpr,SPLIT:gl.constexpr,GROUPED:gl.constexpr,M:gl.constexpr,PreW=None):
    gl.amd.gfx1250.tdm.async_wait(0)
    gl.barrier()
    pid=gl.program_id(0);nt=pid//SPLIT;ks=pid%SPLIT
    ML:gl.constexpr=gl.amd.AMDWMMALayout(version=3,transposed=True,warp_bases=[[0,1],[0,2]],instr_shape=[16,16,128])
    DA:gl.constexpr=gl.DotOperandLayout(0,ML,16);DB:gl.constexpr=gl.DotOperandLayout(1,ML,16)
    ASL:gl.constexpr=gl.amd.gfx1250.get_wmma_scale_layout(DA,[16,4],scale_factor=32)
    BSL:gl.constexpr=gl.amd.gfx1250.get_wmma_scale_layout(DB,[128,4],scale_factor=32)
    AL:gl.constexpr=gl.to_linear_layout(DA,[16,128]);BL:gl.constexpr=gl.to_linear_layout(DB,[128,128])
    SL:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,16]],[128,128],[1,0])
    XL:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,16]],[16,128],[1,0])
    if PreW is None:sh=gl.allocate_shared_memory(W.dtype.element_ty,[2,128,128],SL)
    else:sh=PreW
    xsh=gl.allocate_shared_memory(X.dtype.element_ty,[2,16,128],XL)
    wd=gl.amd.gfx1250.tdm.make_tensor_descriptor(base=W+nt*128*K,shape=[128,K],strides=[K,1],block_shape=[128,128],layout=SL)
    xb=nt//8*K if GROUPED else 0
    xd=gl.amd.gfx1250.tdm.make_tensor_descriptor(base=X+xb,shape=[M,K],strides=[32768 if GROUPED else K,1],block_shape=[16,128],layout=XL)
    rm=gl.arange(0,16,layout=gl.SliceLayout(1,AL));ka=gl.arange(0,128,layout=gl.SliceLayout(0,AL))
    kb=gl.arange(0,128,layout=gl.SliceLayout(1,BL));rn=gl.arange(0,128,layout=gl.SliceLayout(0,BL))
    blocks:gl.constexpr=K//SPLIT//128
    first=ks*blocks
    if PreW is None:
        gl.amd.gfx1250.tdm.async_load(wd,[0,first*128],sh.index(0))
    gl.amd.gfx1250.tdm.async_load(xd,[0,first*128],xsh.index(0))
    if PreW is None:
        gl.amd.gfx1250.tdm.async_load(wd,[0,(first+1)*128],sh.index(1))
    gl.amd.gfx1250.tdm.async_load(xd,[0,(first+1)*128],xsh.index(1))
    acc=gl.full((16,128),0,gl.float32,ML)
    for b in range(blocks):
        if PreW is not None and b==0:gl.amd.gfx1250.tdm.async_wait(0)
        elif b<blocks-1:gl.amd.gfx1250.tdm.async_wait(2)
        else:gl.amd.gfx1250.tdm.async_wait(0)
        bv=gl.amd.cdna4.async_copy.load_shared_relaxed(sh.index(b%2).permute([1,0]),BL)
        av=gl.amd.cdna4.async_copy.load_shared_relaxed(xsh.index(b%2),AL)
        gl.barrier()
        if b<blocks-2:
            gl.amd.gfx1250.tdm.async_load(wd,[0,(first+b+2)*128],sh.index(b%2));gl.amd.gfx1250.tdm.async_load(xd,[0,(first+b+2)*128],xsh.index(b%2))
        sr=gl.arange(0,16,layout=gl.SliceLayout(1,ASL))
        sk=gl.arange(0,4,layout=gl.SliceLayout(0,ASL))
        ax=gl.load(XS+sr[:,None]*(256 if GROUPED else K//128)+xb//128+first+b+sk[None,:]*0,sr[:,None]<M,0).to(gl.uint32)
        bx=gl.load(WS+nt*(K//128)+first+b).to(gl.uint32)
        asc=ax.to(gl.uint8)+gl.full((16,4),0,gl.uint8,ASL)
        bsc=bx.to(gl.uint8)+gl.full((128,4),0,gl.uint8,BSL)
        aa=gl.convert_layout(av.to(gl.float8e4nv),DA);bb=gl.convert_layout(bv,DB)
        acc=gl.amd.gfx1250.wmma_scaled(aa,asc,'e4m3',bb,bsc,'e4m3',acc)
    gl.amd.gfx1250.tdm.async_wait(0)
    gl.barrier()
    OL:gl.constexpr=gl.to_linear_layout(ML,[16,128]);out=gl.convert_layout(acc,OL)
    om=gl.arange(0,16,layout=gl.SliceLayout(1,OL));on=nt*128+gl.arange(0,128,layout=gl.SliceLayout(0,OL))
    dev_store(out,P,ks*M*N+on[None,:]+om[:,None]*N,om[:,None]<M)

@g.jit
def inverse_quant(O,Cos,Sin,Q,S,M:gl.constexpr,Tiles=None,Ready=None,Health=None,Completion=None):
 p=gl.program_id(0);L:gl.constexpr=gl.BlockedLayout([1],[32],[4],[0]);i=gl.arange(0,128,layout=L)
 for task in range(p,M*256,256):
  row=task//256;tile=task%256
  if Ready is not None:consume(Ready+(row*64+tile//4)*64,Health,1,True,1)
  x=dev_load(O,task*128+i).to(gl.float32)
  if tile%4==3:
   mate=gl.gather(x,i^1,0)
   c=dev_load(Cos,row*32+(i-64)//2,i>=64,1.,cache='.ca');sn=dev_load(Sin,row*32+(i-64)//2,i>=64,0.,cache='.ca')
   x=x*c+gl.where(i%2==0,1.,-1.)*mate*sn
  e,scale=quant_scale(x)
  dev_store((x/scale).to(gl.float8e4nv),Q,task*128+i);dev_store(e,S,task)
  if Tiles is not None:publish(Tiles+(tile//8)*64,True)
 if Completion is not None:publish(Completion+(p//32)*64,True)
@g.jit
def reduce_quant(P,Y,Q,S,N:gl.constexpr,SPLIT:gl.constexpr,QUANT:gl.constexpr,M:gl.constexpr,Ready=None,Health=None,Produced=None,ReadyEpoch=1,ReadyCount:gl.constexpr=0,Completion=None,Chunks:gl.constexpr=1):
 p=gl.program_id(0);L:gl.constexpr=gl.BlockedLayout([1],[32],[4],[0]);i=gl.arange(0,128,layout=L)
 for task in range(p,M*N//128,256):
  if Ready is not None:consume(Ready+(task%(N//128))*64,Health,ReadyCount if ReadyCount else SPLIT,True,ReadyEpoch)
  x=gl.full((128,),0,gl.float32,L)
  for k in gl.static_range(SPLIT):x+=dev_load(P,k*M*N+task*128+i)
  x=x.to(gl.bfloat16).to(gl.float32);dev_store(x,Y,task*128+i)
  if QUANT:
   e,s=quant_scale(x);dev_store((x/s).to(gl.float8e4nv),Q,task*128+i);dev_store(e,S,task)
  if Produced is not None:
   publish(Produced+((task%(N//128))//8 if QUANT else task%(N//128))*64,True)
 if Completion is not None:publish(Completion+((p%(N//128))//Chunks)*64,True)
@g.jit
def stage_output_weights(W,B,K:gl.constexpr,SPLIT:gl.constexpr):
 p=gl.program_id(0);nt=p//SPLIT;ks=p%SPLIT
 SL:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,16]],[128,128],[1,0])
 wd=gl.amd.gfx1250.tdm.make_tensor_descriptor(W+nt*128*K,[128,K],[K,1],[128,128],SL)
 first=ks*(K//SPLIT//128)
 gl.amd.gfx1250.tdm.async_load(wd,[0,first*128],B.index(0))
 gl.amd.gfx1250.tdm.async_load(wd,[0,(first+1)*128],B.index(1))


@g.jit
def stage_tuned(W,B,K:gl.constexpr,SPLIT:gl.constexpr,BN:gl.constexpr,BK:gl.constexpr,NB:gl.constexpr):
 p=gl.program_id(0);nt=p//SPLIT;ks=p%SPLIT
 SL:gl.constexpr=gl.SwizzledSharedLayout(1,1,1,[1,0]) if OUTPUT_PACKED else gl.PaddedSharedLayout.with_identity_for([[BK,16]],[BN,BK],[1,0])
 wd=gl.amd.gfx1250.tdm.make_tensor_descriptor(W+nt*BN*K,[BN//16,K*16] if OUTPUT_PACKED else [BN,K],[K*16,1] if OUTPUT_PACKED else [K,1],[BN//16,BK*16] if OUTPUT_PACKED else [BN,BK],SL)
 for j in gl.static_range(NB):
  gl.amd.gfx1250.tdm.async_load(wd,[0,(ks*(K//SPLIT)+j*BK)*(16 if OUTPUT_PACKED else 1)],B.index(j))

@g.jit
def split_project_tuned(X,XS,W,WS,P,N:gl.constexpr,K:gl.constexpr,SPLIT:gl.constexpr,GROUPED:gl.constexpr,M:gl.constexpr,BN:gl.constexpr,BK:gl.constexpr,BUFFERS:gl.constexpr,PreW):
 pid=gl.program_id(0);nt=pid//SPLIT;ks=pid%SPLIT
 NB:gl.constexpr=min(BUFFERS,K//SPLIT//BK)
 ML:gl.constexpr=gl.amd.AMDWMMALayout(version=3,transposed=True,warp_bases=[[0,1],[0,2]],instr_shape=[16,16,128])
 DA:gl.constexpr=gl.DotOperandLayout(0,ML,16);DB:gl.constexpr=gl.DotOperandLayout(1,ML,16)
 ASL:gl.constexpr=gl.amd.gfx1250.get_wmma_scale_layout(DA,[16,BK//32],scale_factor=32)
 BSL:gl.constexpr=gl.amd.gfx1250.get_wmma_scale_layout(DB,[BN,BK//32],scale_factor=32)
 AL:gl.constexpr=gl.to_linear_layout(DA,[16,BK]);BL:gl.constexpr=gl.to_linear_layout(DB,[BK,BN])
 SL:gl.constexpr=gl.SwizzledSharedLayout(1,1,1,[1,0]) if OUTPUT_PACKED else gl.PaddedSharedLayout.with_identity_for([[BK,16]],[BN,BK],[1,0])
 XL:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[BK,16]],[16,BK],[1,0])
 sh=PreW
 xsh=gl.allocate_shared_memory(X.dtype.element_ty,[NB,16,BK],XL)
 wd=gl.amd.gfx1250.tdm.make_tensor_descriptor(W+nt*BN*K,[BN//16,K*16] if OUTPUT_PACKED else [BN,K],[K*16,1] if OUTPUT_PACKED else [K,1],[BN//16,BK*16] if OUTPUT_PACKED else [BN,BK],SL)
 xb=nt//(1024//BN)*K if GROUPED else 0
 xd=gl.amd.gfx1250.tdm.make_tensor_descriptor(X+xb,[M,K],[32768 if GROUPED else K,1],[16,BK],XL)
 blocks:gl.constexpr=K//SPLIT//BK
 first=ks*blocks
 for j in gl.static_range(NB):gl.amd.gfx1250.tdm.async_load(xd,[0,(first+j)*BK],xsh.index(j))
 acc=gl.full((16,BN),0,gl.float32,ML)
 sr=gl.arange(0,16,layout=gl.SliceLayout(1,ASL));sk=gl.arange(0,BK//32,layout=gl.SliceLayout(0,ASL))
 br=gl.arange(0,BN,layout=gl.SliceLayout(1,BSL));bk=gl.arange(0,BK//32,layout=gl.SliceLayout(0,BSL))
 # Initial weights were issued before the preceding operator. Activation loads
 # are issued only after their publication; first wait drains both streams.
 for b in range(blocks):
  if OUTPUT_COEFF:
   asc=gl.load(XS+sr[:,None]*(256 if GROUPED else K//128)+xb//128+(first+b)*(BK//128)+sk[None,:]//4,sr[:,None]<M,0)
   bsc=gl.load(WS+(nt*BN+br[:,None])//128*(K//128)+(first+b)*(BK//128)+bk[None,:]//4)
  if b==0:gl.amd.gfx1250.tdm.async_wait(0)
  elif b<blocks-NB+1:gl.amd.gfx1250.tdm.async_wait((NB-1)*2)
  else:gl.amd.gfx1250.tdm.async_wait(0)
  av=gl.amd.cdna4.async_copy.load_shared_relaxed(xsh.index(b%NB),AL)
  if OUTPUT_PACKED:
   reshaped=sh.index(b%NB).reshape((BN//16,BK//16,16,16)).permute((0,2,1,3)).reshape((BN,BK))
   bv=gl.amd.cdna4.async_copy.load_shared_relaxed(reshaped.permute([1,0]),BL)
  else:bv=gl.amd.cdna4.async_copy.load_shared_relaxed(sh.index(b%NB).permute([1,0]),BL)
  if not OUTPUT_COEFF:
   asc=gl.load(XS+sr[:,None]*(256 if GROUPED else K//128)+xb//128+(first+b)*(BK//128)+sk[None,:]//4,sr[:,None]<M,0)
   bsc=gl.load(WS+(nt*BN+br[:,None])//128*(K//128)+(first+b)*(BK//128)+bk[None,:]//4)
  gl.barrier()
  if b<blocks-NB:
   gl.amd.gfx1250.tdm.async_load(wd,[0,(first+b+NB)*BK*(16 if OUTPUT_PACKED else 1)],sh.index(b%NB))
   gl.amd.gfx1250.tdm.async_load(xd,[0,(first+b+NB)*BK],xsh.index(b%NB))
  acc=gl.amd.gfx1250.wmma_scaled(gl.convert_layout(av,DA),asc,'e4m3',gl.convert_layout(bv,DB),bsc,'e4m3',acc)
 gl.amd.gfx1250.tdm.async_wait(0);gl.barrier()
 OL:gl.constexpr=gl.to_linear_layout(ML,[16,BN]);out=gl.convert_layout(acc,OL)
 om=gl.arange(0,16,layout=gl.SliceLayout(1,OL));on=nt*BN+gl.arange(0,BN,layout=gl.SliceLayout(0,OL))
 dev_store(out,P,ks*M*N+on[None,:]+om[:,None]*N,om[:,None]<M)

@g.jit
def output_tuned(O,Cos,Sin,OA,OAS,OB,OBS,XQ,XS,AP,AY,AQ,AS,BP,Y,Counter,Health,M:gl.constexpr):
 p=gl.program_id(0)
 if OUTPUT_READY:
  R=Counter+512*64
  epoch=dev_load(R,(256+p)*64)+1
  dev_store(epoch,R,(256+p)*64)
 else:
  R=None
  epoch=0
 gl.static_assert(8192//OA_BN*OA_SPLIT<=256 and 4096//OB_BN*OB_SPLIT<=256)
 ANB:gl.constexpr=min(OA_NB,4096//OA_SPLIT//OA_BK)
 BNB:gl.constexpr=min(OB_NB,8192//OB_SPLIT//OB_BK)
 ASL:gl.constexpr=gl.SwizzledSharedLayout(1,1,1,[1,0]) if OUTPUT_PACKED else gl.PaddedSharedLayout.with_identity_for([[OA_BK,16]],[OA_BN,OA_BK],[1,0])
 BSL:gl.constexpr=gl.SwizzledSharedLayout(1,1,1,[1,0]) if OUTPUT_PACKED else gl.PaddedSharedLayout.with_identity_for([[OB_BK,16]],[OB_BN,OB_BK],[1,0])
 earlya=gl.allocate_shared_memory(OA.dtype.element_ty,[ANB,OA_BN//16,OA_BK*16] if OUTPUT_PACKED else [ANB,OA_BN,OA_BK],ASL)
 if p<8192//OA_BN*OA_SPLIT:stage_tuned(OA,earlya,4096,OA_SPLIT,OA_BN,OA_BK,ANB)
 inverse_quant(O,Cos,Sin,XQ,XS,M,Completion=R)
 if OUTPUT_READY:consume(R+(p//OA_SPLIT//(1024//OA_BN))*64,Health,32,p<8192//OA_BN*OA_SPLIT,epoch)
 else:sync(Counter,Health,256,3)
 if p<8192//OA_BN*OA_SPLIT:split_project_tuned(XQ,XS,OA,OAS,AP,8192,4096,OA_SPLIT,True,M,OA_BN,OA_BK,OA_NB,earlya)
 earlyb=gl.allocate_shared_memory(OB.dtype.element_ty,[BNB,OB_BN//16,OB_BK*16] if OUTPUT_PACKED else [BNB,OB_BN,OB_BK],BSL)
 if OUTPUT_EARLIER and p<4096//OB_BN*OB_SPLIT:stage_tuned(OB,earlyb,8192,OB_SPLIT,OB_BN,OB_BK,BNB)
 if OUTPUT_READY:publish(R+(8+p//OA_SPLIT//(128//OA_BN))*64,p<8192//OA_BN*OA_SPLIT)
 else:sync(Counter,Health,256,3)
 if not OUTPUT_EARLIER and p<4096//OB_BN*OB_SPLIT:stage_tuned(OB,earlyb,8192,OB_SPLIT,OB_BN,OB_BK,BNB)
 reduce_quant(AP,AY,AQ,AS,8192,OA_SPLIT,True,M,Ready=R+8*64 if OUTPUT_READY else None,Health=Health,ReadyEpoch=epoch,ReadyCount=(128//OA_BN)*OA_SPLIT,Completion=R+72*64 if OUTPUT_READY else None,Chunks=64//OB_SPLIT)
 if OUTPUT_READY:consume(R+(72+p%OB_SPLIT)*64,Health,min(M*64,256)//OB_SPLIT,p<4096//OB_BN*OB_SPLIT,epoch)
 else:sync(Counter,Health,256,3)
 if p<4096//OB_BN*OB_SPLIT:split_project_tuned(AQ,AS,OB,OBS,BP,4096,8192,OB_SPLIT,False,M,OB_BN,OB_BK,OB_NB,earlyb)
 if OUTPUT_READY:publish(R+(80+p//OB_SPLIT//(128//OB_BN))*64,p<4096//OB_BN*OB_SPLIT)
 else:sync(Counter,Health,256,3)
 reduce_quant(BP,Y,AQ,AS,4096,OB_SPLIT,False,M,Ready=R+80*64 if OUTPUT_READY else None,Health=Health,ReadyEpoch=epoch,ReadyCount=(128//OB_BN)*OB_SPLIT)

@g.jit
def output_body(O,Cos,Sin,OA,OAS,OB,OBS,XQ,XS,AP,AY,AQ,AS,BP,Y,Counter,Health,M:gl.constexpr,SA:gl.constexpr=4,SB:gl.constexpr=8,STOP:gl.constexpr=0,OVERLAP:gl.constexpr=False,Tiles=None,InputTiles=None):
 if OUTPUT_TUNE:
  output_tuned(O,Cos,Sin,OA,OAS,OB,OBS,XQ,XS,AP,AY,AQ,AS,BP,Y,Counter,Health,M)
  return
 if Tiles is not None:
  gl.static_assert(SA==4 and SB==8)
 if OVERLAP & 1:
  prefetch_bytes(OB,4096*8192)
  prefetch_bytes(OBS,4096*8192//128//128)
 if OVERLAP & 2:
  SL:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,16]],[128,128],[1,0])
  early=gl.allocate_shared_memory(OA.dtype.element_ty,[2,128,128],SL)
  stage_output_weights(OA,early,4096,SA)
 else:early=None
 inverse_quant(O,Cos,Sin,XQ,XS,M,Tiles+32*64 if Tiles is not None else None,InputTiles,Health)
 if Tiles is None:sync(Counter,Health,256,3)
 if STOP==1:return
 if gl.program_id(0)<64*SA:
  if Tiles is not None:consume(Tiles+(32+(gl.program_id(0)//SA)//8*4+gl.program_id(0)%SA)*64,Health,M*8,True,1)
  split_project_old(XQ,XS,OA,OAS,AP,8192,4096,SA,True,M,early)
  if Tiles is not None:publish(Tiles+(64+gl.program_id(0)//SA)*64,True)
 if Tiles is None:sync(Counter,Health,256,3)
 if STOP==2:return
 if OVERLAP & 2:stage_output_weights(OB,early,8192,SB)
 reduce_quant(AP,AY,AQ,AS,8192,SA,True,M,Tiles+64*64 if Tiles is not None else None,Health,Tiles+128*64 if Tiles is not None else None)
 if Tiles is None:sync(Counter,Health,256,3)
 if STOP==3:return
 if gl.program_id(0)<32*SB:
  if Tiles is not None:consume(Tiles+(128+gl.program_id(0)%SB)*64,Health,M*8,True,1)
  split_project_old(AQ,AS,OB,OBS,BP,4096,8192,SB,False,M,early)
  if Tiles is not None:publish(Tiles+(136+gl.program_id(0)//SB)*64,True)
 if Tiles is None:sync(Counter,Health,256,3)
 if STOP==4:return
 reduce_quant(BP,Y,AQ,AS,4096,SB,False,M,Tiles+136*64 if Tiles is not None else None,Health,Tiles+168*64 if Tiles is not None else None)
