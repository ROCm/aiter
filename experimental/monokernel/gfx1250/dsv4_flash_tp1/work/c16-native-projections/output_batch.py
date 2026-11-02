import os
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from native_tiles import gemm,weight_first
from previous.output_batch import inverse_quant
from edge_sync import publish,consume,prefetch_bytes
from device_scope_helpers import dev_load,dev_store
from combined_scaled import sync,wait_target
OB_EARLY_STAGE=gl.constexpr(int(os.getenv("OB_EARLY_STAGE","0")))
O_SUBGROUP=gl.constexpr(int(os.getenv("O_SUBGROUP","0")))
from ffn_batch import quant_body
from native_quant import quant_groups,quant_column
ELECT_QUANT=gl.constexpr(int(os.getenv("ELECT_QUANT","0")))
VECTOR_QUANT=gl.constexpr(int(os.getenv("VECTOR_QUANT","1")))
ADV=gl.constexpr(int(os.getenv('O_ADV',os.getenv('ADV','0'))))
OB_BUFFERS=gl.constexpr(int(os.getenv('OB_BUFFERS','6' if int(os.getenv('ADV','0'))&16 else '4')))
OA_M=gl.constexpr(int(os.getenv("OA_M","32")))
OA_K=gl.constexpr(int(os.getenv("OA_K","1024")))
OA_BUFFERS=gl.constexpr(int(os.getenv('OA_BUFFERS','4')))

@g.jit
def output_body(O,Cos,Sin,OA,OAS,OB,OBS,XQ,XS,AP,AY,AQ,AS,BP,Y,Counter,Health,M:gl.constexpr,SA:gl.constexpr=4,SB:gl.constexpr=8,STOP:gl.constexpr=0,OVERLAP:gl.constexpr=False,Tiles=None,InputTiles=None):
 p=gl.program_id(0)
 gl.static_assert(M==16)
 gl.static_assert(Tiles is None and InputTiles is None)
 if ADV&12 and not ELECT_QUANT:
  epoch=dev_load(Counter,(1024+p)*64)+1
  gl.barrier()
  dev_store(epoch,Counter,(1024+p)*64)
 else:epoch=0
 if O_SUBGROUP and p<64:
  ob_epoch=dev_load(Counter,(1280+p)*64)+1
  gl.barrier()
  dev_store(ob_epoch,Counter,(1280+p)*64)
 else:ob_epoch=0
 if ADV&1:earlya=weight_first(OA,8192,4096,32,OA_K,p)
 else:earlya=None
 if VECTOR_QUANT:quant_groups(O,Cos,Sin,XQ,XS,M,32768,2048,True,Completion=Counter+256*64 if ADV&8 and not ELECT_QUANT else None)
 else:inverse_quant(O,Cos,Sin,XQ,XS,M,Completion=Counter+256*64 if ADV&8 and not ELECT_QUANT else None)
 if ADV&2:prefetch_bytes(OB,4096*8192)
 if ADV&8:
  if ELECT_QUANT:sync(Counter+(256+p//32)*64,Health,32,3)
  else:consume(Counter+(256+p//32)*64,Health,32,True,epoch)
 else:sync(Counter,Health,256,3)
 if ADV&1 and OB_EARLY_STAGE:earlyb=weight_first(OB,4096,8192,64,512,gl.minimum(p,63),p<64)
 gemm(XQ,XS,OA,OAS,AY,M,8192,4096,OA_M,32,OA_K,OA_BUFFERS,p,GROUPED=True,Early=earlya)
 if ADV&1:
  if not OB_EARLY_STAGE:earlyb=weight_first(OB,4096,8192,64,512,gl.minimum(p,63),p<64)
 else:earlyb=None
 if ELECT_QUANT:
  gl.inline_asm_elementwise('s_wait_storecnt 0\n s_mov_b32 $0, 0',constraints='=s,~{memory}',args=[],dtype=gl.int32,is_pure=False,pack=1)
  gl.barrier()
  ticket=gl.atomic_add(Counter+(264+p//4)*64,1,sem='relaxed',scope='gpu')
  if ticket%4==3:
   quant_column(AY,AQ,AS,M,8192,p//4)
   if O_SUBGROUP:
    gl.inline_asm_elementwise('s_wait_storecnt 0\n s_mov_b32 $0, 0',constraints='=s,~{memory}',args=[],dtype=gl.int32,is_pure=False,pack=1)
    gl.barrier()
    gl.atomic_add(Counter+384*64,1,sem='relaxed',scope='gpu')
  if O_SUBGROUP:
   if p<64:wait_target(Counter+384*64,Health,ob_epoch*64,3)
  else:sync(Counter,Health,256,3)
 else:
  if ADV&8:
   publish(Counter+(264+p//(16 if VECTOR_QUANT else 4))*64,True)
   consume(Counter+(264+(p//16 if VECTOR_QUANT else p%64))*64,Health,16 if VECTOR_QUANT else 4,True,epoch)
  else:sync(Counter,Health,256,3)
  if VECTOR_QUANT:quant_groups(AY,Cos,Sin,AQ,AS,M,8192,512)
  else:quant_body(AY,AQ,AS,8192,128,1,M)
  if ADV&8:publish(Counter+(328+(p//16 if VECTOR_QUANT else (p%64)//4))*64,True)
  elif ADV&4:
   publish(Counter+344*64,True)
   consume(Counter+344*64,Health,256,p<64,epoch)
  else:sync(Counter,Health,256,3)
 if p<64:gemm(AQ,AS,OB,OBS,Y,M,4096,8192,16,64,512,OB_BUFFERS,p,Early=earlyb,Ready=Counter+328*64 if ADV&8 and not ELECT_QUANT else None,Health=Health,Epoch=epoch,ReadyCount=16)
