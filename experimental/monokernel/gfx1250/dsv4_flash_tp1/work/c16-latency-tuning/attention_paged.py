from edge_sync import publish,consume
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from aiter_paged import _pa_decode_sparse, _pa_decode_sparse_reduce
from combined import sync
from masked_prefetch import masked_line_prefetch
@g.jit
def compute_attention(Q,KV,Cache,Indices,Indptr,Sink,M,L,A,O,Counts,T:gl.constexpr,PAGES,SPLITS:gl.constexpr,BH:gl.constexpr=16,BK:gl.constexpr=32,WRITE_SLOT=0,DIRECT:gl.constexpr=False,PID=0):
    pid=PID
    if pid<(64//BH)*SPLITS:
        h=pid//SPLITS;k=pid%SPLITS
        n=gl.load(Indptr+1)-gl.load(Indptr)
        if k*gl.cdiv(n,SPLITS*BK)<gl.cdiv(n,BK):
            _pa_decode_sparse(Q,Cache,Cache,Indices,Indptr,M,L,A,Sink,O,PAGES,
                64*512,512,1,512,1,1,
                SPLITS*64,64,1,SPLITS*64,64,1,
                SPLITS*64*512,64*512,512,1,64*512,512,1,
                64,512,SPLITS,0.04419417382415922,BH,512,BK,True,False,128,4,True,4,h,k,KV,WRITE_SLOT,DIRECT)


@g.jit
def attention_paged(Q,KV,Cache,Indices,Indptr,Sink,Max,Sum,Acc,Out,Counts,Counter,Health,Dest,Pages,M:gl.constexpr,SPLITS:gl.constexpr=8,Tiles=None,LayerTiles=None,COMP:gl.constexpr=0):
 pid=gl.program_id(0)
 for task in range(pid,M*4*SPLITS,256):
  row=task%M;head=(task//M)%4;split=task//(M*4)
  if LayerTiles is not None:
   consume(LayerTiles+(1088+row*4+head)*64,Health,16,True,1)
   consume(LayerTiles+(1120+row)*64,Health,1,True,1)
   if COMP:consume(LayerTiles+(1664+row)*64,Health,COMP,True,1)
  beg=gl.load(Indptr+row);n=gl.load(Indptr+row+1)-beg
  first=split*gl.cdiv(n,SPLITS*16)*16
  if first<n:
   slot=gl.load(Indices+beg+first)
   PS:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,8]],[1,512],[1,0])
   pd=gl.amd.gfx1250.tdm.make_tensor_descriptor(Cache+slot.to(gl.int64)*512,[1,512],[512,1],[1,512],PS)
   gl.amd.gfx1250.tdm.prefetch(pd,[0,0],pred=(slot>=0)&(slot<Pages),speculative=False)
   dest=gl.load(Dest+row)
   compute_attention(Q+row*32768,KV+row*512,Cache,Indices,Indptr+row,Sink,
    Max+row*SPLITS*64,Sum+row*SPLITS*64,Acc+row*SPLITS*32768,Out+row*32768,Counts,
    0,Pages,SPLITS,16,16,dest,True,head*SPLITS+split)
  if Tiles is not None:publish(Tiles+(row*4+head)*64,True)
 if Tiles is None:sync(Counter,Health,256,3)
 for task in range(pid,M*64,256):
  row=task//64;head=task%64
  if Tiles is not None:consume(Tiles+(row*4+head//16)*64,Health,SPLITS,True,1)
  _pa_decode_sparse_reduce(Max+row*SPLITS*64,Sum+row*SPLITS*64,Acc+row*SPLITS*32768,Sink,Indptr+row,Out+row*32768,
   SPLITS*64,64,1,SPLITS*64,64,1,SPLITS*64*512,64*512,512,1,64*512,512,1,
   64,512,SPLITS,1,512,16,True,4,head)

  if LayerTiles is not None:publish(LayerTiles+(1136+row*64+head)*64,True)
