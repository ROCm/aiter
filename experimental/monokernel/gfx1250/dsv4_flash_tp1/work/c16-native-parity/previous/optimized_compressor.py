import os
from triton.experimental.gluon import language as gl
from previous.edge_sync import publish,consume
from device_scope_helpers import dev_load,dev_store
CSA_POOL = gl.constexpr(int('0'))
HCA_READY = gl.constexpr(int('0'))
HCA_CONCURRENT = gl.constexpr(int('1'))
HCA_WIDTH = gl.constexpr(int('64'))
HCA_SLICE = gl.constexpr(int('1'))
# Compressor math adapted from ROCm/ATOM v4_kernels/fused_compress.py (MIT).
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

@g.jit
def row(P, N:gl.constexpr):
    S:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,8]],[1,N],[1,0])
    sm=gl.allocate_shared_memory(P.dtype.element_ty,[1,N],S)
    desc=gl.amd.gfx1250.tdm.make_tensor_descriptor(P,[1,N],[N,1],[1,N],S)
    gl.amd.gfx1250.tdm.async_load(desc,[0,0],sm)
    gl.amd.gfx1250.tdm.async_wait(0)
    v=sm.load(gl.BlockedLayout([1,1],[1,32],[1,4],[1,0])).to(gl.float32)
    return gl.convert_layout(gl.sum(v,0),gl.BlockedLayout([1],[32],[4],[0]))

@g.jit
def compressor_body(KV,Score,APE,Norm,Cos,Sin,Plan,WritePlan,SlotIn,SlotOut,
                    KS,SS,Cache,Scale,Blocks,
                    D:gl.constexpr,R:gl.constexpr,OVERLAP:gl.constexpr,
                    STATE:gl.constexpr,KSS:gl.constexpr,KSP:gl.constexpr,
                    SSS:gl.constexpr,SSP:gl.constexpr,CBS:gl.constexpr,
                    CTS:gl.constexpr,SBS:gl.constexpr,QUANT:gl.constexpr,
                    OWNER=0,PARALLEL_POOL:gl.constexpr=False,Partial=None,EARLY:gl.constexpr=False):
    if gl.program_id(0)==OWNER:
        L:gl.constexpr=gl.BlockedLayout([1],[32],[4],[0])
        DF:gl.constexpr=D*(2 if OVERLAP else 1)
        K:gl.constexpr=R*(2 if OVERLAP else 1)
        gl.static_assert(STATE % (4 if OVERLAP else 8) == 0, "ring must align with TDM tile")
        pos=gl.load(Plan+2)
        if pos>=0:
            if EARLY:
                CL:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,8]],[1,D],[1,0])
                coeffn=gl.allocate_shared_memory(Norm.dtype.element_ty,[1,D],CL)
                coeffa=gl.allocate_shared_memory(APE.dtype.element_ty,[1,D],CL)
                nd=gl.amd.gfx1250.tdm.make_tensor_descriptor(Norm,[1,D],[D,1],[1,D],CL)
                ad=gl.amd.gfx1250.tdm.make_tensor_descriptor(APE+(R-1)*DF+(D if OVERLAP else 0),[1,D],[D,1],[1,D],CL)
                gl.amd.gfx1250.tdm.async_load(nd,[0,0],coeffn)
                gl.amd.gfx1250.tdm.async_load(ad,[0,0],coeffa)
            # C1 decode has one current input and K-1 historical rows.
            slot=gl.load(SlotIn).to(gl.int64)
            m=gl.full((D,),float('-inf'),gl.float32,L)
            a=gl.full((D,),0.,gl.float32,L)
            z=gl.full((D,),0.,gl.float32,L)
            if Partial is not None:
                LL:gl.constexpr=gl.BlockedLayout([1,4],[1,32],[4,1],[1,0])
                rr=gl.arange(0,4,layout=gl.SliceLayout(1,LL));dd=gl.arange(0,D,layout=gl.SliceLayout(0,LL))
                pm=gl.load(Partial+rr[:,None]*3*D+dd[None,:]);pa=gl.load(Partial+rr[:,None]*3*D+D+dd[None,:]);pz=gl.load(Partial+rr[:,None]*3*D+2*D+dd[None,:])
                mm=gl.max(pm,0);ww=gl.exp(pm-mm[None,:])
                m=gl.convert_layout(mm,L);a=gl.convert_layout(gl.sum(pa*ww,0),L);z=gl.convert_layout(gl.sum(pz*ww,0),L)
            else:
                # Boundary windows align to ratio; tile the historical ring reads.
                TILE:gl.constexpr=4 if OVERLAP else 8
                SL:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,8]],[TILE,D],[1,0])
                sk=gl.allocate_shared_memory(gl.float32,[TILE,D],SL)
                sg=gl.allocate_shared_memory(gl.float32,[TILE,D],SL)
                for base in range(0,K,TILE):
                    start=pos-K+1+base
                    off=(gl.cast(base>=R,gl.int32)*D) if OVERLAP else 0
                    if start>=0:
                        ring=start%STATE
                        kd=gl.amd.gfx1250.tdm.make_tensor_descriptor(KS+slot*KSS+ring*KSP+off,[TILE,D],[KSP,1],[TILE,D],SL)
                        sd=gl.amd.gfx1250.tdm.make_tensor_descriptor(SS+slot*SSS+ring*SSP+off,[TILE,D],[SSP,1],[TILE,D],SL)
                        gl.amd.gfx1250.tdm.async_load(kd,[0,0],sk)
                        gl.amd.gfx1250.tdm.async_load(sd,[0,0],sg)
                        gl.amd.gfx1250.tdm.async_wait(0)
                        if PARALLEL_POOL:
                            LL:gl.constexpr=gl.BlockedLayout([1,4],[4,8],[4,1],[1,0])
                            vv=sk.load(LL);sv=sg.load(LL)
                            rr=gl.arange(0,TILE,layout=gl.SliceLayout(1,LL))
                            sv=gl.where((base+rr[:,None])<K-1,sv,float('-inf'))
                            cm=gl.max(sv,0)
                            cw=gl.exp(sv-cm[None,:])
                            ca=gl.convert_layout(gl.sum(cw*vv,0),L)
                            cz=gl.convert_layout(gl.sum(cw,0),L)
                            cm=gl.convert_layout(cm,L)
                            mn=gl.maximum(m,cm)
                            f=gl.where(m==float('-inf'),0.,gl.exp(m-mn))
                            w=gl.exp(cm-mn)
                            a=a*f+ca*w;z=z*f+cz*w;m=mn
                        else:
                            for j in gl.static_range(TILE):
                                if base+j<K-1:
                                    v=gl.convert_layout(gl.sum(sk.slice(j,1,0).load(gl.BlockedLayout([1,1],[1,32],[1,4],[1,0])),0),L)
                                    score=gl.convert_layout(gl.sum(sg.slice(j,1,0).load(gl.BlockedLayout([1,1],[1,32],[1,4],[1,0])),0),L)
                                    mn=gl.maximum(m,score)
                                    f=gl.where(m==float('-inf'),0.,gl.exp(m-mn))
                                    w=gl.where(score==float('-inf'),0.,gl.exp(score-mn))
                                    a=a*f+w*v;z=z*f+w;m=mn
            off:gl.constexpr=D if OVERLAP else 0
            v=row(KV+off,D)
            if EARLY:
                gl.amd.gfx1250.tdm.async_wait(0)
                CLD:gl.constexpr=gl.BlockedLayout([1,1],[1,32],[1,4],[1,0])
                nv=gl.convert_layout(gl.sum(coeffn.load(CLD),0),L)
                av=gl.convert_layout(gl.sum(coeffa.load(CLD),0),L)
                score=row(Score+off,D)+av
            else:score=row(Score+off,D)+row(APE+(R-1)*DF+off,D)
            mn=gl.maximum(m,score)
            f=gl.where(m==float('-inf'),0.,gl.exp(m-mn));w=gl.exp(score-mn)
            a=a*f+w*v;z=z*f+w
            pooled=a/z
            normed=pooled*gl.rsqrt(gl.sum(pooled*pooled,0)/D+1.e-6)*(nv if EARLY else row(Norm,D))
            d=gl.arange(0,D,layout=L)
            # Interleaved GPT-J rotation over final 64 features.
            mate=gl.gather(normed,d^1,0)
            pair=gl.maximum((d-(D-64))//2,0)
            rp=(pos//R)*R
            c=gl.load(Cos+rp*32+pair,d>=D-64,1.).to(gl.float32)
            s=gl.load(Sin+rp*32+pair,d>=D-64,0.).to(gl.float32)
            rotated=gl.where(d%2==0,normed*c-mate*s,normed*c+mate*s)
            ci=pos//R;kb:gl.constexpr=256//R
            block=gl.load(Blocks+ci//kb).to(gl.int64);token=ci%kb
            if QUANT:
                scale=gl.maximum(gl.max(gl.abs(rotated),0),1.e-4)/448.
                scale=gl.exp2(gl.ceil(gl.log2(scale)))
                q=gl.minimum(gl.maximum(rotated/scale,-448.),448.)
                addr=block*CBS+token*D+d  # packed FP8 rows precede the per-block scale tail
                if Cache.dtype.element_ty == gl.uint8:
                    gl.store(Cache+addr,q.to(gl.float8e4nv).to(gl.uint8,bitcast=True))
                else:
                    gl.store(Cache+addr,q.to(Cache.dtype.element_ty))
                gl.store(Scale+block*SBS+token,scale)
            else:
                gl.store(Cache+block*CBS+token*CTS+d,rotated.to(gl.bfloat16))
        # Must follow all historical reads, including when input/output slots alias.
        wp=gl.load(WritePlan+2)
        if wp>=0:
            slot=gl.load(SlotOut).to(gl.int64)
            d=gl.arange(0,DF,layout=L)
            v=row(KV,DF);s=row(Score,DF)+row(APE+(wp%R)*DF,DF)
            gl.store(KS+slot*KSS+(wp%STATE)*KSP+d,v)
            gl.store(SS+slot*SSS+(wp%STATE)*SSP+d,s)

# The body is directly launchable and can also be inlined in the layer.
compressor = compressor_body

@g.jit
def pool_partial(KS,SS,Plan,Slots,Work,KSS:gl.constexpr,KSP:gl.constexpr,SSS:gl.constexpr,SSP:gl.constexpr,M:gl.constexpr=8,PLAN_STRIDE:gl.constexpr=4,Task=None):
 if HCA_SLICE:
  pool_sliced(KS,SS,Plan,Slots,Work,KSS,KSP,SSS,SSP,M,PLAN_STRIDE,Task)
  return
 p=gl.program_id(0);row=p//4;part=p%4
 if p<M*4:
  pos=gl.load(Plan+row*PLAN_STRIDE+2)
  if pos>=0:
   slot=gl.load(Slots+row).to(gl.int64)
   SL:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,8]],[32,512],[1,0])
   kb=gl.allocate_shared_memory(gl.float32,[32,512],SL);sb=gl.allocate_shared_memory(gl.float32,[32,512],SL)
   kd=gl.amd.gfx1250.tdm.make_tensor_descriptor(KS+slot*KSS,[128,512],[KSP,1],[32,512],SL)
   sd=gl.amd.gfx1250.tdm.make_tensor_descriptor(SS+slot*SSS,[128,512],[SSP,1],[32,512],SL)
   gl.amd.gfx1250.tdm.async_load(kd,[part*32,0],kb);gl.amd.gfx1250.tdm.async_load(sd,[part*32,0],sb);gl.amd.gfx1250.tdm.async_wait(0)
   LL:gl.constexpr=gl.BlockedLayout([1,4],[1,32],[4,1],[1,0])
   rr=gl.arange(0,32,layout=gl.SliceLayout(1,LL));dd=gl.arange(0,512,layout=gl.SliceLayout(0,LL))
   vv=kb.load(LL);sc=sb.load(LL);sc=gl.where(rr[:,None]+part*32<127,sc,float('-inf'))
   mm=gl.max(sc,0);ww=gl.exp(sc-mm[None,:]);aa=gl.sum(ww*vv,0);zz=gl.sum(ww,0)
   gl.store(Work+p*1536+dd,mm);gl.store(Work+p*1536+512+dd,aa);gl.store(Work+p*1536+1024+dd,zz)

@g.jit
def pool_sliced(KS,SS,Plan,Slots,Work,KSS:gl.constexpr,KSP:gl.constexpr,SSS:gl.constexpr,SSP:gl.constexpr,M:gl.constexpr,PLAN_STRIDE:gl.constexpr,Task=None):
 p=gl.program_id(0) if Task is None else Task
 row=p//(512//HCA_WIDTH);feat=p%(512//HCA_WIDTH)
 if (p>=0)&(row<M):
  pos=gl.load(Plan+row*PLAN_STRIDE+2)
  if pos>=0:
   slot=gl.load(Slots+row).to(gl.int64)
   SL:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,8]],[128,HCA_WIDTH],[1,0])
   kb=gl.allocate_shared_memory(gl.float32,[128,HCA_WIDTH],SL);sb=gl.allocate_shared_memory(gl.float32,[128,HCA_WIDTH],SL)
   kd=gl.amd.gfx1250.tdm.make_tensor_descriptor(KS+slot*KSS,[128,512],[KSP,1],[128,HCA_WIDTH],SL)
   sd=gl.amd.gfx1250.tdm.make_tensor_descriptor(SS+slot*SSS,[128,512],[SSP,1],[128,HCA_WIDTH],SL)
   gl.amd.gfx1250.tdm.async_load(kd,[0,feat*HCA_WIDTH],kb);gl.amd.gfx1250.tdm.async_load(sd,[0,feat*HCA_WIDTH],sb);gl.amd.gfx1250.tdm.async_wait(0)
   LL:gl.constexpr=gl.BlockedLayout([1,4],[4,8],[4,1],[1,0])
   rr=gl.arange(0,128,layout=gl.SliceLayout(1,LL))
   vv=kb.load(LL).reshape((4,32,HCA_WIDTH));sc=sb.load(LL)
   sc=gl.where(rr[:,None]<127,sc,float('-inf')).reshape((4,32,HCA_WIDTH))
   mm=gl.max(sc,1);ww=gl.exp(sc-mm[:,None,:]);aa=gl.sum(ww*vv,1);zz=gl.sum(ww,1)
   OL:gl.constexpr=gl.BlockedLayout([1,4],[4,8],[4,1],[1,0])
   ii=gl.arange(0,4,layout=gl.SliceLayout(1,OL));dd=gl.arange(0,HCA_WIDTH,layout=gl.SliceLayout(0,OL))
   off=row*6144+ii[:,None]*1536+feat*HCA_WIDTH+dd[None,:]
   gl.store(Work+off,gl.convert_layout(mm,OL));gl.store(Work+off+512,gl.convert_layout(aa,OL));gl.store(Work+off+1024,gl.convert_layout(zz,OL))

@g.jit
def hca_ready(Counter,Health,M:gl.constexpr):
 p=gl.program_id(0)
 epoch=dev_load(Counter,(256+p)*64)+1
 dev_store(epoch,Counter,(256+p)*64)
 publish(Counter+64,p<64)
 publish(Counter+(2+(p-64)//(512//HCA_WIDTH))*64,(p>=64)&(p<64+M*(512//HCA_WIDTH)))
 consume(Counter+64,Health,64,p<M,epoch)
 consume(Counter+(2+p)*64,Health,512//HCA_WIDTH,p<M,epoch)
