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
                    OWNER:gl.constexpr=0):
    if gl.program_id(0)==OWNER:
        L:gl.constexpr=gl.BlockedLayout([1],[32],[4],[0])
        DF:gl.constexpr=D*(2 if OVERLAP else 1)
        K:gl.constexpr=R*(2 if OVERLAP else 1)
        pos=gl.load(Plan+2)
        if pos>=0:
            # C1 decode has one current input and K-1 historical rows.
            slot=gl.load(SlotIn).to(gl.int64)
            m=gl.full((D,),float('-inf'),gl.float32,L)
            a=gl.full((D,),0.,gl.float32,L)
            z=gl.full((D,),0.,gl.float32,L)
            for k in range(K-1):
                s=pos-K+1+k
                off=(gl.cast(k>=R,gl.int32)*D) if OVERLAP else 0
                if s>=0:
                    v=row(KS+slot*KSS+(s%STATE)*KSP+off,D)
                    score=row(SS+slot*SSS+(s%STATE)*SSP+off,D)
                    mn=gl.maximum(m,score)
                    f=gl.where(m==float('-inf'),0.,gl.exp(m-mn))
                    w=gl.where(score==float('-inf'),0.,gl.exp(score-mn))
                    a=a*f+w*v;z=z*f+w;m=mn
            off:gl.constexpr=D if OVERLAP else 0
            v=row(KV+off,D)
            score=row(Score+off,D)+row(APE+(R-1)*DF+off,D)
            mn=gl.maximum(m,score)
            f=gl.where(m==float('-inf'),0.,gl.exp(m-mn));w=gl.exp(score-mn)
            a=a*f+w*v;z=z*f+w
            pooled=a/z
            normed=pooled*gl.rsqrt(gl.sum(pooled*pooled,0)/D+1.e-6)*row(Norm,D)
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
                addr=block*CBS+(token//16)*(16*D)+(d//16)*256+(token%16)*16+d%16
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
