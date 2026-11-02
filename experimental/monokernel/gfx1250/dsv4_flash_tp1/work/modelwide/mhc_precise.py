from mhc import *
@g.jit
def rcp(x):
    return gl.inline_asm_elementwise("v_rcp_f32 $0, $1",constraints="=v,v",args=[x],dtype=gl.float32,is_pure=True,pack=1)


@g.jit
def tail_fused(R,S,B,AN,M,Temp,Raw,X,rbuf,abuf,EARLY:gl.constexpr,MODE:gl.constexpr,FP32=None):
    pid=gl.program_id(0)
    if pid<1+MODE:
        L:gl.constexpr=gl.BlockedLayout([1],[32],[4],[0])
        i=gl.arange(0,4096,layout=L)
        if EARLY:
            gl.amd.gfx1250.tdm.async_wait(0)
            RRL:gl.constexpr=gl.BlockedLayout([1,1],[1,32],[1,4],[1,0])
            RDATA=rbuf.load(RRL)
            r0=gl.sum(gl.where(gl.arange(0,4,layout=gl.SliceLayout(1,RRL))[:,None]==0,RDATA,0).to(gl.float32),0)
            r1=gl.sum(gl.where(gl.arange(0,4,layout=gl.SliceLayout(1,RRL))[:,None]==1,RDATA,0).to(gl.float32),0)
            r2=gl.sum(gl.where(gl.arange(0,4,layout=gl.SliceLayout(1,RRL))[:,None]==2,RDATA,0).to(gl.float32),0)
            r3=gl.sum(gl.where(gl.arange(0,4,layout=gl.SliceLayout(1,RRL))[:,None]==3,RDATA,0).to(gl.float32),0)
            r0=gl.convert_layout(r0,L);r1=gl.convert_layout(r1,L);r2=gl.convert_layout(r2,L);r3=gl.convert_layout(r3,L)
        else:
            r0=dev_load(R,i).to(gl.float32);r1=dev_load(R,4096+i).to(gl.float32)
            r2=dev_load(R,8192+i).to(gl.float32);r3=dev_load(R,12288+i).to(gl.float32)
        if FP32 is not None:
            FSL:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,8]],[4,4096],[1,0])
            fb=gl.allocate_shared_memory(gl.float32,[4,4096],FSL)
            fd=gl.amd.gfx1250.tdm.make_tensor_descriptor(FP32,[4,4096],[4096,1],[4,4096],FSL)
            gl.amd.gfx1250.tdm.async_load(fd,[0,0],fb);gl.amd.gfx1250.tdm.async_wait(0)
            FL:gl.constexpr=gl.BlockedLayout([1,1],[1,32],[1,4],[1,0])
            fr=fb.load(FL);fh=gl.arange(0,4,layout=gl.SliceLayout(1,FL))
            f0=gl.convert_layout(gl.sum(gl.where(fh[:,None]==0,fr,0.),0),L)
            f1=gl.convert_layout(gl.sum(gl.where(fh[:,None]==1,fr,0.),0),L)
            f2=gl.convert_layout(gl.sum(gl.where(fh[:,None]==2,fr,0.),0),L)
            f3=gl.convert_layout(gl.sum(gl.where(fh[:,None]==3,fr,0.),0),L)
        else:f0=r0;f1=r1;f2=r2;f3=r3
        # Preserve the original 16 sequential 1024-element RMS reductions.
        # A different F32 reduction order can flip a later BF16/FP8 threshold.
        ri=gl.arange(0,1024,layout=L)
        CHL:gl.constexpr=gl.BlockedLayout([1,1],[1,32],[1,4],[1,0])
        sq=0.
        for h in gl.static_range(4):
            if h==0:rv=f0
            elif h==1:rv=f1
            elif h==2:rv=f2
            else:rv=f3
            xv=gl.convert_layout(rv.reshape((4,1024)),CHL)
            parts=gl.sum(xv*xv,1)
            ch=gl.arange(0,4,layout=gl.SliceLayout(1,CHL))
            for b in gl.static_range(4):
                sq+=gl.sum(gl.where(ch==b,parts,0.),0)
        rs=gl.rsqrt(sq/16384+1.e-6)
        si=gl.arange(0,128,layout=L)
        p0=1./(1.+gl.exp(-(gl.sum(dev_load(Temp,si*24),0)*rs*dev_load(S,0)+dev_load(B,0))))+1.e-6
        p1=1./(1.+gl.exp(-(gl.sum(dev_load(Temp,si*24+1),0)*rs*dev_load(S,0)+dev_load(B,1))))+1.e-6
        p2=1./(1.+gl.exp(-(gl.sum(dev_load(Temp,si*24+2),0)*rs*dev_load(S,0)+dev_load(B,2))))+1.e-6
        p3=1./(1.+gl.exp(-(gl.sum(dev_load(Temp,si*24+3),0)*rs*dev_load(S,0)+dev_load(B,3))))+1.e-6
        if pid==0:
            mixture=((r0*p0+r1*p1)+r2*p2)+r3*p3
            # ATOM computes RMS from FP32 mixture, but normalizes its BF16-rounded value.
            y=mixture.to(gl.bfloat16).to(gl.float32)
            dev_store(y.to(gl.bfloat16),Raw,i)
            if EARLY:
                av=abuf.load(gl.BlockedLayout([1,1],[1,32],[1,4],[1,0]))
                an=gl.convert_layout(gl.sum(av.to(gl.float32),0),L)
            else:an=dev_load(AN,i).to(gl.float32)
            x=y*gl.rsqrt(gl.sum(mixture*mixture,0)/4096+1.e-6)*an
            dev_store(x.to(gl.bfloat16),X,i)
        if pid==MODE:
            dev_store(p0,M,0);dev_store(p1,M,1);dev_store(p2,M,2);dev_store(p3,M,3)
            DL:gl.constexpr=gl.BlockedLayout([1,1],[4,8],[4,1],[1,0])
            rr=gl.arange(0,4,layout=gl.SliceLayout(1,DL));cc=gl.arange(0,4,layout=gl.SliceLayout(0,DL))
            RL:gl.constexpr=gl.BlockedLayout([1,4],[4,8],[4,1],[1,0])
            kk=gl.arange(0,128,layout=gl.SliceLayout(1,RL))
            nn=gl.arange(0,32,layout=gl.SliceLayout(0,RL))
            dots=gl.sum(dev_load(Temp,kk[:,None]*24+nn[None,:],nn[None,:]<24,0),0)
            dd=gl.convert_layout(dots,gl.BlockedLayout([1],[32],[4],[0]))
            # Gather is entirely in registers; projected outputs have only 24 entries.
            flat=gl.arange(0,16,layout=gl.BlockedLayout([1],[32],[4],[0]))
            zz=gl.gather(dd,flat+8,0)
            z=gl.convert_layout(zz.reshape((4,4)),DL)
            post=gl.gather(dd,gl.arange(0,4,layout=gl.BlockedLayout([1],[32],[4],[0]))+4,0)
            post=gl.convert_layout(post,gl.SliceLayout(1,DL))
            z=z*rs*dev_load(S,2)+dev_load(B,8+rr[:,None]*4+cc[None,:])
            post=post*rs*dev_load(S,1)+dev_load(B,4+rr)
            dev_store(2./(1.+gl.exp(-post)),M,4+rr)
            z=gl.exp(z-gl.max(z,1)[:,None]);z=z/gl.sum(z,1)[:,None]+1.e-6;z=z/(gl.sum(z,0)[None,:]+1.e-6)
            for _ in range(19):
                z=z*rcp(gl.sum(z,1)[:,None]+1.e-6);z=z*rcp(gl.sum(z,0)[None,:]+1.e-6)
            dev_store(z,M,8+rr[:,None]*4+cc[None,:])

@g.jit
def candidate_body(R,W,S,B,AN,M,Temp,Raw,X,Counter,Health,EARLY:gl.constexpr=True,FP32=None):
    SL:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,8]],[4,4096],[1,0])
    ASL:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,8]],[1,4096],[1,0])
    rbuf=gl.allocate_shared_memory(R.dtype.element_ty,[4,4096],SL)
    abuf=gl.allocate_shared_memory(AN.dtype.element_ty,[1,4096],ASL)
    if EARLY==2:mhc_project(R,W,Temp)
    if EARLY:
        if gl.program_id(0)<2:
            rd=gl.amd.gfx1250.tdm.make_tensor_descriptor(base=R,shape=[4,4096],strides=[4096,1],block_shape=[4,4096],layout=SL)
            ad=gl.amd.gfx1250.tdm.make_tensor_descriptor(base=AN,shape=[1,4096],strides=[4096,1],block_shape=[1,4096],layout=ASL)
            gl.amd.gfx1250.tdm.async_load(rd,[0,0],rbuf,cache_modifier='.cg')
            gl.amd.gfx1250.tdm.async_load(ad,[0,0],abuf,cache_modifier='.cg')
    if EARLY!=2:mhc_project(R,W,Temp)
    sync(Counter,Health,256,3)
    tail_fused(R,S,B,AN,M,Temp,Raw,X,rbuf,abuf,EARLY,1,FP32)

@g.jit
def candidate(R,W,S,B,AN,M,Temp,Raw,X,Counter,Health,EARLY:gl.constexpr=True,FP32=None):
    candidate_body(R,W,S,B,AN,M,Temp,Raw,X,Counter,Health,EARLY)
