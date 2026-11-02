from mhc import *

@g.jit
def tail_fused(R,S,B,AN,M,Temp,Raw,X,MODE:gl.constexpr):
    pid=gl.program_id(0)
    if pid<1+MODE:
        L:gl.constexpr=gl.BlockedLayout([1],[32],[4],[0])
        i=gl.arange(0,4096,layout=L)
        r0=dev_load(R,i).to(gl.float32);r1=dev_load(R,4096+i).to(gl.float32)
        r2=dev_load(R,8192+i).to(gl.float32);r3=dev_load(R,12288+i).to(gl.float32)
        sq=gl.sum(r0*r0,0)+gl.sum(r1*r1,0)+gl.sum(r2*r2,0)+gl.sum(r3*r3,0)
        rs=gl.rsqrt(sq/16384+1.e-6)
        si=gl.arange(0,128,layout=L)
        p0=1./(1.+gl.exp(-(gl.sum(dev_load(Temp,si*24),0)*rs*dev_load(S,0)+dev_load(B,0))))+1.e-6
        p1=1./(1.+gl.exp(-(gl.sum(dev_load(Temp,si*24+1),0)*rs*dev_load(S,0)+dev_load(B,1))))+1.e-6
        p2=1./(1.+gl.exp(-(gl.sum(dev_load(Temp,si*24+2),0)*rs*dev_load(S,0)+dev_load(B,2))))+1.e-6
        p3=1./(1.+gl.exp(-(gl.sum(dev_load(Temp,si*24+3),0)*rs*dev_load(S,0)+dev_load(B,3))))+1.e-6
        if pid==0:
            y=(((r0*p0+r1*p1)+r2*p2)+r3*p3).to(gl.bfloat16).to(gl.float32)
            dev_store(y.to(gl.bfloat16),Raw,i)
            x=y*gl.rsqrt(gl.sum(y*y,0)/4096+1.e-6)*dev_load(AN,i).to(gl.float32)
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
                z=z/(gl.sum(z,1)[:,None]+1.e-6);z=z/(gl.sum(z,0)[None,:]+1.e-6)
            dev_store(z,M,8+rr[:,None]*4+cc[None,:])

@g.jit
def optimized(R,W,S,B,AN,M,Temp,Raw,X,Counter,Health,MODE:gl.constexpr=1):
    mhc_project(R,W,Temp)
    sync(Counter,Health,256,3)
    tail_fused(R,S,B,AN,M,Temp,Raw,X,MODE)
