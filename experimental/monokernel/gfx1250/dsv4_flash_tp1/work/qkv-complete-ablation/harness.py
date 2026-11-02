from pathlib import Path
import json,torch,statistics,re
from qa import project_fused
from qb import qb_kernel
from combined import complete
from tail import quant,norm_rope,norm_only
from base.opt import project_opt,reduce_opt
root=Path(__file__).resolve().parent
f=torch.load(root/'fixture.pt',weights_only=True)
assert torch.cuda.get_device_properties(0).gcnArchName.startswith('gfx1250')
W=f['W_bytes'].cuda().view(torch.float8_e4m3fn);WS=f['WS'].cuda();NW=f['NW'].cuda()
BW=f['QB_bytes'].cuda().view(torch.float8_e4m3fn);BS=f['QBS'].cuda()
Cos=f['cos'].cuda();Sin=f['sin'].cuda();X=f['inputs'][0].cuda()
kw=dict(num_warps=4,enable_fp_fusion=False)
cbase=dict(BN=128,SPLIT=16,DIRECT=True,WAVES=4,BK=128,UNROLL=1,UPSCALE=True,REPLICATE=True,CACHE='.ca',SCALAR=True,PACKED=False,MODE=1,OUTLAY=1,NVEC=1,COUNTSTRIDE=128,NATIVE=0,NORMCACHE='.ca')
class State:
 def __init__(self):
  def z(n,dtype=torch.float32):return torch.zeros(n,device='cuda',dtype=dtype)
  self.xp=z(4096,torch.float8_e4m3fn);self.xs=z(32,torch.uint8)
  self.lp=z(1024,torch.float8_e4m3fn);self.ls=z(8,torch.uint8)
  self.p=z((16,1536));self.acc=z(1536);self.count=z(129,torch.int32);self.y=z(1536)
  self.qp=z(32768);self.qn=z(32768);self.qc=z(64,torch.int32)
  self.q=z(32768,torch.bfloat16);self.kv=z(512,torch.bfloat16)
  self.bar=z(1,torch.int32);self.health=z(1,torch.int32);self.obj=None
 def qa(self,inline,tail):
  self.obj=project_fused[(192,)](X if inline else self.xp,self.xs,W,WS,self.p,NW,self.acc,self.count,self.y,self.lp,self.ls,Cos,Sin,self.kv,INLINE=inline,TAIL=tail,**cbase,**kw)
 def qb(self,tail,bn=128):
  self.obj=qb_kernel[(32768//bn,)](self.lp,self.ls,BW,BS,self.qp,self.qc,Cos,Sin,self.q,TAIL=tail,BN=bn,**kw)
 def run(self,mode,bn=128):
  if mode in ['single','single256']:
   grid=256 if mode=='single256' else 192
   assert torch.cuda.get_device_properties(0).multi_processor_count>=grid
   self.obj=complete[(grid,)](X,self.xs,W,WS,self.p,NW,self.acc,self.count,self.y,self.lp,self.ls,Cos,Sin,self.kv,BW,BS,self.qp,self.qc,self.q,self.bar,self.health,BN=bn,GRID=grid,**kw)
  elif mode=='three':
   quant[(32,)](X,self.xp,self.xs,**kw)
   self.qa(False,True);self.qb(True,bn)
  elif mode=='two':self.qa(True,True);self.qb(True,bn)
  else:
   quant[(32,)](X,self.xp,self.xs,**kw)
   if mode=='separate':
    project_opt[(384,)](self.xp,self.xs,W,WS,self.p,BN=32,SPLIT=8,DIRECT=True,WAVES=2,BK=128,UNROLL=1,UPSCALE=True,REPLICATE=True,CACHE='.ca',SCALAR=True,PACKED=False,num_warps=2,enable_fp_fusion=False)
    reduce_opt[(2,)](self.p,NW,self.y,SPLIT=8,**kw)
   else:self.qa(False,False)
   quant[(8,)](self.y,self.lp,self.ls,**kw)
   self.qb(False,bn)
   norm_only[(64,)](self.qp,self.qn,**kw)
   norm_rope[(64,)](self.qn,Cos,Sin,self.q,DO_NORM=False,**kw)
   norm_rope[(1,)](self.y[1024:],Cos,Sin,self.kv,DO_NORM=False,**kw)
 def check_state(self):
  return dict(acc_zero=not torch.count_nonzero(self.acc).item(),count_zero=not torch.count_nonzero(self.count).item(),qb_count_zero=not torch.count_nonzero(self.qc).item(),health=self.health.item())
def metric(a,b):
 a=a.float();b=b.float();return dict(relative_rms=((a-b).square().mean()/b.square().mean().clamp_min(1e-30)).sqrt().item(),max_abs=(a-b).abs().max().item(),finite=torch.isfinite(a).all().item())
def tq(x):
 y=x.float().reshape(-1,128);a=y.abs().amax(-1).clamp_min(1e-10)/448.
 e=(((a.view(torch.int32)&0x7fffffff)+0x7fffff)>>23).clamp_max(254);s=(e<<23).view(torch.float32)
 q=(y/s[:,None]).to(torch.float8_e4m3fn)
 return q.flatten(),e.to(torch.uint8),(q.double()*s.double()[:,None]).flatten()
def rotary(x):
 y=x.clone().reshape(-1,512);z=y[:,448:].reshape(-1,32,2).clone()
 y[:,448::2]=z[:,:,0]*Cos-z[:,:,1]*Sin;y[:,449::2]=z[:,:,1]*Cos+z[:,:,0]*Sin
 return y.flatten().bfloat16()
wf=W.double()*torch.pow(2.,WS.double()-127).repeat_interleave(128,0).repeat_interleave(128,1)
bwf=BW.double()*torch.pow(2.,BS.double()-127).repeat_interleave(128,0).repeat_interleave(128,1)
def reference(x):
 xp,xs,xf=tq(x);a=(wf@xf).bfloat16().float()
 y=torch.cat([a[:1024]*torch.rsqrt(a[:1024].square().mean()+1e-6)*NW[:1024],a[1024:]*torch.rsqrt(a[1024:].square().mean()+1e-6)*NW[1024:]])
 lp,ls,lf=tq(y[:1024]);q=(bwf@lf).bfloat16().float();qn=q.reshape(64,512);qn=qn*torch.rsqrt(qn.square().mean(-1,keepdim=True)+1e-6)
 return dict(xp=xp,xs=xs,y=y,lp=lp,ls=ls,qp=q,q=rotary(qn),kv=rotary(y[1024:]))
