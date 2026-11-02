import sys,json
from pathlib import Path
root=Path(__file__).resolve().parent
attnroot=root.parent/'full-attention-native'
sys.path[:0]=[str(attnroot),str(attnroot/'prefix'),str(attnroot/'post-kernel'),str(root.parent/'ffn-c1/moe'),str(root.parent/'ffn-c1/shared-router')]
import full_setup as fa
import torch,aiter
import importlib.util
sp=importlib.util.spec_from_file_location("shared_native",root.parent/"ffn-c1/shared-router/native.py");mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(mod);NativeSharedRouter=mod.NativeSharedRouter
from ffn_kernel import ffn_body
class FFNState:
 def __init__(self):
  self.shared=NativeSharedRouter()
  src=(root.parent/'ffn-c1/moe/bench.py').read_text().split('ref=native()')[0]
  self.mo={};exec(compile(src,'moe_fixture_setup','exec'),self.mo)
  m=self.mo
  f=torch.load(root.parent/'ffn-c1/mhc/fixture.pt',weights_only=True)
  self.hfn,self.hfs,self.hfb,self.fn=[f[k].cuda().contiguous() for k in ['fn','scale','base','norm']]
  self.nm=fa.NativeMHC(self.hfn,self.hfs,self.hfb,self.fn)
  def z(sh,dtype=torch.float32):return torch.zeros(sh,device='cuda',dtype=dtype)
  self.r=z((1,4,4096),torch.bfloat16);self.out=torch.empty_like(self.r);self.mix=z(24);self.temp=z((128,24));self.raw=z(4096,torch.bfloat16)
  self.x=z((1,4096),torch.bfloat16);self.xq=z((1,4096),torch.float8_e4m3fn);self.xs=z((32,8),torch.uint8)
  self.mx=torch.empty_like(self.xq);self.ms=z((1,128),torch.uint8)
  self.gu=z((1,4096),torch.bfloat16);self.act=z((1,2048),torch.bfloat16);self.aq=z((1,2048),torch.float8_e4m3fn);self.asc=z((16,8),torch.uint8);self.so=torch.empty_like(self.x);self.logits=z((1,256),torch.bfloat16)
  self.hist=z(256,torch.int32);self.offs=z(256,torch.int32);self.pad=z(1,torch.int32);self.map=z(6,torch.int32);self.gather=z(6,torch.int32);self.scatter=z(6,torch.int32);self.gamma=z(6,torch.bfloat16)
  self.sum=torch.empty_like(self.x);self.counter=z(1,torch.uint32);self.health=z(1,torch.int32)
 def args(self,r=None):
  m=self.mo;sh=self.shared.shared
  return (self.r if r is None else r,self.out,self.hfn,self.hfs,self.hfb,self.fn,self.mix,self.temp,self.raw,self.x,self.xq,self.xs,self.mx,self.ms,self.gu,self.act,self.aq,self.asc,self.so,self.shared.hash_gate.weight,self.logits,
   sh.gate_up_proj.weight,sh.gate_up_proj.weight_scale.view(torch.uint8),sh.w2.weight,sh.w2.weight_scale.view(torch.uint8),m['a'],m['b'],m['c'],m['d'],m['i'],m['is_'],m['y'],self.hist,self.offs,self.pad,self.map,self.gather,self.scatter,self.gamma,m['count'],m['z'],self.sum,self.counter,self.health)
 def meta(self):
  return dict(zip(['W1S','S1S','W2S','S2S'],[self.mo[k].stride() for k in ['a','b','c','d']]))
 def fused(self,r=None,pre=True):
  self.obj=ffn_body[(256,)](*self.args(r),**self.meta(),DO_PRE=pre,num_warps=4,enable_fp_fusion=False)
  return self.out
 def native_x(self,x):
  m=self.mo
  logits=self.shared.hash_gate(x)
  rd,gather,scatter=m['routing'](logits,6,score_mode='sqrtsoftplus',renorm=True,routed_scaling_factor=1.5)
  routed=m['_fused_experts_silu_gugu'](x,m['w1'],m['w2'],rd,gather,scatter,m['s1'],m['s2'],'GFX1250_SCALE','GFX1250_SCALE',swiglu_limit=10.,preshuffled=True)
  shared=self.shared.shared(x)
  self.native_logits=logits;self.rd=rd;self.native_routed=routed;self.native_shared=shared
  return routed+shared
 def native(self,r):
  rsh=aiter.mhc_res_shuffle(r.view(1,4,4096));x,p,c=self.nm(rsh);y=self.native_x(x)
  out=torch.empty_like(r.view(1,4,4096));aiter.mhc_post(out,y,rsh,p.view(1,4,1),c,res_preshuffle=True)
  self.native_norm=x
  return out
if __name__=='__main__':
 f=FFNState();f.r.copy_(fa.ps.R.view_as(f.r));ref=f.native(f.r).clone();out=f.fused();torch.cuda.synchronize()
 def err(a,b):return float(((a.float()-b.float()).square().mean()/b.float().square().mean().clamp_min(1.e-30)).sqrt())
 print({'full_error':err(out,ref),'norm':err(f.x,f.native_norm),'logits':err(f.logits,f.native_logits),'shared':err(f.so,f.native_shared),'routed':err(f.mo['z'],f.native_routed),'health':f.health.item(),'moehealth':f.mo['health'].item(),'ids':f.map.tolist(),'refids':f.rd.topk_ids.tolist(),'gamma':f.gamma.tolist(),'refgamma':f.rd.gate_scal.tolist(),'resources':[f.obj.n_regs,f.obj.metadata.shared,f.obj.n_spills]},flush=True)
 (root/'ffn.amdgcn').write_text(f.obj.asm['amdgcn'])
