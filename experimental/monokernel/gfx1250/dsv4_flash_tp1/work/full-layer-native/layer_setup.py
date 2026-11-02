from ffn_setup import *
from layer_kernel import full_layer
class LayerState:
 def __init__(self):
  self.a=fa.FullState();self.f=FFNState();self.r=aiter.mhc_res_shuffle(fa.ps.R.view(1,4,4096));self.f.r=self.a.rout.view(1,4,4096)
 def attn_args(self):
  f=self.a;s=f.s;a=s.attn;p=fa.ps
  return (self.r,f.rplain,f.rout,p.HAF,p.HAS,p.HAB,p.AN,s.mix,s.temp,s.raw,s.x,s.ls,s.W,s.WS,s.p,s.NW,s.acc,s.count,s.y,s.lp,s.ls,f.cos,f.sin,s.kv,s.QB,s.QBS,s.qp,s.qc,s.q,s.counter,s.health,
   a.cache,a.indices,a.indptr,a.sink,a.m,a.l,a.a,a.out,a.counts,
   f.oa,f.oas,f.ob,f.obs,f.oxq,f.oxs,f.oap,f.oay,f.oaq,f.oasc,f.obp,f.oy)
 def fused(self):
  self.obj=full_layer[(256,)](self.attn_args(),self.f.args(),**self.f.meta(),num_warps=4,enable_fp_fusion=False)
  return self.f.out
 def staged(self):
  self.a.fused(self.r);return self.f.fused()
 def native(self):
  r=self.a.native(self.r);return self.f.native(r)
 def health(self):return {'attention':self.a.check(),'ffn':self.f.health.item(),'moe':self.f.mo['health'].item()}
if __name__=='__main__':
 f=LayerState();ref=f.native().clone();out=f.fused().clone();torch.cuda.synchronize();split=f.staged().clone();torch.cuda.synchronize()
 def err(a,b):return float(((a.float()-b.float()).square().mean()/b.float().square().mean().clamp_min(1.e-30)).sqrt())
 result={'native_error':err(out,ref),'staged_error':err(out,split),'staged_equal':torch.equal(out,split),'health':f.health(),'resources':{'vgpr':f.obj.n_regs,'lds':f.obj.metadata.shared,'spills':f.obj.n_spills}}
 print(result,flush=True);(root/'smoke.json').write_text(json.dumps(result,indent=2));(root/'full.amdgcn').write_text(f.obj.asm['amdgcn'])
