import sys,json
from pathlib import Path
root=Path(__file__).resolve().parent
sys.path[:0]=[str(root/'output-native'),'/work/atom-actual-microbench/qkv','/work/atom-actual-microbench/mhc','/app/aiter','/app/ATOM']
from native import NativeQKV
from native_api import NativeMHC
from native_output import NativeOutput
sys.path[:0]=[str(root),str(root/'prefix'),str(root/'output-kernel'),str(root/'post-kernel')]
import setup as ps
import torch,aiter
from atom.model_ops.v4_kernels.paged_decode import sparse_attn_v4_paged_decode
from full_kernel import full_attention
from output_region import output_kernel
from post import post
from layout import unshuffle_kernel
class FullState:
 def __init__(self):
  self.s=ps.State(128);s=self.s
  f=torch.load(root/'output-native/fixture.pt',weights_only=True)
  self.oa=f['OA'].cuda().contiguous();self.oas=f['OAS'].cuda().view(torch.uint8).contiguous();self.ob=f['OB'].cuda().contiguous();self.obs=f['OBS'].cuda().view(torch.uint8).contiguous()
  def z(sh,dtype=torch.float32):return torch.zeros(sh,device='cuda',dtype=dtype)
  self.rplain=z((4,4096),torch.bfloat16);self.rout=z((4,4096),torch.bfloat16)
  self.oxq=z(32768,torch.float8_e4m3fn);self.oxs=z(256,torch.uint8);self.oap=z((4,8192));self.oay=z(8192,torch.bfloat16);self.oaq=z(8192,torch.float8_e4m3fn);self.oasc=z(64,torch.uint8);self.obp=z((8,4096));self.oy=z(4096,torch.bfloat16)
  self.nm=NativeMHC(ps.HAF,ps.HAS,ps.HAB,ps.AN);self.nq=NativeQKV(cache=s.attn.cache.clone(),dest=127);self.no=NativeOutput();self.pos=1024
  self.idx=z(16384,torch.int32);self.idx[:128]=torch.arange(128,device='cuda',dtype=torch.int32);self.ptr=torch.tensor([0,128],device='cuda',dtype=torch.int32)
  self.setpos(1024)
 def setpos(self,pos):
  self.pos=pos;self.nq.pos.fill_(pos);self.no.positions.fill_(pos)
  self.cos=self.nq.cos[pos].flatten().float().contiguous();self.sin=self.nq.sin[pos].flatten().float().contiguous()
  ps.Cos=self.cos;ps.Sin=self.sin
 def native(self,rsh,do_post=True):
  s=self.s;x,p,c=self.nm(rsh);q,kv=self.nq(x)
  o=sparse_attn_v4_paged_decode(q,self.nq.cache,self.idx,self.ptr,s.attn.sink,512**-.5)
  y=self.no(o)
  self.native_x=x;self.native_q=q;self.native_kv=kv;self.native_attn=o;self.native_y=y
  return self.no.post(y,rsh,p,c,res_preshuffle=True) if do_post else y
 def output_args(self):
  s=self.s
  return (s.attn.out,self.cos,self.sin,self.oa,self.oas,self.ob,self.obs,self.oxq,self.oxs,self.oap,self.oay,self.oaq,self.oasc,self.obp,self.oy)
 def fused(self,rsh,do_post=True):
  s=self.s;a=s.attn
  self.obj=full_attention[(256,)](rsh,self.rplain,self.rout,ps.HAF,ps.HAS,ps.HAB,ps.AN,s.mix,s.temp,s.raw,s.x,s.ls,s.W,s.WS,s.p,s.NW,s.acc,s.count,s.y,s.lp,s.ls,self.cos,self.sin,s.kv,s.QB,s.QBS,s.qp,s.qc,s.q,s.counter,s.health,
   a.cache,a.indices,a.indptr,a.sink,a.m,a.l,a.a,a.out,a.counts,
   self.oa,self.oas,self.ob,self.obs,self.oxq,self.oxs,self.oap,self.oay,self.oaq,self.oasc,self.obp,self.oy,
   T=128,PAGES=128,WRITE_SLOT=127,SPLITS=4,SHUFFLED_INPUT=True,DO_POST=do_post,**ps.kw)
  return self.rout if do_post else self.oy
 def staged(self,rsh,do_post=True):
  s=self.s
  unshuffle_kernel[(256,)](rsh,self.rplain,**ps.kw)
  s.run(self.rplain,'opt_both')
  output_kernel[(256,)](*self.output_args(),s.counter,s.health,**ps.kw)
  if do_post:post[(256,)](self.rplain,self.oy,s.mix,self.rout,B=128,SHUFFLED=False,FMA=2,**ps.kw)
  return self.rout if do_post else self.oy
 def check(self):return self.s.check()
if __name__=='__main__':
 f=FullState();r=aiter.mhc_res_shuffle(ps.R.view(1,4,4096))
 with torch.inference_mode():
  ref=f.native(r).clone();our=f.fused(r).clone();torch.cuda.synchronize();split=f.staged(r).clone();torch.cuda.synchronize()
  result={'native_rms':ps.err(our.flatten(),ref.flatten()),'staged_rms':ps.err(our.flatten(),split.flatten()),'staged_bitwise':torch.equal(our,split),'health':f.check(),'resources':{'vgpr':f.obj.n_regs,'lds':f.obj.metadata.shared,'spills':f.obj.n_spills}}
  print(result,flush=True);(root/'smoke.json').write_text(json.dumps(result,indent=2));(root/'full.amdgcn').write_text(f.obj.asm['amdgcn'])
