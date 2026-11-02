from common import *
from faithful_gemm import entry
kind=sys.argv[1]
# Reuse saved model tensors, not prior fused outputs, for standalone controls.
if kind in ['qa','qb']:
 exec(Path('native_qkv_cache.py').read_text().split('rr=native();')[0])
 aa=n.qa(x);qq,ss=n.norm(aa[:,:1024]);native()
 if kind=='qa':
  from atom.model_ops.linear import per_1x128_e8m0_quant
  xx,ss=per_1x128_e8m0_quant(x,quant_dtype=torch.float8_e4m3fn,transpose_scale=True)
  lin=n.qa;ww=wp;sw=ws;nn=1536;kk=4096;bm=16;bn=32;bk=1024;nb=4;nwarps=2
 else:xx=qq;lin=n.qb;ww=qbp;sw=qbs;nn=32768;kk=1024;bm=16;bn=128;bk=512;nb=2;nwarps=4
 rf=lambda:lin(xx,x_scale=ss)
 yy=torch.empty((16,nn),device='cuda',dtype=torch.bfloat16)
 fn=lambda:entry[(nn//bn,)](xx,ss.view(torch.uint8),ww,sw,yy,16,nn,kk,bm,bn,bk,nb,num_warps=nwarps,enable_fp_fusion=False)
else:
 exec(Path('native_output.py').read_text().split('rr=n(o);')[0])
 from atom.models.deepseek_v4 import batched_gemm_a8w8_mxscale_bpreshuffle as bmm
 if kind=='oa':
  from atom.models.deepseek_v4 import inverse_rope_group_quant
  print('inverse signature',inverse_rope_group_quant,flush=True)
  # Exact native intermediate quantization is produced by wo_a_grouped_lora hook below.
  oa=n.oa(o)
  import atom.models.deepseek_v4 as dv
  orig=dv.batched_gemm_a8w8_mxscale_bpreshuffle
  captured={}
  def hook(*args,**kwargs):
   captured['args']=args;captured['kwargs']=kwargs
   return orig(*args,**kwargs)
  dv.batched_gemm_a8w8_mxscale_bpreshuffle=hook
  n.oa(o);dv.batched_gemm_a8w8_mxscale_bpreshuffle=orig
  xx,ww,ss,sw=captured['args'][:4]
  yy=a[3];rf=lambda:orig(*captured['args'],**captured['kwargs'])
  fn=lambda:entry[(256,)](xx,ss.view(torch.uint8),ww,sw.view(torch.uint8),yy,16,8192,4096,32,32,1024,4,True,**kw)
 else:
  xx=n.oa(o)
  from atom.model_ops.linear import per_1x128_e8m0_quant
  xx,ss=per_1x128_e8m0_quant(xx,quant_dtype=torch.float8_e4m3fn,transpose_scale=True)
  rf=lambda:n.wo_b(xx,x_scale=ss);yy=a[7]
  fn=lambda:entry[(64,)](xx,ss.view(torch.uint8),wbp,sb,yy,16,4096,8192,16,64,512,4,**kw)
rr=rf();obj=fn();torch.cuda.synchronize()
val=err(yy,rr);print('VALIDATION',val,flush=True)
(ROOT/('part_'+kind+'.amdgcn')).write_text(obj.asm['amdgcn'])
(ROOT/('part_'+kind+'_resource.json')).write_text(json.dumps(dict(regs=obj.n_regs,spills=obj.n_spills,lds=obj.metadata.shared)))
assert val['finite'] and val['relative_rms']<1e-6,val
bench('part_'+kind,{'native':rf,'faithful':fn},val)
