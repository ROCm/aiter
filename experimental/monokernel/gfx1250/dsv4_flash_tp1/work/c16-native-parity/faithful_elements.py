from common import *
from faithful_ops import *
kind=sys.argv[1]
if kind!='inverse':
 exec(Path('native_qkv_cache.py').read_text().split('rr=native();')[0])
 from atom.model_ops.linear import per_1x128_e8m0_quant
 aa=n.qa(x);qr,qs=n.norm(aa[:,:1024]);raw=n.qb(qr,x_scale=qs)
 if kind=='quant':
  rf=lambda:per_1x128_e8m0_quant(x,torch.float8_e4m3fn,True)
  fn=lambda:quant_entry[(32,)](x,xq,xs,16,4096,num_warps=2,enable_fp_fusion=False)
  rr=rf();obj=fn();torch.cuda.synchronize();val=dict(q=err(xq,rr[0]),s=err(xs.reshape(32,16).T,rr[1].view(torch.uint8).reshape(-1).reshape(-1,16).T))
 elif kind=='norm':
  rf=lambda:n.norm(aa[:,:1024]);fn=lambda:norm_entry[(16,)](aa,n.norm.weight,q,s,16,**kw)
  rr=rf();obj=fn();torch.cuda.synchronize();val=dict(q=err(q,rr[0]),s=err(s.reshape(8,16).T,rr[1].view(torch.uint8).reshape(-1).reshape(-1,16).T))
 else:
  rf=lambda:qk_norm_rope_maybe_quant(raw,aa[:,1024:],n.nw,n.cos,n.sin,pos,64,512,64,1e-6,swa_kv=cache,swa_dest_rows=dest,batch_id_per_q_token=bid)
  fn=lambda:rope_entry[(260,)](aa,raw,n.nw,cos,sin,qo,kv,cache2,dest,bid,16,16*128,**kw)
  rr=rf();obj=fn();torch.cuda.synchronize();val=dict(q=err(qo,rr.q_sa),kv=err(kv,rr.kv),cache=err(cache2,cache))
else:
 exec(Path('native_output.py').read_text().split('rr=n(o);')[0])
 import atom.models.deepseek_v4 as dv
 orig=dv.inverse_rope_group_quant;captured={}
 def hook(*args,**kwargs):
  captured['args']=args;captured['kwargs']=kwargs
  return orig(*args,**kwargs)
 dv.inverse_rope_group_quant=hook;n.oa(o);dv.inverse_rope_group_quant=orig
 rf=lambda:orig(*captured['args'],**captured['kwargs'])
 fn=lambda:inverse_entry[(512,)](o,c,sn,a[0],a[1],16,num_warps=1,enable_fp_fusion=False)
 rr=rf();obj=fn();torch.cuda.synchronize();print('REFERENCE',[(v.shape,v.stride()) for v in rr],flush=True)
 val=dict(q=err(a[0],rr[0]),s=err(a[1],rr[1].view(torch.uint8)))
print('VALIDATION',val,flush=True)
(ROOT/(kind+'.amdgcn')).write_text(obj.asm['amdgcn'])
(ROOT/(kind+'_resource.json')).write_text(json.dumps(dict(regs=obj.n_regs,spills=obj.n_spills,lds=obj.metadata.shared)))
bench(kind,{'native':rf,'faithful':fn},val)
