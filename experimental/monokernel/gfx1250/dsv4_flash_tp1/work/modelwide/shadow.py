"""Eager-only numerical localization; returns native outputs, never benchmark this mode."""
import json,os
from pathlib import Path
import torch,aiter

def shadow(obj,residual,positions,fc):
 from atom.models.deepseek_v4 import HCState
 b=obj.block
 fused=obj.execute(residual,positions,fc.attn_metadata).clone()
 captured={};saved={}
 for name in ['_qk_norm_rope','_sparse_attention','_attn_post']:
  original=getattr(b.attn,name);saved[name]=original
  def wrap(*args,_name=name,_fn=original,**kw):
   out=_fn(*args,**kw);captured[_name]=out
   return out
  setattr(b.attn,name,wrap)
 handles=[]
 handles.append(b.attn.register_forward_pre_hook(lambda mod,args:captured.update(attn_input=args[0])))
 handles.append(b.ffn.register_forward_pre_hook(lambda mod,args:captured.update(ffn_input=args[0])))
 handles.append(b.ffn.gate.register_forward_hook(lambda mod,args,out:captured.update(logits=out)))
 old=b._experimental_full_layer;b._experimental_full_layer=None
 try:
  result=b.forward(HCState(residual,None,None,None,True),positions)
  plain=b.hc_post(result.x_prev,result.residual,result.post_mix,result.comb_mix,res_preshuffle=result.res_preshuffle)
  native=aiter.mhc_res_shuffle(plain)
 finally:
  b._experimental_full_layer=old
  for name,fn in saved.items():setattr(b.attn,name,fn)
  for h in handles:h.remove()
 def err(a,c):
  a=a.float().flatten();c=c.float().flatten()
  return dict(rms=((a-c).square().mean()/c.square().mean().clamp_min(1e-30)).sqrt().item(),max_abs=(a-c).abs().max().item(),exact=torch.equal(a,c),finite=torch.isfinite(a).all().item())
 A,F=obj.weights.args(obj.state,residual)
 report=dict(layer=b.layer_id,ratio=obj.weights.ratio,position=positions.item(),output=err(fused,native))
 report['same_input_ffn']=err(F[41],b.ffn(F[9].clone()))
 report['same_input_post']=err(A[51],saved['_attn_post'](A[38].reshape(1,64,512),positions))
 md=fc.attn_metadata;r=obj.weights.ratio;kind={0:'swa',4:'csa',128:'hca'}[r]
 ptr=getattr(md,'kv_indptr_'+kind);idx=getattr(md,'kv_indices_'+kind)
 from atom.model_ops.v4_kernels.paged_decode import sparse_attn_v4_paged_decode
 report['same_query_attention']=err(A[38],sparse_attn_v4_paged_decode(A[28].reshape(1,64,512),b.attn.unified_kv,idx,ptr,A[34],512**-.5))
 report['current_kv']=err(A[23],b.attn.unified_kv[md.swa_dest_rows[r].long()].reshape(-1))
 for name,ours in [('attn_input',A[10]),('ffn_input',F[9]),('logits',F[20]),('_sparse_attention',A[38]),('_attn_post',A[51])]:
  if name in captured:report[name]=err(ours,captured[name])
 qkn=captured.get('_qk_norm_rope')
 if qkn is not None and qkn.q_sa is not None:report['query']=err(A[28],qkn.q_sa)
 with open('/work/modelwide/shadow-validation'+os.getenv('SHADOW_TAG','')+'.jsonl','a') as f:f.write(json.dumps(report)+'\n')
 print('SHADOW',json.dumps(report),flush=True)
 return native
