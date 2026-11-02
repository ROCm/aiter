"""Independent torch equations for the isolated full-layer Gluon experiment.

Not a substitute for ATOM live-state or task-level validation. Uses the same
packed checkpoint, but no Gluon helpers or intermediate outputs as inputs.
"""
import json
import torch
import torch.nn.functional as F


def quant(x,group):
 v=x.float().reshape(-1,group)
 a=v.abs().amax(-1,keepdim=True).clamp_min(1.e-10)/448.
 exponent=((a.view(torch.int32)&0x7fffffff)+0x7fffff)>>23
 scale=(exponent.clamp_max(254)<<23).view(torch.float32)
 return ((v/scale).to(torch.float8_e4m3fn).float()*scale).reshape(x.shape)


def fp8(w,s):
 return w.float()*s.view(torch.float8_e8m0fnu).float().repeat_interleave(128,0).repeat_interleave(128,1)


def fp4(w,s):
 b=w.to(torch.uint8)
 v=torch.stack([b&15,b>>4],dim=-1).flatten(-2).long()
 lut=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.],device=w.device)
 return lut[v]*s.view(torch.float8_e8m0fnu).float().repeat_interleave(32,-1)


def norm(x,w=None):
 y=x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1.e-6)
 if w is not None:y=y*w.float()
 return y.bfloat16()


def mhc(r,w,s,b,norm_weight):
 flat=r.flatten().float()
 m=(w.float()@flat)*torch.rsqrt(flat.square().mean()+1.e-6)
 pre=torch.sigmoid(m[:4]*s[0]+b[:4])+1.e-6
 post=2*torch.sigmoid(m[4:8]*s[1]+b[4:8])
 comb=(m[8:]*s[2]+b[8:]).reshape(4,4).softmax(-1)+1.e-6
 comb=comb/(comb.sum(0,keepdim=True)+1.e-6)
 for _ in range(19):
  comb=comb/(comb.sum(1,keepdim=True)+1.e-6)
  comb=comb/(comb.sum(0,keepdim=True)+1.e-6)
 x=(pre[:,None]*r.float().reshape(4,4096)).sum(0).bfloat16()
 return norm(x,norm_weight),torch.cat([pre,post,comb.flatten()])


def post(x,r,m):
 v=x.float()[None,:]*m[4:8,None]
 for i in range(4):v=v+r.reshape(4,4096)[i].float()[None,:]*m[8:].reshape(4,4)[i,:,None]
 return v.bfloat16()


def rope(x,c,s,inverse=False):
 x=x.reshape(-1,512).float()
 a=x[:,-64:].reshape(-1,32,2)
 sign=-1 if inverse else 1
 y=torch.stack([a[...,0]*c-sign*a[...,1]*s,a[...,1]*c+sign*a[...,0]*s],-1)
 return torch.cat([x[:,:448],y.flatten(-2)],-1).bfloat16()


def act(g,u):
 return (F.silu(g.float().clamp_max(10))*u.float().clamp(-10,10)).bfloat16()


def check(d,root):
 expected={}
 def put(k,v):expected[k]=v;return v
 def mv(x,w,s):return (fp8(d[w],d[s])@x.float()).bfloat16()
 r=d['R'].clone()
 x,ma=mhc(r,d['HAF'],d['HAS'],d['HAB'],d['AN']);put('X',x);put('MixA',ma)
 xq=put('XQ',quant(x,128))
 ql=norm(mv(xq,'QA','QAS'),d['QN']);put('QL',ql)
 kv=norm(mv(xq,'KVW','KVS'),d['KN']);put('KV',kv)
 qlq=put('QLQ',quant(ql,128))
 q=put('Q',norm(mv(qlq,'QB','QBS').reshape(64,512)))
 qr=put('QRot',rope(q,d['Cos'],d['Sin']))
 kr=put('KVRot',rope(kv,d['Cos'],d['Sin']))
 cache=d['Cache'].clone();cache[-1]=kr.flatten()
 logits=qr.float()@cache.float().T*(512**-.5)
 prob=torch.cat([logits,d['Sink'][:,None]],-1).softmax(-1)[:,:128]
 ao=put('AO',(prob@cache.float()).bfloat16())
 ar=put('AR',rope(ao,d['Cos'],d['Sin'],True))
 aq=put('AQ',quant(ar,128))
 ol=put('OL',torch.einsum('gnk,gk->gn',fp8(d['OA'],d['OAS']).reshape(8,1024,4096),aq.reshape(8,4096)).flatten().bfloat16())
 olq=put('OLQ',quant(ol,128))
 attn=put('Attn',mv(olq,'OB','OBS'))
 r2=put('R2',post(attn,r,ma))
 ff,mf=mhc(r2,d['HFN'],d['HFS'],d['HFB'],d['FN']);put('FF',ff);put('MixF',mf)
 fq=put('FFQ',quant(ff,128));fmx=put('FFMX',quant(ff,32))
 logits=put('Logits',(d['Gate'].float()@ff.float()).bfloat16())
 ids=put('Ids',d['Tid'][d['Token'].item()].int())
 weights=F.softplus(logits[ids.long()].float()).sqrt()
 weights=put('Gamma',1.5*weights/weights.sum())
 sg=put('SG',mv(fq,'SW1','SS1'));su=put('SU',mv(fq,'SW3','SS3'))
 sa=put('SA',act(sg,su));saq=put('SAQ',quant(sa,128));so=put('SO',mv(saq,'SW2','SS2'))
 eg=[];eu=[];ea=[];eaq=[];eo=[]
 for e in ids.tolist():
  g=(fp4(d['EW1'][e],d['ES1'][e])@fmx).bfloat16()
  u=(fp4(d['EW3'][e],d['ES3'][e])@fmx).bfloat16()
  a=act(g,u);aq=quant(a,32)
  o=(fp4(d['EW2'][e],d['ES2'][e])@aq).bfloat16()
  eg.append(g);eu.append(u);ea.append(a);eaq.append(aq);eo.append(o)
 for key,values in [('EG',eg),('EU',eu),('EA',ea),('EAQ',eaq),('EO',eo)]:put(key,torch.stack(values))
 summed=torch.zeros(4096,device='cuda')
 for i in range(6):summed+=eo[i].float()*weights[i]
 summed=put('Sum',(summed.bfloat16().float()+so.float()).bfloat16())
 out=put('Out',post(summed,r2,mf))
 result={}
 for key,ref in expected.items():
  actual=d[key].flatten();ref=ref.flatten()
  diff=(actual.float()-ref.float()).abs()
  result[key]={'exact':bool(torch.equal(actual,ref)), 'max_abs':diff.max().item(),'rms_relative':(diff.square().mean().sqrt()/ref.float().square().mean().sqrt().clamp_min(1.e-20)).item(),'nonfinite':int((~torch.isfinite(actual)).sum())}
 (root/'reference-comparison.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(result,indent=2),flush=True)
 assert all(v['nonfinite']==0 for v in result.values())
 assert result['Ids']['exact']
 assert result['Out']['rms_relative']<.01, 'Full-layer reference mismatch; not validated for serving'
 print('Independent isolated full-layer reference: output relative RMS <1%; not ATOM exact equivalence.',flush=True)
