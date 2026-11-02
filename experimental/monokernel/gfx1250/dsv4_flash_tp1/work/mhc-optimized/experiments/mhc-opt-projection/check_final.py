exec(open(__file__.replace('check_final.py','tune_projection.py')).read().split('entries=[]')[0])
from projection import final_region
Wh=W.bfloat16();Wp=torch.cat([Wh,(W-Wh.float()).bfloat16()],0).contiguous()
variants=[('original',False,0),('prefetch',False,1),('localprefetch',False,2),('packed',True,0),('packedprefetch',True,1)]
states=[state() for _ in variants];maximum={n:0. for n,*_ in variants}
for c in range(64):
 gen=torch.Generator(device='cuda').manual_seed(c+786)
 rr=f['inputs'][c].cuda() if c<9 else (torch.randn((4,4096),device='cuda',generator=gen)*(10**((c%5)-2))).bfloat16()
 mhc_region[(256,)](rr,W,S,B,AN,*ref,**kw)
 for (name,pk,pf),st in zip(variants,states):
  final_region[(256,)](rr,Wp if pk else W,S,B,AN,*st,PACKED=pk,PREFETCH=pf,**kw)
  torch.cuda.synchronize()
  for field in [0,2,3]:
   err=((st[field].float()-ref[field].float()).square().mean()/ref[field].float().square().mean().clamp_min(1e-30)).sqrt().item()
   maximum[name]=max(maximum[name],err);assert err<.001,(c,name,field,err)
  assert st[-1].item()==0
(root/'final-validation.json').write_text(json.dumps(dict(cases=64,checks=64*5*3,max_errors=maximum),indent=2));print(maximum)
