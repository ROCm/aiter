from full_setup import *
import math
f=FullState();results=[]
with torch.inference_mode():
 for rnd in [1,2]:
  rows=[]
  for pos in [1024,1536,2047]:
   f.setpos(pos)
   for i in range(137):
    r=ps.f2['inputs'][i].cuda().contiguous() if i<9 else torch.randn((4,4096),device='cuda',generator=torch.Generator(device='cuda').manual_seed(171000+i)).bfloat16()
    rsh=aiter.mhc_res_shuffle(r.view(1,4,4096))
    expected=f.native(rsh).clone();ob=f.native_y.clone()
    f.rout.fill_(float('nan'));f.oy.fill_(float('nan'));f.s.attn.out.fill_(float('nan'))
    fused=f.fused(rsh).clone();fob=f.oy.clone();torch.cuda.synchronize()
    staged=f.staged(rsh).clone();torch.cuda.synchronize()
    rows.append({'case':i,'position':pos,'native_residual_rms':ps.err(fused.flatten(),expected.flatten()),'native_ob_rms':ps.err(fob.flatten(),ob.flatten()),'staged_rms':ps.err(fused.flatten(),staged.flatten()),'staged_bitwise':torch.equal(fused,staged),'finite':bool(torch.isfinite(fused).all()),'health':f.check()})
   print(rnd,pos,'done',flush=True)
  res={'round':rnd,'count':len(rows),'max_native_residual_rms':max(x['native_residual_rms'] for x in rows),'native_failures':sum(x['native_residual_rms']>.001 for x in rows),'max_native_ob_rms':max(x['native_ob_rms'] for x in rows),'staged_nonexact':sum(not x['staged_bitwise'] for x in rows),'unhealthy':sum(not x['health'] for x in rows),'nonfinite':sum(not x['finite'] for x in rows),'rows':rows}
  results.append(res);(root/'full-validation.json').write_text(json.dumps(results,indent=2));print({k:v for k,v in res.items() if k!='rows'},flush=True)
 k=f.obj;(root/'full-resources.json').write_text(json.dumps({'vgpr':k.n_regs,'lds':k.metadata.shared,'spills':k.n_spills},indent=2));(root/'full.amdgcn').write_text(k.asm['amdgcn']);(root/'full.llir').write_text(k.asm['llir'])
