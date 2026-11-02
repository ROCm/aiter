from harness import *
from combined import complete
out=root/'smoke';out.mkdir(exist_ok=True)
assert torch.cuda.get_device_properties(0).multi_processor_count>=256
r=reference(X);rows=[]
for style,sem,u in [(0,False,1),(1,False,1),(2,False,1),(0,False,4),(1,False,4),(2,False,4),(1,True,4),(2,True,4)]:
 s=State();ready=torch.zeros(1,device='cuda',dtype=torch.int32);epoch=torch.zeros(256,device='cuda',dtype=torch.int32)
 args=(X,s.xs,W,WS,s.p,NW,s.acc,s.count,s.y,s.lp,s.ls,Cos,Sin,s.kv,BW,BS,s.qp,s.qc,s.q,s.bar,s.health)
 opts=dict(BN=128,GRID=256,STYLE=style,U=u,SEM=sem,Ready=ready,Epoch=epoch,**kw)
 name=f'style{style}_sem{int(sem)}_u{u}'
 obj=complete.warmup(*args,**opts,grid=(256,))
 obj._init_handles()
 res=dict(vgpr=obj.n_regs,spills=obj.n_spills,lds=obj.metadata.shared)
 print('COMPILED',name,res,flush=True)
 # Four waves: at most1024VGPR/wave. Retain low LDS and zero spills;
 # kernel resource feasibility is then additionally checked by HIP dispatch.
 assert obj.n_regs<=512 and obj.n_spills==0 and obj.metadata.shared<=65536
 (out/(name+'.amdgcn')).write_text(obj.asm['amdgcn'])
 complete[(256,)](*args,**opts);torch.cuda.synchronize()
 row=dict(name=name,resources=res,q=metric(s.q,r['q']),kv=metric(s.kv,r['kv']),state=s.check_state(),ready=ready.item(),epoch_min=epoch.min().item(),epoch_max=epoch.max().item())
 print(row,flush=True);rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2))
 assert row['state']['health']==0 and row['q']['relative_rms']<.001 and row['kv']['relative_rms']<.001
