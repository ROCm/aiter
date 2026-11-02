from harness import *
out=root/'smoke';out.mkdir(exist_ok=True)
r=reference(X);rows=[]
for mode in ['separate','qa_fused','two','single']:
 s=State();s.run(mode);torch.cuda.synchronize()
 row=dict(mode=mode,metrics={n:metric(getattr(s,n),r[n]) for n in ['y','qp','q','kv']},state=s.check_state())
 o=s.obj
 row['resources']=dict(vgpr=o.n_regs,spills=o.n_spills,lds=o.metadata.shared)
 (out/(mode+'.amdgcn')).write_text(o.asm['amdgcn'])
 rows.append(row);print(row,flush=True);(out/'results.json').write_text(json.dumps(rows,indent=2))
