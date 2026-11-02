from common import *
rows=[]
for fused in [False,True]:
 c=makecache();idx=torch.arange(128,device='cuda',dtype=torch.int32)
 s=QKVAttentionRegion(W,WS,NW,BW,BS,c,idx,Sink,fused=fused)
 o=s(X,Cos,Sin);torch.cuda.synchronize();r=reference(s.q,c,idx,Sink);e=error(o,r)
 obj=s.obj if fused else s.attn.obj
 row=dict(fused=fused,error=e,health=s.health.item(),vgpr=obj.n_regs,lds=obj.metadata.shared,spills=obj.n_spills);print(row,flush=True);rows.append(row)
 (root/f'chain-{fused}.amdgcn').write_text(obj.asm['amdgcn']);assert e<.001 and s.check()
(root/'chain-smoke.json').write_text(json.dumps(rows,indent=2))
