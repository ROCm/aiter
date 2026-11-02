import torch,json
from pathlib import Path
from attention import attention_kernel
p=Path(__file__).resolve().parent
assert torch.cuda.get_device_properties(0).gcnArchName.startswith('gfx1250')
torch.manual_seed(3)
q=torch.randn((64,512),device='cuda',dtype=torch.bfloat16);kv=torch.randn(512,device='cuda',dtype=torch.bfloat16)
c=torch.randn((128,512),device='cuda',dtype=torch.bfloat16);idx=torch.arange(128,device='cuda',dtype=torch.int32);ip=torch.tensor([0,128],device='cuda',dtype=torch.int32);sink=torch.randn(64,device='cuda')
c[-1].copy_(kv);scores=q.double()@c.double().T/(512**.5);mx=torch.maximum(scores.max(1).values,sink.double());probs=(scores-mx[:,None]).exp();ref=(probs@c.double()/(probs.sum(1)+(sink.double()-mx).exp())[:,None]).bfloat16().float()
rows=[]
for s in [1,2,4]:
 m=torch.zeros((s,64),device='cuda');l=torch.zeros_like(m);a=torch.zeros((s,64,512),device='cuda');out=torch.empty_like(q);ct=torch.zeros(4,device='cuda',dtype=torch.int32);bar=torch.zeros(1,device='cuda',dtype=torch.int32);health=torch.zeros_like(bar)
 obj=attention_kernel[(4*s,)](q,kv,c,idx,ip,sink,m,l,a,out,ct,bar,health,T=128,PAGES=128,WRITE_SLOT=127,SPLITS=s,GRID=4*s,num_warps=4,enable_fp_fusion=False)
 torch.cuda.synchronize();err=((out.float()-ref).square().mean()/ref.square().mean()).sqrt().item()
 row=dict(split=s,error=err,health=health.item(),count=ct.tolist(),vgpr=obj.n_regs,lds=obj.metadata.shared,spills=obj.n_spills);rows.append(row);print(row,flush=True)
 (p/f'smoke-s{s}.amdgcn').write_text(obj.asm['amdgcn'])
 assert err<.001 and not health.item() and not ct.any()
(p/'smoke.json').write_text(json.dumps(rows,indent=2))
