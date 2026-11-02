from publication_diagnostic import publish_test
from combined import *
import torch,json
from pathlib import Path

@g.jit
def once(Payload,Ready,Health,Nonce,STYLE:gl.constexpr):
    pid=gl.program_id(0)
    i=gl.arange(0,512,layout=gl.BlockedLayout([1],[32],[4],[0]))
    if pid==0:
        dev_store(Nonce*100000+i,Payload,i)
        gl.inline_asm_elementwise('s_wait_storecnt 0\n s_mov_b32 $0,0',constraints='=s,~{memory}',args=[],dtype=gl.int32,is_pure=False,pack=1)
        gl.barrier()
        gl.atomic_add(Ready,1,sem='relaxed',scope='gpu')
    else:
        wait_target(Ready,Health,1,STYLE)
        v=dev_load(Payload,i)
        if gl.sum((v!=Nonce*100000+i).to(gl.int32),0)>0:
            gl.atomic_or(Health,4,sem='relaxed',scope='gpu')

root=Path(__file__).resolve().parent;out=root/'publication-acyclic';out.mkdir(exist_ok=True)
p=torch.zeros(512,device='cuda',dtype=torch.int32);ready=torch.zeros(1,device='cuda',dtype=torch.int32);health=torch.zeros_like(ready)
rows=[]
for style in [0,1,3]:
 for nonce in range(1,65):
  ready.zero_();once[(256,)](p,ready,health,nonce,STYLE=style,num_warps=4)
 torch.cuda.synchronize();assert health.item()==0
 cg=torch.cuda.CUDAGraph()
 with torch.cuda.graph(cg):
  for nonce in range(1,65):
   ready.zero_();once[(256,)](p,ready,health,nonce,STYLE=style,num_warps=4)
 for _ in range(160):cg.replay()
 torch.cuda.synchronize()
 row=dict(style=style,publications=64*160,consumer_payload_checks=64*160*255*512,health=health.item());rows.append(row);print(row,flush=True)
 assert row['health']==0
(out/'results.json').write_text(json.dumps(rows,indent=2))
