from combined import *
import torch,json
from pathlib import Path

@g.jit
def publish_test(Payload,Ready,Ack,Health,STYLE:gl.constexpr,ITERS:gl.constexpr):
    pid=gl.program_id(0)
    i=gl.arange(0,512,layout=gl.BlockedLayout([1],[32],[4],[0]))
    for epoch in range(1,ITERS+1):
        if pid==0:
            if epoch>1:wait_target(Ack,Health,(epoch-1)*255,STYLE)
            dev_store(epoch*100000+i,Payload,i)
            gl.inline_asm_elementwise('s_wait_storecnt 0\n s_mov_b32 $0,0',constraints='=s,~{memory}',args=[],dtype=gl.int32,is_pure=False,pack=1)
            gl.barrier()
            gl.atomic_add(Ready,1,sem='relaxed',scope='gpu')
        else:
            wait_target(Ready,Health,epoch,STYLE)
            v=dev_load(Payload,i)
            if gl.sum((v!=epoch*100000+i).to(gl.int32),0)>0:
                gl.atomic_or(Health,4,sem='relaxed',scope='gpu')
            gl.inline_asm_elementwise('s_wait_loadcnt 0\n s_wait_storecnt 0\n s_mov_b32 $0,0',constraints='=s,~{memory}',args=[],dtype=gl.int32,is_pure=False,pack=1)
            gl.barrier()
            gl.atomic_add(Ack,1,sem='relaxed',scope='gpu')

if __name__=='__main__':
 root=Path(__file__).resolve().parent;out=root/'publication';out.mkdir(exist_ok=True)
 p=torch.zeros(512,device='cuda',dtype=torch.int32);ready=torch.zeros(1,device='cuda',dtype=torch.int32);ack=torch.zeros_like(ready);health=torch.zeros_like(ready)
 rows=[]
 for style in [0,1,2]:
  for run in range(10):
   ready.zero_();ack.zero_();health.zero_()
   obj=publish_test[(256,)](p,ready,ack,health,STYLE=style,ITERS=128,num_warps=4,enable_fp_fusion=False)
   torch.cuda.synchronize()
   row=dict(style=style,run=run,health=health.item(),ready=ready.item(),ack=ack.item(),payload_checks=128*255*512)
   rows.append(row);assert row['health']==0 and row['ready']==128 and row['ack']==128*255,row
  (out/f'style{style}.amdgcn').write_text(obj.asm['amdgcn']);print('PASS',style,flush=True)
 (out/'results.json').write_text(json.dumps(rows,indent=2))
