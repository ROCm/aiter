from pathlib import Path
p=Path(__file__).resolve().parent
s=(p/'base/opt.py').read_text().split('@g.jit\ndef rg')[0]
s=s.replace('def project_opt(X,XS,W,WS,P,','def project_qb(X,XS,W,WS,P,Count,Cos,Sin,O,TILE,TAIL:gl.constexpr,')
s=s.replace('tile=gl.program_id(0)//SPLIT;ks=gl.program_id(0)%SPLIT','tile=TILE;ks=0')
s=s.replace('4096','1024').replace('*32+','*8+')
s=s.replace('dev_store(out,P,ks*1536+on[None,:]+om[:,None]*0,om[:,None]==0)','''dev_store(out.to(gl.bfloat16).to(gl.float32),P,on[None,:]+om[:,None]*0,om[:,None]==0)
    if TAIL:
        gl.inline_asm_elementwise('s_wait_loadcnt 0\\n s_wait_storecnt 0\\n s_mov_b32 $0, 0',constraints='=s,~{memory}',args=[],dtype=gl.int32,is_pure=False,pack=1)
        gl.barrier()
        head=tile//(512//BN)
        ticket=gl.atomic_add(Count+head,1,sem='relaxed',scope='gpu')
        if ticket==512//BN-1:
            norm_rope_body(P,Cos,Sin,O,head,True)
            gl.atomic_xchg(Count+head,0,sem='relaxed',scope='gpu')''')
s=s.replace('from device_scope_helpers import dev_load,dev_store','from device_scope_helpers import dev_load,dev_store\nfrom tail import norm_rope_body')
s+='''
@g.jit
def qb_kernel(X,XS,W,WS,P,Count,Cos,Sin,O,TAIL:gl.constexpr,BN:gl.constexpr=128):
    project_qb(X,XS,W,WS,P,Count,Cos,Sin,O,gl.program_id(0),TAIL,BN,1,True,4,128,1,True,True,'.ca',True,False)
'''
(p/'qb.py').write_text(s)
# reuse existing bounded grid barrier only
k=(p.parent/'full-layer-c1/regions/wmma/kernel.py').read_text()
bar=k[k.index('@g.jit\ndef clock'):k.index('@g.jit\ndef quant')]
(p/'combined.py').write_text('''from qa import project_fused
from qb import project_qb
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
'''+bar+'''
@g.jit
def complete(X,XS,W,WS,P,NW,Acc,Count,Y,LP,LS,Cos,Sin,KVOut,QBW,QBS,QP,QC,QOut,Counter,Health,BN:gl.constexpr=128):
    project_fused(X,XS,W,WS,P,NW,Acc,Count,Y,LP,LS,Cos,Sin,KVOut,True,True,1,128,16,True,4,128,1,True,True,'.ca',True,False,1,1,128,0,'.ca')
    sync(Counter,Health,192)
    for tile in range(gl.program_id(0),32768//BN,192):
        project_qb(LP,LS,QBW,QBS,QP,QC,Cos,Sin,QOut,tile,True,BN,1,True,4,128,1,True,True,'.ca',True,False)
''')
