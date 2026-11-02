"""Generate a separate resident-task MoE experiment; never modifies serving."""
from pathlib import Path
import ast
r=Path(__file__).parent
s=(r/'moe_aiter_scoped.py').read_text();tree=ast.parse(s)
for node in tree.body:
 if isinstance(node,ast.FunctionDef) and node.name=='_moe_gemm_a8w4_decode':
  node.args.args.append(ast.arg(arg='TaskID'));node.args.defaults.append(ast.Constant(None))
  for part in ast.walk(node):
   if isinstance(part,ast.Assign) and len(part.targets)==1 and isinstance(part.targets[0],ast.Name) and part.targets[0].id=='pid' and ast.unparse(part.value)=='gl.program_id(0)':
    part.value=ast.parse('gl.program_id(0) if TaskID is None else TaskID',mode='eval').body
(r/'moe_task_stage.py').write_text(ast.unparse(tree)+'\n')
s='''import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from moe_task_stage import _moe_gemm_a8w4_decode as stage

@g.jit
def phase(Count,Health,GRID:gl.constexpr):
    pid=gl.program_id(0)
    gl.amd.gfx1250.tdm.async_wait(0)
    gl.inline_asm_elementwise("s_wait_storecnt 0\\n s_mov_b32 $0, 0",constraints="=s,s,~{memory}",args=[pid],dtype=gl.int32,is_pure=False,pack=1)
    gl.barrier()
    ticket=gl.atomic_add(Count,1,sem='relaxed',scope='gpu')
    target=(ticket//GRID+1)*GRID
    seen=gl.atomic_add(Count,0,sem='relaxed',scope='gpu')
    spins=0
    while (seen!=target)&(spins<1000000):
        seen=gl.atomic_add(Count,0,sem='relaxed',scope='gpu')
        spins+=1
    ok=seen==target
    if not ok:
        gl.atomic_or(Health,1,sem='relaxed',scope='gpu')
    return ok

@g.jit
def fused(X,XS,W1,S1,W2,S2,I,IS,Y,Gather,Hist,Offs,Pad,Map,Gamma,Count,Health,Z,Scatter,
          W1S:gl.constexpr,S1S:gl.constexpr,W2S:gl.constexpr,S2S:gl.constexpr,
          LIMIT:gl.constexpr,M:gl.constexpr,GRID:gl.constexpr,BUFFERS:gl.constexpr=4):
    pid=gl.program_id(0)
    for task in range(pid,M*6*32,GRID):
        stage(I,2048,1,X,4096,1,XS,128,1,
              W1,W1S[0],W1S[1],W1S[2],S1,S1S[0],S1S[1],S1S[2],
              None,None,None,None,None,M,4096,4096,Gather,Hist,Offs,Pad,Map,M*6,32,
              True,1.0,LIMIT,2,False,6,16,128,256,1,BUFFERS,
              'GFX1250_SCALE',True,True,False,4,
              YMxScale=IS,stride_y_mx_m=64,stride_y_mx_n=1,HAS_MX_OUT=True,TaskID=task)
    ok=phase(Count,Health,GRID)
    if ok:
        for task in range(pid,M*6*32,GRID):
            stage(Y,4096,1,I,2048,1,IS,64,1,
                  W2,W2S[0],W2S[1],W2S[2],S2,S2S[0],S2S[1],S2S[2],
                  None,None,None,None,Gamma,M*6,4096,2048,None,Hist,Offs,Pad,Map,M*6,32,
                  False,1.0,LIMIT,1,False,6,16,128,256,1,BUFFERS,
                  'GFX1250_SCALE',True,True,False,4,TaskID=task)
    # Every CTA enters phase2 even after timeout, avoiding a second deadlock.
    ok2=phase(Count+1,Health,GRID)
    for task in range(pid,M*32,GRID):
        token=task//32;tile=task%32
        cols=tile*128+gl.arange(0,128,layout=gl.BlockedLayout([1],[32],[4],[0]))
        acc=gl.full((128,),0,gl.float32,layout=gl.BlockedLayout([1],[32],[4],[0]))
        if ok and ok2:
            for j in gl.static_range(6):
                row=gl.load(Scatter+token*6+j).to(gl.int32)
                vals=gl.amd.gfx1250.buffer_load(Y,row*4096+cols,cache='.cg')
                acc+=vals.to(gl.float32)
        else:
            acc=gl.full((128,),float('nan'),gl.float32,layout=gl.BlockedLayout([1],[32],[4],[0]))
        gl.store(Z+token*4096+cols,acc)
'''
(r/'moe_multitoken_prototype.py').write_text(s)
