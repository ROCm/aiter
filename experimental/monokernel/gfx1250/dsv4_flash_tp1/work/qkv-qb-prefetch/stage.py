from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

@g.jit
def stage_qb(W, pid, Desc, Shared, PREFETCH:gl.constexpr, TDM:gl.constexpr):
    if PREFETCH:
        tid=gl.arange(0,128,layout=gl.BlockedLayout([1],[32],[4],[0]))
        # LLVM names TH=1 as TH_LOAD_NT; for GLOBAL_PREFETCH_B8 at
        # device scope it encodes regular-temporal NON-SPECULATIVE prefetch.
        if PREFETCH==2:
            ASM:gl.constexpr="global_prefetch_b8 $1, off scope:SCOPE_DEV th:TH_LOAD_NT\n v_mov_b32 $0, 0"
        else:
            ASM:gl.constexpr="global_prefetch_b8 $1, off scope:SCOPE_DEV th:TH_LOAD_RT\n v_mov_b32 $0, 0"
        for line in gl.static_range(4):
            addr=(W+pid*131072+tid*1024+line*256).to(gl.uint64)
            gl.inline_asm_elementwise(
                ASM,
                constraints="=v,v,~{memory}",args=[addr],dtype=gl.int32,is_pure=False,pack=1)
    if TDM:
        gl.amd.gfx1250.tdm.async_load(Desc,[0,0],Shared)
