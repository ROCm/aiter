"""Device-scope (sc1) tagged-pair put asm: masked lanes must not store."""
import torch
from triton.experimental import gluon
import triton.experimental.gluon.language as gl

L = gl.constexpr(gl.BlockedLayout([2], [64], [4], [0]))
PUT_A = gl.constexpr("v_cmp_ne_u32 vcc, 0, $2\ns_and_saveexec_b64 $0, vcc\ns_cbranch_execz 1f\n"
                     "global_store_dwordx2 $3, $1, off sc1\n1:\ns_or_b64 exec, exec, $0")
PUT_B = gl.constexpr("s_mov_b64 $0, exec\ns_nop 4\nv_cmpx_ne_u32_e32 0, $2\ns_nop 4\n"
                     "global_store_dwordx2 $3, $1, off sc1\ns_nop 4\ns_mov_b64 exec, $0\ns_nop 4")


@gluon.jit
def k(dst, n, V: gl.constexpr):
    i = gl.arange(0, 512, layout=L)
    w = (i.to(gl.int64) << 32) | 7
    m = (i < n).to(gl.int32)
    a = (dst + i).to(gl.int64, bitcast=True) + w * 0
    if V == 0:
        gl.inline_asm_elementwise(PUT_A, "=&s,v,v,v,~{vcc},~{memory}", [w, m, a], dtype=gl.int64, is_pure=False, pack=1)
    else:
        gl.inline_asm_elementwise(PUT_B, "=&s,v,v,v,~{vcc},~{memory}", [w, m, a], dtype=gl.int64, is_pure=False, pack=1)


for V in (0, 1):
    x = torch.zeros(512, dtype=torch.int64, device="cuda")
    try:
        k[(1,)](x, 300, V, num_warps=4)
        torch.cuda.synchronize()
        exp = torch.where(torch.arange(512, device="cuda") < 300, (torch.arange(512, device="cuda") << 32) | 7, 0)
        print(V, "ok" if torch.equal(x, exp) else f"WRONG {(x != exp).sum().item()} words")
    except Exception as e:
        print(V, "ERR", str(e)[:100])
        break
c = k.warmup(torch.zeros(1, dtype=torch.int64, device="cuda"), 1, 0, grid=(1,), num_warps=4)
print("\n".join(l for l in c.asm["amdgcn"].splitlines() if "global_store" in l or "saveexec" in l or "v_cmp" in l or "exec" in l))
