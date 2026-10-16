import sys, torch, triton
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from vllm.models.minimax_m3.amd.mono.gluon.kernel import (FLAT4, qkv_in, qkv_quant, qkv_amax, sum3, FLAT3, XL3, i32p, hw_rsq)

@gluon.jit
def p(hid, res, g, out, psc, li, MODE: gl.constexpr, S: gl.constexpr):
    SP: gl.constexpr = triton.next_power_of_2(S)
    errors = i32p(out)
    xa_lo, xa_hi, ga, pa, oka = qkv_in(li, 0, 32, SP, S, hid, res, psc, li + 5, errors, li * 0, i32p(g))
    if MODE == 0:
        gl.store(out.to(gl.pointer_type(gl.float32), bitcast=True) + pa, xa_lo + xa_hi, mask=oka)
    else:
        ss = sum3(xa_lo * xa_lo + xa_hi * xa_hi)
        rstd = hw_rsq(ss / 6144.0 + 1e-6)
        amax = qkv_amax(xa_lo, xa_hi, ga, rstd)
        x_scale = gl.where(amax == 0.0, 1.0, amax / 448.0)
        if MODE == 1:
            gl.store(out.to(gl.pointer_type(gl.float32), bitcast=True) + gl.arange(0, SP, layout=gl.SliceLayout(1, gl.SliceLayout(2, XL3))), x_scale)
        elif MODE == 2:
            xsa = gl.allocate_shared_memory(gl.uint8, [4, 8, 16, 128], FLAT3)
            qkv_quant(xa_lo, xa_hi, ga, rstd, x_scale, xsa, 32, SP)
        else:
            q_lo = (xa_lo / x_scale[:, None, None]).to(gl.float8e4nv).to(gl.uint8, bitcast=True)
            q_hi = (xa_hi / x_scale[:, None, None]).to(gl.float8e4nv).to(gl.uint8, bitcast=True)
            x8 = gl.reshape(gl.join(q_lo, q_hi), [SP, 32, 128])
            if MODE == 3:
                b = gl.allocate_shared_memory(gl.uint8, [32, 16, 128], FLAT3)
                b.slice(0, SP, dim=1).store(gl.permute(x8, (1, 0, 2)))
            elif MODE == 4:
                b = gl.allocate_shared_memory(gl.uint8, [32, SP, 128], FLAT3)
                b.store(gl.permute(x8, (1, 0, 2)))
            elif MODE == 5:
                b = gl.allocate_shared_memory(gl.uint8, [SP, 32, 128], FLAT3)
                b.store(x8)
            elif MODE == 7:
                d_lo = gl.div_rn(xa_lo, x_scale[:, None, None] + xa_lo * 0).to(gl.float8e4nv).to(gl.uint8, bitcast=True)
                d8 = gl.reshape(gl.join(d_lo, q_hi), [SP, 32, 128])
                b = gl.allocate_shared_memory(gl.uint8, [32, 16, 128], FLAT3)
                b.slice(0, SP, dim=1).store(gl.permute(d8, (1, 0, 2)))
            elif MODE == 8:
                x4 = gl.reshape(gl.join(q_lo, q_hi), [SP, 4, 8, 128])
                b = gl.allocate_shared_memory(gl.uint8, [4, 8, 16, 128], FLAT3)
                b.slice(0, SP, dim=2).store(gl.permute(x4, (1, 2, 0, 3)))
            elif MODE == 9:
                x4 = gl.reshape(gl.join(q_lo, q_hi), [SP, 4, 8, 128])
                b = gl.allocate_shared_memory(gl.uint8, [4, 8, 16, 128], FLAT4)
                b.slice(0, SP, dim=2).store(gl.permute(x4, (1, 2, 0, 3)))
            elif MODE == 6:
                b = gl.allocate_shared_memory(gl.uint8, [16, 32, 128], FLAT3)
                b.slice(0, SP, dim=0).store(x8)

h = torch.zeros(16, 6144, dtype=torch.bfloat16, device="cuda")
for mode in [int(x) for x in sys.argv[1:]]:
    try:
        cc = p.warmup(h, h, h, h, h, 0, MODE=mode, S=1, grid=(1,), num_warps=8)
        print("mode", mode, "ok", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc()
