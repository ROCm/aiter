"""Probe gfx1250 WMMA operand / accumulator layouts against torch (one wave a launch).

A hypothesis is a packing of row-major A[M][K], B[N][K] into per-lane words and a
reading of the 8 (or 16) f32 accumulators a lane holds; a hypothesis holds when the
WMMA result equals A @ B.T exactly (small values: the f32 sums are exact).
"""

import itertools
import sys

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

WAVE = 32


def rsrc(addr):
    return bo.create_buffer_resource_from_addr(addr)


def build(op: str, wa: int, wb: int, nacc: int):
    """Kernel: lane l loads wa / wb dwords of A / B at l * wa / l * wb, one WMMA, stores nacc f32."""

    @flyc.kernel(name=f"probe_{op}", known_block_size=[WAVE, 1, 1])
    def probe(a: Int64, b: Int64, d: Int64, sa: Int32, sb: Int32):
        lane = fx.thread_idx.x

        def words(base, n):
            parts = []
            for c in range_constexpr(n // 4):
                v = fx.Vector(bo.buffer_load(rsrc(base), lane * n + 4 * c, vec_width=4, dtype=T.i32))
                parts += [v[e] for e in range(4)]
            return fx.Vector.from_elements(parts, fx.Int32)

        va, vb = words(a, wa), words(b, wb)
        acc = fx.Vector.filled(nacc, 0.0, fx.Float32)
        rt = fx.Vector.make_type(nacc, fx.Float32)
        if const_expr(op == "fp8_128"):
            r = rocdl.wmma_f32_16x16x128_fp8_fp8(rt, va, vb, acc)
        elif const_expr(op == "fp8_64"):
            r = rocdl.wmma_f32_16x16x64_fp8_fp8(rt, fx.as_ir_value(va), fx.as_ir_value(vb), fx.as_ir_value(acc)).result
        elif const_expr(op == "bf16_32"):
            r = rocdl.wmma_f32_16x16x32_bf16(
                rt, fx.as_ir_value(va.bitcast(fx.BFloat16)), fx.as_ir_value(vb.bitcast(fx.BFloat16)),
                fx.as_ir_value(acc)).result
        elif const_expr(op == "scale_fp8"):
            r = rocdl.wmma_scale_f32_16x16x128_f8f6f4(rt, va, vb, acc, sa, sb, fmtA=0, fmtB=0)
        elif const_expr(op == "scale_fp4a"):
            r = rocdl.wmma_scale_f32_16x16x128_f8f6f4(rt, va, vb, acc, sa, sb, fmtA=4, fmtB=0)
        elif const_expr(op == "scale32_f4"):
            r = rocdl.wmma_scale_f32_32x16x128_f4(rt, va, vb, acc, sa, sb)
        out = fx.Vector(r)
        for c in range_constexpr(nacc // 4):
            bo.buffer_store(
                fx.Vector.from_elements([out[4 * c + e] for e in range(4)], fx.Float32),
                rsrc(d), lane * nacc + 4 * c)

    @flyc.jit
    def launch(a: Int64, b: Int64, d: Int64, sa: Int32, sb: Int32, stream: fx.Stream = fx.Stream(None)):
        probe(a, b, d, sa, sb).launch(grid=(1,), block=(WAVE,), stream=stream)

    return launch


def run(launch, a_words, b_words, nacc, sa=0x7F7F7F7F, sb=0x7F7F7F7F):
    a = a_words.contiguous().cuda()
    b = b_words.contiguous().cuda()
    d = torch.zeros(WAVE * nacc, dtype=torch.float32, device="cuda")
    _run_compiled(launch, a.data_ptr(), b.data_ptr(), d.data_ptr(), sa, sb, torch.cuda.current_stream())
    torch.cuda.synchronize()
    return d.view(WAVE, nacc).cpu()


VALS = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])


def rand_mat(rows, k):
    return VALS[torch.randint(0, len(VALS), (rows, k))]


# --- hypotheses: (lane, slot) -> (row, k); slot = byte for fp8, element for bf16, nibble for fp4


def k_chunked(lane, slot, chunk, kstep):
    """slot's chunk c of `chunk` elements: k = kstep * c + chunk * (lane // 16) + slot % chunk."""
    c, e = divmod(slot, chunk)
    return kstep * c + chunk * (lane // 16) + e


def pack(mat, slots, kmap, rows=16, elem="fp8"):
    """mat[rows][K] -> [32 lanes][slots] elements via kmap(lane, slot) -> k, row = lane % 16."""
    out = torch.zeros(WAVE, slots)
    for lane in range(WAVE):
        for s in range(slots):
            out[lane, s] = mat[lane % rows if rows == 16 else lane % 16][kmap(lane, s)]
    if elem == "fp8":
        return out.to(torch.float8_e4m3fn).view(torch.uint8).view(WAVE, slots // 4, 4).contiguous().view(torch.int32).view(-1)
    if elem == "bf16":
        return out.to(torch.bfloat16).view(torch.int16).contiguous().view(torch.int32).view(-1)
    raise ValueError(elem)


def acc_reads(d, kind):
    """The 16x16 result read from per-lane accumulators d[32][8]."""
    res = torch.zeros(16, 16)
    for lane in range(WAVE):
        for i in range(8):
            if kind == "m8":  # lane: col lane % 16, rows 8 * (lane // 16) + i
                res[8 * (lane // 16) + i, lane % 16] = d[lane, i]
            elif kind == "n8":  # lane: row lane % 16, cols 8 * (lane // 16) + i
                res[lane % 16, 8 * (lane // 16) + i] = d[lane, i]
    return res


def check(name, op, wa, wb, k, slots, elem, chunk_opts):
    launch = build(op, wa, wb, 8)
    torch.manual_seed(0)
    A, B = rand_mat(16, k), rand_mat(16, k)
    ref = A @ B.T
    for chunk, kstep in chunk_opts:
        km = lambda lane, s, c=chunk, ks=kstep: k_chunked(lane, s, c, ks)
        d = run(launch, pack(A, slots, km, elem=elem), pack(B, slots, km, elem=elem), 8)
        for kind in ("m8", "n8"):
            got = acc_reads(d, kind)
            if torch.equal(got, ref):
                print(f"{name}: OK  A/B lane l: row l%16, k = {kstep}*(slot//{chunk}) + {chunk}*(l//16) + slot%{chunk};"
                      f" acc '{kind}'")
                return (chunk, kstep, kind)
    print(f"{name}: no hypothesis matched")
    return None


if __name__ == "__main__" and "scales" not in sys.argv[1:]:
    which = sys.argv[1:] or ["fp8_128", "fp8_64", "bf16_32", "scale_fp8"]
    if "fp8_128" in which:
        check("wmma_f32_16x16x128_fp8_fp8", "fp8_128", 16, 16, 128, 64, "fp8",
              [(16, 32), (8, 16), (32, 64), (4, 8)])
    if "fp8_64" in which:
        check("wmma_f32_16x16x64_fp8_fp8", "fp8_64", 8, 8, 64, 32, "fp8",
              [(16, 32), (8, 16), (4, 8), (32, 64)])
    if "bf16_32" in which:
        check("wmma_f32_16x16x32_bf16", "bf16_32", 8, 8, 32, 16, "bf16",
              [(8, 16), (4, 8), (2, 4), (16, 32)])
    if "scale_fp8" in which:
        check("wmma_scale_f32_16x16x128_f8f6f4 (fp8 x fp8, unit scales)", "scale_fp8", 16, 16, 128, 64, "fp8",
              [(16, 32), (8, 16), (32, 64), (4, 8)])


# --- scaled: E8M0 per (row, 32-k block); FP4 A


def e8m0(codes):
    return torch.exp2(codes.float() - 127)


def scale_words(codes, layout):
    """codes[16 rows][4 kblocks] -> one i32 a lane, by layout."""
    w = torch.zeros(WAVE, 4, dtype=torch.uint8)
    for lane in range(WAVE):
        r = lane % 16
        if layout == "row_4blocks":  # every lane: its row's 4 blocks
            w[lane] = codes[r]
        elif layout == "half_2blocks":  # lane < 16: blocks 0,1; lane >= 16: blocks 2,3 (lo half)
            h = lane // 16
            w[lane, 0], w[lane, 1] = codes[r, 2 * h], codes[r, 2 * h + 1]
            w[lane, 2], w[lane, 3] = 127, 127
        elif layout == "halfk_interleave":  # lane half h holds blocks h, h+2
            h = lane // 16
            w[lane, 0], w[lane, 1] = codes[r, h], codes[r, h + 2]
            w[lane, 2], w[lane, 3] = 127, 127
    return int.from_bytes(bytes(0), "little"), w.view(torch.int32).view(-1)


def check_scaled_fp8():
    """Per-lane scale VGPRs need a per-lane value: pass them as an extra word of A / B."""
    print("scale layout probe needs per-lane scale VGPRs; see check_scaled_fp8_lanes")


def build_scaled(op, wa, wb, fmt_a):
    @flyc.kernel(name=f"probe_s_{op}_{fmt_a}", known_block_size=[WAVE, 1, 1])
    def probe(a: Int64, b: Int64, d: Int64, sa: Int64, sb: Int64):
        lane = fx.thread_idx.x

        def words(base, n):
            parts = []
            for c in range_constexpr(n // 4):
                v = fx.Vector(bo.buffer_load(rsrc(base), lane * n + 4 * c, vec_width=4, dtype=T.i32))
                parts += [v[e] for e in range(4)]
            return fx.Vector.from_elements(parts, fx.Int32)

        va, vb = words(a, wa), words(b, wb)
        s_a = fx.Int32(bo.buffer_load(rsrc(sa), lane, vec_width=1, dtype=T.i32))
        s_b = fx.Int32(bo.buffer_load(rsrc(sb), lane, vec_width=1, dtype=T.i32))
        acc = fx.Vector.filled(8, 0.0, fx.Float32)
        rt = fx.Vector.make_type(8, fx.Float32)
        r = rocdl.wmma_scale_f32_16x16x128_f8f6f4(rt, va, vb, acc, s_a, s_b, fmtA=fmt_a, fmtB=0)
        out = fx.Vector(r)
        for c in range_constexpr(2):
            bo.buffer_store(fx.Vector.from_elements([out[4 * c + e] for e in range(4)], fx.Float32),
                            rsrc(d), lane * 8 + 4 * c)

    @flyc.jit
    def launch(a: Int64, b: Int64, d: Int64, sa: Int64, sb: Int64, stream: fx.Stream = fx.Stream(None)):
        probe(a, b, d, sa, sb).launch(grid=(1,), block=(WAVE,), stream=stream)

    return launch


def run_scaled(launch, aw, bw, saw, sbw):
    t = [x.contiguous().cuda() for x in (aw, bw, saw, sbw)]
    d = torch.zeros(WAVE * 8, dtype=torch.float32, device="cuda")
    _run_compiled(launch, t[0].data_ptr(), t[1].data_ptr(), d.data_ptr(), t[2].data_ptr(),
                  t[3].data_ptr(), torch.cuda.current_stream())
    torch.cuda.synchronize()
    return d.view(WAVE, 8).cpu()


FP4_VALS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])  # E2M1 codes 0..7, sign bit 8


def fp4_code(x):
    mag = (FP4_VALS - x.abs()).abs().argmin()
    return int(mag) | (8 if x < 0 else 0)


def pack_fp4(mat, kmap):
    """mat[16][128] (E2M1-representable) -> [32 lanes][32 B]: nibble s of a lane is k = kmap(lane, s),
    low nibble first."""
    out = torch.zeros(WAVE, 32, dtype=torch.uint8)
    for lane in range(WAVE):
        for s in range(64):
            c = fp4_code(mat[lane % 16][kmap(lane, s)])
            out[lane, s // 2] |= (c << (4 * (s % 2)))
    return out.view(torch.int32).view(-1)


def check_scales():
    torch.manual_seed(1)
    A, B = rand_mat(16, 128), rand_mat(16, 128)
    ca = torch.randint(125, 130, (16, 4))
    cb = torch.randint(125, 130, (16, 4))
    Ad = A * e8m0(ca).repeat_interleave(32, 1)
    Bd = B * e8m0(cb).repeat_interleave(32, 1)
    ref = Ad @ Bd.T
    km = lambda lane, s: k_chunked(lane, s, 16, 32)
    launch = build_scaled("fp8", 16, 16, 0)
    for lay in ("row_4blocks", "half_2blocks", "halfk_interleave"):
        _, saw = scale_words(ca, lay)
        _, sbw = scale_words(cb, lay)
        d = run_scaled(launch, pack(A, 64, km), pack(B, 64, km), saw, sbw)
        if torch.equal(acc_reads(d, "m8"), ref):
            print(f"scaled fp8 E8M0: OK scale VGPR layout '{lay}' (opsel 0)")
            break
    else:
        print("scaled fp8 E8M0: no scale layout matched")
    # FP4 A x FP8 B, unit scales
    FA = FP4_VALS[torch.randint(0, 8, (16, 128))] * (torch.randint(0, 2, (16, 128)) * 2 - 1)
    ref = FA @ B.T
    launch4 = build_scaled("fp4", 8, 16, 4)
    ones = torch.full((WAVE,), 0x7F7F7F7F, dtype=torch.int32)
    for chunk, kstep in ((32, 64), (16, 32), (64, 128), (8, 16)):
        kmf = lambda lane, s, c=chunk, ks=kstep: k_chunked(lane, s, c, ks)
        d = run_scaled(launch4, pack_fp4(FA, kmf), pack(B, 64, km), ones, ones)
        if torch.equal(acc_reads(d, "m8"), ref):
            print(f"scaled FP4 A x FP8 B: OK A lane l: row l%16, k = {kstep}*(nib//{chunk}) + {chunk}*(l//16) + nib%{chunk}"
                  " (low nibble first)")
            break
    else:
        print("scaled FP4 A x FP8 B: no A layout matched")


if __name__ == "__main__" and "scales" in sys.argv[1:]:
    check_scales()
