"""Compile one stage of the Gluon M3 kernel alone; print resources."""
import os, sys, torch
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
import vllm.models.minimax_m3.amd.mono.gluon.kernel as K

@gluon.jit
def p_qkv(ptab, hid, res, scratch, errors, li, S: gl.constexpr):
    bid = gl.program_id(0)
    K.stage_qkv(bid, li, ptab, hid, res, scratch, scratch, li + 1, li, errors, bid, 1e-6, S)

@gluon.jit
def p_o(ptab, scratch, peers, errors, hm_out, li, S: gl.constexpr):
    bid = gl.program_id(0)
    K.stage_o(bid, li, ptab, 57, scratch, peers, scratch.to(gl.int64, bitcast=True), li.to(gl.int64), 0, hm_out, li + 1, errors, bid, S, 4096)

@gluon.jit
def p_split(ptab, cs, pos, slots, bt, sl, scratch, errors, li, S: gl.constexpr):
    bid = gl.program_id(0)
    K.stage_split(bid, li, ptab, cs.to(gl.pointer_type(gl.int32), bitcast=True), pos, slots, bt, 256, sl, scratch, li + 1, errors, bid, 0.088, 1e-6)

@gluon.jit
def p_router(ptab, scratch, errors, li, S: gl.constexpr):
    bid = gl.program_id(0)
    K.stage_router(bid, li, ptab, scratch, li + 1, errors, bid, 1e-6, S)

@gluon.jit
def p_moe(ptab, scratch, peers, errors, ar_out, li, S: gl.constexpr):
    bid = gl.program_id(0)
    K.stage_moe(li, ptab, 57, scratch, peers, scratch.to(gl.int64, bitcast=True), li.to(gl.int64), 0, ar_out, li + 1, errors, bid, 2.0, 1.0, 7.0, S, 4096)

z64 = torch.zeros(64, dtype=torch.int64, device="cuda")
i32 = torch.zeros(64, dtype=torch.int32, device="cuda")
u8 = torch.zeros(64, dtype=torch.uint8, device="cuda")
h = torch.zeros(16, 6144, dtype=torch.bfloat16, device="cuda")
S = int(os.environ.get("S", "1"))
specs = {
    "qkv": (p_qkv, (z64, h, h, u8, i32, 0)),
    "o": (p_o, (z64, u8, z64, i32, h, 0)),
    "split": (p_split, (z64, h, z64, z64, i32, i32, u8, i32, 0)),
    "router": (p_router, (z64, u8, i32, 0)),
    "moe": (p_moe, (z64, u8, z64, i32, h, 0)),
}
for name in sys.argv[1:]:
    fn, args = specs[name]
    try:
        cc = fn.warmup(*args, S=S, grid=(256,), num_warps=8, enable_fp_fusion=False)
        cc._init_handles()
        a = cc.asm["amdgcn"]
        print(f"{name}: VGPRs {cc.n_regs} spills {cc.n_spills} LDS {cc.metadata.shared} waterfall {a.count('s_and_saveexec')} "
              f"dwordx4 {a.count('buffer_load_dwordx4')} scratch {a.count('scratch_')}", flush=True)
        open(f"/home/vigokhal/m3_monokernel/tests/probes/stage_{name}.s", "w").write(a)
    except Exception as e:
        print(name, "FAIL", "\n".join(str(e).splitlines()[-int(os.environ.get("ELINES", "1")):])[:3000], flush=True)


@gluon.jit
def p_route(ptab, scratch, errors, out, li, S: gl.constexpr):
    bid = gl.program_id(0)
    E, W = K.route_all(scratch, bid, li + 1, errors, 2.0, bid >= 0, S)
    t = 0.0
    for i in gl.static_range(4 * S):
        t = t + E[i].to(gl.float32) + W[i]
    gl.store(out + bid, t)


@gluon.jit
def p_down(ptab, scratch, peers, errors, ar_out, li, S: gl.constexpr):
    bid = gl.program_id(0)
    E, W = K.route_all(scratch, bid, li + 1, errors, 2.0, bid >= 0, S)
    e2 = K.lrow(ptab, li, (14, 15))
    w2 = K.lp(e2[0], gl.uint8)
    s2 = K.lp(e2[1], gl.uint8)
    ms8 = gl.allocate_shared_memory(gl.uint8, [8, 16, 128], K.BSW)
    mc8 = gl.allocate_shared_memory(gl.uint8, [8, 16, 4], K.FLAT3)
    msh = gl.allocate_shared_memory(gl.uint8, [2, 16, 128], K.BSW)
    mch = gl.allocate_shared_memory(gl.uint8, [2, 16, 4], K.FLAT3)
    red = gl.allocate_shared_memory(gl.float32, [128, 4], K.FLAT2)
    K.stage_down(bid, li, 57, E, W, w2, s2, ms8, mc8, msh, mch, red, scratch, peers, scratch.to(gl.int64, bitcast=True),
                 li.to(gl.int64), 0, ar_out, li + 1, errors, bid, 1.0, S, 4096)


@gluon.jit
def p_mids(scratch, errors, li, S: gl.constexpr):
    bid = gl.program_id(0)
    ms8 = gl.allocate_shared_memory(gl.uint8, [8, 16, 128], K.BSW)
    mc8 = gl.allocate_shared_memory(gl.uint8, [8, 16, 4], K.FLAT3)
    msh = gl.allocate_shared_memory(gl.uint8, [2, 16, 128], K.BSW)
    mch = gl.allocate_shared_memory(gl.uint8, [2, 16, 4], K.FLAT3)
    K.stage_mids(scratch, li + 1, errors, bid, ms8, mc8, msh, mch, S)


specs2 = {
    "route": (p_route, (z64, u8, i32, h, 0)),
    "down": (p_down, (z64, u8, z64, i32, h, 0)),
    "mids": (p_mids, (u8, i32, 0)),
}
for name in os.environ.get("EXTRA", "").split(","):
    if not name:
        continue
    fn, args = specs2[name]
    try:
        cc = fn.warmup(*args, S=S, grid=(256,), num_warps=8, enable_fp_fusion=False)
        cc._init_handles()
        a = cc.asm["amdgcn"]
        print(f"{name}: VGPRs {cc.n_regs} spills {cc.n_spills} LDS {cc.metadata.shared} writelane {a.count('v_writelane')}", flush=True)
        open(f"/home/vigokhal/m3_monokernel/tests/probes/stage_{name}.s", "w").write(a)
    except Exception as e:
        print(name, "FAIL", "\n".join(str(e).splitlines()[-int(os.environ.get("ELINES", "1")):])[:3000], flush=True)
