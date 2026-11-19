"""Build opt-in fine cycle instrumentation from the current S1F sources."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = Path(__file__).resolve().parents[3] / "aiter/ops/flydsl/kernels/mega_moe_gfx1250"


def replace(text, old, new, count=1):
    assert text.count(old) == count, (old[:100], text.count(old), count)
    return text.replace(old, new, count)


for name in ("s1f_stage1.py", "s1f_pull.py", "dispatch_pull.py"):
    (ROOT / ("original_" + name)).write_text((SRC / name).read_text())

(ROOT / "cycle_trace.py").write_text(
    '''"""Wave-local shader-cycle probes; no hardware wait added by a probe."""
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr.typing import T


def cycles():
    # No typed FX shader-cycle API. gfx1250 ISA supports this scalar clock;
    # memory clobber orders compiler memory motion, not hardware completion.
    return fx.Int64(llvm.InlineAsmOp(
        T.i64, [], "s_get_shader_cycles_u64 $0", "=s,~{memory}",
        has_side_effects=True,
    ).result)


def store(address, index, value):
    pointer = fx.inttoptr(
        fx.PointerType.get(T.i64, fx.AddressSpace.Global, 8), address
    )
    pointer[index] = fx.Int64(value)
'''
)

s = (SRC / "dispatch_pull.py").read_text()
s = replace(
    s,
    "from .tdm_gather_shim import make_tensor_gather_descriptor",
    """from .tdm_gather_shim import make_tensor_gather_descriptor
from .cycle_trace import cycles""",
)
s = replace(
    s,
    "        end: fx.Int32,\n    ):",
    """        end: fx.Int32,
        trace_sample: fx.Constexpr[bool] = False,
    ):
        if const_expr(trace_sample):
            c0 = cycles()""",
)
s = replace(
    s,
    "        for gather in range_constexpr(2):",
    """        if const_expr(trace_sample):
            c1 = cycles()
        for gather in range_constexpr(2):""",
)
s = replace(
    s,
    "        tdm_ops.tensor_wait(0)\n        for item",
    """        if const_expr(trace_sample):
            c2 = cycles()
        tdm_ops.tensor_wait(0)
        if const_expr(trace_sample):
            c3 = cycles()
        for item""",
)
s = replace(
    s,
    "        lds_address = fx.Int64(fx.ptrtoint(lds_rows))",
    """        if const_expr(trace_sample):
            c4 = cycles()
        lds_address = fx.Int64(fx.ptrtoint(lds_rows))""",
)
s = replace(
    s,
    "\n    return copy_tile",
    """
        if const_expr(trace_sample):
            c5 = cycles()
            return c0, c1, c2, c3, c4, c5

    return copy_tile""",
)
(ROOT / "dispatch_pull.py").write_text(s)

s = (SRC / "s1f_pull.py").read_text()
s = replace(
    s,
    "from . import ready_queue as rq",
    """from . import ready_queue as rq
from .cycle_trace import cycles, store as trace_store""",
)
# Only the external-ready baseline is instrumented; reject other modes.
s = replace(
    s,
    "    validate_producer_group(producer_group, workers, producers, tile_m)",
    """    if ready_first or tile_ready or producer_group != 1 or tile_m != 256:
        raise ValueError("fine trace requires external-ready, M64 flags, M256/group1")
    validate_producer_group(producer_group, workers, producers, tile_m)""",
)
s = replace(
    s,
    "        generation: fx.Int32,\n    ):",
    """        generation: fx.Int32,
        trace_address: fx.Int64,
    ):""",
)
s = replace(
    s,
    "\n        for item in range(loop_first, loop_end, loop_stride):",
    """
        for item in range(loop_first, loop_end, loop_stride):
            p0 = cycles()
            c0 = c1 = c2 = c3 = c4 = c5 = fx.Int64(0)""",
)
old = """                    copy_tile(
                        lds_rows,
                        output,
                        scale_rows,
                        local + off_rowmap,
                        window.base,
                        window.stride,
                        row0,
                        end,
                    )"""
new = """                    if const_expr(chunk == 0):
                        c0, c1, c2, c3, c4, c5 = copy_tile(
                            lds_rows, output, scale_rows, local + off_rowmap,
                            window.base, window.stride, row0, end,
                            trace_sample=True,
                        )
                    else:
                        copy_tile(
                            lds_rows, output, scale_rows, local + off_rowmap,
                            window.base, window.stride, row0, end,
                        )"""
s = replace(s, old, new)
s = replace(
    s,
    "            comm_ops.waitcnt_stores()\n            if const_expr(tile_ready):",
    """            p1 = cycles()
            comm_ops.waitcnt_stores()
            p2 = cycles()
            if const_expr(tile_ready):""",
)
anchor = """                        generation,
                    )

    @flyc.jit
    def producer_ready_first"""
s = replace(
    s,
    anchor,
    """                        generation,
                    )
            p3 = cycles()
            if (lane == 0) & (count > 0):
                base = tile * 16
                for column, value in enumerate((
                    p0, p1, p2, p3, slot, generation,
                    c0, c1, c2, c3, c4, c5, wave, count,
                )):
                    trace_store(trace_address, base + column, value)

    @flyc.jit
    def producer_ready_first""",
)
(ROOT / "s1f_pull.py").write_text(s)

s = (SRC / "s1f_stage1.py").read_text()
s = replace(
    s,
    "from . import ready_queue as rq",
    """from . import ready_queue as rq
from .cycle_trace import cycles, store as trace_store""",
)
s = replace(s, 'f"s1f_{producer_name}_p{P}', 'f"s1f_{producer_name}_finecycles_p{P}')
s = replace(
    s,
    """                        slot_id,
                        gen,
                    )""",
    """                        slot_id,
                        gen,
                        inp_scale,
                    )""",
)
s = replace(
    s,
    """                mask = tinfo & fx.Int32(0xFF)

                def local_a_ready""",
    """                mask = tinfo & fx.Int32(0xFF)
                trace_base = fx.Int32(tile_cap * 16) + (
                    m * fx.Int32(NT) + n
                ) * fx.Int32(4 * 20) + wave * fx.Int32(20)

                def local_a_ready""",
)
s = replace(
    s,
    """                def local_a_ready(m=m, mask=mask, ready=ready, seq=seq):""",
    """                def local_a_ready(m=m, mask=mask, ready=ready, seq=seq):
                    r0 = cycles()""",
)
s = replace(
    s,
    """                def a_ready(seq=seq):""",
    """                    r1 = cycles()
                    if lane == 0:
                        trace_store(inp_scale, trace_base + 2, r0)
                        trace_store(inp_scale, trace_base + 3, r1)

                def a_ready(seq=seq):""",
)
s = replace(
    s,
    """                def drain_start(queue=queue, nxt=nxt):
                    nxt["claim"] = claim_issue(queue)""",
    """                def drain_start(queue=queue, nxt=nxt):
                    d0 = cycles()
                    nxt["claim"] = claim_issue(queue)
                    d1 = cycles()
                    if lane == 0:
                        trace_store(inp_scale, trace_base + 4, d0)
                        trace_store(inp_scale, trace_base + 5, d1)""",
)
s = replace(
    s,
    """                def epi_start(queue=queue, nxt=nxt):
                    c = claim_resolve(queue, nxt["claim"])""",
    """                def epi_start(queue=queue, nxt=nxt):
                    e0 = cycles()
                    c = claim_resolve(queue, nxt["claim"])
                    e1 = cycles()""",
)
s = replace(
    s,
    """                    nxt["flags"] = fl_n

                def epi_end""",
    """                    nxt["flags"] = fl_n
                    e2 = cycles()
                    if lane == 0:
                        trace_store(inp_scale, trace_base + 6, e0)
                        trace_store(inp_scale, trace_base + 7, e1)
                        trace_store(inp_scale, trace_base + 8, e2)

                def epi_end""",
)
s = replace(
    s,
    """                def epi_end(queue=queue, nxt=nxt, m=m):
                    if is_w0:""",
    """                def epi_end(queue=queue, nxt=nxt, m=m):
                    f0 = cycles()
                    if is_w0:""",
)
s = replace(
    s,
    """                        rdy = pre_ready(nxt["flags"], nxt["tsub"])
                        xq_n""",
    """                        rdy = pre_ready(nxt["flags"], nxt["tsub"])
                        f1 = cycles()
                        if lane == 0:
                            trace_store(inp_scale, trace_base + 10, f1)
                        xq_n""",
)
s = replace(
    s,
    """                        publish_slot(c, ti, rdy, xq_n)

                set_holders""",
    """                        publish_slot(c, ti, rdy, xq_n)
                    f2 = cycles()
                    if lane == 0:
                        trace_store(inp_scale, trace_base + 9, f0)
                        trace_store(inp_scale, trace_base + 11, f2)

                set_holders""",
)
s = replace(
    s,
    """                g1_task(
                    arg_a2,""",
    """                s0 = cycles()
                g1_task(
                    arg_a2,""",
)
s = replace(
    s,
    """                code, tinfo, ready, queue = read_slot()
                seq = seq + fx.Int32(1)""",
    """                s1 = cycles()
                if lane == 0:
                    for column, value in enumerate((slot_id, seq, gen, ready)):
                        trace_store(inp_scale, trace_base + 12 + column, value)
                    trace_store(inp_scale, trace_base, s0)
                    trace_store(inp_scale, trace_base + 1, s1)
                code, tinfo, ready, queue = read_slot()
                seq = seq + fx.Int32(1)""",
)
(ROOT / "s1f_stage1.py").write_text(s)
print("Built overlay from current sources:", ROOT)
