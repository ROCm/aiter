"""Standalone correctness test for the TDM gather primitive + the per-wave /
per-sub descriptor pattern used by the gather-MoE stage1 kernel.

Loads M rows of K bytes from a token-order source ``src[gather_index[r]]`` into
LDS via per-wave gather descriptors (mirroring gemm_mxscale_gfx1250's gather A
path), then stores LDS -> out. Compares against the torch reference
``out[r] = src[gather_index[r]]`` (padding rows -> 0).
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, range_constexpr, rocdl, tdm_ops, vector
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T
from flydsl.compiler.kernel_function import CompilationContext
from flydsl._mlir import ir
from flydsl._mlir.dialects import gpu as gpu_d
from flydsl.utils.smem_allocator import SmemAllocator
from aiter.ops.flydsl.kernels.gemm_common_gfx1250 import lds_load_b128

M = 128
KFULL = 7168  # full row width (bytes) of the source matrix (= model_dim)
KTILE = 256  # row_width per K-tile (bytes)
NKTILES = KFULL // KTILE  # 28
PAD = 16  # LDS padding bytes per row (matches LDS_PAD_A_BYTES)
K = KTILE + PAD  # LDS row stride (bytes), one tile's worth + pad
CHUNKS = KTILE // 16  # b128 chunks per row tile
NUM_WARPS = 4
WAVE_SIZE = 32
BLOCK_THREADS = NUM_WARPS * WAVE_SIZE
RPW = M // NUM_WARPS  # 32
RPS = 8
SUBOPS = RPW // RPS  # 4


def build():
    alloc = SmemAllocator(None, arch="gfx1250", global_sym_name="gather_test_lds")
    lds_bytes = M * K
    alloc.ptr = lds_bytes

    def _advance_addr(lo, hi, adv):
        new_lo = arith.addi(lo, adv)
        wrapped = arith.cmpi(arith.CmpIPredicate.ult, new_lo, lo)
        hi_inc = arith.addi(hi, arith.constant(1, type=T.i32))
        return new_lo, arith.select(wrapped, hi_inc, hi)

    @flyc.kernel(name="tdm_gather_test", known_block_size=[BLOCK_THREADS, 1, 1])
    def kernel(out: fx.Tensor, src: fx.Tensor, gidx: fx.Tensor, n_tok: fx.Int32):
        lds_base = alloc.get_base()
        gi_rsrc = buffer_ops.create_buffer_resource(gidx, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(out, max_size=True)
        wid = arith.index_cast(T.index, _raw(rocdl.wave_id()))
        tx = arith.index_cast(T.index, _raw(fx.thread_idx.x))
        a_pred = arith.constant(1 | (1 << 30) | (1 << 31), type=T.i32)
        adv = arith.constant(KTILE, type=T.i32)

        # Pre-fill the whole LDS A region with a sentinel (0x7F7F7F7F) so that any
        # row the gather fails to write shows up as sentinel garbage.
        sentinel = arith.constant(0x7F7F7F7F, type=T.i32)
        svec = vector.from_elements(
            T.vec(4, T.i32), [sentinel, sentinel, sentinel, sentinel]
        )
        from aiter.ops.flydsl.kernels.gemm_common_gfx1250 import lds_store_b128

        _txs = arith.index_cast(T.index, _raw(fx.thread_idx.x))
        for _c in range_constexpr(K // 16):
            lds_store_b128(lds_base, _txs * arith.index(K) + arith.index(_c * 16), svec)
        gpu_d.barrier()

        # Build init descriptors (mirrors desc_a_init) and capture shared/per-sub
        # state, then rebuild dgroup0 each K-step like the gemm kernel does.
        lds_addr = []
        dg2 = []
        dg3 = []
        addr_lo = None
        addr_hi = None
        dg1 = None
        for s in range_constexpr(SUBOPS):
            sub_row0 = wid * arith.index(RPW) + arith.index(s * RPS)
            rows = []
            for i in range_constexpr(RPS):
                rows.append(
                    _raw(
                        buffer_ops.buffer_load(
                            gi_rsrc,
                            arith.index_cast(T.i32, sub_row0 + arith.index(i)),
                            vec_width=1,
                            dtype=T.i32,
                        )
                    )
                )
            lds_off = wid * arith.index(RPW * K) + arith.index(s * RPS * K)
            desc = tdm_ops.make_tensor_gather_descriptor(
                global_ptr=src,
                lds_memref=lds_base,
                row_indices=rows,
                row_width=KTILE,
                tensor_dim0=KFULL,
                tensor_dim1=n_tok.ir_value(),
                stride=KFULL,
                elem_bytes=1,
                pad_interval=KTILE,
                pad_amount=PAD,
                index_size=32,
                lds_byte_offset=lds_off,
                global_byte_offset=arith.index(0),
            )
            lds_addr.append(
                vector.extract(desc.dgroup0, static_position=[1], dynamic_position=[])
            )
            dg2.append(desc.dgroup2)
            dg3.append(desc.dgroup3)
            if s == 0:
                addr_lo = vector.extract(
                    desc.dgroup0, static_position=[2], dynamic_position=[]
                )
                addr_hi = vector.extract(
                    desc.dgroup0, static_position=[3], dynamic_position=[]
                )
                dg1 = desc.dgroup1

        for kt in range_constexpr(NKTILES):
            for s in range_constexpr(SUBOPS):
                dg0 = vector.from_elements(
                    T.vec(4, T.i32), [a_pred, lds_addr[s], addr_lo, addr_hi]
                )
                tdm_ops.tensor_load_gather(
                    tdm_ops.TDMGatherDescriptor(dg0, dg1, dg2[s], dg3[s])
                )
            tdm_ops.tensor_wait(0)
            gpu_d.barrier()
            # thread t copies row t of this K-tile (CHUNKS b128 chunks).
            for c in range_constexpr(CHUNKS):
                vec = lds_load_b128(
                    lds_base, tx * arith.index(K) + arith.index(c * 16)
                )
                out_dword = (
                    tx * arith.index(KFULL // 4)
                    + arith.index(kt * (KTILE // 4) + c * 4)
                )
                buffer_ops.buffer_store(
                    vec, out_rsrc, arith.index_cast(T.i32, out_dword)
                )
            gpu_d.barrier()
            addr_lo, addr_hi = _advance_addr(addr_lo, addr_hi, adv)

    @flyc.jit
    def launch(
        out: fx.Tensor,
        src: fx.Tensor,
        gidx: fx.Tensor,
        n_tok: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            alloc.finalized = False
            alloc.finalize()
        launcher = kernel(out, src, gidx, n_tok)
        launcher.launch(grid=(1, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)

    return launch


def main():
    torch.manual_seed(0)
    dev = "cuda"
    n_tok = 4096
    src = torch.randint(0, 256, (n_tok, KFULL), dtype=torch.uint8, device=dev)
    gidx = torch.randint(0, n_tok, (M,), dtype=torch.int32, device=dev)
    # inject some padding rows (-1 -> zero fill)
    gidx[5] = -1
    gidx[40] = -1
    gidx[127] = -1
    out = torch.empty((M, KFULL), dtype=torch.uint8, device=dev)

    launch = build()
    launch(
        out.view(M, KFULL),
        src.view(n_tok, KFULL),
        gidx,
        int(n_tok),
        stream=torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()

    ref = torch.zeros((M, KFULL), dtype=torch.uint8, device=dev)
    valid = gidx >= 0
    ref[valid] = src[gidx[valid].to(torch.long)]

    eq = (out == ref)
    nbad = int((~eq).sum().item())
    print(f"mismatched bytes: {nbad} / {M*KFULL}")
    bad_rows = (~eq).any(dim=1).nonzero().flatten().tolist()
    print(f"bad rows ({len(bad_rows)}): {bad_rows[:20]}")
    if nbad == 0:
        print("GATHER PRIMITIVE OK")
    else:
        for r in bad_rows[:4]:
            print(f"row {r}: gidx={int(gidx[r])}")
            print(f"  out={out[r].tolist()}")
            print(f"  ref={ref[r].tolist()}")


if __name__ == "__main__":
    main()
