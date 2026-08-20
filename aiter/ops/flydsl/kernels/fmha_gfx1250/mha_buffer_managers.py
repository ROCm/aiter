# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""16-bit (bf16/fp16) Q/K/V staging managers for the gfx1250 MHA kernels.

Each manager owns the ``global -> LDS (async) -> VGPR (WMMA fragment)`` path for
one 16-bit-element operand (Q, K or V): the LDS swizzle, the async copy schedule
and the fragment read. They are self-contained — the only things a caller passes
in are the *configuration* it already maintains (hdim, gqa_ratio, kv block width,
number of waves) via the constructor, and the runtime ``warp_idx`` / ``lane_idx``
into the specific member functions that need them. Nothing here reads a shared
module-level tiling constant from the kernel; the WMMA/tiling facts intrinsic to
the managed layout live below as private module constants.

The ``16b`` suffix names the element width (16-bit): every swizzle here assumes a
``b128 == 8`` element chunk (``_BF16_BYTES``). An 8-bit (fp8) variant would need
its own manager family (different chunk arithmetic), hence the explicit width tag.

Contents:
  - ``QManager16b`` — Q loader (ring-buffered async stage, natural ``ds_load_b128``).
  - ``KManager16b`` — K loader (one-block stage, natural ``ds_load_b128`` B-fragment).
  - ``VManager16b`` — V loader (V-specific swizzle, transpose ``ds_load_tr16_b128``).
  - ``OManager16b`` — O writer (WMMA accumulator -> LDS reshape -> coalesced store).

Target: gfx1250 (MI400 / mi450), wave32, 8 waves per threadgroup (256 threads).
"""

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl._mlir.dialects import rocdl as rocdl_dialect
from flydsl.expr import buffer_ops, gpu, rocdl

# ============================================================================
# Manager-intrinsic tiling constants (private — not the caller's config).
#
# These are fixed by the WMMA instruction + the swizzles implemented here, so
# they are NOT parameters. Anything the caller genuinely chooses (hdim, gqa_ratio,
# kv block width, wave count) arrives through a constructor argument instead.
# ============================================================================

# v_wmma_f32_16x16x32_bf16/f16 shape.
_WMMA_M = 16
_WMMA_K = 32
_BF16_BYTES = 2
_CHUNK_ELEMS = 8  # b128 = 8 bf16
_CHUNK_BYTES = _CHUNK_ELEMS * _BF16_BYTES  # 16

# gfx1250 is wave32; every swizzle/reshape here assumes 32 lanes per wave. FlyDSL
# only exposes the wave size compiler-side (GPUTarget.warp_size, from the arch),
# not as a trace-time Python int, so it is a named constant (== main file
# WAVE_SIZE). An fp8/CDNA (wave64) port would revisit the whole file, not just this.
_WAVE_LANES = 32

# Default 8-wave ("m16x8") threadgroup; override via the ``num_waves`` ctor arg.
_DEFAULT_NUM_WAVES = 8

# KV sequence block choices (columns of one QK GEMM tile).
_N_BLOCK_CHOICES = (32, 64, 128, 256)
_DEFAULT_N_BLOCK = 64

# O epilogue software pipeline (OManager16b, see its docstring). 64-col transpose
# units keep each coalesced global store a full 128B row; _O_INFLIGHT_UNITS units
# stay resident in an LDS ring and overlap.
_O_COLS_PER_TILE = 64
_O_INFLIGHT_UNITS = 2
_O_DSCNT_MAX = 63  # s_wait_dscnt SIMM16[5:0]

# LDS budget the O ring fits inside (the caller's non-current K|V slot):
# 8 waves * 2 units * 2KB = 32KB.
_O_LDS_BUDGET_BYTES = 32 * 1024

# K global->LDS async write tile: 8(kv) x 32(hdim) per warp call (one b128/lane).
_K_WR_TILE_KV = 8
_K_WR_TILE_HD = _WMMA_K

# V staging swizzle granularity (see VManager16b). A V block is stored as stacked
# 32(kv) x v_hdim sub-blocks; each is split into 32x32 tiles, each into 4(kv)x16(d)
# subtiles, with the subtile col index XOR-swizzled by (subtile row index & 1) to
# make the transpose load (ds_load_tr16_b128) bank-conflict-free.
_V_BLK_KV = 32  # kv rows per swizzle sub-block
_V_TILE = 32  # square tile side within a sub-block (kv and d)
_V_SUB_KV = 4  # subtile kv rows
_V_SUB_HD = 16  # subtile d cols
_V_WR_TILE_KV = 8  # async global->LDS write tile: kv rows per warp call
_V_WR_TILE_HD = _WMMA_K  # 32 d cols per write tile

# O staging (OManager16b): no padding -- an XOR swizzle on the 8-bf16 chunk index
# makes both the b128 store and b128 read bank-conflict-free (same idea as the
# K/QManager swizzle). MI400 LDS = 64 banks x 4 B, so an 8-bf16 (16 B) chunk spans
# one 4-bank group and bank_group(q, slot) = (q*G + slot) % 16 with G = v_hdim/8
# chunks per row. The store touches a fixed chunk-column across all 16 q-rows, so
# the swizzle must spread the slot over all 16 q; the read touches one full row
# (all G chunks) so any within-row bijection is already conflict-free. The swizzle
# ``slot = chunk ^ (q >> shift)``, shift = max(0, 4 - log2(G)), satisfies both:
# for v_hdim=128 (G=16) it is the full ``chunk ^ q``; the shift folds q's low bits
# already carried by the ``q*G`` term when G < 16. Verified conflict-free for
# v_hdim in {64,128,256} (see /tmp/async_probe/o_bank_check.py).


def _assert_multiple(name, val, mult):
    if isinstance(val, int):
        assert val % mult == 0, f"{name} must be a multiple of {mult}; got {val}"


# ---- gfx1250 Expert Scheduling Mode 2 covers -------------------------------
# DEP_MODE=2 turns the HW VA_VDST/VM_VSRC issue interlocks OFF. It is enabled via
# the `amdgpu-expert-scheduling-mode` LLVM hint the kernel passes at jit time (see
# _ensure_*_kernel); LLVM then emits the DEP_MODE=2 setreg AND covers the SSA-visible
# producer RAW hazards (WMMA->consumer, VALU-address->load) with its own post-RA
# depctr waits. What LLVM does NOT cover: WAR hazards (register reuse after a memory
# op) and anything around the opaque inline-asm memory ops it won't schedule into.
# Those we still cover explicitly here -- `s_wait_alu 0` (drain ALL depctr counters)
# plus field-targeted vm_vsrc/xcnt/dscnt -- immediately before/after the memory op
# that reuses a VALU-produced address/value.
#
# This flag is the single source of truth and is imported by the kernel module,
# which flips the LLVM hint in lockstep. Default False -> the _mem_* helpers emit
# the plain flydsl intrinsic and the kernel runs in normal mode 0 (bit-identical).
ENABLE_SCHED_MODE2 = True

# Field-targeted depctr cover reference (mnemonics + encodings verified via llvm-mc
# -mcpu=gfx1250). Under DEP_MODE=2 the HW inserts NO issue interlocks, but the
# depctr dependency COUNTERS remain maintained, so an explicit field-targeted
# s_wait_alu is a real *conditional* cover (waits only if the dep is live), not an
# unconditional s_nop delay. Used by the _mem_* helpers below:
#   depctr_va_vdst(0) [0xBF880F9F] — drain pending VALU->VGPR dest writes. Covers a
#     pre-op RAW: a VALU-produced load/store address or data read too early.
#   depctr_vm_vsrc(0) [0xBF88FF83] — wait until every prior VMEM/async op has READ
#     its VGPR sources. Covers a post-store WAR: the next op reuses (overwrites) the
#     store's address VGPR before the memory engine latched it.
#   s_wait_dscnt 0x0 — wait until all DS ops have COMPLETED (implies address read);
#     the ds post-op WAR (stronger than vm_vsrc; the only ds WAR measured non-harmful).
#   s_wait_xcnt 0x0 — wait until store addresses are TRANSLATED; the compiler's own
#     buffer_store WAR cover.


def _wait_tie(val, asm):
    """Emit an inline-asm cover TIED to ``val`` (an LLVM pointer or integer): ``val``
    is threaded in/out and the tied result returned. Tying forces the producer of
    ``val`` ABOVE the cover and every consumer BELOW it — LLVM has no data dep on a
    bare operand-less volatile asm and freely reorders pure VALU across it, so an
    UNTIED cover does not hold placement. No-op unless mode 2."""
    if not ENABLE_SCHED_MODE2:
        return val
    raw = val.ir_value() if hasattr(val, "ir_value") else val
    tied = llvm_dialect.inline_asm(raw.type, [raw], asm, "=v,0", has_side_effects=True)
    return type(val)(tied) if hasattr(val, "ir_value") else tied


# ===========================================================================
# Memory-op helpers — the SINGLE dispatch point for every load/store.
#
# mode 2 (ENABLE_SCHED_MODE2 True): emit the memory op AND its depctr covers
#   inside ONE inline-asm string. The compiler treats the string as opaque, so
#   the waits are PROVABLY adjacent to the op. (A separate `has_side_effects`
#   inline-asm only preserves ORDER, not adjacency — the compiler slips VALU
#   between the op and the wait; observed on the 5th O buffer_store where the
#   vm_vsrc drifted 3 VALU past the store.) Covers embedded per op:
# The covers (each a distinct depctr wait, individually toggleable):
#     RAW    `s_wait_alu depctr_va_vdst(0)` BEFORE — drain the VALU that produced
#            the address/data so the op reads final values (DEP_MODE=2 drops the
#            HW VA_VDST interlock).
#     DSCNT  `s_wait_dscnt 0x0` AFTER a ds op — the DS has COMPLETED (so its
#            address was read). Stronger than vm_vsrc for the ds address WAR, and
#            the ONLY ds post-WAR measured non-harmful: vm_vsrc(0) after the V
#            ds_load_tr16 was MEASURED HARMFUL (8192nc 4.4%->30.6%). This is the
#            default ds WAR.
#     VMVSRC `s_wait_alu depctr_vm_vsrc(0)` AFTER a buffer_store — the store has
#            READ its VGPR sources before RA reuses the address reg.
#     XCNT   `s_wait_xcnt 0x0` AFTER a buffer_store — address TRANSLATED; the
#            compiler's own buffer_store WAR cover, kept inside the block so it
#            can't drift. vm_vsrc and xcnt are SEPARATE waits, toggled separately.
# mode 0 (False): the plain flydsl intrinsic — HW issue interlocks handle it.
#
# Each cover is INDIVIDUALLY OPTIONAL via the _COV_* knobs below, so the mode-2
# cover set can be minimized empirically (per user 2026-08-19) without touching
# call sites. Defaults reproduce the current known-good cover set EXACTLY (no
# regression); flip a knob to drop that wait everywhere it applies. Every kernel
# memory op routes through one of these helpers; no site emits an intrinsic or a
# raw depctr wait directly. addr/offset args are the raw i32/i64 index values
# (NOT pre-wrapped llvm ptrs) so the asm operands bind to VGPRs directly.
# ===========================================================================

# RAW cover = FULL depctr drain (``s_wait_alu 0`` == s_waitcnt_depctr 0, drains ALL
# depctr fields to 0: va_vdst, vm_vsrc, sa_sdst, ...) + ``_RAW_NOP_COUNT`` s_nop 15
# pads. This is the committed-baseline (HEAD) RAW cover and is what makes mode 2
# correct at scale -- a field-targeted ``depctr_va_vdst(0)`` alone (no pad) MEASURED
# a MASSIVE regression (8192nc: 102/160 clean -> 0/160, all NaN-garbage). The full
# drain also covers the pre-op vm_vsrc WAR that the field cover misses. Narrow this
# (``s_wait_alu depctr_va_vdst(0)`` / _RAW_NOP_COUNT=0) ONLY with a >=8192nc
# NaN/max-abs-delta re-validation -- the default allclose gate hides the regression.
_W_RAW = "s_wait_alu 0"                     # FULL depctr drain BEFORE op (RAW+pre-WAR)
_RAW_NOP_COUNT = 8                          # s_nop 15 (~16 cyc) pads after the drain
_W_VMVSRC = "s_wait_alu depctr_vm_vsrc(0)"  # op read its VGPR sources AFTER op (WAR)
_W_DSCNT = "s_wait_dscnt 0x0"              # ds op completed AFTER op (WAR, implies addr read)
_W_XCNT = "s_wait_xcnt 0x0"                # store address translated AFTER store

# Per-cover minimization knobs (mode 2 only). Each is independent so vm_vsrc and
# xcnt can be toggled separately. Defaults = committed-baseline (HEAD) covers.
_COV_RAW = True          # full-drain (+nop pad) before every load/store
_COV_DS_VMVSRC = True    # vm_vsrc(0) fused BACK-TO-BACK after EVERY ds op (load+store):
                         # WAR on the ds addr/data VGPRs. Closes the epilogue
                         # ds_store -> v_cvt reg-reuse race (addr reg clobbered before
                         # the store reads it). NOTE: adding vm_vsrc(0) after V's tr16
                         # ds_load was previously measured HARMFUL (see _read_frag);
                         # re-validate at >=8192nc with the NaN/max-abs gate.
_COV_DSLOAD_WAR = True   # dscnt(0) after every ds_load / ds_load_tr16 (known-good)
_COV_DSSTORE_WAR = False # dscnt(0) after ds_store (good config had none)
_COV_BUF_VMVSRC = True   # vm_vsrc(0) after buffer_store
_COV_BUF_XCNT = True      # xcnt(0) after buffer_store
_COV_GLOAD_WAR = False   # vm_vsrc(0) after global_load (good config had none)


def _raw_cover():
    """The RAW cover block: full depctr drain + the s_nop 15 pad (HEAD baseline)."""
    return _W_RAW + "".join("\n\ts_nop 15" for _ in range(_RAW_NOP_COUNT))


def _mem_asm(op_line, post_waits, *, raw=None):
    """Assemble one inline-asm string: the RAW cover (if enabled) BEFORE the op
    mnemonic line, then each enabled post-wait AFTER. ``post_waits`` is a list of
    (asm_string, enabled_bool) pairs — a wait is emitted only when its flag is set,
    so a call site never changes when a knob flips. ``raw`` overrides _COV_RAW for
    this one asm (None -> use the global knob)."""
    parts = []
    if _COV_RAW if raw is None else raw:
        parts.append(_raw_cover())
    parts.append(op_line)
    for w, en in post_waits:
        if en:
            parts.append(w)
    return "\n\t".join(parts)


def _mem_ds_load(addr_i32, out_ty, *, tr16=False, raw=None, war=None):
    """LDS b128 load into VGPR vector ``out_ty`` (natural ``ds_load_b128`` or, when
    ``tr16``, the transpose ``ds_load_tr16_b128``). ``addr_i32`` = LDS byte address
    (fx.Int32). Returns an fx.Vector. ``raw``/``war`` override the global cover knobs
    for this one call (None -> use _COV_RAW / _COV_DSLOAD_WAR); the Q part2 reads pass
    both False (their ordering is handled by the surrounding asynccnt/dscnt pipeline).
    Under mode 2 EVERY ds op is emitted as inline asm (no intrinsic fast-path) so a
    vm_vsrc(0) WAR can be fused back-to-back after the op regardless of raw/war."""
    raw = _COV_RAW if raw is None else raw
    war = _COV_DSLOAD_WAR if war is None else war
    if not ENABLE_SCHED_MODE2:
        lds_ptr = buffer_ops.create_llvm_ptr(addr_i32, address_space=3)
        if tr16:
            return fx.Vector(rocdl.ds_load_tr16_b128(out_ty, lds_ptr))
        return fx.Vector(llvm_dialect.load(out_ty, lds_ptr))
    mnem = "ds_load_tr16_b128" if tr16 else "ds_load_b128"
    # vm_vsrc(0) FIRST (back-to-back after the op) = WAR on the addr reg; dscnt(0) next.
    asm = _mem_asm(
        f"{mnem} $0, $1", [(_W_VMVSRC, _COV_DS_VMVSRC), (_W_DSCNT, war)], raw=raw
    )
    res = llvm_dialect.inline_asm(
        out_ty, [_ir(addr_i32)], asm, "=v,v", has_side_effects=True
    )
    return fx.Vector(res)


def _mem_ds_store(addr_i32, data_vec):
    """LDS b128 store of ``data_vec`` (v8 bf16) to LDS byte address ``addr_i32``.
    The single RAW cover drains the VALU that produced BOTH the address and the
    data before the store issues."""
    if not ENABLE_SCHED_MODE2:
        lds_ptr = buffer_ops.create_llvm_ptr(addr_i32, address_space=3)
        llvm_dialect.store(_ir(data_vec), lds_ptr, alignment=_CHUNK_BYTES)
        return
    # vm_vsrc(0) FIRST (back-to-back after the store) = WAR on the addr/data regs
    # (closes the epilogue store -> v_cvt reg-reuse race); dscnt(0) optional next.
    asm = _mem_asm(
        "ds_store_b128 $0, $1",
        [(_W_VMVSRC, _COV_DS_VMVSRC), (_W_DSCNT, _COV_DSSTORE_WAR)],
    )
    llvm_dialect.inline_asm(
        None, [_ir(addr_i32), _ir(data_vec)], asm, "v,v", has_side_effects=True
    )


def _mem_buffer_store(data, rsrc, voffset_i32, *, width):
    """``buffer_store_b{width}`` of ``data`` to VRAM via ``rsrc`` (!llvm.ptr<8> SRD)
    at BYTE ``voffset_i32``. ``width`` in {32, 128}. Caller must have pre-masked the
    offset (OOB -> 0x7fffffff); ``mask`` is not used so the store maps 1:1 to the
    hand-written mnemonic. vm_vsrc and xcnt are separate, separately-toggled WARs."""
    if not ENABLE_SCHED_MODE2:
        buffer_ops.buffer_store(
            data, rsrc, voffset_i32, mask=None, offset_is_bytes=True
        )
        return
    mnem = f"buffer_store_b{width}"
    asm = _mem_asm(
        f"{mnem} $0, $1, $2, null offen",
        [(_W_VMVSRC, _COV_BUF_VMVSRC), (_W_XCNT, _COV_BUF_XCNT)],
    )
    llvm_dialect.inline_asm(
        None,
        [_ir(data), _ir(voffset_i32), _ir(rsrc)],
        asm,
        "v,v,s",
        has_side_effects=True,
    )


def _mem_global_load_b32(addr_i64, out_ty):
    """Flat/global 32-bit load from VRAM byte address ``addr_i64`` (fx.Int64) into
    scalar-per-lane ``out_ty``. Used for the per-head sink logit. Returns the raw
    loaded value (wrap at the call site)."""
    if not ENABLE_SCHED_MODE2:
        gptr = buffer_ops.create_llvm_ptr(addr_i64, address_space=1)
        return llvm_dialect.load(out_ty, gptr)
    asm = _mem_asm("global_load_b32 $0, $1, off", [(_W_VMVSRC, _COV_GLOAD_WAR)])
    return llvm_dialect.inline_asm(
        out_ty, [_ir(addr_i64)], asm, "=v,v", has_side_effects=True
    )


# --- Mode-2 async-load ASM GLUE (THE fix for the large-seqlen WAR NaN) --------
# A frontend-placed cover (any counter, tied or not) cannot fix the async-load
# address-VGPR WAR: post-RA, the register allocator inserts the next-chunk offset
# overwrite (`v_mov`/`v_add`) at load+1, BEFORE the earliest slot a separate cover
# can occupy (load+2). Mode 0's HW VM_VSRC interlock held that overwrite until the
# async engine latched the address; mode 2 removes it -> the overwrite races the
# latch -> corrupt address -> NaN / wild-read / page-fault, but ONLY at scale
# (>=8192, when many async loads are queued and the latch is delayed).
#
# The only guaranteed fix is to make the load and its WAR wait ATOMIC: emit both
# inside one inline-asm string. The compiler treats the string as opaque and will
# not insert instructions into it, so the wait is provably adjacent to the load
# and the overwrite is forced strictly after.
#
# The load's three address regs ($0 LDS-dest, $1 global offset, $2 base) have TWO
# distinct release points; the RA reuses them immediately (the very next instr
# often overwrites the LDS-dest reg $0), so the WAR must cover BOTH:
#   (a) GLOBAL source address ($1 offset, $2 base): safe once TRANSLATED.
#       s_wait_xcnt 0 -> XCNT counts "issued but not-yet-translated" mem ops; this
#       is the compiler's OWN cover for the identical buffer_store address-reuse WAR
#       in this same ISA. vm_vsrc(0) [source-read] is included as its companion
#       (sources entered the pipe). Together they clear all NaN/page-faults and are
#       0-clean up to 8192 causal and 16384 NON-causal.
#   (b) LDS-DEST address ($0): the async engine does NOT re-read $0 from the VGPR
#       at data-arrival — it latches $0 into the async queue entry a few cycles
#       AFTER issue. Neither xcnt nor vm_vsrc waits for that internal latch, so at
#       16384 CAUSAL the immediate $0 overwrite still raced it (262/16.7M finite-
#       wild, no NaN). A short fixed delay lets the latch complete: s_nop 8 (~9 cyc)
#       closes it with margin. asynccnt 0 also fixes it but costs 100s of cycles
#       (full data-arrival) and is NOT needed -- the latch, not the write, is the
#       dependency. Measured 0-clean across 512..16384, causal + non-causal.
# Costs: vm_vsrc/xcnt ~<=10 cyc each, s_nop 8 ~9 cyc (vs asynccnt 0 ~100s).
_COVER_GLUE_WAR = "s_wait_alu depctr_vm_vsrc(0)\ns_wait_xcnt 0x0"  # buffer_store WAR
# Async-load WAR: drop s_wait_xcnt 0 (measured ~155 cyc in the hot loop; the global
# source-address WAR it guarded is instead held by vm_vsrc alone, re-validate @scale).
# vm_vsrc is itself optional per-caller: Q (QMgr) disables it because its graduated
# s_wait_asynccnt culminates in asynccnt(0) before any address-VGPR reuse, which
# already covers the WAR; K/V keep vm_vsrc.
_COVER_GLUE_WAR_ASYNC = "s_wait_alu depctr_vm_vsrc(0)"


def _ir(x):
    """Unwrap an fx value to its raw MLIR ir.Value (pass-through if already raw)."""
    return x.ir_value() if hasattr(x, "ir_value") else x


def _async_load_to_lds(
    base_i64, g_offs, lds_offs, imm_offs=None, *, cluster, war=_COVER_GLUE_WAR_ASYNC
):
    """Emit a BATCH of async 16B (b128) global->LDS loads sharing one global base.

    ``base_i64`` uniform global base (fx.Int64), shared by every load in the batch.
    ``g_offs`` / ``lds_offs`` are equal-length lists — one entry per load:
      g_offs[i]   per-lane global byte offset (fx.Int32, divergent)
      lds_offs[i] LDS byte offset (fx.Int32)
    ``imm_offs`` optional equal-length list of compile-time immediate BYTE offsets
      (Python ints; default all 0). The b128 async instruction's ``offset:`` imm is
      applied by HW to BOTH the VRAM source AND the LDS destination in lockstep
      (24-bit signed; see memory ``gfx1250-async-load-to-lds``), so a nonzero imm
      shifts src and dst together — use it only for a delta that is identical on
      both sides. ``cluster`` selects the MCAST form (K/V) vs plain global (Q).

    A scalar (non-list) g_off/lds_off/imm_off is accepted and treated as a 1-load
    batch (back-compat).

    mode 0: each load is the plain rocdl intrinsic — HW issue interlocks handle
      RAW/WAR. The imm is folded into BOTH the global and LDS addresses (address
      arithmetic), matching the HW lockstep, since the intrinsic form here takes
      no separate imm field we rely on.
    mode 2: emitted as ONE inline-asm block so the compiler cannot slip a next-
      chunk address-VGPR overwrite between a load and its WAR wait. Layout:
        s_wait_alu depctr_va_vdst(0)                    # RAW once: addr VALUs retired
        <mnem> $0,     $n,     $2n  offset:imm0         # load 0
        <mnem> $1,     $n+1,   $2n  offset:imm1         # load 1
        ...                                             # ... N loads back-to-back
        <war>                                           # WAR once, after LAST load
      ``war`` is the trailing WAR-cover string (default ``_COVER_GLUE_WAR_ASYNC`` =
      vm_vsrc only; pass None/"" to omit it and let the caller own the WAR, e.g. Q
      whose graduated asynccnt(0) already covers address-VGPR reuse).
      Operands: $0..$(n-1) = LDS byte offsets (v,i32); $n..$(2n-1) = global
      voffsets (v,i32, divergent); $2n = uniform global base (s, i64 sgpr-pair),
      shared. HW computes each global address as base + voffset(+imm); the divergent
      voffsets stay distinct per-chunk regs, the uniform high bits live in the
      scalar base and can never fault. All operand regs are held live across the
      whole block, so the single trailing WAR covers every load's address WAR."""
    if not isinstance(g_offs, (list, tuple)):
        g_offs = [g_offs]
    if not isinstance(lds_offs, (list, tuple)):
        lds_offs = [lds_offs]
    n = len(g_offs)
    if len(lds_offs) != n:
        raise ValueError(f"g_offs/lds_offs length mismatch: {n} vs {len(lds_offs)}")
    if imm_offs is None:
        imm_offs = [0] * n
    elif not isinstance(imm_offs, (list, tuple)):
        imm_offs = [imm_offs]
    if len(imm_offs) != n:
        raise ValueError(f"imm_offs length mismatch: {len(imm_offs)} vs {n}")

    if not ENABLE_SCHED_MODE2:
        for g_off, lds_off, imm in zip(g_offs, lds_offs, imm_offs):
            # imm applies to both sides in lockstep -> fold into each address.
            g_full = g_off + fx.Int32(imm) if imm else g_off
            lds_full = lds_off + fx.Int32(imm) if imm else lds_off
            gptr = buffer_ops.create_llvm_ptr(
                base_i64 + fx.Int64(g_full), address_space=1
            )
            lds_ptr = buffer_ops.create_llvm_ptr(lds_full, address_space=3)
            if cluster:
                rocdl.cluster_load_async_to_lds(gptr, lds_ptr, _CHUNK_BYTES)
            else:
                rocdl_dialect.global_load_async_to_lds_b128(gptr, lds_ptr, 0, 0)
        return

    mnem = (
        "cluster_load_async_to_lds_b128" if cluster else "global_load_async_to_lds_b128"
    )
    lds_rs = [_ir(x) for x in lds_offs]
    off_rs = [_ir(x) for x in g_offs]
    base_r = _ir(base_i64)
    operands = lds_rs + off_rs + [base_r]
    constraints = ",".join(["v"] * (2 * n) + ["s"])
    lines = []
    for i, imm in enumerate(imm_offs):
        off_field = f" offset:{imm}" if imm else ""
        lines.append(f"\t{mnem} ${i}, ${n + i}, ${2 * n}{off_field}")
    if war:  # WAR after LAST load
        lines.append(f"\t{war}")
    llvm_dialect.inline_asm(
        None, operands, "\n".join(lines), constraints, has_side_effects=True
    )


# ============================================================================
# Q loader (global -> LDS async -> VGPR WMMA fragments)
# ============================================================================


class QManager16b:
    """Owns everything about Q: its LDS footprint and the global->LDS->VGPR load.

    Keeping this behind one object means the compute core just asks the manager
    how much LDS to reserve (``get_lds_size_in_byte``) and then hands the raw
    allocation base to ``load_q_to_vgpr`` — the per-warp sub-offset and the
    swizzled staging layout are entirely the manager's business. A future Q
    strategy (different tiling / dtype) is a drop-in replacement.

    Staging layout (per warp, tile-major): each of ``qk_hdim // _WMMA_K`` K-tiles
    is 16 rows x _WMMA_K cols bf16 (1024 B). Within a tile, 4x4 subtiles of
    4 rows x 8 cols; the 8-col b128 chunk index is XOR-swizzled with the 4-row
    subtile index to spread LDS banks. The waves stack their tiles contiguously.

    Config (caller-maintained) arrives via the constructor: ``qk_hdim``,
    ``gqa_ratio``, ``num_waves`` and the ring depth ``lds_tiles``.
    """

    def __init__(self, *, qk_hdim, gqa_ratio, num_waves=_DEFAULT_NUM_WAVES, lds_tiles=None):
        if qk_hdim % _WMMA_K != 0:
            raise ValueError(f"qk_hdim must be a multiple of {_WMMA_K}; got {qk_hdim}")
        self.qk_hdim = qk_hdim  # compile-time
        self.gqa_ratio = gqa_ratio  # compile-time
        self.num_waves = num_waves  # compile-time (threadgroup wave count)
        self.block_m = _WMMA_M * num_waves  # Q rows per threadgroup
        self.k_tiles = qk_hdim // _WMMA_K

        # Ring-buffer depth: how many K-tiles of a wave's Q live in LDS at once
        # (2 async b128 per tile). Sets both the LDS footprint and the inflight-
        # async budget so they can't drift. Default == k_tiles fully buffers Q, so
        # every async copy is in flight at once and none waits on a ds_load to free
        # a slot; smaller trades that overlap for less LDS (freeing it for K/V).
        if lds_tiles is None:
            lds_tiles = self.k_tiles
        if not 1 <= lds_tiles <= self.k_tiles:
            raise ValueError(
                f"lds_tiles must be in [1, {self.k_tiles}]; got {lds_tiles}"
            )
        self.lds_tiles = lds_tiles
        self._warp_stride = _WMMA_M * self.lds_tiles * _WMMA_K * _BF16_BYTES

    def get_lds_size_in_byte(self):
        """LDS bytes the caller must reserve for Q staging (all waves)."""
        return self.num_waves * self._warp_stride

    def _lds_byte(self, row, col_chunk, tile):
        """Swizzled LDS byte offset (within a warp region) for a 8-col b128 chunk."""
        sw = col_chunk ^ (row // 4)  # 4x4 subtile XOR swizzle
        return (
            tile * (_WMMA_M * _WMMA_K * _BF16_BYTES)
            + row * (_WMMA_K * _BF16_BYTES)
            + sw * _CHUNK_BYTES
        )

    def _async_load_vram_to_lds(self, q_base_i64, g_offs, lds_offs, imm_offs=None, war=None):
        """gfx1250 async 16B global->LDS copy — accepts a batch (equal-length lists,
        or scalars for one load). ``imm_offs`` shifts BOTH src and dst by the same
        bytes in lockstep (default 0). mode-2 emits the batch as ONE atomic inline-asm
        block. Q passes ``war=None`` to omit the trailing async WAR cover: the
        graduated ``s_wait_asynccnt`` in ``load_q_to_vgpr`` reaches asynccnt(0) before
        any address-VGPR reuse (phase-3 shuffle/scale), which already covers the WAR.
        Q uses the plain (non-MCAST) global form."""
        _async_load_to_lds(
            q_base_i64, g_offs, lds_offs, imm_offs, cluster=False, war=war
        )

    def global_load_params(
        self,
        *,
        lds_q_base,  # fx.Int32: byte base of THIS warp's LDS region
        warp_row0,  # fx.Int32: global Q-row of this warp's row 0
        kv_head,
        q_start,
        q_len,
        stride_q_seq,
        stride_q_head,
        lane_idx,
    ):
        """Params for EVERY ``global_load_async_to_lds_b128`` of this warp's Q tile.

        Returns ``(g_offs, lds_offs, imm_offs)`` — three equal-length lists, one
        entry per async b128 group: ``k_tiles`` tiles x 2 half-loads = 8 / 12 / 16
        groups for qk_hdim 128 / 192 / 256. Pure index arithmetic (no memory op) so
        the caller can hoist ALL address VALU ahead of the load burst. ``g_offs`` /
        ``lds_offs`` are the per-lane global (VRAM) and LDS byte offsets; ``imm_offs``
        is 0 (the intra-tile hdim shift is already folded into ``g_off``). Rows with
        seq >= q_len are clamped in-bounds (masked later in softmax)."""
        g_offs, lds_offs = [], []
        for tile in fx.range_constexpr(self.k_tiles):
            for half in fx.range_constexpr(2):
                row = lane_idx // 4 + half * 8  # row within warp [0,16)
                col_chunk = lane_idx % 4  # 8-col chunk within the 32-col tile [0,4)
                pr = warp_row0 + row
                q_head = kv_head * self.gqa_ratio + pr % self.gqa_ratio
                seq = pr // self.gqa_ratio
                safe_seq = (seq < q_len).select(seq, fx.Int32(0))  # clamp OOB
                token = q_start + safe_seq
                g_off = (
                    token * stride_q_seq
                    + q_head * stride_q_head
                    + fx.Int32(tile * _WMMA_K * _BF16_BYTES)
                    + col_chunk * _CHUNK_BYTES
                )
                g_offs.append(g_off)
                # LDS slot wraps mod lds_tiles: tile t reuses slot t % lds_tiles.
                slot = tile % self.lds_tiles
                lds_offs.append(lds_q_base + self._lds_byte(row, col_chunk, slot))
        imm_offs = [0] * len(g_offs)
        return g_offs, lds_offs, imm_offs

    def ds_load_params(self, *, lds_q_base, lane_idx):
        """LDS byte offsets for EVERY ``ds_load_b128`` read of this warp's Q tile.

        Returns a flat list of ``k_tiles`` x 2 (lo, hi) = 8 / 12 / 16 read offsets:
        read ``2t`` is tile ``t``'s low 8-col half, read ``2t+1`` its high half; the
        pair shuffles into the 16x32 WMMA A fragment. Pure index math (no memory
        op)."""
        row = lane_idx % _WMMA_M
        klane = lane_idx // _WMMA_M  # 0 or 1
        offs = []
        for tile in fx.range_constexpr(self.k_tiles):
            slot = tile % self.lds_tiles  # match global_load_params ring slot
            offs.append(lds_q_base + self._lds_byte(row, klane, slot))  # lo
            offs.append(lds_q_base + self._lds_byte(row, klane + 2, slot))  # hi
        return offs

    def load_q_to_vgpr_part1(
        self,
        *,
        ptr_Q,
        stride_q_seq,
        stride_q_head,
        q_start,
        q_len,
        kv_head,
        block_x,  # fx.Int32: this workgroup's grid-x tile index
        warp_idx,
        lane_idx,
        ptr_lds,  # fx.Int32: base byte addr of the caller's Q allocation
    ):
        """Part 1 of the Q load: compute all global/LDS offsets (stashed as members),
        then issue the prime chunk of async global->LDS loads. A trailing
        ``sched_barrier`` pins these above the caller's SALU so that work fills the
        load's shadow. Call ``load_q_to_vgpr_part2`` after to drain + read."""
        lds_q_base = ptr_lds + warp_idx * self._warp_stride
        q_base_i64 = fx.Int64(fx.ptrtoint(fx.get_iter(ptr_Q)))
        warp_row0 = block_x * self.block_m + warp_idx * _WMMA_M

        # All address VALU up front (2 async b128 + 2 ds_load per tile).
        g_offs, lds_wr_offs, imm_offs = self.global_load_params(
            lds_q_base=lds_q_base,
            warp_row0=warp_row0,
            kv_head=kv_head,
            q_start=q_start,
            q_len=q_len,
            stride_q_seq=stride_q_seq,
            stride_q_head=stride_q_head,
            lane_idx=lane_idx,
        )
        ds_offs = self.ds_load_params(lds_q_base=lds_q_base, lane_idx=lane_idx)

        # Prime: issue the first lds_tiles tiles (2 loads each).
        n_prime = 2 * self.lds_tiles
        self._async_load_vram_to_lds(
            q_base_i64, g_offs[:n_prime], lds_wr_offs[:n_prime], imm_offs[:n_prime],
            war=_COVER_GLUE_WAR_ASYNC,
        )
        rocdl.sched_barrier(0)  # pin the prime loads above the caller's SALU

        # Stash for part 2 (drain + reads, and steady-loop refills when lds_tiles<k_tiles).
        self._q_base_i64 = q_base_i64
        self._q_g_offs = g_offs
        self._q_lds_wr_offs = lds_wr_offs
        self._q_imm_offs = imm_offs
        self._q_ds_offs = ds_offs

    def load_q_to_vgpr_part2(self, *, scale):
        """Part 2 of the Q load: drain the async loads issued in part 1 and read the
        tiles into WMMA A fragments (``scale`` folded in). A leading ``sched_barrier``
        keeps the waits/reads below the caller's SALU so it stays in the load shadow.
        Async is drained GRADUALLY (in-issue-order assumption); ``lds_tiles ==
        k_tiles`` (default) skips the steady loop = fully-resident drain-only."""
        rocdl.sched_barrier(0)  # keep waits/reads below the caller's SALU
        k_tiles = self.k_tiles
        lds_tiles = self.lds_tiles
        ds_offs = self._q_ds_offs
        v8_ty = fx.Vector.make_type(_CHUNK_ELEMS, fx.BFloat16)

        def _read_tile(tile):
            # Self-scheduled reads: the surrounding s_wait_asynccnt / s_wait_dscnt
            # pipeline (below) already orders these against the async refills, and the
            # tile offsets are loop-invariant, so no per-op mode-2 cover is needed.
            # raw=war=False -> plain intrinsic under mode 2 too (matches known-good).
            lo = _mem_ds_load(ds_offs[2 * tile], v8_ty, raw=False, war=False)
            hi = _mem_ds_load(ds_offs[2 * tile + 1], v8_ty, raw=False, war=False)
            return lo, hi

        def _refill(tile):
            lo = 2 * tile
            self._async_load_vram_to_lds(
                self._q_base_i64, self._q_g_offs[lo:lo + 2],
                self._q_lds_wr_offs[lo:lo + 2], self._q_imm_offs[lo:lo + 2],
            )

        scale_bf16 = _wait_tie(scale.to(fx.BFloat16), "s_wait_alu depctr_va_vdst(0)")

        q_frags = []
        # Steady loop: read+evict tile (i-lds_tiles), refill tile i into its slot.
        for i in fx.range_constexpr(lds_tiles, k_tiles):
            rocdl.s_wait_asynccnt((lds_tiles - 1) * 2)  # oldest tile's 2 loads landed
            lo, hi = _read_tile(i - lds_tiles)
            rocdl.s_wait_dscnt(0)  # slot free to overwrite
            _refill(i)
            q_frags.append(lo.shuffle(hi, list(range(16))) * scale_bf16)
        # Drain loop: read the last lds_tiles tiles, no refill; overlap lo scale w/ hi load.
        for i in fx.range_constexpr(0, lds_tiles):
            rocdl.s_wait_asynccnt((lds_tiles - 1 - i) * 2)
            lo, hi = _read_tile(k_tiles - lds_tiles + i)
            rocdl.s_wait_dscnt(1)  # lo landed (in-order LDS return)
            lo = lo * scale_bf16
            rocdl.s_wait_dscnt(0)  # hi landed
            hi = hi * scale_bf16
            q_frags.append(lo.shuffle(hi, list(range(16))))
        return q_frags


# ============================================================================
# K loader (global -> LDS async -> VGPR WMMA B-fragments)
# ============================================================================


class KManager16b:
    """Owns K's LDS staging and the global->LDS->VGPR B-fragment load.

    Unlike QManager16b there is no ring buffer here: KManager16b only reports the
    byte size of ONE ``n_block x qk_hdim`` K block (``get_lds_size_in_byte``). The
    caller reserves however many ping-pong buffers it wants and passes the chosen
    buffer base (``ptr_lds``) into every method — the manager is not bound to a
    buffer. The K block is shared by all waves (each computes S[16, n_block]).

    LDS layout mirrors QManager16b: ``(n_block/16)`` kv-subtiles x ``(qk_hdim/32)``
    hdim-units, each a 16x32 bf16 tile (1024 B) with the 4x4 XOR swizzle
    (``sw = chunk ^ row//4``). The global->LDS write API streams 8(kv)x32(hdim)
    tiles (``row_idx`` = kv, mult of 8; ``col_idx`` = hdim, mult of 32) — one b128
    per lane, each an 8-row half of a unit. The VGPR read API pulls 16x16 tiles
    (``col_idx`` mult of 16); a ``col_idx``/``col_idx+16`` pair combines into one
    16x32 WMMA B operand. Loads use ``cluster_load_async_to_lds_b128``
    (MCAST-ready; mask 0 for now).

    Config (caller-maintained) arrives via the constructor: ``qk_hdim``,
    ``n_block`` and ``num_waves``.
    """

    def __init__(self, *, qk_hdim, n_block=_DEFAULT_N_BLOCK, num_waves=_DEFAULT_NUM_WAVES):
        if qk_hdim % _WMMA_K != 0:
            raise ValueError(f"qk_hdim must be a multiple of {_WMMA_K}; got {qk_hdim}")
        if n_block not in _N_BLOCK_CHOICES:
            raise ValueError(f"n_block must be one of {_N_BLOCK_CHOICES}; got {n_block}")
        self.qk_hdim = qk_hdim  # compile-time
        self.n_block = n_block  # compile-time
        self.num_waves = num_waves  # compile-time
        self.hd_units = qk_hdim // _WMMA_K  # 32-hdim WMMA units
        # 8x32 write-tile grid (async global->LDS): rows = kv, cols = hdim.
        self.n_wr_tile_rows = n_block // _K_WR_TILE_KV
        self.n_wr_tile_cols = qk_hdim // _K_WR_TILE_HD

    def get_lds_size_in_byte(self):
        """LDS bytes for one n_block x qk_hdim K block (one ping-pong buffer)."""
        return self.n_block * self.qk_hdim * _BF16_BYTES

    def _lds_byte(self, tile_row, tile_col, row_in_tile, chunk):
        """Swizzled LDS byte offset within one K block. LDS is a grid of 16x32
        WMMA tiles; ``tile_row``/``tile_col`` index that grid (row = kv,
        col = hdim). ``row_in_tile`` (0..15) is the row inside the tile and
        ``chunk`` (0..3) the 8-hdim b128 within the 32-wide tile. Same 4x4 XOR
        swizzle as QManager16b."""
        tile_base = (
            (tile_row * self.hd_units + tile_col) * (_WMMA_M * _WMMA_K * _BF16_BYTES)
        )
        sw = chunk ^ (row_in_tile // 4)
        return (
            fx.Int32(tile_base)
            + row_in_tile * (_WMMA_K * _BF16_BYTES)
            + sw * _CHUNK_BYTES
        )

    def async_load_vram_to_lds_wr_tile(
        self,
        *,
        ptr_lds,
        k_base_i64,  # fx.Int64: ptrtoint(get_iter(ptr_K))
        stride_k_seq,
        stride_k_head,
        kv_head,
        kv_row0,  # fx.Int32: global token of this block's kv-row 0
        kv_valid,  # fx.Int32: valid kv rows in this block (only read if check_oob)
        row_idx,  # kv offset within block, mult of 8, < n_block
        col_idx,  # hdim offset, mult of 32, < qk_hdim
        lane_idx,
        check_oob=True,  # compile-time: clamp rows >= kv_valid in-bounds
    ):
        """One warp streams an 8(kv) x 32(hdim) tile into ``ptr_lds`` — one b128
        (8 bf16) per lane: lane -> (wr_row = lane//4, chunk = lane%4). ``row_idx``/
        ``col_idx`` may be Python ints or fx values (all addressing below is plain
        integer arithmetic). When ``check_oob`` is False the caller guarantees the
        whole block is in-bounds and the clamp is skipped."""
        _assert_multiple("row_idx", row_idx, _K_WR_TILE_KV)
        _assert_multiple("col_idx", col_idx, _K_WR_TILE_HD)
        wr_row = lane_idx // 4  # kv row within the 8-row write tile [0,8)
        chunk = lane_idx % 4  # which 8-hdim b128 [0,4) -> spans the 32-wide tile
        tile_row = row_idx // _WMMA_M  # which 16-kv LDS tile
        row_in_tile = (row_idx % _WMMA_M) + wr_row  # row within that tile [0,16)
        tile_col = col_idx // _WMMA_K

        kv_row = fx.Int32(row_idx) + wr_row
        if check_oob:
            kv_row = (kv_row < kv_valid).select(kv_row, fx.Int32(0))  # clamp OOB
        token = kv_row0 + kv_row
        g_off = (
            token * stride_k_seq
            + kv_head * stride_k_head
            + (fx.Int32(col_idx) + chunk * _CHUNK_ELEMS) * _BF16_BYTES
        )
        lds_off = ptr_lds + self._lds_byte(tile_row, tile_col, row_in_tile, chunk)
        # mode-2: load + RAW/WAR covers emitted as ONE atomic inline-asm block so
        # the RA cannot slip the next-chunk offset overwrite between load and wait
        # (see _async_load_to_lds). The scalar-base + per-chunk voffset form is kept
        # by passing base/offset separately.
        _async_load_to_lds(k_base_i64, g_off, lds_off, cluster=True)

    def async_load_vram_to_lds(
        self,
        *,
        ptr_lds,
        ptr_K,
        stride_k_seq,
        stride_k_head,
        kv_head,
        kv_row0,
        kv_valid,
        warp_idx,
        lane_idx,
        check_oob=True,
    ):
        """Load the whole ``n_block x qk_hdim`` K block into ``ptr_lds``.

        The 8x32 write-tile grid (``n_wr_tile_rows x n_wr_tile_cols``) is spread
        round-robin across the waves: warp ``w`` streams tiles ``w, w+num_waves, ...``.
        ``row_idx``/``col_idx`` are derived from the runtime warp id, so the tile
        method runs with runtime indices here (plain integer arithmetic).
        ``check_oob`` is forwarded to each tile (skip the clamp for full blocks)."""
        n_tiles = self.n_wr_tile_rows * self.n_wr_tile_cols
        if n_tiles % self.num_waves != 0:
            raise NotImplementedError(
                f"K tile grid ({n_tiles}) must be divisible by {self.num_waves} warps; "
                f"got n_block={self.n_block}, qk_hdim={self.qk_hdim}"
            )
        k_base_i64 = fx.Int64(fx.ptrtoint(fx.get_iter(ptr_K)))
        for i in fx.range_constexpr(n_tiles // self.num_waves):
            tile_id = warp_idx + fx.Int32(i * self.num_waves)
            row_idx = (tile_id // self.n_wr_tile_cols) * _K_WR_TILE_KV
            col_idx = (tile_id % self.n_wr_tile_cols) * _K_WR_TILE_HD
            self.async_load_vram_to_lds_wr_tile(
                ptr_lds=ptr_lds,
                k_base_i64=k_base_i64,
                stride_k_seq=stride_k_seq,
                stride_k_head=stride_k_head,
                kv_head=kv_head,
                kv_row0=kv_row0,
                kv_valid=kv_valid,
                row_idx=row_idx,
                col_idx=col_idx,
                lane_idx=lane_idx,
                check_oob=check_oob,
            )

    # ------------------------------------------------------------------
    def load_lds_to_vgpr_tile_as_k(self, *, ptr_lds, row_idx, col_idx, lane_idx):
        """Read one 16x16 tile as a WMMA fragment via natural ``ds_load_b128``
        (row_idx=kv, col_idx=hdim; both mult of 16) -> v8 bf16 per lane.

        lane -> (row_in_tile = lane%16, col_half = lane//16); returns this lane's 8
        hdim cols [col_idx + col_half*8 .. +8) of kv-row (row_idx + lane%16). Two
        adjacent calls (col_idx, col_idx+16) shuffle into a 16x32 WMMA fragment."""
        _assert_multiple("row_idx", row_idx, _WMMA_M)
        _assert_multiple("col_idx", col_idx, _WMMA_M)
        row_in_tile = lane_idx % _WMMA_M
        col_half = lane_idx // _WMMA_M  # 0 or 1: which 8-hdim half of the 16 cols
        tile_row = row_idx // _WMMA_M
        tile_col = col_idx // _WMMA_K
        chunk_base = (col_idx % _WMMA_K) // _CHUNK_ELEMS  # 0 or 2
        chunk = fx.Int32(chunk_base) + col_half
        v8_ty = fx.Vector.make_type(_CHUNK_ELEMS, fx.BFloat16)
        addr = ptr_lds + self._lds_byte(tile_row, tile_col, row_in_tile, chunk)
        return _mem_ds_load(addr, v8_ty)


# ============================================================================
# V loader (global -> LDS async -> VGPR WMMA A-fragments via transpose load)
# ============================================================================


class VManager16b:
    """Owns V's LDS staging and the global->LDS->VGPR **transpose** load.

    PV computes O^T = V^T @ P^T, so V is the WMMA A-operand ``V^T[d, kv]``. Since V
    is stored ``[kv, d]`` (d contiguous) but PV contracts over kv, the LDS->VGPR read
    uses ``ds_load_tr16_b128`` (transpose) rather than K's natural ``ds_load_b128``.
    See [[ds-load-tr16-b128-behavior]]: the load is (1) a per-lane b128 fetch where
    bank conflicts live, then (2) a fixed lane-indexed 8x8 transpose crossbar.

    K's swizzle gives a 2-way conflict on that fetch, so V needs its OWN LDS layout:
    a V block (``n_block`` kv x ``v_hdim``) is stored as stacked ``_V_BLK_KV`` x v_hdim
    sub-blocks; each sub-block is tiled into ``_V_TILE`` x ``_V_TILE`` tiles, each tile
    into ``_V_SUB_KV`` x ``_V_SUB_HD`` (4x16) subtiles, and the subtile column index is
    XOR-swizzled by ``(subtile_row_index & 1)``. That makes the transpose-load fetch
    bank-conflict-free (verified analytically for every 16x16 tile offset).

    Config (caller-maintained) arrives via the constructor: ``v_hdim``,
    ``n_block`` and ``num_waves``.
    """

    def __init__(self, *, v_hdim, n_block=_DEFAULT_N_BLOCK, num_waves=_DEFAULT_NUM_WAVES):
        if v_hdim % _V_TILE != 0:
            raise ValueError(f"v_hdim must be a multiple of {_V_TILE}; got {v_hdim}")
        if n_block % _V_BLK_KV != 0:
            raise ValueError(f"n_block must be a multiple of {_V_BLK_KV}; got {n_block}")
        self.v_hdim = v_hdim  # compile-time
        self.n_block = n_block  # compile-time
        self.num_waves = num_waves  # compile-time
        # 8x32 write-tile grid (async global->LDS): rows = kv, cols = d.
        self.n_wr_tile_rows = n_block // _V_WR_TILE_KV
        self.n_wr_tile_cols = v_hdim // _V_WR_TILE_HD

    def get_lds_size_in_byte(self):
        """LDS bytes for one ``n_block`` x ``v_hdim`` V block (one ping-pong buffer)."""
        return self.n_block * self.v_hdim * _BF16_BYTES

    def _lds_byte(self, kv_row, d_col):
        """Swizzled LDS byte offset of V element ``(kv_row, d_col)`` within one block.

        Layout: sub-block ``kv_row // 32`` -> 32x32 tile ``d_col // 32`` -> 4x16
        subtile ``(ridx = (kv_row%32)//4, cidx = (d_col%32)//16)``, with the stored
        column index ``cidx ^ (ridx & 1)``. Works for Python-int or fx operands (plain
        integer arithmetic); 8 contiguous ``d_col`` land contiguously in LDS."""
        blk = kv_row // _V_BLK_KV
        r32 = kv_row % _V_BLK_KV
        c32 = d_col % _V_TILE
        tile = d_col // _V_TILE
        ridx = r32 // _V_SUB_KV
        cidx = c32 // _V_SUB_HD
        sidx = cidx ^ (ridx % 2)  # XOR swizzle (avoid fx '&': use %2)
        rloc = r32 % _V_SUB_KV
        cloc = c32 % _V_SUB_HD
        return (
            blk * (_V_BLK_KV * self.v_hdim * _BF16_BYTES)
            + tile * (_V_TILE * _V_TILE * _BF16_BYTES)
            + (ridx * 2 + sidx) * (_V_SUB_KV * _V_SUB_HD * _BF16_BYTES)
            + (rloc * _V_SUB_HD + cloc) * _BF16_BYTES
        )

    def async_load_vram_to_lds_wr_tile(
        self,
        *,
        ptr_lds,
        v_base_i64,  # fx.Int64: ptrtoint(get_iter(ptr_V))
        stride_v_seq,
        stride_v_head,
        kv_head,
        kv_row0,  # fx.Int32: global token of this block's kv-row 0
        kv_valid,  # fx.Int32: valid kv rows in this block (only read if check_oob)
        row_idx,  # kv offset within block, mult of 8, < n_block
        col_idx,  # d offset, mult of 32, < v_hdim
        lane_idx,
        check_oob=True,  # compile-time: clamp rows >= kv_valid in-bounds
    ):
        """One warp streams an 8(kv) x 32(d) tile into ``ptr_lds`` (new V swizzle) —
        one b128 (8 bf16) per lane: lane -> (wr_row = lane//4, chunk = lane%4). The
        global read is coalesced (d contiguous); the LDS write is swizzled. When
        ``check_oob`` is False the caller guarantees the block is in-bounds."""
        _assert_multiple("row_idx", row_idx, _V_WR_TILE_KV)
        _assert_multiple("col_idx", col_idx, _V_WR_TILE_HD)
        wr_row = lane_idx // 4  # kv row within the 8-row write tile [0,8)
        chunk = lane_idx % 4  # which 8-d b128 [0,4) -> spans the 32-wide tile
        kv_row = fx.Int32(row_idx) + wr_row
        d_col = fx.Int32(col_idx) + chunk * _CHUNK_ELEMS

        safe_kv = kv_row
        if check_oob:
            safe_kv = (kv_row < kv_valid).select(kv_row, fx.Int32(0))  # clamp OOB
        token = kv_row0 + safe_kv
        g_off = token * stride_v_seq + kv_head * stride_v_head + d_col * _BF16_BYTES
        # LDS position uses the UNCLAMPED kv_row (global read uses clamped safe_kv).
        lds_off = ptr_lds + self._lds_byte(kv_row, d_col)
        # mode-2: load + RAW/WAR covers as ONE atomic inline-asm block (see K loader
        # / _async_load_to_lds). Scalar-base + per-chunk voffset form preserved.
        _async_load_to_lds(v_base_i64, g_off, lds_off, cluster=True)

    def async_load_vram_to_lds(
        self,
        *,
        ptr_lds,
        ptr_V,
        stride_v_seq,
        stride_v_head,
        kv_head,
        kv_row0,
        kv_valid,
        warp_idx,
        lane_idx,
        check_oob=True,
    ):
        """Load the whole ``n_block`` x ``v_hdim`` V block into ``ptr_lds``.

        The 8x32 write-tile grid is spread round-robin across the waves (warp ``w``
        streams tiles ``w, w+num_waves, ...``); ``check_oob`` forwards to each tile."""
        n_tiles = self.n_wr_tile_rows * self.n_wr_tile_cols
        if n_tiles % self.num_waves != 0:
            raise NotImplementedError(
                f"V tile grid ({n_tiles}) must be divisible by {self.num_waves} warps; "
                f"got n_block={self.n_block}, v_hdim={self.v_hdim}"
            )
        v_base_i64 = fx.Int64(fx.ptrtoint(fx.get_iter(ptr_V)))
        for i in fx.range_constexpr(n_tiles // self.num_waves):
            tile_id = warp_idx + fx.Int32(i * self.num_waves)
            row_idx = (tile_id // self.n_wr_tile_cols) * _V_WR_TILE_KV
            col_idx = (tile_id % self.n_wr_tile_cols) * _V_WR_TILE_HD
            self.async_load_vram_to_lds_wr_tile(
                ptr_lds=ptr_lds,
                v_base_i64=v_base_i64,
                stride_v_seq=stride_v_seq,
                stride_v_head=stride_v_head,
                kv_head=kv_head,
                kv_row0=kv_row0,
                kv_valid=kv_valid,
                row_idx=row_idx,
                col_idx=col_idx,
                lane_idx=lane_idx,
                check_oob=check_oob,
            )

    # ------------------------------------------------------------------
    def load_lds_to_vgpr_tile_as_v(self, *, ptr_lds, kv_idx, d_idx, lane_idx):
        """Transpose-load one 16(kv) x 16(d) tile as a PV A-fragment via
        ``ds_load_tr16_b128`` (kv_idx, d_idx both mult of 16) -> v8 bf16 per lane.

        Each lane feeds one b128 fetch; the fixed 8x8 crossbar then yields:
        lane l holds ``V[kv_idx + (l//16)*8 + e, d_idx + l%16]`` for e in 0..7 (v8[e]),
        i.e. 8 consecutive kv values at a fixed d = d_idx + l%16 -> the transposed
        ``V^T`` fragment. Fetch address per lane:
        ``V[kv_idx + (l//16)*8 + l%8, d_idx + ((l//8)%2)*8]`` (see class docstring)."""
        _assert_multiple("kv_idx", kv_idx, _WMMA_M)
        _assert_multiple("d_idx", d_idx, _WMMA_M)
        fetch_kv = fx.Int32(kv_idx) + (lane_idx // 16) * 8 + lane_idx % 8
        fetch_d = fx.Int32(d_idx) + ((lane_idx // 8) % 2) * 8
        v8_ty = fx.Vector.make_type(_CHUNK_ELEMS, fx.BFloat16)
        addr = ptr_lds + self._lds_byte(fetch_kv, fetch_d)
        # NOTE (mode-2): _mem_ds_load now fuses vm_vsrc(0) back-to-back after every
        # ds op (_COV_DS_VMVSRC) AND keeps dscnt(0) (_COV_DSLOAD_WAR). HISTORICAL
        # WARNING: adding vm_vsrc(0) after this V ds_load_tr16 was previously MEASURED
        # HARMFUL (8192nc 4.4% -> 30.6%, garbage 2/160 -> 27/160) -- that test was
        # vm_vsrc alone (no epilogue ds_store WAR fix, older RAW cover). Re-validate
        # this config at >=8192nc with the NaN/max-abs gate before trusting it; if the
        # V tr16 vm_vsrc regresses, gate it per-callsite instead of the global knob.
        return _mem_ds_load(addr, v8_ty, tr16=True)


# ============================================================================
# O writer (VGPR WMMA accumulator -> LDS reshape -> coalesced global store)
# ============================================================================


class OManager16b:
    """Owns the O epilogue: fp32 WMMA accumulator -> bf16 -> global VRAM.

    PV leaves each wave's 16 x v_hdim tile in the accumulator layout: for d-tile
    ``k`` lane ``l`` holds ``O[q = l%16, d = (l//16)*8 + 16*k + {0..7}]``. Adjacent
    lanes hold different q rows, so O is transposed through per-warp staging LDS --
    store accumulator-indexed, re-read giving each lane 8 contiguous d of one q,
    then ``buffer_store`` coalesced.

    The 16 x v_hdim warp tile splits into ``n_units = v_hdim / cols_per_tile``
    self-contained transpose units. Staging LDS is an ``inflight_units``-deep ring
    of ``cols_per_tile``-wide slots, each 16(q) x cols_per_tile bf16, row-major with
    the 8-bf16 chunk XOR-swizzled (``slot = chunk ^ (q >> shift)``) -> no padding,
    b128 store and read bank-conflict-free. The units software-pipeline: unit u+1's
    ds_stores are issued ahead of unit u's ds_load re-read + coalesced buffer_store,
    hiding the LDS round-trip. ``cols_per_tile=64`` keeps each global store a full
    128B row. The caller points ``ptr_lds`` at the non-current K|V slot (holding a
    dead prefetch no wave reads), so no threadgroup barrier is needed.

    Ordering uses ``s_wait_dscnt``. DSCNT is a single in-order 6-bit counter shared
    by ds_store and ds_load, so every DS op's global issue index is tracked at
    compile time and a dependency is awaited via ``clamp(issued - 1 - gidx, 0, 63)``.

    Constructor config: ``v_hdim``, ``gqa_ratio``, ``num_waves``, ``cols_per_tile``,
    ``inflight_units``, ``lds_budget_bytes``.
    """

    def __init__(
        self,
        *,
        v_hdim,
        gqa_ratio,
        num_waves=_DEFAULT_NUM_WAVES,
        cols_per_tile=_O_COLS_PER_TILE,
        inflight_units=_O_INFLIGHT_UNITS,
        lds_budget_bytes=_O_LDS_BUDGET_BYTES,
    ):
        if v_hdim % _WMMA_M != 0:
            raise ValueError(f"v_hdim must be a multiple of {_WMMA_M}; got {v_hdim}")
        self.v_hdim = v_hdim
        self.gqa_ratio = gqa_ratio
        self.num_waves = num_waves
        self.block_m = _WMMA_M * num_waves  # Q/O rows per threadgroup
        self.d_tiles = v_hdim // _WMMA_M  # WMMA output tiles == frags/lane

        cpt = min(v_hdim, cols_per_tile)
        cpt = (cpt // _WMMA_M) * _WMMA_M  # 16-col aligned
        if cpt < _WMMA_M:
            raise ValueError(f"cols_per_tile={cols_per_tile} too small")
        if v_hdim % cpt != 0:
            raise NotImplementedError(
                f"v_hdim={v_hdim} not a multiple of cols_per_tile={cpt}"
            )
        self.cols_per_tile = cpt
        self.n_units = v_hdim // cpt
        self.tiles_per_unit = cpt // _WMMA_M  # ds_store ops per unit == ds_load ops
        self.inflight_units = min(inflight_units, self.n_units)

        # LDS geometry, per unit slot; the ring holds inflight_units slots.
        self._row_bytes = cpt * _BF16_BYTES
        self._slot_stride = _WMMA_M * self._row_bytes
        self._warp_stride = self.inflight_units * self._slot_stride
        self._chunks_per_row = cpt // _CHUNK_ELEMS  # G, b128 chunks per slot row
        # XOR swizzle: slot = chunk ^ (q // _sw_div), shift = max(0, 4 - log2(G)).
        shift = max(0, 4 - int(self._chunks_per_row).bit_length() + 1)
        self._sw_div = 1 << shift

        total = num_waves * self._warp_stride
        if total > lds_budget_bytes:
            raise ValueError(
                f"O ring {total}B (waves={num_waves} x inflight={self.inflight_units} x "
                f"slot={self._slot_stride}B) exceeds budget {lds_budget_bytes}B"
            )

    def get_lds_size_in_byte(self):
        """LDS bytes the caller must reserve for O staging (all waves, whole ring)."""
        return self.num_waves * self._warp_stride

    def _lds_byte(self, slot_idx, q_row, d_col):
        """Swizzled LDS byte offset (within a warp region) of ring-slot ``slot_idx``'s
        O(q_row, d_col) (``d_col`` is slot-local, in [0, cols_per_tile))."""
        chunk = d_col // _CHUNK_ELEMS
        sw = chunk ^ (q_row // self._sw_div)
        return slot_idx * self._slot_stride + q_row * fx.Int32(self._row_bytes) + sw * _CHUNK_BYTES

    def store_o_to_vram(
        self,
        *,
        o_rsrc,  # buffer resource over the O tensor (buffer_ops.create_buffer_resource)
        o_base_elems,  # fx.Int32: element offset of this (batch, ...) origin (0 for thd)
        stride_o_seq,  # elements per token step
        stride_o_head,  # elements per q-head step
        q_start,  # fx.Int32: first token of this request (varlen) / 0 (batch)
        q_len,  # fx.Int32: valid query rows; rows with seq >= q_len are masked off
        kv_head,  # fx.Int32: this workgroup's kv head
        block_x,  # fx.Int32: this workgroup's grid-x tile index
        warp_idx,
        lane_idx,
        ptr_lds,  # fx.Int32: base byte addr of the caller's O staging allocation
        o_frags,  # list[d_tiles] of v8 f32 (pre-normalized) WMMA accumulators
    ):
        """Reshape this warp's 16 x v_hdim fp32 accumulator to bf16 and store it.

        ``o_frags[k]`` is this lane's v8 fp32 for d-tile ``k``, already normalized
        (O / row-sum). Rows with seq >= q_len are dropped via the buffer_store mask,
        which redirects to offset ``0x7FFFFFFF`` -- so ``o_rsrc`` MUST carry the real
        ``num_records`` (``max_size=False``) or it faults.
        """
        if len(o_frags) != self.d_tiles:
            raise ValueError(
                f"expected {self.d_tiles} O frags (v_hdim//{_WMMA_M}); got {len(o_frags)}"
            )
        lds_warp = ptr_lds + warp_idx * self._warp_stride
        q_st = lane_idx % _WMMA_M
        d_half = (lane_idx // _WMMA_M) * _CHUNK_ELEMS  # 0 or 8
        v8_ty = fx.Vector.make_type(_CHUNK_ELEMS, fx.BFloat16)
        base_row = block_x * self.block_m + warp_idx * _WMMA_M
        G = self._chunks_per_row
        TPU = self.tiles_per_unit  # ds_store ops per unit == ds_load rounds per unit
        NSL = self.inflight_units  # ring depth (units resident at once)

        # DSCNT is one in-order 6-bit counter for both ds_store and ds_load: op X has
        # retired <=> DSCNT <= issued-1-X. Track each DS op's global index and await a
        # dependency ``dep`` via s_wait_dscnt(clamp(issued-1-dep, 0, 63)).
        issued = 0  # DS ops emitted so far (global issue index)
        unit_last_store = {}  # unit -> gidx of its final ds_store (all TPU stores done)
        unit_loads = {}  # unit -> list of gidx of its TPU ds_loads

        def emit_write(u):
            """Issue unit ``u``'s TPU ds_stores (accumulator -> ring slot u%NSL)."""
            nonlocal issued
            slot = u % NSL
            prev = u - NSL  # last occupant of this slot
            if prev >= 0:
                # WAR: prev unit's re-reads must retire before we overwrite the slot.
                dep = unit_loads[prev][-1]
                rocdl.s_wait_dscnt(max(0, min(_O_DSCNT_MAX, issued - 1 - dep)))
            last = None
            for kk in range(TPU):
                k = u * TPU + kk
                d_col = d_half + fx.Int32(kk * _WMMA_M)  # slot-local column
                bf = o_frags[k].to(fx.BFloat16)
                addr = lds_warp + self._lds_byte(slot, q_st, d_col)
                # mode-2: the single RAW cover in _mem_ds_store drains the VALU that
                # produced BOTH the addr and the bf16 data (a WMMA-acc cvt) before
                # the ds_store issues.
                _mem_ds_store(addr, bf)
                last = issued
                issued += 1
            unit_last_store[u] = last

        def emit_read(u):
            """Re-read unit ``u`` coalesced (ring slot u%NSL) and buffer_store to VRAM."""
            nonlocal issued
            slot = u % NSL
            # RAW: every re-read pulls d-columns spanning all TPU stores of the unit,
            # so wait for the unit's final store before the first load.
            dep = unit_last_store[u]
            rocdl.s_wait_dscnt(max(0, min(_O_DSCNT_MAX, issued - 1 - dep)))
            loaded = []
            gidxs = []
            for r in range(TPU):
                f = fx.Int32(r * _WAVE_LANES) + lane_idx  # 32 b128 chunks per round
                q_out = f // G  # q-row within this warp [0,16)
                d_local = (f % G) * _CHUNK_ELEMS
                addr = lds_warp + self._lds_byte(slot, q_out, d_local)
                data = _mem_ds_load(addr, v8_ty)
                loaded.append((data, q_out, d_local))
                gidxs.append(issued)
                issued += 1
            unit_loads[u] = gidxs
            d_base = fx.Int32(u * self.cols_per_tile)  # global d origin of this unit
            for r in range(TPU):
                data, q_out, d_local = loaded[r]
                # RAW: this load's data must be in VGPR before storing it to VRAM.
                rocdl.s_wait_dscnt(max(0, min(_O_DSCNT_MAX, issued - 1 - gidxs[r])))
                pr = base_row + q_out  # global packed row
                q_head = kv_head * self.gqa_ratio + pr % self.gqa_ratio
                seq = pr // self.gqa_ratio
                valid = seq < q_len
                token = q_start + seq
                off_elems = (
                    o_base_elems
                    + token * stride_o_seq
                    + q_head * stride_o_head
                    + d_base
                    + d_local
                )
                # mode-2: buffer_store lowers `mask=valid` to an INTERNAL
                # select(valid, off, 0x7fffffff) that lands right before the store, so
                # covering `off_bytes` alone leaves that select uncovered -> stale
                # reused offset -> wild VRAM address (page fault). Replicate the
                # mask-select here (valid ? off : OOB) and pass it to _mem_buffer_store
                # with mask=None, so the RAW cover inside the asm block drains the
                # select immediately before the store with no VALU between. (data is
                # dscnt-covered above.)
                off_bytes = off_elems * fx.Int32(_BF16_BYTES)
                off_masked = valid.select(off_bytes, fx.Int32(0x7FFFFFFF))
                _mem_buffer_store(data, o_rsrc, off_masked, width=128)

        # Two-stage pipeline: prime unit 0, then overlap unit u+1's write with unit
        # u's read. n_units == 1 collapses to a single write+read with one RAW wait.
        emit_write(0)
        for u in range(self.n_units):
            if u + 1 < self.n_units:
                emit_write(u + 1)
            emit_read(u)
