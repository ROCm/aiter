# FlashKDA K2 pass A on gfx950: MIR edits in Gluon source

This document is about pass A of the segmented K2 recurrence of FlashKDA on gfx950 (MI350 series). For each (segment, head), pass A runs two recurrences over the chunks of the segment: h_b starts from zero with the real v, and h_a starts from the identity with v = 0.

A study made 10 MIR edits to a research version of the pass-A kernel. Each edit changes the instruction order before the machine scheduler, or removes instructions. With all 10 edits (MIR b1n2s4t1 below), the kernel is 4.3 to 4.7 µs faster than the normal compile at T = 16384, on the same GPU in the same time period. The ATT cycles per chunk go from about 2150 to 1663. MIR edits cannot go into production code: the replacement MIR file is matched by the sha256 of the LLIR, so a change of the source or of the compiler makes it invalid.

This document uses the latest upstream Triton main as the reference. For each of the 10 edits, it tells whether Gluon source can express the edit, how, and the effect. For each edit that source cannot express, it tells what blocks it and which compiler change is necessary. Section 3 describes the kernel that is in aiter: `k2_ab_fused_pipe_gluon` in `aiter/ops/triton/_gluon_kernels/gfx950/kimi_delta_attn/flash_kda_k2.py`.

**Versions:**

- The aiter kernel is tested on upstream Triton main 0df169311f (2026-11-12) and on Triton 3.8.0 (`3.8.0+amd.rocm7.2.0.git7cb7b059`), which aiter CI uses. With Triton 3.8.0, aiter does not use the new kernel (section 3.1).
- The research numbers of sections 2 and 4 to 6 are from upstream main f8cddb947b (2026-11-04, `3.9.0+gitf8cddb94`). Section 7.1 and the repeat runs of section 2 use 173580104f (2026-11-08). Between the two, there is no commit to the AMD part of Gluon. The two use the same LLVM for the Triton core (18f9e623) and the same AMD codegen library (08b05a33), and they compile each research mode to byte-identical ISA. 0df169311f also uses the codegen library 08b05a33.
  - [triton-lang/triton#12656](https://github.com/triton-lang/triton/pull/12656) changes only the pin that builds the codegen LLVM from source (`cmake/amd-llvm-build-info.json`, 293a85c0). The prebuilt library that the build downloads comes from `cmake/amd-llvm-info.json`, which is still 08b05a33. `SIFoldOperands` of 293a85c0 also does not fold `V_ADD_U32 0, x`, so the result of step 4 does not change with it.
- The MIR edits were made on ROCm Triton 3.7. A Triton 3.7 source variant of the same kernel implements two of the items with inline asm (sections 4.4 and 5.3).

## Summary

### The 10 MIR edits in Gluon source

Each MIR edit adds to the edits before it (section 1.2). The result is the change of the kernel time at T = 16384, H = 12, on GPU 3 if the cell does not give a different GPU. "Over `g_lt`" means: with this switch on, in addition to the switches of `g_lt`. Section 2 has all the data.

| MIR edit | Source-level approach | Result (GPU 3) | Verdict |
| --- | --- | --- | --- |
| 1. dot1: VALU in the MFMA shadows (cyc03) | `D1=3`: one MFMA per sched region, in the MIR order. `D1=1/2`: per piece | `D1=3`: +0.65 µs over `g_lt`. `D1=1/2`: +1.12 / +0.69 µs (GPU 2) | Possible in source. Slower |
| 2. mem2: address adds first (cyc03h) | No source op: the lowering of the LDS read makes the adds. `M2=1/2` moves the full reads | `M2=1/2`: +1.28 / +0.60 µs (GPU 2) | Not possible in source |
| 3a. mem1: LDS reads before the copies (cyc03hm1spread, first half) | `M1=3`: two `sched_barrier`s split mem1 into the kr reads, the gt reads and the copies | −0.7 µs (GPU 2 and GPU 3) | **Landed, in aiter** |
| 3b. mem1: copies between the reads (cyc03hm1spread, second half) | `M1=4`: kr is read in 4 slices, and one copy group follows each slice | +0.95 µs over `g_lt` (GPU 2) | Possible in source. No gain |
| 4. Remove the `+0` copies (s_z) | None. `llvm_cl_opts="disable-cgp"` removes them, but the kernel spills | 195 µs (spills) | Not possible in source. Needs [llvm/llvm-project#163298](https://github.com/llvm/llvm-project/pull/163298) in the AMD codegen LLVM of Triton |
| 5. Back-edge SALU before the last `s_barrier` (s_zl) | `LT=1`: the copy bases are loop-carried values, moved on in mem2 of p3 | −1.1 µs. Back edge: 12 → 4 SALU | **Landed, in aiter** |
| 6. dot2: all `pk` first, then the MFMAs (d2s0) | `D2=1`: `sched_barrier`s between the `g * gt` products, the B MFMAs, and the A MFMA + cvt pairs | Same order as the MIR. −0.55 µs alone. VGPR 221 → 229 | **Landed, in aiter** |
| 7. gt reads earlier (gt3k2) | `GTE=1/2`: mem2 reads the first 1 or 2 gt pieces of the next chunk | −0.4 to +0.9 µs; the sign changes between runs (GPU 2) | Possible in source. No gain |
| 8. β clamp in dot1 (b1n2) | `BD1=2`: dot1 computes the offset of the β copy | ATT per chunk: 2060 → 2008. Time: in the noise | Landed in the research kernel. Not necessary in aiter: the aiter kernel has no β clamp |
| 9. dot2: A multiplies in the B MFMA shadows (b1n2s3, b1n2s4) | `D2=4`, `D2N=3/4`: one MFMA per sched region. Same instruction string as the MIR | +0.36 / +1.05 µs over `g_lt` | Possible in source. Slower |
| 10. MFMA result in the srcC registers (tie, b1n2s4t1) | None. A 16x16x32 MFMA has no tied form, and `cd_regclass` selects only the register class | — | Not possible in source. Needs an MFMA option |

### Compiler options and research features

| Item | Research kernel | aiter kernel | Result |
| --- | --- | --- | --- |
| SILoadStoreOptimizer off | A research compiler patch | `llvm_fn_attrs="target-features=-load-store-opt"` at the launch | T = 16384, H = 12: 56.27 → 44.58 µs. T = 4096, H = 12: 21.79 → 18.71 µs. **Landed** |
| VGPRs only, no AGPRs | A research compiler patch (`amdgpu-agpr-alloc=0,0`) | Nothing | Not necessary: upstream Triton gives the kernel no AGPRs |
| B128 reads of kd and inv | K1 writes kd and inv in the column order of the reads, so a lane reads its 16 bytes with one `ds_read_b128` | Not used: it changes K1, which is in a different file | No measurable cost: 46.42 µs (graph replay) against 47.06 µs for the research `g_lt` |
| Layouts | Kernel arguments | Module constants | Host time of a launch: 27.5 → 13.5 µs. **Landed** |
| Gluon ops | `buffer_load_to_shared`, `load_shared_relaxed`, `buffer_store` (deprecated on Triton main) | Buffer descriptor ops and `smem.load` | Same time. **Landed** |

### The kernel in aiter

`k2_ab_fused_pipe_gluon` has the landed steps 3a, 5 and 6, and it compiles without SILoadStoreOptimizer. On gfx950, it is pass A if Triton has `gl.amd.hint.sched_barrier` and the buffer descriptor ops (Triton main since 2026-10-25). If not, pass A is `k2_ab_fused_gluon`, with no change.

Pass A only, MI350X (GPU 3), Triton main 0df169311f, against `k2_ab_fused_gluon`, GPU time from CUDA graph replay:

| T, H (B = 1) | Segments × chunks | New (µs) | Old (µs) | Speedup |
| --- | --- | --- | --- | --- |
| 2048, 12 | 16 × 4 | 14.13 | 18.09 | 1.28× |
| 4096, 12 | 16 × 8 | 18.49 | 25.03 | 1.35× |
| 8192, 12 | 16 × 16 | 27.65 | 40.04 | 1.45× |
| 16384, 12 | 16 × 32 | 46.42 | 73.64 | 1.59× |
| 65536, 4 | 32 × 64 | 74.07 | 124.25 | 1.68× |

- Eager launches are 1.30× to 1.89× faster. The host time of a launch is about 13.5 µs, against 25.5 µs (section 3.3).
- 239 or 240 VGPRs, no AGPRs, no VGPR spills.
- The landed steps give the time of the research `g_lt`: 46.42 µs against 47.06 µs at T = 16384, H = 12.

## Contents

1. [Background](#1-background)
2. [Step-by-step comparison](#2-step-by-step-comparison)
3. [The kernel in aiter](#3-the-kernel-in-aiter)
4. [dot2 order (step 6)](#4-dot2-order-step-6)
5. [mem1: reads before copies (step 3)](#5-mem1-reads-before-copies-step-3)
6. [Back-edge SALU and β clamp (steps 5 and 8)](#6-back-edge-salu-and-β-clamp-steps-5-and-8)
7. [Steps not landed (steps 1, 2, 4, 9 and 10)](#7-steps-not-landed-steps-1-2-4-9-and-10)
8. [References](#8-references)

## 1. Background

### 1.1 The main loop

The research kernel is `k2_ab_fused_wp2_pipe4u`. Its description:

```text
Per chunk j the loop body is four ``warp_pipeline_stage`` clusters:

  dot1[j]  DOT1 (kd @ -h, B chain seeded with v) and INV (inv @ u)
  mem1[j]  LDS read kr/gt[j]; big -> bf16; async copy kd/inv/v/beta[j+3]
  dot2[j]  h * gt and UPD (kr^T @ big); -h_B -> bf16 for chunk j+1
  mem2[j]  LDS read kd/inv/v/beta[j+1]; async copy kr/gt[j+3]
```

- All research numbers in this document come from the mode `full_u_b128_tr`. It unrolls the main loop 4 times, so one iteration does 4 chunks. Below, these are p0 to p3.
- An `s_barrier` separates two clusters. Thus one iteration has 17 regions: 0 to 3 are dot1, mem1, dot2 and mem2 of p0, 4 to 7 are p1, 8 to 11 are p2, 12 to 15 are p3, and 16 is the back edge (pointer increments, loop count and branch). p1 to p3 have the same code. p0 has its own overhead.
- A workgroup has 8 waves in two groups (sl0 and sl1), with one wave of each group on each SIMD. The two groups run one cluster apart: when one group runs a dot cluster, the other group runs a mem cluster. The two share the issue cycles of the SIMD.
- The aiter kernel has the same four stages and the same unroll by 4 (section 3.1).

### 1.2 The 10 MIR edits

The method: dump the MIR before the machine scheduler (`TRITON_DUMP_MIR`), change the order or remove instructions with a script, and compile the changed MIR (`TRITON_SWAP_MIR`). Each step adds one item to the step before it:

| Step | Name | Change | Cumulative (µs) | ATT per chunk |
| --- | --- | --- | --- | --- |
| — | base | Normal compile | 0 | 2152 |
| 1 | cyc03 | dot1 reordered with an issue model, with VALU in the MFMA shadows. The dot1 `pk` split into `v_mul_f32`. The `v_xor` that negates β merged into the neg modifier of the multiply | +0.54 | 2222 |
| 2 | cyc03h | The `v_add_u32` for the read addresses of inv, v and β in mem2 moved to the start of the region | −0.88 | 2116 |
| 3 | cyc03hm1spread | mem1 issues the LDS reads first. The 4 copy groups are spread between the reads (one group after each 4 kr reads) | −1.17 | 2070 |
| 4 | s_z | The 11 `v_add_u32 v, 0, v` of p0 (the `+0` copies) removed | −1.88 | 1962 |
| 5 | s_zl | The 12 SALU of the back edge moved before the last `s_barrier` | −2.55 | 1897 |
| 6 | d2s0 | All 32 `pk` of dot2 first, with one LDS wait, then the MFMAs back to back | −3.81 | 1793 |
| 7 | gt3k2 | The first 2 gt reads of p1 to p3 moved to mem2 of the chunk before | about −4.1 | 1756 |
| 8 | b1n2 | The address calculation of the β copy moved into an `s_nop` of dot1 | about −4.3 | 1732 |
| 9 | b1n2s3 | The multiplies of 3 A tiles (`pk` split into `v_mul_f32`) issued in the MFMA shadows of the B part of dot2 | about −4.5 | 1704 |
| 10 | b1n2s4t1 | The dot1 MFMA results written to the srcC registers (tie), which removes the `s_nop` that register allocation causes. Step 9 with 4 A tiles | about −4.8 | 1663 |

- The µs values are the cumulative change against the normal compile at T = 16384, from different runs. The values of steps 7 to 10 are estimates. In a direct comparison on the same GPU in the same time period, b1n2s4t1 is 4.3 to 4.7 µs faster than the normal compile.
- Step 1 alone makes the kernel slower. It gives a gain only together with step 2.

### 1.3 Terms

- **pk:** `v_pk_mul_f32`, one instruction for 2 f32 multiplies.
- **srcA, srcB, srcC:** the three inputs of an MFMA. The result is A·B + C.
- **MFMA shadow:** after a 16x16x32 MFMA issues, the matrix unit computes for about 16 cycles. In this time, the same wave can issue other instructions.
- **Wait state:** a gap that the hardware requires between two instructions. `s_nop N` gives N + 1 wait states, and each normal instruction between the two gives 1.
- **Piece:** the kernels split each state (g_b, g_a and others) by rows into 4 blocks of 32 rows. One block is a piece (section 4.3).
- **Region:** the code between two `s_barrier`s of the main loop (17 for each iteration, section 1.1).
- **Sched region:** the code between two `sched_barrier`s. The machine scheduler changes the order only in a sched region and does not move an instruction out of it.
- **Slot:** the two wave groups run one cluster apart. Thus, between two `s_barrier`s, the dot cluster of one group runs together with the mem cluster of the other group. This time is a slot. X is dot1 of sl0 with mem2 of sl1, Y is mem1 of sl0 with dot1 of sl1, Z is dot2 of sl0 with mem1 of sl1, and W is mem2 of sl0 with dot2 of sl1. The subscript is the position: for example, Y_1 is the Y slot of p1.
- **ATT per chunk:** the main-loop cycles from an ATT capture, divided by the number of chunks.
- **Instruction string:** one letter for each instruction, in issue order from left to right:

| Letter | Instruction | Letter | Instruction |
| --- | --- | --- | --- |
| `M` | bf16 MFMA | `r` | LDS read (`ds_read*`) |
| `I` | f16 MFMA | `g` | Global-to-LDS copy (`buffer_load … lds`) |
| `p` | `v_pk_mul_f32` | `w` | `s_waitcnt` |
| `m` | `v_mul_f32` | `n` | `s_nop` |
| `b`, `h` | cvt to bf16, f16 | `s` | Other SALU, `s_barrier` included |
| `x` | `v_xor` | `.` | Other VALU |
| `f` | bf16 to f32 | | |

## 2. Step-by-step comparison

The reference is upstream Triton. The research kernel on upstream Triton is `k2_ab_fused_wp2_pipe4g`: `k2_ab_fused_wp2_pipe4u` with constexpr switches. A switch value of 0 is the original order of `k2_ab_fused_wp2_pipe4u`. A mode name gives a set of switches (section 5.2 tells how to read the names). "Over `g_lt`" means: with this switch on, in addition to `g_lt`, in the same run, on GPU 2, median of 10 rounds. For the one-MFMA forms of steps 1 and 9 (`D1=3`, `D2=4`), the value is the mean of 5 runs on each of GPU 2 and GPU 3 (section 7.1).

| Step | MIR change | Source form | Result | Verdict |
| --- | --- | --- | --- | --- |
| 1 | dot1 reordered one instruction at a time | `D1=3`: one MFMA per sched region, about the MIR order (`g_d1t`). `D1=1/2`: per piece | `D1=3`: 0.6 to 0.8 µs slower. `D1=1/2`: 1.12 / 0.69 µs slower | Possible, slower; 7.1 |
| 2 | mem2 address adds first | `M2=1/2`: read inv, v and β first, then kd | 1.28 / 0.60 µs slower | Not possible; 7.2 |
| 3, first half | mem1 reads before the copies | `M1=3`: read kr, then gt, then issue the copies | 0.7 µs faster on both GPUs (5 repeat runs; in the run of the cumulative table, GPU 3 0.9 µs faster and GPU 2 0.2 µs faster) | Landed; 5.2 |
| 3, second half | Copies between the reads | `M1=4`: kr in 4 LDS slices, one copy group after each slice | 0.95 µs slower | Possible, no gain; 5.4 |
| 4 | Remove the `+0` copies | No source form. `disable-cgp` removes them, but the kernel spills | 195 µs | Not possible; 7.3 |
| 5 | Back-edge SALU earlier | `LT=1`: the copy bases are loop-carried values, moved on in mem2 of p3 | Back edge 12 → 4 instructions. About 1.1 µs faster on both GPUs | Landed; 6.1 |
| 6 | All dot2 `pk` first | `D2=1` | Same order as the MIR. VGPR 221 → 229 | Landed; section 4 |
| 7 | gt reads earlier | `GTE=1/2`: mem2 reads the first 1 or 2 gt pieces of the next chunk | −0.4 to +0.9 µs; the sign changes between runs | Possible, no gain |
| 8 | β clamp into dot1 | `BD1=2` | ATT per chunk 2060 → 2008 | Landed; 6.2 |
| 9 | A multiplies in the B part of dot2 | `D2=4`, `D2N=N`: one MFMA per sched region. The `g·gt` of the last N A tiles is split in halves after the B MFMAs. The instruction string is the same as the MIR (`g_t3`, `g_t4`). `D2S=N`: per piece | `D2N=3/4`: 0.4 to 0.5 / 1.0 to 1.5 µs slower. `D2N=0` (`g_t0`): equal. `D2S=3`: 0.90 µs slower | Possible, slower; 7.1 |
| 10 | MFMA tie | No source form | — | Not possible; 7.4 |

**Cumulative table:** T = 16384, GPU 2 and GPU 3 at the same time, median of 20 rounds. Each row turns on one more switch.

| Mode | New switch | GPU 2 (µs) | GPU 3 (µs) | VGPR | Back-edge SALU | ATT per chunk |
| --- | --- | --- | --- | --- | --- | --- |
| `g_base` | Upstream compile of the original | 50.81 | 49.25 | 221 | 12 | 2156 |
| `g_p` | Split into pieces | 50.34 | 49.41 | 221 | 12 | 2136 |
| `g_d2` | `D2=1` | 50.32 | 48.86 | 229 | 12 | 2144 |
| `g_d2m3` | `M1=3` | 50.08 | 47.93 | 229 | 12 | 2060 |
| `g_d2b2m3` | `BD1=2` | 49.50 | 48.18 | 229 | 12 | 2008 |
| **`g_lt`** | `LT=1` | **48.35** | **47.06** | 229 | 4 | 1978 |
| ROCm normal compile | — | 50.37 / 50.39 | 49.59 / 50.27 | 225 | 12 | — |
| ROCm + MIR b1n2s4t1 | — | 46.14 / 46.07 | 45.03 / 45.44 | 207 | 0 | 1663 |

- The error of each upstream row is (5.43e-04, 1.88e-03), and of the two ROCm rows (5.59e-04, 1.90e-03). There are no spills. The compilers are different, so the errors are a little different. The two ROCm rows are two alternating runs in the same time period.
- Two runs of the same mode can differ by 0.3 to 0.5 µs. Thus the single steps in the middle rows are not precise (on GPU 3, `g_d2b2m3` is even slower than `g_d2m3`). The first and the last rows agree on the two GPUs.
- The ATT is from GPU 2. For `g_lt` and MIR b1n2s4t1, it is the mean of two captures; for the other rows, one capture.
- This table is from f8cddb947b. On 173580104f, these modes compile to byte-identical ISA. Repeat runs on it (5 runs on each of GPU 2 and GPU 3, one process at a time): each mode is in 0.5 µs of the table, and `g_lt` is 2.16 and 2.01 µs faster than `g_base`. ATT per chunk: `g_base` 2156, `g_lt` 1976.

**Other items that did not help:** over `g_lt` (same run, GPU 2, median of 10 rounds).

| Mode | Switch or option | Over `g_lt` (µs) | Note |
| --- | --- | --- | --- |
| `g_ltms` | `schedule_hint="mfma-schedule"` | +0.16 | The ISA is byte-identical to `g_lt`. This hint interleaves MFMAs and memory operations between two sched barriers, with the memory operations as anchors. The dot clusters of K2 have no memory operations |
| `g_ltilp` | `llvm_fn_attrs="amdgpu-sched-strategy=max-ilp"` | +1.16 | |
| `g_ltiilp` | `…=iterative-ilp` | +0.58 | VGPR 210 |
| `g_ltmc` | `…=max-memory-clause` | −0.02 | |
| `g_ltl` | `llvm_cl_opts="disable-lsr"` | +0.41 | Section 6.1 |
| `g_vp` | `VP=1`: the v of p3 is also converted to f32 in mem2 | +0.13 | |
| `g_ltvd` | Build option `v_cvt_dot1=True`: the v of all 4 positions is converted to f32 in dot1 | +0.39 | |

## 3. The kernel in aiter

### 3.1 Structure

`k2_ab_fused_pipe_gluon` in `flash_kda_k2.py` does pass A with the main loop of section 1.1.

- One program computes one (segment, head), with the full [K, V] = [128, 128] state of both chains. The grid is (1, segments × H), with 8 warps and `waves_per_eu=2`. The kernel keeps the state negated (g = −h).
- Each chunk is four warp-pipeline stages: dot1, mem1, dot2 and mem2. The kernel copies the chunks to LDS in a ring of 4 slots, so the copies of chunks j + 1 to j + 3 are in flight while chunk j computes. Each chunk has two commit groups: kd, inv, v and β (issued in mem1), and kr and gt (issued in mem2).
- Chunk j uses ring slot (j − n mod 4) mod 4, where n is the number of chunks of the segment. The first n mod 4 chunks run in a loop with run-time slots. The other chunks run in groups of 4 with compile-time slots (p0 to p3), and the last group is straight-line code. With n < 4, the copies past the end repeat chunk n − 1, so the wait counts stay correct.
- Segments can have different lengths, and the last chunk can be partial. Only the copy of chunk n − 1 masks the rows past the end of the segment: these rows of v and β load as 0. kd, inv, kr and gt come from the K1 workspace.
- From section 2, the kernel has step 6 (`_dot2`, the `D2=1` order), step 3, first half (mem1 in `_step`, the `M1=3` order) and step 5 (`pk` and `_advance`, the `LT=1` form).
- The launch sets `llvm_fn_attrs="target-features=-load-store-opt"` (section 3.2).

**Gate and fallback:** `PASS_A_PIPELINED` is true if Triton has `gl.amd.hint.sched_barrier` and the buffer descriptor ops (`gl.amd.cdna4.buffer`). Then `k2_ab_fused_fast` launches `k2_ab_fused_pipe_gluon`. If not, it launches `k2_ab_fused_gluon`, with no change.

- The wrapper `gluon_k2_ab_fused` keeps its signature. On the pipelined path, the schedule is fixed, and the wrapper does not use `BW`, `num_warps` or `waves_per_eu`. It raises `ValueError` if (C, K, V) is not (32, 128, 128). The dispatcher (`_gluon_k2_usable` in `flash_kda.py`) selects the Gluon K2 only for these values.
- Pass A runs only if the segment scheduler (`_choose_chunks_per_seg` in `flash_kda.py`) splits the sequences into segments. It does this if the longest sequence has 64 chunks or more (T ≥ 2048) and the grid without segments does not fill the CUs. `CHUNK_DELTA_ATTN_FLASH_KDA_SEG` overrides the segment length.

**Triton versions:** These two PRs added the two ops to upstream main. The other Gluon ops of the kernel and the `llvm_fn_attrs` option are older, and the upstream Triton 3.8.0 has them.

- `gl.amd.hint.sched_barrier`: [triton-lang/triton#11932](https://github.com/triton-lang/triton/pull/11932), 2026-09-24. This PR also added the `gl.amd.hint` module.
- The buffer descriptor ops: [triton-lang/triton#12345](https://github.com/triton-lang/triton/pull/12345), 2026-10-25. This PR added `make_buffer_descriptor`, `load_to_shared` and `store`, which the kernel uses.

| Triton | Branched from main | Released | `sched_barrier` | Buffer descriptor ops | Pass A |
| --- | --- | --- | --- | --- | --- |
| [3.8.0](https://github.com/triton-lang/triton/tree/v3.8.0) | 2026-06-23 (3bb23dde71) | 2026-08-28 (PyPI) | No | No | `k2_ab_fused_gluon` |
| 3.8.0, AMD build | Unknown | 2026-09-17 (pypi.amd.com) | No | No | `k2_ab_fused_gluon` |
| [3.9.0](https://github.com/triton-lang/triton/tree/v3.9.0) | 2026-10-01 (aad2a60d95) | 2026-10-27 (PyPI) | Yes | No | `k2_ab_fused_gluon` |
| main 0df169311f | | | Yes | Yes | `k2_ab_fused_pipe_gluon` |

- The 3.8 release branch starts before the two PRs, and it does not cherry-pick them. Thus Triton 3.8 has neither of the two.
- The 3.9 release branch starts between the two PRs, and it does not cherry-pick #12345. Thus Triton 3.9.0 has only `sched_barrier`, and its pass A is also `k2_ab_fused_gluon`. On 2026-11-13, no release branch starts after 2026-10-25. Thus no Triton release has the two ops.
- The rows for the upstream 3.8.0 and 3.9.0 come from the source at the tags. These two versions are not tested.
- aiter CI installs the AMD build `3.8.0+amd.rocm7.2.0.git7cb7b059` (directory `release_0914` on pypi.amd.com; the date in the table is the date of the wheel file), not the upstream 3.8.0. A test on this build shows that it has neither `gl.amd.hint` nor `gl.amd.cdna4.buffer`. Its codegen LLVM is 8aa7100b, not the LLVM of the upstream 3.8.0. This LLVM also spills the kernel: an earlier version of the kernel, with the old ops and no `sched_barrier`, spilled 122 VGPRs on this build and took 112 µs at T = 16384, H = 12.

### 3.2 Differences from the research kernel

| Item | Research kernel (`g_lt`) | aiter kernel |
| --- | --- | --- |
| Chunks of a program | A fixed multiple of 4. All chunks full | Any n ≥ 0 for each segment. The last chunk can be partial |
| β copy | 2C wide from the [T, H] input. The offsets are clamped to T − 1 (`_beta_off`, step 8) | 2C wide from the K1 workspace. Only the copy of the last chunk masks. There is no clamp, so step 8 has nothing to move |
| Values of `LT=1` | 6 copy bases and the β limit (int32) | 6 copy bases |
| kd and inv reads | B128: K1 writes kd and inv in the column order of the reads, and a lane reads its 16 bytes with one `ds_read_b128` | Two `ds_read_b64`. K1 does not change |
| Compiler | A research compiler patch: no AGPRs (`amdgpu-agpr-alloc=0,0`) and no SILoadStoreOptimizer | `llvm_fn_attrs="target-features=-load-store-opt"`. Upstream Triton gives the kernel no AGPRs without a patch |
| Gluon ops | `buffer_load_to_shared`, `load_shared_relaxed`, `buffer_store` | `cdna4.buffer.load_to_shared` and `cdna4.buffer.store` with buffer descriptors, and `smem.load` |
| Layouts | Kernel arguments | Module constants |

- **SILoadStoreOptimizer:** with this pass, LLVM merges the kd and inv reads of rows r and r + 16 into `ds_read2st64_b64`, and `v_mov`s then rebuild the register pairs of the MFMA operands. With the pass, the ISA of the aiter kernel has 90 `ds_read2st64_b64`. Event timing on GPU 3: at T = 16384, H = 12, 56.27 µs with the pass and 44.58 µs without it; at T = 4096, H = 12, 21.79 µs and 18.71 µs.
- **B128:** the aiter kernel takes 46.42 µs (graph replay) at T = 16384, H = 12, and the research `g_lt` took 47.06 µs on the same GPU, in a different run. Two runs can differ by 0.3 to 1.3 µs (section 3.4). Thus B128 has no measurable effect here, and K1 can stay as it is.
- **Layouts:** `fast_launch` makes a cache key from all keyword arguments of each launch. With the 12 layouts as arguments, the hash of the layout objects takes about 5.5 µs, and a launch takes about 27.5 µs of host time. As module constants, the layouts are in the JIT cache key (by their repr), and a launch takes about 13.5 µs.
- **Gluon ops:** on Triton main, `load_shared_relaxed` is deprecated since 2026-10-26 ([triton-lang/triton#12478](https://github.com/triton-lang/triton/pull/12478)), and the pointer-based buffer ops since 2026-10-27 ([triton-lang/triton#12527](https://github.com/triton-lang/triton/pull/12527)). The aiter kernel uses the replacements. Each descriptor has `num_records = 2**31 - 2`, the bound of the old pointer-based ops; the masks keep the accesses in bounds. The time is the same as with the old ops (A/B in ±0.45 µs). With the descriptor ops, 12 to 14 SGPRs spill to VGPR lanes (`v_writelane`, `v_readlane`). None of these are in the loops.

### 3.3 Results

MI350X (GPU 3), Triton main 0df169311f, pass A only. "Old" is `k2_ab_fused_gluon` with its tuned config. The KFD eviction count was 0 during the eager runs.

**GPU time (CUDA graph replay) and host time of a launch:**

| T, H (B = 1) | Segments × chunks | GPU, new (µs) | GPU, old (µs) | Speedup | Host, new (µs) | Host, old (µs) |
| --- | --- | --- | --- | --- | --- | --- |
| 2048, 12 | 16 × 4 | 14.13 | 18.09 | 1.28× | 13.58 | 25.57 |
| 4096, 12 | 16 × 8 | 18.49 | 25.03 | 1.35× | 13.28 | 25.37 |
| 2048, 32 | 8 × 8 | 21.54 | 26.83 | 1.25× | 13.45 | 25.60 |
| 8192, 12 | 16 × 16 | 27.65 | 40.04 | 1.45× | 13.44 | 25.32 |
| 8192, 16 | 16 × 16 | 32.00 | 43.61 | 1.36× | 13.51 | 25.80 |
| 16384, 12 | 16 × 32 | 46.42 | 73.64 | 1.59× | 13.40 | 25.33 |
| 32768, 8 | 16 × 64 | 74.09 | 124.18 | 1.68× | 13.46 | 25.41 |
| 65536, 4 | 32 × 64 | 74.07 | 124.25 | 1.68× | 13.43 | 25.97 |
| 4096, 64 | 4 × 32 | 58.97 | 82.26 | 1.40× | 14.07 | 26.43 |

**Eager launches** (back-to-back launches with event timing, median):

| Inputs | Segments × chunks | New (µs) | Old (µs) | Speedup |
| --- | --- | --- | --- | --- |
| T = 2048, H = 4 | 16 × 4 | 13.65 | 25.77 | 1.89× |
| T = 2048, H = 12 | 16 × 4 | 13.77 | 25.70 | 1.87× |
| T = 2048, H = 32 | 8 × 8 | 21.13 | 27.39 | 1.30× |
| T = 4096, H = 4 | 16 × 8 | 14.68 | 25.96 | 1.77× |
| T = 4096, H = 8 | 16 × 8 | 16.38 | 26.19 | 1.60× |
| T = 4096, H = 12 | 16 × 8 | 18.33 | 26.25 | 1.43× |
| T = 4096, H = 32 | 8 × 16 | 30.86 | 43.92 | 1.42× |
| T = 4096, H = 64 | 4 × 32 | 57.56 | 84.41 | 1.47× |
| T = 6144, H = 12 | 12 × 16 | 25.22 | 38.05 | 1.51× |
| T = 8192, H = 12 | 16 × 16 | 26.73 | 39.46 | 1.48× |
| T = 8192, H = 16 | 16 × 16 | 30.95 | 43.84 | 1.42× |
| T = 8192, H = 64 | 4 × 64 | 122.29 | 171.64 | 1.40× |
| T = 16384, H = 12 | 16 × 32 | 44.88 | 73.80 | 1.64× |
| T = 32768, H = 8 | 16 × 64 | 74.56 | 124.66 | 1.67× |
| T = 65536, H = 4 | 32 × 64 | 73.81 | 124.42 | 1.69× |
| B = 4, T = 4096, H = 8 | 32 × 16 | 30.98 | 43.44 | 1.40× |
| Lengths 5000, 3001, 8191, 777; H = 8 | 17 × 25 to 32 | 40.60 | 67.97 | 1.67× |

- With few chunks, an eager launch is host-bound: the time is the host time of the launch. Thus the eager speedup at T = 2048 is larger than the GPU speedup.
- On Triton 3.8.0, pass A is `k2_ab_fused_gluon` with no change, and its host and GPU times do not change. `k2_ab_fused_gluon` is faster on Triton 3.8.0 than on Triton main: 67.39 µs (graph replay) at T = 16384, H = 12, and 120.96 µs at T = 65536, H = 4. Against these times, the new kernel on Triton main is 1.45× and 1.63× faster.
- Registers: 239 or 240 VGPRs, no AGPRs, no VGPR spills, occupancy 2 (all 17 variants that the tests and the benchmarks compiled).
- Error against an fp64 reference: the same as `k2_ab_fused_gluon`.

### 3.4 How to test and time

**Unit tests** (with Triton main or Triton 3.8.0):

```bash
python3 -m pytest op_tests/triton_tests/kimi_delta_attn/test_flash_kda.py \
  op_tests/triton_tests/kimi_delta_attn/test_fast_launch.py -q -rs
```

- With Triton main 0df169311f: 116 passed, 1 skipped (the skipped test needs a second GPU). With Triton 3.8.0: the same.
- The tests run pass A through `flash_kda_fwd`: segmented runs, variable lengths with uneven segments, partial last chunks and weak gates (`lower_bound` −1, −0.1 and −0.01). With the default lower bound −5, gt is about 1e-35 and the A chain is all zero. The weak-gate tests check the A chain.
- `PASS_A_PIPELINED` in `flash_kda_k2.py` tells which kernel is pass A. The info log line `flash_kda route: ...` of `flash_kda.py` tells the route, once for each new route.

**Timing:**

- Compare two kernels alternately, in the same process on the same GPU. Two runs of the same kernel can differ by 0.3 to 1.3 µs, and two GPUs by more than 1 µs.
- CUDA graph replay gives the GPU time. Eager back-to-back launches also include the host time: with few chunks, a launch is host-bound.
- Run one GPU process at a time, and monitor the KFD eviction count: while the process runs, `/sys/class/kfd/kfd/proc/<host pid>/stats_*/evicted_ms` must stay 0. On 2026-11-09, a stuck CI run caused queue evictions for more than 90% of the time in all GPU processes, and a new stream or a graph capture took tens of seconds. Two processes at the same time can also evict each other.
- Use a different `TRITON_CACHE_DIR` for each Triton version. The Triton cache key does not include all environment variables that change the compile. For an A/B of compiler options, use one cache directory for each variant, or set `TRITON_ALWAYS_COMPILE=1`.
- To examine the ISA, read the `.amdgcn` file in the Triton cache. Do a check of `.vgpr_count`, `.vgpr_spill_count`, `.sgpr_spill_count` and of `ds_read2st64` (there must be none).

### 3.5 Limits and open items

- **CI coverage:** aiter CI uses Triton 3.8.0, so CI tests only the fallback. CI tests the new kernel only when the CI Triton has `gl.amd.hint.sched_barrier` and `gl.amd.cdna4.buffer`. Triton 3.9.0 is not sufficient, because it does not have the buffer descriptor ops (section 3.1).
- **Untested Triton versions:** the kernel is tested on Triton main 0df169311f. Triton main from 2026-10-25 (buffer descriptor ops, [triton-lang/triton#12345](https://github.com/triton-lang/triton/pull/12345)) to 2026-10-27 (codegen LLVM 08b05a33, [triton-lang/triton#12496](https://github.com/triton-lang/triton/pull/12496)) passes the gate with an older codegen LLVM. These versions are not tested. Triton 3.8 shows that an older codegen LLVM can spill this kernel.
- **No dedicated unit test or benchmark:** the `flash_kda_fwd` tests cover pass A (section 3.4).
- **Host time of the other K2 kernels:** `k2_c_gluon` and the fallback `k2_ab_fused_gluon` still get their layouts as kernel arguments. A launch of `k2_ab_fused_gluon` takes about 25.5 µs of host time. Module constants (section 3.2) can make this time about half. The host time of `k2_c_gluon` was not measured.
- **Old ops in the other kernels:** `k2_ab_fused_gluon` and `k2_c_gluon` still use the deprecated ops, because they must run on Triton 3.8.
- **Tuned configs:** `k2_ab_fused_gluon_wide` and `k2_ab_fused_gluon_narrow` in `aiter/ops/triton/configs/gfx950/gluon/attention/chunk_delta_attn/DEFAULT.json` apply only to the fallback.
- **`sched_barrier` masks:** `gl.amd.hint.sched_barrier()` accepts only mask 0, and Gluon has no `sched_group_barrier` (section 7.1).
- **Sequences of length 0:** `prepare_chunk_indices` (`kimi_delta_attn/utils/index.py`) gets the sequence ID of a chunk from the positions where the chunk index is 0. A sequence of length 0 has no chunks, so all chunks after it get the sequence ID before theirs: with `cu_seqlens = [0, 64, 64, 128]` and C = 32, the IDs are `[0, 0, 1, 1]`, not `[0, 0, 2, 2]`. The old and the new pass A get the same incorrect inputs. This problem is in main before this change, and this change does not touch it.

## 4. dot2 order (step 6)

This is MIR step 6. On upstream Triton, `sched_barrier` expresses it (`D2=1`, section 4.3). The Triton 3.7 variant uses inline asm (`d2pk16w`, section 4.4). In aiter, `_dot2` in `flash_kda_k2.py` has the `D2=1` order.

### 4.1 What dot2 computes

In `_step` of the research kernel, the source of the dot2 stage is:

```python
g_b = gl.amd.cdna4.mfma(kr, big_b_op, g_b * gt[:, None])
if FULL_COMPUTE:
    g_a = gl.amd.cdna4.mfma(kr, big_a_op, g_a * gt[:, None])
gb_op = gl.convert_layout(g_b.to(gl.bfloat16), B_OP)
```

- g_b and g_a each have 8 output tiles of 16x16, B0 to B7 and A0 to A7 below. Each tile is one `v_mfma_f32_16x16x32_bf16`, so there are 16 MFMAs.
- srcC is `g * gt`. Each thread has 4 f32 values in each tile, which need 2 `pk`. The two chains need 32 `pk`.
- mem1 reads gt with 8 `ds_read_b128`. B_j and A_j use the same part of gt.
- At the end, 16 cvt convert g_b to bf16 for dot1 of the next chunk.

### 4.2 Target order

The instruction strings of region 6 (dot2 of p1):

```text
MIR step 6           wppppppppppppppppppppppppppppppppMMMMMMMMMbbMbbMbbMbbMbbMbbMbbMbbws
Upstream, g_p        ppppMmmpppppppppppMmmppMmmppMmmppMmmppMmmpbMbbbMbbbppMbbbMbbbMbbbMMMMMws
ROCm normal compile  wppwppwppwppwppwppwppwppppppMmmppMmmMMMMMnmmpMmmpMnmmpMnmmpMMnbbbMbbbMbbbMbbbMbbbbws
```

MIR step 6 puts all 32 `pk` first. The first `pk` uses the gt that is read last, so `SIInsertWaitcnts` inserts only one `lgkmcnt(0)`. Then the MFMAs issue back to back, with only cvt between them. The order of the compiler has these problems:

1. **The `pk` are split.** A packed f32 instruction stalls about 16 cycles if an MFMA has not completed, and a scalar `v_mul_f32` does not stall. Thus `SIPreEmitPeephole` of LLVM splits each `pk` in an MFMA shadow into two `v_mul_f32` (`mm` in the string), which adds instructions. Upstream `g_p` splits 6 in each dot2. The ROCm normal compile splits 28 in the 4 dot2, which adds 28 instructions.
2. **VALU in the MFMA shadows.** In the experiment of MIR step 6, an empty dot2 shadow is faster than a shadow with VALU: when one wave group issues VALU in the shadow, the mem cluster of the other wave group on the same SIMD becomes slower.
3. **The ROCm normal compile also waits for LDS 8 times.** The start is 8 groups of `wpp`: after each part of gt comes back, one wait and 2 `pk`. The upstream compile completes the LDS wait at the end of mem1, before the `s_barrier` (`s_waitcnt lgkmcnt(0)`), so dot2 does not wait at its start.

### 4.3 `D2=1`

**First, split the state into pieces.** A `sched_barrier` can only go between two ops. The original `mfma(kr, big_b_op, g_b * gt)` of dot2 makes 8 output tiles at a time, which are 8 MFMAs, and a `sched_barrier` cannot go between them. The research kernel keeps g_b, g_a and gb_op (g_b converted to the bf16 B operand) as 4 pieces of 32 rows (a tuple) in the full loop. One piece is two output tiles of dot2, and it is also one K step of the next dot1. After the LDS read of kr, `gl.amd.slice` splits it by rows into 4 pieces in registers (`krp`), with no instructions:

```python
@gluon.jit
def _sb(ON: gl.constexpr):
    if ON:
        gl.amd.hint.sched_barrier()


@gluon.jit
def _rows(x, i: gl.constexpr, R: gl.constexpr):
    return gl.amd.slice(x, [R, x.shape[1]], [i * R, 0])
```

With only the split into pieces and no `sched_barrier` (`g_p`), the instruction count and the VGPRs do not change, and the time is in the noise.

**The `D2=1` part of `_dot2`** (without `D2S`): first all `g * gt`, then the MFMAs of the 4 B pieces back to back, and at the end the A-piece MFMAs alternate with the B cvt, one pair for each sched region:

```python
cb0 = g_b[0] * gt[0][:, None]          # cb1..cb3, ca0..ca3 likewise: 32 pk
...
_sb(True)
nb0 = gl.amd.cdna4.mfma(krp[0], big_b_op, cb0)
nb1 = gl.amd.cdna4.mfma(krp[1], big_b_op, cb1)
nb2 = gl.amd.cdna4.mfma(krp[2], big_b_op, cb2)
nb3 = gl.amd.cdna4.mfma(krp[3], big_b_op, cb3)
_sb(True)
na0 = gl.amd.cdna4.mfma(krp[0], big_a_op, ca0)
ob0 = _bop(nb0, B_OP)                  # g_b to bf16, the B operand of the next dot1
_sb(True)
na1 = gl.amd.cdna4.mfma(krp[1], big_a_op, ca1)
ob1 = _bop(nb1, B_OP)
_sb(True)
...                                    # (na2, ob2), (na3, ob3) likewise
```

**Result:**

```text
Upstream, g_d2  ppppppppppppppppppppppppppppppppMMMMMMMMMbbbMbMbbbMbMbbbMbMbbbMbws
MIR step 6      wppppppppppppppppppppppppppppppppMMMMMMMMMbbMbbMbbMbbMbbMbbMbbMbbws
```

- The 32 `pk` come first, and 9 MFMAs issue back to back, as in MIR step 6. No `pk` is in an MFMA shadow, so none is split. Region 6 goes from 72 to 66 instructions. VGPR 221 → 229.
- `D2=1` alone (`g_p` → `g_d2`): GPU 3 is 0.55 µs faster, and GPU 2 does not change (near the noise). In the ATT, the Z slot (dot2 of sl0 with mem1 of sl1) goes from 537 to 616 cycles, and with `M1=3` and `BD1=2` it goes down to 464. The probable cause: while dot2 issues its 32 `pk` back to back, the mem1 of the other wave group on the same SIMD cannot issue its VALU (gt address, β clamp). `M1=3` issues the kr reads first, and `BD1=2` moves the β clamp. This was not examined instruction by instruction in the ATT.

### 4.4 Inline asm form: `d2pk16w` (Triton 3.7 variant)

The `g * gt` of g_b and of g_a are each one non-pure inline asm of 16 `v_pk_mul_f32`. The asm is one unit: the scheduler does not split it and does not put an MFMA in it. The cost: the code must handle the hazards between the asm and the MFMAs outside it. On ROCm Triton 3.7, GPU 3: 48.17 to 48.49 µs, about 0.9 µs faster than the normal compile (49.23 to 49.35 µs). VGPR 225 → 227, no spills, the same error.

```text
d2pk16w         wppppppppppppppppnnMMMMMMMMnppppppppppppppppnnMMMbbbMbbbMbbbMbbbMbbbMbws
```

Against `D2=1`: the 16 `pk` of A are between the B and the A MFMAs (`D2=1` has all 32 `pk` at the start), and each dot2 has 5 more `s_nop`. VGPR goes up by only 2 (`D2=1`: by 8).

#### Source

The two switches at the top of the file:

```python
D2_PK16 = gl.constexpr("d2pk16" in _SRC or "d2pk16w" in _SRC)
D2_WAR = gl.constexpr("d2pk16w" in _SRC)
```

The call in the dot2 stage:

```python
with _stage("dot2", PRIO_DOT):
    if D2_PK16 and FULL_COMPUTE:
        gb_g = _gate_pk16n(g_b, gt)
        ga_g = _gate_pk16n(g_a, gt, D2_WAR)
        g_b = gl.amd.cdna4.mfma(kr, big_b_op, gb_g)
        g_a = gl.amd.cdna4.mfma(kr, big_a_op, ga_g)
    ...
    gb_op = gl.convert_layout(g_b.to(gl.bfloat16), B_OP)
```

The first asm computes `g_b * gt`, and the second computes `g_a * gt`. With `d2pk16w`, the second asm starts with one more `s_nop 2`.

The template and the constraints of the inline asm:

```python
_PK16N_ASM = gl.constexpr("\n".join(
    [f"v_pk_mul_f32 ${i}, ${16 + i}, ${32 + i}" for i in range(16)] + ["s_nop 1"]))
_PK16W_ASM = gl.constexpr("s_nop 2\n" + _PK16N_ASM.value)
_PK16N_CON = gl.constexpr(",".join(["=&v"] * 16 + ["v"] * 32))
```

`_PK16N_ASM` expands to:

```text
v_pk_mul_f32 $0, $16, $32
v_pk_mul_f32 $1, $17, $33
...
v_pk_mul_f32 $15, $31, $47
s_nop 1
```

- `$0` to `$15` are the 16 outputs. The constraint `=&v` means VGPR, early-clobber.
- `$16` to `$31` are the 16 register pairs of g, and `$32` to `$47` are the 16 pairs of the broadcast gt.
- Each operand is an `int64`, that is, a pair of adjacent VGPRs, which is one 64-bit operand of a `pk`.
- `_PK16W_ASM` only adds an `s_nop 2` at the start.

`_gate_pk16n`:

```python
@gluon.jit
def _gate_pk16n(g, gt, WAR: gl.constexpr = False):
    # _gate_pk16 plus the MFMA hazards the asm is invisible to: s_nop 1 after
    # (VALU write -> MFMA SrcC read) and, with WAR, s_nop 2 before (the outputs
    # may reuse SrcC registers of a just-issued MFMA; LLVM requires 3 wait states).
    gtb, g = gl.broadcast(gt[:, None], g)
    if WAR:
        rp = gl.inline_asm_elementwise(_PK16W_ASM, _PK16N_CON, [_pairs(g), _pairs(gtb)],
                                       dtype=gl.int64, is_pure=False, pack=16)
    else:
        rp = gl.inline_asm_elementwise(_PK16N_ASM, _PK16N_CON, [_pairs(g), _pairs(gtb)],
                                       dtype=gl.int64, is_pure=False, pack=16)
    return _unpairs(rp, g.type.layout)
```

- `gl.broadcast` expands gt to the shape of g.
- `pack=16`: one asm instance does 16 elements. The g of each thread is 16 int64 (32 f32: 8 tiles of 4), so each thread has one asm instance.
- `dtype=gl.int64`: the outputs are also register pairs. Triton packs automatically only elements of less than 32 bits, not f32, so the code must put two f32 into one int64.

`_pairs`, `_unpairs` and `_pk64`:

```python
@gluon.jit
def _pairs(x):
    # [K, BW] -> [K // 2, BW] int64 of (row 2m, row 2m + 1): adjacent MFMA output registers.
    K: gl.constexpr = x.shape[0]
    BW: gl.constexpr = x.shape[1]
    e, o = gl.split(gl.permute(gl.reshape(x, [K // 2, 2, BW]), [0, 2, 1]))
    return _pk64(e, o)


@gluon.jit
def _unpairs(rp, layout: gl.constexpr):
    K: gl.constexpr = 2 * rp.shape[0]
    BW: gl.constexpr = rp.shape[1]
    re = rp.to(gl.uint32).to(gl.float32, bitcast=True)
    ro = (rp >> 32).to(gl.uint32).to(gl.float32, bitcast=True)
    return gl.convert_layout(gl.reshape(gl.permute(gl.join(re, ro), [0, 2, 1]), [K, BW]), layout)


@gluon.jit
def _pk64(lo, hi):
    return ((hi.to(gl.uint32, bitcast=True).to(gl.int64) << 32)
            | lo.to(gl.uint32, bitcast=True).to(gl.int64))
```

- `_pairs` puts rows 2m and 2m + 1 into one int64, with the even row in the low 32 bits. In the MFMA output layout, these two rows of a thread are in two adjacent VGPRs, so this VGPR pair is already one operand of a `pk`.
- `_unpairs` splits the int64 back into two f32 and puts them back in the `[K, BW]` order.
- These shifts, joins and `convert_layout`s only give a different view of the same registers, and LLVM removes all of them: in the ISA below, there is no `v_mov` before or after the asm.

#### ISA

Region 6 (dot2 of p1), without the middle lines that have the same pattern:

```text
        s_waitcnt lgkmcnt(0)                              ; one LDS wait only
        v_pk_mul_f32 v[162:163], v[32:33], v[170:171]     ; first asm: g_b * gt
v_pk_mul_f32 v[164:165], v[34:35], v[172:173]
...                                                       ; 16 in total, outputs v[162:169], v[202:225]
v_pk_mul_f32 v[224:225], v[62:63], v[200:201]
s_nop 1                                                   ; RAW guard in the asm
        s_nop 0                                           ; added by LLVM, see the s_nop part below
        v_mfma_f32_16x16x32_bf16 v[32:35], v[96:99], v[100:103], v[162:165]   ; B0
        ...                                                                   ; B1..B6
        v_mfma_f32_16x16x32_bf16 v[60:63], v[68:71], v[100:103], v[222:225]   ; B7
        s_nop 2                                           ; first line of the second asm: WAR guard
v_pk_mul_f32 v[100:101], v[0:1], v[170:171]               ; g_a * gt
...                                                       ; 16 in total, outputs v[100:103], v[162:169], v[202:221]
v_pk_mul_f32 v[220:221], v[30:31], v[200:201]
s_nop 1
        s_nop 0
        v_mfma_f32_16x16x32_bf16 v[0:3], v[96:99], v[64:67], v[100:103]       ; A0
        v_mfma_f32_16x16x32_bf16 v[4:7], v[92:95], v[64:67], v[162:165]       ; A1
        v_mfma_f32_16x16x32_bf16 v[8:11], v[88:91], v[64:67], v[166:169]      ; A2
        v_cvt_pk_bf16_f32 v88, v32, v33                   ; g_b to bf16, between the A MFMAs
        ...
        s_waitcnt vmcnt(10) lgkmcnt(0)
        s_barrier
```

- LLVM indents only the first line of an inline asm. The lines without indent are the asm text.
- The first asm waits only once (`lgkmcnt(0)`): the asm is one unit, so it can start only after all 8 parts of gt are back.
- The outputs of the second asm use registers again that the first asm does not need any more (v[162:169], v[202:221]), and also the srcB of the B MFMAs (v[100:103], that is, `big_b_op`, which is not used after B7).

#### Why this form

**16 `pk` in one asm.** The asm is one unit, which has three effects: no MFMA can go into it; LLVM does not split the `pk` into `v_mul_f32`; `SIInsertWaitcnts` waits for LDS only once, before the asm.

**Two asm of 16, not one asm of 32.** MIR step 6 can put all 32 `pk` first without more registers because each `pk` writes back to the registers of its own g (`v_pk_mul_f32 v[62:63], v[62:63], v[164:165]`): this pair of g is not necessary after the `pk` reads it. Inline asm cannot express "output i can share a register with input i, but not with the other inputs":

- With early-clobber, the 32 outputs (64 VGPRs) must be live at the same time and must not overlap any input. The VGPRs go to 256, and the kernel spills.
- With the outputs tied to the inputs (`d2asm`), register allocation adds 63 `v_mov` in the loop.

With two asm, the outputs of the first asm are dead after the B MFMAs read them, and the second asm can use these registers again. VGPR goes up by only 2.

**Early-clobber (`=&v`).** Without `&`, LLVM thinks that the asm reads all inputs before it writes the outputs. Thus LLVM lets an output share a register with an input that is dead after the asm. But this asm has 16 instructions: if output 0 gets the registers of pair 1 of g, instruction 0 writes it, and then instruction 1 reads an incorrect value. `&` makes sure that no output shares a register with an input. The cost is VGPR 225 → 227.

**`is_pure=False`.** An asm with side effects is not removed, duplicated or moved out of its stage, and the order of the two asm does not change. The MFMAs can still move: the machine scheduler puts the B MFMAs between the two asm, which lets the second asm use the registers of the first again.

**int64 register pairs.** Rows 2m and 2m + 1 of a thread are already in two adjacent VGPRs. As an int64, they are one operand of a `pk`, with no `v_mov`.

#### `s_nop`: the code must handle the MFMA hazards of the asm

The hazard recognizer of LLVM does not look into an asm, and for the worst case it counts an asm as 0 wait states (`getCountedWaitStates` in `GCNHazardRecognizer.cpp`: "Inline asm may be empty, so it adds none, which is the worst case"). Thus the code must give the wait states between the `pk` in the asm and the MFMAs outside it. The table comes from the hazard pass of the ROCm `llc` for gfx950 (the procedure is in the folded block below):

| Case | LLVM inserts | Wait states | In d2pk16w |
| --- | --- | --- | --- |
| RAW: a `pk` writes a register, and the next MFMA reads it as srcC | `S_NOP 1` | 2 | The `s_nop 1` at the end of each asm |
| WAR: an MFMA reads a register as srcC, and the next `pk` writes it | `S_NOP 2` | 3 | The `s_nop 2` at the start of the second asm |
| An MFMA reads srcB, and the next `pk` writes it | Nothing | 0 | Not necessary |

- The WAR wait states depend on the number of passes of the MFMA. LLVM counts a 16x16x32 bf16 MFMA as a 4-pass XDL instruction (`GFX940_XDL4PassReadVgprVALUWarWaitStates = 3`). 2, 8 and 16 passes need 1, 7 and 15. For a different MFMA shape, do the check again.
- Of the registers that the second asm writes, the nearest one is the srcC of B6. Between them are B7, `s_nop 2` and 14 `pk`, which is much more than 3 wait states. The `s_nop 2` is for the worst case: if the register allocation changes, the first `pk` of the second asm can write the srcC of B7.
- At the end of each asm, LLVM adds one more `s_nop 0`. This is the dst forwarding rule of gfx940 and later: LLVM assumes that all outputs of an inline asm have a forwarding hazard (the source comment is "Assume inline asm has dst forwarding hazard"), so a VALU (an MFMA too) that reads them immediately needs 1 wait state. LLVM does not see the `s_nop 1` in the asm, so this `s_nop 0` is not necessary, but the source cannot remove it.
- Without the WAR guard (`d2pk16`), the kernel is equally fast, and with the current register allocation the result is also correct. But a change of the register allocation can make the result incorrect.
- The first version used 12 wait states for 8 passes (`s_nop 7` + `s_nop 3`), and it was about 1.8 µs slower than `d2pk16`. The check with `llc` showed that 3 are sufficient, so the code now has `s_nop 2`.

<details>
<summary>Reproduce the table with <code>llc</code></summary>

Save this as `t.mir`:

```text
---
name: war
tracksRegLiveness: false
body: |
  bb.0:
    $vgpr0_vgpr1_vgpr2_vgpr3 = V_MFMA_F32_16X16X32_BF16_vgprcd_e64 $vgpr4_vgpr5_vgpr6_vgpr7, $vgpr8_vgpr9_vgpr10_vgpr11, $vgpr12_vgpr13_vgpr14_vgpr15, 0, 0, 0, implicit $mode, implicit $exec
    $vgpr12_vgpr13 = V_PK_MUL_F32 8, $vgpr20_vgpr21, 8, $vgpr22_vgpr23, 0, 0, 0, 0, 0, implicit $mode, implicit $exec
    S_ENDPGM 0
...
---
name: war_srcb
tracksRegLiveness: false
body: |
  bb.0:
    $vgpr0_vgpr1_vgpr2_vgpr3 = V_MFMA_F32_16X16X32_BF16_vgprcd_e64 $vgpr4_vgpr5_vgpr6_vgpr7, $vgpr8_vgpr9_vgpr10_vgpr11, $vgpr12_vgpr13_vgpr14_vgpr15, 0, 0, 0, implicit $mode, implicit $exec
    $vgpr8_vgpr9 = V_PK_MUL_F32 8, $vgpr20_vgpr21, 8, $vgpr22_vgpr23, 0, 0, 0, 0, 0, implicit $mode, implicit $exec
    S_ENDPGM 0
...
---
name: raw
tracksRegLiveness: false
body: |
  bb.0:
    $vgpr12_vgpr13 = V_PK_MUL_F32 8, $vgpr20_vgpr21, 8, $vgpr22_vgpr23, 0, 0, 0, 0, 0, implicit $mode, implicit $exec
    $vgpr0_vgpr1_vgpr2_vgpr3 = V_MFMA_F32_16X16X32_BF16_vgprcd_e64 $vgpr4_vgpr5_vgpr6_vgpr7, $vgpr8_vgpr9_vgpr10_vgpr11, $vgpr12_vgpr13_vgpr14_vgpr15, 0, 0, 0, implicit $mode, implicit $exec
    S_ENDPGM 0
...
```

Then run:

```bash
/opt/rocm/llvm/bin/llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -run-pass=post-RA-hazard-rec \
  -verify-machineinstrs=0 t.mir -o - | grep -E "^name:|S_NOP|V_MFMA|V_PK_MUL"
```

In `war`, there is an `S_NOP 2` between the two instructions. In `raw`, there is an `S_NOP 1`. In `war_srcb`, there is none.

</details>

#### Forms that did not help

| Form | Change | Result |
| --- | --- | --- |
| `d2pin` | Normal multiplies, with the results pinned by an empty non-pure asm (`_pin2`) | All `pk` are split into `v_mul_f32` (64 in each dot2). VGPR 213, 50.9 µs |
| `d2asm`, early-clobber | 32 `pk` in one asm | VGPR 256, spills |
| `d2asm`, outputs tied to the inputs (the current code) | As above, but the outputs write back to g | Alone, it still spills (VGPR 256, 7 spills). With `m1spread`, VGPR 227 and no spills, but 63 more `v_mov` in the loop: 247 instructions for each chunk (`d2pk16`: 234), about the speed of the normal compile |
| `d2pk16` | Two asm of 16, without `s_nop 2` at the start of the second | About −1.0 µs, but not safe |
| `d2pk16w`, 12 wait states | The first guarded version (`s_nop 7` + `s_nop 3`) | About 1.8 µs slower than `d2pk16` |

#### Results

| | ROCm normal compile | d2pk16w | MIR steps 1 to 6 |
| --- | --- | --- | --- |
| Instructions in the 4 dot2 | 328 | 289 | 269 |
| Of these, `s_waitcnt` | 27 | 8 | 8 |
| `v_pk_mul_f32` | 100 | 128 | 128 |
| `v_mul_f32` | 56 | 0 | 0 |
| `s_nop` | 13 | 21 | 1 |
| VGPR | 225 | 227 | 199 |
| T = 16384, GPU 3 (µs) | 49.23 to 49.35 | 48.17 to 48.49 | 45.8 to 46.1 |

- All three have 64 MFMAs and 64 cvt. The VGPRs and the time of the MIR column include all changes of MIR steps 1 to 6, not only the dot2 changes.
- The error is the same as the normal compile, at T = 16384 and at T = 8192.
- Of the 21 `s_nop` of d2pk16w, 20 come from the asm: 5 in each dot2, that is, one `s_nop 1` at the end of each asm, one `s_nop 0` that LLVM adds after each, and the `s_nop 2` at the start of the second asm.
- With `m1spread`, there is no more gain.

#### Notes for changes

- `pack=16` assumes that the g of each thread is 16 int64. If the shape, the number of warps or the layout changes, calculate `pack` again, and make sure in the ISA that each asm still has 16 `pk` and that the 8 B MFMAs are still between the two asm.
- The `s_nop` values depend on the MFMA shape and on the hazard rules of LLVM. After a change of the MFMA or of the compiler version, do the `llc` check above again.
- The outputs must stay early-clobber (`=&v`).
- A possible improvement: the second asm comes immediately after the 8 B MFMAs. From the MIR experiments (a packed f32 instruction stalls about 16 cycles if an MFMA has not completed), the first `pk` of the second asm can stall once in each dot2. In MIR step 6 and in `D2=1`, all `pk` are before the MFMAs, so there is no such stall. The ATT does not confirm this yet.

## 5. mem1: reads before copies (step 3)

This is MIR step 3. For the first half (the LDS reads before the copies), upstream Triton uses `sched_barrier` (`M1=3`, section 5.2), and the Triton 3.7 variant uses an empty inline asm (`_fence`, section 5.3). Section 5.4 is about the second half (the copies between the reads). In aiter, mem1 of `_step` in `flash_kda_k2.py` has the `M1=3` order.

### 5.1 Problem

mem1 issues 24 LDS reads (16 for kr, 8 for gt) and 4 groups of global-to-LDS copies (kd, inv, v and β, each with its address SALU and `m0`). The issue is in order: a stalled `buffer_load` blocks all LDS reads after it. Thus the reads must issue first.

```text
Region 5 (mem1 of p1)
  ROCm normal compile  ssssgssssgssssgs.s.ss.sbgrrrrrrrrrrrrrrrr.rrrrrrrrbbbbbbbs
  Upstream, g_d2       ssssssssss.s.rrrrrrrrrrrrrrrrrrrrrrrrgss.gsssgss.sbgbbbbbbbws
  MIR step 3           rrrrssssgrrrrssssgrrrrssssgrrrrssssgrrrrrrbbbbbbbbs
```

The MIR line is b1n2s4t1, which also has step 7, so only 6 gt reads stay.

- The ROCm normal compile puts the copies first. The first LDS read issues only about 156 cycles after the barrier releases. After dot1 is reordered as in the MIR and becomes denser, the next instruction after the second copy waits 80 to 100 more cycles.
- **The machine scheduler makes this order.** The source of the research kernel reads kr and gt first and then issues the copies. The MIR before the machine scheduler has the same order (24 reads, then 4 copy groups). The machine scheduler moves the copies to the front. It can do this because Triton gives the two types of memory access alias scopes that do not alias: the LDS reads have `!alias.scope !75, !noalias !49`, and the copies have `!alias.scope !49` (`!75` is `amdg.LocalLoads`, `!49` is `amdg.AsyncCopies`). Thus there is no dependency between them.
- The upstream Triton compile already reads before it copies, but it starts with 13 preparation instructions (the copy addresses and others, section 5.2).
- MIR step 3 issues the LDS reads first and spreads the 4 copy groups between the reads: one group after each 4 kr reads. It uses the issue cycles in which the wave is idle between the reads.

### 5.2 `M1=3`

**How to read a mode name:** a mode name gives the switches that are on. `g_` means the research kernel compiled with upstream Triton, and each part after it is one switch: `d2` is `D2=1`, `m3` is `M1=3`, `b2` is `BD1=2`, and `lt` is `LT=1`. The switches that are not in the name are 0.

- `g_d2`: only `D2=1`, that is, the dot2 order of section 4: first all 32 `pk`, then the MFMAs back to back. mem1 does not change: `M1` is 0.
- `g_d2m3`: `g_d2` with `M1=3` in addition.

The two modes differ only in `M1`, so all changes from `g_d2` to `g_d2m3` come from `M1=3`. The cumulative table of section 2 also turns on one more switch in each row: `g_d2` → `g_d2m3` → `g_d2b2m3` → `g_lt`, and `g_lt` has all 4 switches on.

**The value of `M1` is the number of an order:** `M1` sets the order of the LDS reads and the copies in mem1. Each value is one order, and the value is only a number; `D2` and `BD1` are the same. 0 is the original code, and 1 to 4 are the 4 orders that were tried:

| `M1` | Order | Instruction string of mem1 of p1 (region 5) |
| --- | --- | --- |
| 0 | No `sched_barrier`; the compiler sets the order | `ssssssssss.s.rrrrrrrrrrrrrrrrrrrrrrrrgss.gsssgss.sbgbbbbbbbws` |
| 1 | First the 4 copy groups, then the kr and gt reads. The same order as the ROCm normal compile, for comparison | `sssssssssss.gss.gsssgss.sngrrrrrrrrrrrrrrrr.rrrrrrrrbbbbbbbbws` |
| 2 | 1 copy group after the kr reads, the other 3 groups between the gt reads | `rrrrrrrrrrrrrrrrsssssssss..ssng.rrssngrrssngrrrrssss.gbbbbbbbbws` |
| **3** | **First the kr reads, then the gt reads, then the 4 copy groups.** The first half of MIR step 3. This order is landed | `rrrrrrrrrrrrrrrrssssssss.rrrrrrrrsss.gss.gsssgss.sbgbbbbbbbws` |
| 4 | kr in 4 slices, with 1 copy group after each slice. The full order of MIR step 3, section 5.4 | `ssssssssssrrrrgs..ssssrrrrgssrrrrgssss.rrrrg.rrrrrrrrbbbbbbbbws` |

`r` is an LDS read, `g` is a copy, `s` is SALU, `.` is other VALU, and `b` is cvt (section 1.3).

**`M1 == 3` in the code:** `_sb(ON)` puts a `sched_barrier` if `ON` is true, and nothing if it is false (section 4.3). A `sched_barrier` is a wall for the machine scheduler: no instruction can move from one side of it to the other side. `M1` is a compile-time constant, so `_sb(M1 == 3)` means "if the order is number 3, put a wall here". `M1=3` puts two walls, which split mem1 into three parts. From `_step_g`:

```python
kr = gl.amd.cdna4.async_copy.load_shared_relaxed(kr_r.index(S).permute([1, 0]), A_OP)
krp = (_rows(kr, 0, R), _rows(kr, 1, R), _rows(kr, 2, R), _rows(kr, 3, R))
_sb(M1 == 3)                           # wall 1; above it, part 1: the 16 kr reads
gt0 = _gt_piece(gt_s, S, 0, R, MMA)
gt1 = _gt_piece(gt_s, S, 1, R, MMA)
gt2 = _gt_piece(gt_s, S, 2, R, MMA)
gt3 = _gt_piece(gt_s, S, 3, R, MMA)
_sb(M1 >= 3)                           # wall 2; above it, part 2: the 8 gt reads
# part 3: big to bf16, the clamp of the beta address, the 4 copy groups
big_b_op = gl.convert_layout(big_b.to(ws_kr.dtype.element_ty), B_OP)
big_a_op = gl.convert_layout(big_a.to(ws_kr.dtype.element_ty), B_OP)
if M1 == 3 and BD1 == 0:
    boff = _beta_off(beta_o, blim, i_h32, H)
for w in gl.static_range(4):
    _ac_d_g(kd_w, inv_w, v_w, beta_s, pd, S3, kd_off, inv_off, v_off, boff, w)
```

- All 4 copy groups are in part 3, so the machine scheduler cannot move them before the reads.
- With `M1=0`, the clamp of the β address (`_beta_off`) is at the start of mem1. `M1=3` moves it to part 3.
- Wall 2 is `M1 >= 3` because `M1=4` also needs it: `M1=4` has its own wall after each kr slice, and after the gt reads it also must separate the reads from the cvt.
- The excerpt does not show the branches of `GTE` (step 7) and `COPY`. `COPY` is false for the last 3 chunks, because then there is no data to prefetch.
- In aiter, the reads use `smem.load(layout)`, because `load_shared_relaxed` is deprecated on Triton main. There is no β clamp (section 3.2).

**Changes in the ISA** (region 5, rows 0 and 3 of the table above):

- `g_d2`: after the barrier, 13 preparation instructions issue first, and the first LDS read is instruction 14. These 13 are the copy addresses and `m0` (10 SALU), the clamp of the β address (2), and the gt read address (1). Of these, `v_add_u32` (gt read address) and `v_min_i32` (β clamp) are VALU.
- `g_d2m3`: the first instruction after the barrier is a kr read, and the 16 kr reads issue in sequence. The copy addresses and the gt read address move between the kr reads and the gt reads. `m0` and the β clamp move after the gt reads.
- Both have 61 instructions and 229 VGPRs. Only the order is different.

**Why it is faster:** the problem occurs when mem1 runs together with dot2 of the other wave group (the Z slot, section 1.3). With `D2=1`, dot2 starts with 32 `pk` back to back. Below are the issue times of the two wave groups on SIMD 0 in the Z slot of p1, in the ATT. The cycles start when the barrier releases (`g_d2` is from a repeat run on 173580104f, `g_d2m3` is from the run of the cumulative table in section 2):

```text
g_d2
  sl0 (dot2)  0 to 124: one pk each 4 cycles, 32 in total; the MFMAs start at 128
  sl1 (mem1)  0 to 36: 10 SALU; 40: the first VALU (the gt read address);
              then no instruction for 100 cycles; 140 and 144: 1 each; 164: the first kr read
g_d2m3
  sl1 (mem1)  0 to 116: all 16 kr reads; 124: the first VALU; 152: the first gt read
```

- In `g_d2`, mem1 stops after its first VALU, until the other wave group has issued its 32 `pk`. A wave issues its instructions in order, so the 24 LDS reads after the VALU also wait.
- The same mem1 code does not stop in the Y slot. In the Y slot, the other wave group runs dot1, whose VALU are between the MFMAs and not in a long sequence. Thus the first LDS read of mem1 issues at cycle 60.
- `g_d2m3` puts the kr reads first. The kr read addresses are in registers before mem1 starts, so they need no VALU, and the reads can all issue while the other wave group issues its `pk`.
- Thus only the Z slot becomes shorter in the ATT: in the run of the cumulative table in section 2, the Z slot of p1 to p3 goes from 616 to 529 cycles, and X, Y and W change by less than 5 cycles.

**Results:**

- Repeat runs on 173580104f (5 runs on each of GPU 2 and GPU 3, one process at a time): GPU 2 50.34 → 49.63 µs, GPU 3 49.10 → 48.39 µs, 0.7 µs faster on both. In all 10 runs, `g_d2m3` is faster, by 0.04 to 1.47 µs.
- In the run of the cumulative table of section 2 (f8cddb947b, both GPUs at the same time): GPU 2 50.32 → 50.08 µs, GPU 3 48.86 → 47.93 µs. ATT per chunk 2144 → 2060 cycles.

### 5.3 Inline asm form: `_fence` (Triton 3.7 variant)

The Triton 3.7 variant separates the reads and the copies with an empty non-pure inline asm:

```python
@gluon.jit
def _fence(x):
    # Empty non-pure asm: the machine scheduler keeps memory operations on their
    # side of it (ALU instructions still move freely).
    return gl.inline_asm_elementwise("", "=s,s", [x], dtype=gl.int32, is_pure=False, pack=1)
```

- `m1late`: one `_fence(i_h32)` after the kr and gt reads, and the 4 copy groups after it.
- `m1spread`: kr is read in 4 slices (`krv.slice`), with one copy group after each slice and a `_fence` between each read and copy. At the end, `_cat0` puts the 4 slices back together into kr, and then the gt reads follow.

**The arguments:**

- Template `""`: no instructions in the asm.
- Constraints `"=s,s"`: one SGPR output and one SGPR input.
- `[x]`: the call gives `i_h32` (`program_id(1) % H`), a scalar that is already in an SGPR. The input is a scalar, so the call makes only one asm instance and uses no VGPR.
- `is_pure=False`: the asm has side effects. Nothing uses the result, but an asm with side effects is not removed.
- `dtype=gl.int32`, `pack=1`: one int32 output, and each asm instance does one element.

**LLIR** (`m1late`, mem1 of p1, between the last gt read and the first copy group):

```llvm
%1095 = load <4 x float>, ptr addrspace(3) %359, align 16, !dbg !95, !alias.scope !75, !noalias !49
%1096 = tail call i32 asm sideeffect "", "=s,s"(i32 %.decomposed) #5, !dbg !96
; ... the fptrunc of big to bf16
tail call void @llvm.amdgcn.raw.ptr.buffer.load.async.lds(ptr addrspace(8) %1125, ptr addrspace(3) %119, i32 16, i32 %120, i32 0, i32 0, i32 0), !dbg !107, !alias.scope !49

attributes #5 = { nounwind }
!49 = !{!50}
!50 = !{!"amdg.AsyncCopies", !51, !"Scope containing all AsyncCopyGlobalToLocal and BufferLoadToLocal ops"}
!75 = !{!76}
!76 = !{!"amdg.LocalLoads", !51, !"Scope containing all LocalLoad ops"}
```

- `asm sideeffect` comes from `is_pure=False`. `#5` has only `nounwind` and no `memory(...)`, so LLVM thinks that the call can read and write all memory.
- `%.decomposed` is the value of `i_h32`.
- The alias scopes of the LDS reads and of the copies do not overlap, so there is no dependency between them. `_fence` has a dependency on both sides.

**MIR before the machine scheduler** (without `implicit $exec` and the memory operands):

```text
%2179:vreg_128_align2 = DS_READ_B128_gfx9 %2157, 448, 0                ; the last gt read
INLINEASM &"", sideeffect attdialect, regdef:SReg_32, def dead %1240, reguse:SReg_32, %3830
undef %2284.sub0:vreg_128_align2 = V_CVT_PK_BF16_F32_e64 0, %2103.sub0, 0, %2103.sub1, 0, 0
...                                                                     ; 8 cvt in total
$m0 = COPY %20
BUFFER_LOAD_DWORDX4_LDS_OFFEN %714, %841, 0, 0, 0, 0, 1                 ; the first copy group (kd)
```

- `sideeffect` makes `hasUnmodeledSideEffects()` of this `INLINEASM` true. The output is `def dead`: nothing uses it.
- At this point, the order is the source order (the normal compile also reads before it copies at this point). `_fence` has its effect in the machine scheduler that follows.

**ISA:**

```text
	ds_read_b128 v[204:207], v16 offset:448
	;;#ASMSTART
	;;#ASMEND
	s_add_u32 s0, s51, 0x8000
	s_addc_u32 s1, s52, 0
	s_mov_b32 m0, s28
	s_and_b32 s1, s1, 0xffff
	buffer_load_dwordx4 v143, s[0:3], 0 offen lds
```

- The asm has no instructions, so the ISA has only the two comment lines `;;#ASMSTART` and `;;#ASMEND`, which show the position of `_fence`. The ISA of `m1late` has 5 such empty asm (4 in the loop, 1 outside it), and `m1spread` has 40.
- VGPR stays 225, the same as the normal compile.

**Why it stops memory operations but not ALU:** when the machine scheduler builds the dependency graph (`ScheduleDAGInstrs::buildSchedGraph`), it uses an instruction with unmodeled side effects as a barrier chain: each memory operation before it gets an edge to it, and it gets an edge to each memory operation after it. Thus no LDS read or copy can move past it. ALU instructions do not access memory and have no edge to it, so they can move freely. The IR level is the same: a call without `memory(...)` can read and write all memory, so no IR pass moves a load or a store past it.

ALU instructions do move. `m1spread` shows this: in the source, the 8 cvt of big to bf16 are after all `_fence`s, but in the ISA, 3 of them move past some `_fence`s into copy groups 2 to 4 (`ssssbg`):

```text
Region 5 (mem1 of p1)
  ROCm normal compile  ssssgssssgssssgs.s.ss.sbgrrrrrrrrrrrrrrrr.rrrrrrrrbbbbbbbs
  m1late               .rrrrrrrrrrrrrrrrrrrrrrrrssssgssssbgssssbgs.s.ss.sbgbbbbbs
  m1spread             rrrrssssgrrrrssssbgrrrrssssbgrrrrss.s.s.sbg.rrrrrrrrbbbbbs
```

**Comparison with the two other walls:**

| | Instructions | Memory operations can move past it | ALU can move past it | Used in |
| --- | --- | --- | --- | --- |
| `_fence(x)` | None | No | Yes | `m1late` and `m1spread` of the Triton 3.7 variant |
| `_s_barrier(x)` | One `s_barrier` | No | Yes | Workgroup synchronization outside the loop |
| `gl.amd.hint.sched_barrier()` | None (`SCHED_BARRIER 0` in the MIR) | No | No | `_sb` of the research kernel |

`_fence` is almost a `sched_barrier` that lets ALU move and does not let memory operations move. The LLVM `sched_barrier` can do this with a mask, but `hint.py` of upstream Gluon accepts only mask 0.

**Results** (ROCm Triton 3.7, T = 16384, GPU 3, alternately with the normal compile): `m1late` −0.3 µs, `m1spread` −0.2 µs. With `d2pk16w`, there is no more gain.

### 5.4 Second half: copies between the reads

- `M1=4`: kr is split along K into 4 memdesc slices in LDS (`.slice`). Each slice has 4 reads followed by one copy group, each in its own sched region. The ISA is almost the same as MIR step 3, but it is 0.95 µs slower than `g_lt`. The cause was not examined.
- The kr of these slices has a (16,1,8) swizzle, and slices along K cross the phases of the swizzle. Upstream `MemDescSubsliceOp` only requires that the offset is a multiple of the slice size. The lowering uses the offset as a logical coordinate and applies the full swizzle map to it, so the addresses are correct (the error does not change).
- `m1spread` (section 5.3) has the same order. On ROCm Triton 3.7 it is about 0.2 µs faster, less than `m1late`.
- The value of this half can be estimated from the Y_1 slot of `g_lt` (section 7.5): with `M1=3`, the copies come after the reads, so the copy SALU cannot use the idle issue cycles between the reads.

## 6. Back-edge SALU and β clamp (steps 5 and 8)

### 6.1 Back-edge SALU (step 5, `LT=1`)

**Symptom:** after the last `s_barrier` of each iteration and before the branch, there are 12 SALU: 5 pairs of 64-bit pointer increments (`s_add_u32`/`s_addc_u32`), one loop count `s_add_i32`, and one compare `s_cmp_lg_u32`. They fall at the start of dot1 of p0 of the next iteration, and both wave groups must issue them first. In the ATT of the ROCm normal compile, the first VALU of dot1 of p0 issues 72 cycles after the barrier releases, and the first MFMA at 162 cycles (after MIR step 5: 20 and 56). The upstream compile of the original has the same 12 instructions.

**Cause:** in the source, each chunk calculates the 6 copy source addresses again with `ws0 + j + 3` and `tok_base + (j + 3) * C`. LSR changes them into 5 pairs of 64-bit pointer IVs and puts the increments before the loop compare. The warp pipeline puts the border at the end of the loop (`s_waitcnt lgkmcnt(0)` + `s_barrier`) at the end of the loop body, before the loop compare. Thus all increments come after the border.

**MIR step 5** (`--latch-early`): it moves these 12 instructions before the last `s_waitcnt lgkmcnt(0)` + `s_barrier` of mem2 of p3, where the LDS wait hides them. Only the copies and the branch of the next iteration read these SGPRs, and between them there are only `s_waitcnt` and `s_barrier`, which do not change SCC. Thus the move is legal.

**`LT=1`:** LSR stays on, and the source writes these IVs itself. The 6 copy bases and the β limit `T − 1 − tok` (int32) are a tuple `pk`, which is a loop-carried value. Position PO (0 to 3, a compile-time constant) uses `pk` plus the offset of PO chunks, and this offset goes into the immediate of the SALU. At the end of mem2 of p3, `pk` moves on by 4 chunks. Thus the increments come before the last `s_barrier`:

```python
@gluon.jit
def _advance(pk, n: gl.constexpr, C: gl.constexpr, K: gl.constexpr, V: gl.constexpr,
             H: gl.constexpr):
    """pk (the six _bases and the beta limit) moved on by n chunks."""
    return (pk[0] + n * (C * K), pk[1] + n * (2 * C * C), pk[2] + n * (C * H * V),
            pk[3] + n * (C * H), pk[4] + n * (C * K), pk[5] + n * K, pk[6] - n * C)

# Outside the loop: the bases of chunk 3
b6 = _bases(ws_kd, ws_inv_mqk, v_input, beta_in, ws_kr, ws_gt, ws0 + 3, tok_base + 3 * C,
            C, K, V, H)
pk = (b6[0], b6[1], b6[2], b6[3], b6[4], b6[5],
      (T_TOTAL - 1 - (tok_base + 3 * C)).to(gl.int32))

# In _step_g, mem1 and mem2 of position PO: the bases of chunk j + 3, which this position copies
pd = _advance(pk, PO, C, K, V, H)
# At the end of mem2 (ADV = 4 only if PO == 3)
if LT and ADV > 0:
    pk = _advance(pk, ADV, C, K, V, H)
```

In aiter, `_advance` and `pk` have the same form, but `pk` has only the 6 copy bases: there is no β limit (section 3.2).

**Result:** the back edge goes from 12 to 4 instructions. `g_d2b2m3` → `g_lt`: GPU 2 49.50 → 48.35, GPU 3 48.18 → 47.06. In the ATT, p0 becomes shorter (the 4 slots together: 2516 → 2290), and p1 to p3 become a little longer (X, Y, Z and W together: 1839 → 1874).

The 4 instructions that stay:

```text
; g_lt, region 16 (back edge)
s_addk_i32   s18, 0xff80        ; beta limit -4C
s_add_u32    s2, s2, 0x8000     ; offset IV shared by kd and kr, +4 chunks
s_addc_u32   s3, s3, 0
s_cmp_lg_u32 s2, 0x38000        ; the loop count also uses this IV
```

- This is the choice of LSR: kd and kr have the same step for each chunk (C·K bf16), so LSR merges the two into one offset IV and also uses it for the loop count. Thus dot1 of p0 starts with 9 more SALU (`MsssMsssMsss`), which calculate the kd and kr bases and the β limit from this IV.
- When the β limit was int64, the back edge had 5 instructions (a 64-bit add needs 2), and the gain was only about 0.5 µs. As int32, the back edge has 4 instructions, and the gain is about 1 µs (different runs).
- In aiter, the back edge has 6 SALU and the branch: two 64-bit pointer increments, the loop count and the compare. The loop count is a run-time value, because the number of chunks changes for each segment.

**LSR off does not help:** with `llvm_cl_opts="disable-lsr"` in addition (`g_ltl`), the back edge has 2 instructions and the 9 SALU of dot1 of p0 disappear, but mem2 of p3 has more SALU, and the kernel is 0.4 µs slower. LSR off without `LT` is also about 0.4 µs slower. On ROCm Triton 3.7, LSR off (`--src nolsr`) is also slower (49.9 µs).

`LT` changes only the source and does not need `sched_barrier`.

### 6.2 β clamp (step 8, `BD1=2`)

**MIR step 8:** each mem1 has 4 instructions that calculate the source address of the β copy: `s_add_i32` (token limit plus offset), `v_min_i32` (clamp), `v_mul_lo_u32` and `v_add_lshl_u32`.

- In the Z slot, mem1 of sl1 and dot2 of sl0 share a SIMD. The 32 `pk` at the start of dot2 and the cvt of the A part issue VALU, so these mem1 instructions can only issue between the 9 MFMAs of the B part.
- `v_mul_lo_u32` is a quarter-rate instruction. One instruction uses many issue cycles, so it delays the MFMAs of dot2, and the β copy issues later too.
- The inputs of these 4 instructions are all available in dot1. The MIR moves them to the end of dot1 of the same position, into the wait of an `s_nop 4`.

**`BD1=2`:** the stage of dot1 calculates the offset of the β copy, and the result `boff` goes to the β copy of mem1:

```python
@gluon.jit
def _beta_off(beta_o, blim, i_h, H: gl.constexpr):
    # The upper half of the 2C beta window runs past T for the last chunk: blim = T - 1 - tok.
    return gl.minimum(beta_o, blim.to(gl.int32)) * H + i_h

# _dot1, D1=0: after the 4 MFMAs of the A chain
if BD1 != 0:
    boff = _beta_off(beta_o, blim, i_h32, H)
```

**Result:** dot1 starts with 1 more SALU (the β limit) and ends with 2 more VALU (`v_min_i32`, `v_mul_lo_u32`, the `III..I` in the instruction string). LLVM merges the third instruction of the β clamp, `v_add_lshl_u32`, with the "multiply by 4 bytes" of the copy, and it stays in mem1. `g_d2m3` → `g_d2b2m3`: ATT per chunk 2060 → 2008 (Z slot 529 → 464). The time: GPU 2 0.58 µs faster, GPU 3 0.25 µs slower, in the noise.

The aiter kernel reads β from the K1 workspace, and only the copy of the last chunk masks (section 3.2). It has no β clamp, so this step does not apply to it.

## 7. Steps not landed (steps 1, 2, 4, 9 and 10)

All verdicts below use upstream Triton main as the reference.

### 7.1 Steps 1 and 9: interleave one MFMA at a time (possible, but slower)

**The MIR method:**

- Step 1: the order of the 20 MFMAs of dot1 does not change. List scheduling with an issue model puts the ready VALU in the MFMA shadows, by urgency, in the pattern one MFMA, two VALU (`MbbMbb…`). The step also splits the `pk` of dot1 into `v_mul_f32`, and merges the `v_xor` that negates β into the neg modifier of the multiply.
- Step 9: after each MFMA of the B part of dot2, 2 `v_mul_f32` issue (`Mmm`). The 4 of each two B MFMAs make the srcC of one A tile.

**All forms by piece are slower:**

```text
Region 4 (dot1 of p1)
  g_d2b2m3  MsMMnbbbMbMMnbbbMbnbbbbMmmphMhnphMMMbbbMbMnMMnmmpppphhhhhIII..Iws
  MIR       ffffffMffMbbMbbMbbMbbMbbMbbMbbMbbMmmMmmMhhMmmMmmMs...hhMmmImmIhhmmmmhhnIIws
```

- `D1=1/2` splits the sched regions by K step. A piece is 2 MFMAs and 4 cvt, so the best order is `MMbbbbMM…`. It is 1.12 / 0.69 µs slower than `g_lt`.
- `D2S=3` can only make `MM........`, which is 0.90 µs slower.

**One MFMA at a time:** a `sched_barrier` can only split sched regions; in a sched region, the machine scheduler sets the order. If each MFMA and each group of VALU is one op in its own sched region, the order is fixed. Upstream Gluon `gl.amd.hint.sched_barrier()` accepts only `allow=None` (mask 0). It also does not expose `llvm.amdgcn.sched.group.barrier(mask, size, syncID)` of LLVM, which can alternate "N MFMAs, N VALU, N DS". With them, fewer sched regions would be necessary, but they are not required (the C++ builder already accepts a non-zero `sched_barrier` mask; only `hint.py` does not). Two problems must be solved, and neither costs instructions.

**Problem one: join two tiles of 16 rows into 32 rows.** In K2, the result of each MFMA is the B operand of the next MFMA, and the rows of the result are the K of the next MFMA. A 16x16x32 MFMA makes only 16 rows, but the next one needs a K of 32 rows. Thus, with one MFMA at a time, the results of two MFMAs must make one tensor:

- tmp → big: the rows of tmp are C (32), and the K of big = inv @ u is also C. A K step of dot1 is one MFMA for each of the two 16-row tiles of C. With one MFMA at a time, tmp splits along C into an upper and a lower tensor, which must join again at u. big into dot2 is the same, because the K of dot2 is also C.
- dot2 → the next dot1: the rows of g are K (128), and the next dot1 has one MFMA for each 32 rows of K. A piece of dot2 is two output tiles (for example B0 and B1), which are the operand of one K step of the next dot1. With one MFMA at a time, each MFMA makes only 16 rows, and two must join into one K step.
- A split along K needs no join: the research kernel splits the K = 128 of dot1 into 4 steps, and each step accumulates into the same tmp. The result is still one tensor.

In the ISA, the join costs no instructions: the cvt of the two tiles write into the same group of 4 VGPRs, and the next MFMA reads these 4 as one operand.

```text
; MIR of the ROCm normal compile, dot1 of p1 (region 4): the B operand %2099 of big_b is made from the two C tiles of tmp_b
undef %2099.sub0 = V_CVT_PK_F16_F32_e64 0, %2068.sub0, 0, %2068.sub1, 0, 0   ; %2068, %2065: tile 0 (%1950) times -beta
%2099.sub1 = V_CVT_PK_F16_F32_e64 0, %2065.sub0, 0, %2065.sub1, 0, 0
%2099.sub2 = V_CVT_PK_F16_F32_e64 0, %2088.sub0, 0, %2088.sub1, 0, 0         ; %2088, %2085: tile 1 (%1965) times -beta
%2099.sub3 = V_CVT_PK_F16_F32_e64 0, %2085.sub0, 0, %2085.sub1, 0, 0
%2100 = V_MFMA_F32_16X16X32_F16 %1910, %2099, 0, 0, 0, 0                      ; the two output tiles of big_b (upper and lower half of inv),
%2106 = V_MFMA_F32_16X16X32_F16 %1911, %2099, 0, 0, 0, 0                      ;   both with all of %2099 as the B operand

; dot2 of p1 (region 6): the B operand %2594 of K step 0 of the next dot1 is made from B0 and B1
undef %2594.sub0 = V_CVT_PK_BF16_F32_e64 0, %2291.sub0, 0, %2291.sub1, 0, 0   ; %2291 is B0
%2594.sub1 = V_CVT_PK_BF16_F32_e64 0, %2291.sub2, 0, %2291.sub3, 0, 0
%2594.sub2 = V_CVT_PK_BF16_F32_e64 0, %2305.sub0, 0, %2305.sub1, 0, 0         ; %2305 is B1
%2594.sub3 = V_CVT_PK_BF16_F32_e64 0, %2305.sub2, 0, %2305.sub3, 0, 0
; region 8 (dot1 of p2): %2607 = V_MFMA_F32_16X16X32_BF16 %2563, %2594, %2606; the other tile of C also reads %2594
```

Gluon has no concat op, but `gl.join` + `permute` + `reshape` is a concat along dimension 0 (`m1spread` of the Triton 3.7 variant also joins the 4 kr slices this way):

```python
@gluon.jit
def _cat0(a, b):
    # [M, N], [M, N] -> [2M, N]; the joined dimension stays in registers.
    return gl.reshape(gl.permute(gl.join(a, b), [2, 0, 1]), [2 * a.shape[0], a.shape[1]])


@gluon.jit
def _mt(a, b, c, t: gl.constexpr, H: gl.constexpr):
    """One MFMA, tile t (H rows) of a @ b + c, in its own sched region."""
    d = gl.amd.cdna4.mfma(_rows(a, t, H), b, c)
    _sb(True)
    return d


@gluon.jit
def _mfma2(a, b, c, H: gl.constexpr):
    """a @ b + c for a 2-tile piece as two _mt, concatenated again."""
    t0 = _mt(a, b, _rows(c, 0, H), 0, H)
    t1 = _mt(a, b, _rows(c, 1, H), 1, H)
    return gl.convert_layout(_cat0(t0, t1), c.type.layout)
```

- `D2=3` (mode `g_d2c`) is `D2=1` with `_mfma2` for the MFMAs of all 4 B pieces: each piece becomes two single MFMAs, each in its own sched region, and they join again into 32 rows. The loop has 235 instructions for each chunk, VGPR 229 and error (5.43e-04, 1.88e-03), all the same as `D2=1`. The count of each instruction type is also the same, with no `v_mov`. The instruction string of dot2 is the same, and of the 17 regions, only dot1 of p1 to p3 each has one MFMA that moves by one position.
- The dimension that `join` adds stays in registers. The joined tensor uses the original registers of the two tiles, and `convert_layout` makes no instructions.
- The research log first said that Gluon has no concat and that an op cannot be split into single MFMAs. That is not correct (research log 4.14.15).

**Problem two: split a tile in halves.** The 2 `v_mul` after each MFMA in the MIR are half of a 16-row tile. `gl.amd.slice` cannot make this split (it must align with the 16-row CTA tile). Use reshape, permute and split:

```python
@gluon.jit
def _halves(x):
    # A lane holds rows 4a + 0..3 of a 16-row MFMA tile in its 4 registers.  Splitting row bit 1
    # gives two halves of one register pair each (a pk, or two v_mul_f32 in an MFMA shadow);
    # splitting bit 0 would need v_mov to form the pairs.
    y = gl.reshape(x, [x.shape[0] // 4, 2, 2, x.shape[1]])
    return gl.split(gl.permute(y, [0, 2, 3, 1]))


@gluon.jit
def _unhalves(lo, hi, L: gl.constexpr):
    y = gl.permute(gl.join(lo, hi), [0, 3, 1, 2])
    return gl.convert_layout(gl.reshape(y, [4 * y.shape[0], y.shape[3]]), L)
```

- In the result of a 16x16 MFMA, the 4 registers of lane l hold 4 rows of column l mod 16: rows 4a + 0 to 3, with a = l / 16. Bit 0 of the register number is row + 1, and bit 1 is row + 2.
- A split by bit 1 of the row number gives halves that are each one pair of adjacent registers, which is the operand of one `pk`. If the `pk` is in an MFMA shadow, `SIPreEmitPeephole` splits it into 2 `v_mul_f32`, which is the `Mmm` of the MIR. A split by bit 0 gives registers that are not adjacent, and `v_mov`s must make the pairs.
- `_mulh(x, s, h)` uses it to calculate half h of `x * s[:, None]`.

**dot2, `D2=4` (step 9):** the code follows the dot2 of the MIR one instruction at a time, and `D2N` is the N of the MIR. First, all `g·gt` of B and of the first 8 − N A tiles. Then one sched region for each of B_0 to B_7; in the first 2N of these, after the MFMA, half of the `g·gt` of one of the last N A tiles (`_mulh`). At the end, one sched region for each A_k, followed by the bf16 cvt of B_k; each two tiles join into one K step of the next dot1. The code is `_dot2` of the research kernel.

```text
Region 6 (dot2 of p1)
  g_t0 (D2N=0)  ppppppppppppppppppppppppppppppppMMMMMMMMMbbMbbMbbMbbMbbMbbMbbMbbws
  MIR b1n2      wppppppppppppppppppppppppppppppppMMMMMMMMMbbMbbMbbMbbMbbMbbMbbMbbws
  g_t3 (D2N=3)  ppppppppppppppppppppppppppMmmMmmMmmMmmMmmMmmMMMbbMbbMbbMbbMbbMbbMbbMbbws
  MIR b1n2s3    wppppppppppppppppppppppppppMmmMmmMmmMmmMmmMmmMMMbbMbbMbbMbbMbbMbbMbbMbbws
  g_t4 (D2N=4)  ppppppppppppppppppppppppMmmMmmMmmMmmMmmMmmMmmMmmMbbMbbMbbMbbMbbMbbMbbMbbws
  MIR b1n2s4    wppppppppppppppppppppppppMmmMmmMmmMmmMmmMmmMmmMmmMbbMbbMbbMbbMbbMbbMbbMbbws
```

- Each pair differs only in the first `s_waitcnt`: the ROCm compile waits for the LDS reads of mem1 at the start of dot2, and the upstream compile waits at the end of mem1, before the `s_barrier`.
- VGPR: `g_t3` and `g_t4` have 221. `g_t0` has 229, the same as `g_lt` (these MIR steps: 207).

**dot1, `D1=3` (step 1):** the code follows MIR step 1, and the order of the 20 MFMAs does not change (the B chain of tile 0 of C, the B chain of tile 1, the two A chains, big_b, big_a). After each MFMA except the first, 2 VALU follow: the bf16 cvt of one tile of g_a, half a tile of tmp times −β, or its f16 cvt. The code is `_dot1` of the research kernel.

```text
Region 4 (dot1 of p1)
  g_lt        MsMMnbbbMbMMnbbbMbMbbbbnppMmmhhMhphMMbbbMbMnMMInmmppphhhhInI..Iws
  g_d1t       MsMbbMbbMbbMbbMbbMbbMbbMbbMmmMmmMnhhMmmMnmmM..hhMmmImmInmmphhhhnIIws
  MIR cyc03   ffffffMffMbbMbbMbbMbbMbbMbbMnbbMbbMmmMmmMhhMmmMmmMhhMmmImmIhhmmmmhhnIIws
  MIR b1n2    ffffffMffMbbMbbMbbMbbMbbMbbMbbMbbMmmMmmMhhMmmMmmMs..n.hhMmmImmIhhmmmmhhnIIws
```

- The `ffffffMff` at the start of the MIR is v to f32. The ROCm compile puts it at the start of dot1, and the upstream compile in mem2. The `..` after A_6 in `g_d1t` is the β clamp (`BD1=2`), the same as the `s..n.` of MIR step 8 (b1n2).
- Differences from the MIR: A_3 and A_5 each have one more `s_nop 4` after them. Register allocation gives the srcC registers that an MFMA just read to the next VALU. This is the same type of `s_nop` that step 10 removes with the tie (section 7.4). At the end, 1 `pk` is not split (`mmp`, the MIR has `mmmm`).

**Instruction counts:** the full loop (4 chunks). Each mode is the switches of `g_lt` with a different dot1 or dot2. All modes have the error (5.43e-04, 1.88e-03) and no spills.

| Mode | Switches | VGPR | Instructions | `s_nop` |
| --- | --- | --- | --- | --- |
| `g_lt` | `D2=1` | 229 | 933 | 21 |
| `g_t0` | `D2=4` | 229 | 932 | 20 |
| `g_t3` | `D2=4`, `D2N=3` | 221 | 957 | 21 |
| `g_t4` | `D2=4`, `D2N=4` | 221 | 962 | 18 |
| `g_d1t` | `D1=3`, `D2=1` | 229 | 946 | 15 |
| `g_d1t3` | `D1=3`, `D2=4`, `D2N=3` | 221 | 973 | 18 |
| `g_d1t4` | `D1=3`, `D2=4`, `D2N=4` | 225 | 977 | 14 |
| MIR b1n2s4t1 | — | 207 | 948 | 8 |

**Timing and ATT:** T = 16384. 13 modes alternate in one process for 20 rounds. GPU 3 and GPU 2 each run 5 times, in turn, one process at a time. The table gives the mean of the difference against `g_lt` in each run; the brackets give the minimum and the maximum of the 5 runs. The ATT is from GPU 2. X, Y, Z and W are the mean of p1 to p3.

| Mode | Over `g_lt`, GPU 2 (µs) | Over `g_lt`, GPU 3 (µs) | ATT per chunk | X / Y / Z / W |
| --- | --- | --- | --- | --- |
| `g_lt` | 0 (48.47) | 0 (47.38) | 1976 | 447 / 505 / 476 / 445 |
| `g_t0` | +0.08 [−0.17, +0.51] | +0.09 [−0.14, +0.39] | 1975 | 437 / 503 / 471 / 431 |
| `g_t3` | +0.53 [+0.41, +0.61] | +0.36 [−0.25, +0.69] | 2037 | 437 / 503 / 560 / 424 |
| `g_t4` | +1.45 [+1.26, +1.69] | +1.05 [+0.54, +1.20] | 2111 | 441 / 499 / 652 / 409 |
| `g_d1t` | +0.77 [+0.67, +0.88] | +0.65 [+0.37, +1.34] | 2064 | 465 / 556 / 495 / 455 |
| `g_d1t3` | +0.87 [+0.80, +0.94] | +0.65 [+0.09, +0.87] | 2129 | 465 / 581 / 575 / 429 |
| `g_d1t4` | +1.60 [+1.45, +1.98] | +1.49 [+1.05, +2.15] | 2122 | 444 / 531 / 617 / 432 |

- Only dot2 as single MFMAs (`g_t0`, `D2N=0`) is equal to `g_lt`.
- The added cycles of step 9 are all in the Z slot (dot2 of sl0 with mem1 of sl1), and the W slot (mem2 of sl0 with dot2 of sl1) even becomes a little shorter. The same dot2 is slower with mem1 in the same slot and faster with mem2. The cause was not examined. On the MIR, step 9 is 0.2 to 0.3 µs faster, but it comes after steps 1 to 8, and `g_lt` does not have steps 1, 2 and 4 and the second half of step 3.
- The added cycles of step 1 are mostly in the Y slot (dot1 of sl1 with mem1 of sl0). On the MIR, step 1 alone is also 0.54 µs slower (70 more ATT cycles per chunk). It needs step 2, which is not possible in source (section 7.2).
- The two steps together (`g_d1t3`, `g_d1t4`) are not better than each step alone.
- `g_d2c` is 0.31 and 0.42 µs slower than `g_d2`. Its ATT per chunk is 2150, and two captures of `g_d2` gave 2106 and 2144, so the difference is in the variation of `g_d2`.
- On 2026-11-09, the node had a problem in the day, and the first timing failed. The numbers in the table are from after the node recovered, with an eviction count of 0 (section 3.4).

**The two other items of step 1:**

- The `v_xor` that negates β: the dot1 of the upstream compile has none.
- The split of the dot1 `pk` into `v_mul_f32`: `SIPreEmitPeephole` splits a `pk` if it is in an MFMA shadow. This follows the order and has no separate switch. On ROCm Triton 3.7, a split of all `pk` of the kernel (`--src scal`, that is, `AMDGCN_SCALARIZE_PACKED_FOPS=1`) gives VGPR 209 and 49.3 to 49.7 µs, slower than the normal compile.

### 7.2 Step 2: the address adds of mem2

**MIR step 2:** in mem2, the addresses of the three reads of inv, v and β need VALU (`v_add_u32`), and these 3 instructions are between the reads. After step 1 makes dot1 denser, the dot1 in the same slot issues densely when these adds must issue. The adds cannot issue, and the reads after them wait too: the read after the inv address waits 172 cycles (24 in the normal compile). The MIR moves these 3 adds to the start of the region and puts a `SCHED_BARRIER 0` after them.

**Why it is not possible in source:** the lowering of the LDS read (`load_shared_relaxed`, now `smem.load`) makes these `v_add_u32`. The source has no op for them, so they cannot move separately from the reads. `M2=1/2` moves the full reads of inv, v and β before kd, and the adds move with them (`.rr.rr.rr…`), but this is 1.28 / 0.60 µs slower. Together with `D1=2` (`g_m2d1`), it is also 0.53 µs slower. The problem that this step solves comes from step 1 (on the MIR, steps 1 and 2 give a gain only together). Until step 1 gives a result, the value of this step is not known.

### 7.3 Step 4: the `+0` copies

**Symptom:** the loop has 11 `v_add_u32 v, 0, v`, all in p0: 9 in mem1 of p0 (the 8 TR read addresses of kr, and the base that gt and β share) and 2 in mem2 of p0 (the read addresses of inv and v). In the ATT of the ROCm normal compile, in the mem1 of p0 of the wave group that runs together with dot2, each of these 9 takes about 12 cycles, and the first LDS read issues at about 144 cycles. With the 9 removed in the MIR, this is about 12 cycles. The upstream compile has the same:

```text
Region 1 (mem1 of p0)
  g_lt      .....rrrr.rrrr.rrrr.rrrr..rrrrrrrrssgsssgsss.gsssbgbbbbbbbws
  MIR       .rrrrssssssgrrrrssssgrrrrssssgrrrrssssgrrrrrrrrbbbbbbbbs
```

The `.....` at the start and the `.` between each 4 reads are these 9, and mem2 of p0 has 2 more. An example from the ROCm Triton 3.7 compile, the first TR read of kr:

```text
; LLIR, outside the loop: LICM has moved the address out of the loop
%296 = getelementptr inbounds nuw i8,
         ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 32768), i32 %295
; LLIR, in the loop
%818 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) %296)

; MIR before the machine scheduler, loop block
%1649 = V_ADD_U32_e32 0, %1375
undef %1776.sub0_sub1 = DS_READ_B64_TR_B16 %1649, -32768, 0 :: (load (s64) from %ir.sunkaddr281)

; ISA
v_add_u32_e32 v134, 0, v153
ds_read_b64_tr_b16 v[198:199], v134 offset:32768
```

The `-32768` in the MIR and the `offset:32768` in the ISA are the same 16-bit immediate. The MIR prints it as a signed number.

**Cause:**

1. Triton declares LDS as an external global of size zero (dynamic LDS): `@global_smem = external addrspace(3) global [0 x i8]`. Its address is 0, the start of LDS, but in LLVM IR it is not a constant. Thus the folds of `add x, 0` in InstSimplify and in the DAGCombiner do not see this 0.
2. CodeGenPrepare sinks the address calculation `@global_smem + %x`, which LICM moved out, back into the loop (`sunkaddr`). AMDGPU `isLegalAddressingMode` accepts "register + register + immediate" for LDS, so the constant part goes into the offset of `ds_read`, and the add that stays is in the loop. In one basic block, CodeGenPrepare sinks the same address only once. The 4 unrolled chunks are in one basic block, so all these adds are at the first use, that is, in p0.
3. Instruction selection changes the address of `@global_smem` into the pseudo-instruction `GET_GROUPSTATICSIZE`, and `finalize-isel` replaces it with `S_MOV_B32 0`. Only at this point is the value 0 known.
4. MachineLICM does not move this add out of the loop: the register pressure is too high ("Can't remat / high reg-pressure").
5. `si-fold-operands` folds the immediate 0 into the add, which gives `V_ADD_U32_e64 0, %x`. Its `tryConstantFoldOp` removes or, and, xor with 0, and the scalar `S_ADD`/`S_SUB` of 0 ([llvm/llvm-project#198410](https://github.com/llvm/llvm-project/pull/198410), merged 2026-05). It does not handle the VALU `V_ADD_U32 0, x`. The K2 addresses are different for each lane, so they use the VALU form. The codegen library 08b05a33 that upstream Triton uses now, and the next pin 293a85c0, both have this behavior.
6. `si-shrink-instructions` shrinks it to `V_ADD_U32_e32 0, %x`, which stays until the ISA.

**Fix:** [llvm/llvm-project#163298](https://github.com/llvm/llvm-project/pull/163298) ("[AMDGPU] Fold dst = v_add 0, src -> src") was submitted on 2025-10-14 and approved on 2026-10-21 (it waits for a rebase). On 2026-11-09, it was not merged. After the merge, Triton must also update the LLVM of the AMD codegen to a version that has it.

**Estimate of the gain** (all on the ROCm Triton 3.7 compile):

- An `llc` simulation of this PR (replace `V_ADD_U32 0, x` with x after `si-fold-operands`): all `+0` copies in the loop disappear. The loop goes from 974 to 958 instructions, and VGPR 225 → 217.
- Removal in the MIR only (`--algo none --fold-zero-add` of the MIR edit script): VGPR 225 → 215. All 12 alternating timings are faster: GPU 2 50.74 → 50.44 µs, GPU 3 49.45 → 49.04 µs, about 0.3 to 0.4 µs faster.
- Removal on top of MIR steps 1 to 3 (step 4): cumulative −1.17 to −1.88 µs (the table of section 1.2).

**All workarounds in source are worse:**

| Method | Result |
| --- | --- |
| CodeGenPrepare off (`llvm_cl_opts="disable-cgp"`, `g_cgp`) | The `+0` copies in the loop disappear, but VGPR 256, 90 spills, 195 µs |
| LDS base as a global with a size (LLIR change) | All 11 disappear, but all addresses move out of the loop: VGPR 255, 30 spills |

### 7.4 Step 10: MFMA tie

**MIR step 10:** the 16 bf16 MFMAs of dot1 are 4 accumulation chains of 4 links. Each link reads the result of the link before it as srcC.

- Without a tie, the result of each link goes into a new group of registers, and register allocation gives the srcC group that was just read to the next cvt.
- But the link before just wrote this group, and this link just read it. A VALU that writes it must be 8 instructions after the link before (WAW) and 3 after this link (WAR). There are only 3 instructions between them, so the compiler adds an `s_nop 4`, and the MFMA after it issues about 16 cycles later.
- Step 10 renames the result of each link in the MIR to the virtual register of its srcC (D = C, accumulation in place). Then register allocation must give the two the same registers, and this type of `s_nop` disappears.

**Why it is not possible in source:**

- A 16x16x32 MFMA has only the opcode with D and C not tied (`V_MFMA_…_vgprcd_e64`). The `_mac` form, with D tied to C, is defined only for MFMAs whose D is more than 4 VGPRs (for example 32x32).
- `mfma(..., cd_regclass=...)` of Gluon can only select the register class of the result. It cannot make the result share registers with srcC.
- The inputs of `inline_asm_elementwise` must broadcast to the same shape and the same layout, so it cannot express an MFMA, whose A, B and C have different shapes.
- An MFMA option "write the result to the accumulator" is necessary.

### 7.5 The gap that stays

`g_lt` has 315 more cycles for each chunk than MIR b1n2s4t1 (ATT, GPU 2):

- **p0, about 147:** the 9 `+0` copies of mem1 of p0 and the 2 of mem2 of p0 (section 7.3); the 9 LSR SALU at the start of dot1 of p0, and the 4 SALU of the back edge (section 6.1).
- **p1 to p3, about 169:** X, Y, Z and W each have 50, 66, 59 and 49 more for each position. The two most typical slots:
  - **Y_1 (mem1 of p1 of sl0 with dot1 of p1 of sl1):** sl0 comes to the barrier 66 cycles after sl1. The 16 kr reads take about 190 cycles (about one each 12 cycles, with idle issue cycles of the wave between two reads), the SALU of the copy bases about 30, the 8 gt reads about 110, the 4 copy groups about 100 (4 to 6 SALU each, with the `m0` write), and the 8 cvt about 40. With `M1=3`, the copies come after the reads, so the copy SALU cannot use the idle cycles between the reads. The "4 reads, 1 copy" of MIR step 3 uses these idle cycles (section 5.4).
  - **X_2 (dot1 of p2 of sl0 with mem2 of p2 of sl1):** sl0 comes to the barrier 104 cycles after sl1. The end of dot1 is a dependency chain: the last bf16 MFMA issues at cycle 312, then `s_nop 4`, the multiply by −β at 348 to 372, the cvt to f16 at 376 to 396, and the 4 f16 MFMAs only at 400 to 448. The reorder of MIR step 1 and the removal of the register-allocation `s_nop` in step 10 make this part shorter (sections 7.1 and 7.4).

## 8. References

- Triton:
  - 0df169311f: the upstream main that the aiter kernel is tested on (2026-11-12).
  - f8cddb947b: the upstream main of the measurements of sections 2 to 6. 173580104f: the main of 2026-11-08, for section 7.1 and the repeat runs.
  - [triton-lang/triton#11932](https://github.com/triton-lang/triton/pull/11932): `gl.amd.hint.sched_barrier` (2026-09-24).
  - [triton-lang/triton#12345](https://github.com/triton-lang/triton/pull/12345): buffer descriptors and the descriptor-based buffer ops (2026-10-25).
  - [triton-lang/triton#12478](https://github.com/triton-lang/triton/pull/12478): `load_shared_relaxed` deprecated (2026-10-26).
  - [triton-lang/triton#12527](https://github.com/triton-lang/triton/pull/12527): the old buffer ops deprecated (2026-10-27).
  - [triton-lang/triton#12496](https://github.com/triton-lang/triton/pull/12496): the AMD codegen LLVM updated to 08b05a33 (2026-10-27).
  - [triton-lang/triton#12656](https://github.com/triton-lang/triton/pull/12656): the pin for a source build of the AMD codegen LLVM changed to 293a85c0. The prebuilt library (`cmake/amd-llvm-info.json`) is still 08b05a33.
  - [triton-lang/triton#12664](https://github.com/triton-lang/triton/pull/12664): a narrowing in `WarpSpecializeUtility.cpp`. A build with GCC 13 needs `-Wno-error=narrowing`.
  - `python/triton/experimental/gluon/language/amd/`: `hint.sched_barrier`, `slice`, `cdna4.mfma` (`cd_regclass`), `cdna4.buffer`.
- LLVM:
  - [llvm/llvm-project#163298](https://github.com/llvm/llvm-project/pull/163298): [AMDGPU] Fold dst = v_add 0, src -> src (not merged).
  - [llvm/llvm-project#198410](https://github.com/llvm/llvm-project/pull/198410): `SIFoldOperands` folds the scalar `S_ADD`/`S_SUB` of 0 (merged).
  - `llvm/lib/Target/AMDGPU/SIFoldOperands.cpp`: `tryConstantFoldOp`.
  - `llvm/lib/Target/AMDGPU/SIISelLowering.cpp`: `LowerGlobalAddress`, `GET_GROUPSTATICSIZE`, `isLegalAddressingMode`.
  - `llvm/lib/Target/AMDGPU/GCNHazardRecognizer.cpp`: `getCountedWaitStates`, `checkMAIVALUHazards`, and the dst forwarding rule of inline asm.
  - `llvm/lib/CodeGen/ScheduleDAGInstrs.cpp`: the barrier chain in `buildSchedGraph`.
- The research kernels, the MIR edit scripts and the full experiment log are not in this repository. The log is `k2_ideal_schedule_budget.md`, section 4.14: 4.14.12 is the source form on ROCm Triton 3.7, 4.14.13 the MIR step-by-step comparison, 4.14.14 the per-piece forms on upstream Triton, and 4.14.15 the one-MFMA forms and the repeat runs on the latest main.
