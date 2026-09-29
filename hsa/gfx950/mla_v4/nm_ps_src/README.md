<!--
SPDX-License-Identifier: MIT
Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
-->
# mla_a8w8_qh64_qseqlen1_gqaratio64_nm_ps: generator sources

This directory regenerates the AITER code object
`hsa/gfx950/mla_v4/mla_a8w8_qh64_qseqlen1_gqaratio64_nm_ps.co` (kernel symbol
`_ZN5aiter39mla_a8w8_qh64_qseqlen1_gqaratio64_nm_psE`) from sources. It is the persistent
DeepSeek-V4 HCA decode kernel (fp8 KV "nm" layout, 128 query heads, qseqlen 1) for gfx950.

## Provenance

* **Base.** The kernel is derived from the AITER #5195 ASM kernel
  `hsa/gfx950/mla_v4/mla_a8w8_qh64_qseqlen1_gqaratio64_nm.co`
  (`_ZN5aiter36mla_a8w8_qh64_qseqlen1_gqaratio64_nmE`). `asm/rt.s` is a disassembly of that code object, made
  re-assemblable (llvm-objdump output plus the original kernel descriptor bytes and metadata). Assembled, it gives
  the same `.text` and kernel descriptor (`.rodata`) as the #5195 `.co` (`gen.py --verify-rt` checks this). `rt.s`
  is kept as generated: `asm/patch.py` finds some of its patch sites by line number, so the file carries no license
  header, and `gen.py` checks its sha256 before it uses the file.
* **Persistent descriptor loop.** `asm/patch.py` rewrites the disassembly with scripts only; nobody edits the
  output by hand. The grid becomes `(2P, 1, 1)`. Each workgroup owns one partition `p` and one head group `hg`,
  and walks the rows `begin_row..end_row` of its descriptor. It runs one tile range `[tb, te)` of a row per pass,
  re-entering the ASM's own prologue each time. The patches retarget the ASM's split-partial epilogue to the slot
  layout of `o_acc` / `lse_acc`, apply the sink once, write the natural-log LSE of unsplit rows, and relax some
  waits. Two never-executed `s_nop` pads keep the hot loop code at its original byte offset mod 256
  (`asm/align.py`). The docstring of `asm/patch.py` lists every patch.
* **In-kernel split planner.** `splice/pre.hip` holds the planner, written in HIP. It runs the closed-form cost-line
  planner, a 64-way wave-cooperative boundary search, and an automatic split or no-split choice. The source is
  compiled with `hipcc -S`, and `splice/splice.py` extracts the kernel body from that assembly. `asm/patch.py`
  then splices the body into the kernel entry, so no separate planner launch is needed. Each workgroup publishes
  its descriptor to `desc[p]`.
* **Fused last-arriver combine.** `splice/fin.hip` is also written in HIP and spliced the same way, at the end of
  every split piece. Each piece counts its tiles into an agent-scope atomic counter (`cnt`). The piece that
  completes a row merges all of that row's partials, adds the sink, and writes `out` / `lse_out`. Wide rows are
  merged in two levels. The partial stores use sc1 (agent-scope write-through), so no separate combine launch is
  needed, and the counters return to zero at the end of every completed call.

## Layout

| path | role |
|---|---|
| `gen.py` | the only build entry point (python3 stdlib only) |
| `asm/rt.s` | re-assemblable disassembly of the #5195 ASM (input, not edited) |
| `asm/patch.py` | rt.s -> persistent kernel (library, called by gen.py) |
| `asm/align.py` | alignment pads (assembles trial passes, reads label offsets with llvm-readelf) |
| `splice/splice.py` | `hipcc -S` + extraction of one kernel body as a splice block |
| `splice/pre.hip` | in-kernel split planner (`ps_plan`) |
| `splice/fin.hip` | fused last-arriver combine (`ps_merge`) |
| `splice/hca_common.hip` | shared definitions: cost line, `Desc`, wave-cooperative search, WG map |

## Build and verify

Run inside a ROCm container, for example with the toolchain listed below:

```bash
python3 gen.py --out /tmp/x.co \
    --check <aiter>/hsa/gfx950/mla_v4/mla_a8w8_qh64_qseqlen1_gqaratio64_nm_ps.co
# optional: keep the intermediates / check the rt.s provenance
python3 gen.py --out /tmp/x.co --keep /tmp/nm_ps_build \
    --verify-rt <aiter>/hsa/gfx950/mla_v4/mla_a8w8_qh64_qseqlen1_gqaratio64_nm.co
```

`gen.py` takes these steps:

1. Check the sha256 of `asm/rt.s`.
2. Compile `splice/pre.hip` and `splice/fin.hip` with
   `hipcc --offload-arch=gfx950 -O3 --offload-device-only -std=c++20 -w -S`.
3. Run `asm/patch.py`, including the alignment passes.
4. Rename the kernel symbol from `_ZN5aiter36..._nmE` to `_ZN5aiter39..._nm_psE`. This covers every occurrence:
   the function symbol, `.kd`, and the metadata `.name` / `.symbol`. The Itanium length prefix is recomputed
   (36 -> 39).
5. Assemble with `clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -c`.
6. Link with `ld.lld -shared`.
7. With `--check`, compare the sha256 against the given file. On a mismatch it exits with status 1.

Tools are taken from `$ROCM_PATH` (default `/opt/rocm`): `llvm/bin/{clang,ld.lld,llvm-readelf,llvm-objcopy}` and
`bin/hipcc`. The output is byte-identical only with the same toolchain. `hipcc` codegen and `clang` / `lld` object
layout can change between ROCm releases.

Toolchain used for the shipped `.co` (container image `rocm/atom-dev:nightly_202609161445`, `/opt/rocm` ->
`/opt/rocm-7.2.4`):

```
$ /opt/rocm/llvm/bin/clang --version
AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.4 26084 f58b06dce1f9c15707c5f808fd002e18c2accf7e)
Target: x86_64-unknown-linux-gnu
Thread model: posix
InstalledDir: /opt/rocm-7.2.4/lib/llvm/bin
$ hipcc --version
HIP version: 7.2.53211-97f5574fe2
AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.4 26084 f58b06dce1f9c15707c5f808fd002e18c2accf7e)
$ /opt/rocm/llvm/bin/ld.lld --version
AMD LLD 22.0.0 (... f58b06dce1f9c15707c5f808fd002e18c2accf7e) (compatible with GNU linkers)
```

Expected sha256 of the output:
`889d8c653c3278f7c0df138476b1ca0d491b7fced25617f816d6818a68392af8`.

## Launch and kernarg ABI

* Grid `(2P, 1, 1)` and block `256`, with no dynamic LDS. `P` is the number of partitions (`P <= 1024`, and
  `P % 8 == 0` selects the XCD-aware WG map). Rows: `N <= 32768`; the byte offsets
  of `out` / `q` wrap above that (inherited from the #5195 ASM). The `cnt` / `arange` sizes below come from a
  65536-row counter layout and are larger than that limit needs.
* The kernarg block is 336 B: 21 slots of 16 B. Offsets are in bytes. Slots the kernel does not read are passed as
  the values shown.

| offset | type | content |
|---|---|---|
| 0x00 | ptr | `o_acc` fp32 `[2P][128][512]`: split-piece partial O (workspace) |
| 0x08 | ptr | `out` bf16 `[N][128][512]` |
| 0x10 | ptr | `lse_acc` fp32 `[2P][128]`: split-piece partial LSE, log2 domain, no sink (workspace) |
| 0x18 | ptr | `lse_out` fp32 `[N][128]`, natural log, sink included; 0 = not written |
| 0x20 | ptr | `q_packed` (Q NoPE, fp8 packed) |
| 0x30 | ptr | `kv_packed` (KV NoPE fp8 + scales, 512 B per token) |
| 0x40 | ptr | `kv_indptr` int32 `[N+1]` |
| 0x50 | ptr | `kv_page_indices` int32 |
| 0x60 | ptr | `desc` int32 `[P][8]`: per-partition descriptor, written by the kernel (workspace) |
| 0x68 | ptr | 0 (unused) |
| 0x70 | f32 | softmax scale `1/sqrt(512)` |
| 0x80 | u32 | 128 (gqa ratio) |
| 0x90 | u32 | 1 (num_kv_splits) |
| 0x98 | ptr | `cnt` int32 `[2*65536 + 16*512 + 4*1024]`, zero at rest: `[0, 2*65536)` row counters, `[2*65536, +16*512)` reserved (unused, but allocated), then `4*1024` group counters of the fused combine |
| 0xa0 | u32 | `P` |
| 0xa4 | u32 | `N` (rows) |
| 0xb0 | u32 | planner config `F | MT << 8` (default `6 | 1 << 8`) |
| 0xb8 | ptr | 0 (unused) |
| 0xc0 | u32 | 0 |
| 0xd0 | ptr | `qo_indptr` = int32 `arange(>= N+1)` |
| 0xe0 | ptr | `split_indptr` = int32 `arange(>= N+1)` (same buffer) |
| 0xf0 | u32 | 1 (bf16 output) |
| 0x100 | ptr | `q_rope` (bf16) |
| 0x110 | ptr | `kv_rope` (bf16) |
| 0x120 | ptr | `sink` fp32 `[128]` |
| 0x130 | ptr | 0 (valid_split) |
| 0x140 | u32 | 0 (use_valid_split) |

Workspace rules:
* One workspace (`o_acc`, `lse_acc`, `desc`, `cnt`) can serve any number of calls that are ordered with respect to
  each other.
* Calls that can run concurrently need separate workspaces.
* Rows with `K = 0` are not written.
* A kernel that aborts can leave `cnt` non-zero. Zero it again before the next call.
