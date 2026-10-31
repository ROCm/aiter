# Round TDM2: padded bf16 K gather (CPU/ISA gated, opt-in)

Base: `45c6c26c113ede7dd0f981974498d663b5825825`. No GPU was exposed to any verification container. Default `QSA_BF16_K_LOAD=raw` is unchanged; no GPU correctness, latency or roofline result is claimed.

## Layout and exact stock API units

The node's `rocm/fw-bringup-mlperf:gfx1250-atom-20261015` image contains FlyDSL **0.3.3.dev886**, at `/usr/local/lib/python3.12/dist-packages/flydsl/expr/rocdl/tdm_ops.py`.
Its `compute_padding_encoding(interval_elems, amount_elems, elem_bits)` converts elements to DWORDs, then encodes `log2(interval_dw)-1` and `amount_dw-1`. `make_tensor_gather_descriptor` also takes **elements**, not encoded fields or DWORD counts.

For bf16, the public arguments are **128 / 8 elements** => **64 / 4 DWORDs** => encoded **5 / 3**, `pad_enable=1`. Group1 config is `0x07510000`, including bf16 `data_size=1` and no multicast. The source row/stride remains 128 bf16 = 256 B. LDS pitch is **272 B**, with 16 B skipped after every 256 B payload row. Tile allocation is **64 * 272 = 17,408 B**, already 128 B aligned. Each eight-row descriptor starts at `local_row * 272`, itself 128 B aligned. Consumer lo/hi vectors remain 16 B at `+0 / +32`; neither can read a padding byte.

Kernel-wiki `mi400-tdm-programming` and `mi400-tdm-sync` are **source-reported**, not timing evidence. They specify skipped LDS padding, both encoded padding fields nonzero, tile_dim0 divisible by 4 B, a 256 B padding interval compliant with the 128 B granularity rule, hardware XACK throttling at 3/wave (6/SIMD), and a separate 63-op completion counter. XACK is not LDS visibility.

The logical 64-bank/4-B address model changes each 16-lane half-wave vector from four banks with 16 accesses each to all 64 banks once (twice per full wave). This proves the address-bank distribution, not a measured replay rate or speedup.

## Supported paths and synchronization

* **r32:** unchanged full-block/single-valid-request guard, one shared padded tile. Four waves issue two 8-row gathers each. Every producer executes tensor_wait(0), then all waves rendezvous before cross-wave consumption. Mixed/partial r32 takes raw K loads.
* **r16:** full 16-row blocks use four disjoint wave-private padded 64-row tiles, **69,632 B total**, allocated by one SharedAllocator. Each scalar-active request pass issues eight 8-index gathers. Invalid/duplicate/invisible passes skip issue using explicit SGPR metadata. All waves execute tensor_wait(0), the pre-consumption workgroup barrier, and a post-consumption barrier before the next of four request passes can overwrite a tile. Partial workgroups take the raw path without reaching these barriers.
* **Fallbacks:** cache bases not 128 B aligned select the actual raw r16 builder specialization; merely selecting r16 geometry would now be insufficient. Copy-atom access and fp8 stay raw. Host API, i64 plane extents, metadata caps, clamped/predicated stores, invalid-page-to-page-zero semantics and visible masks remain.
* Explicit `rocdl.wave_id()` and scalar `_rows_meta_uniform` avoid LLVM's initial VGPR-to-SGPR readfirstlane scalarization in the r16 prologue. The CPU AST gate proves the uniform helper's complete body identical to the raw metadata helper after substituting only the load primitive.

## Real ISA gate

With COMPILE_ONLY=1, FLYDSL_DUMP_IR=1 and FLYDSL_DEBUG_DUMP_ASM=1, the stock compiler produced final gfx1250 assembly, not only a tracing success:

| Static ISA marker | r32 TDM | r16 TDM |
|---|---:|---:|
| tensor_load_to_lds | 2 | 32 (8 x 4 unrolled passes) |
| s_wait_tensorcnt | 2 | 32 |
| explicit wait(0) | 1 | 4 |
| ds_load_b128 | 32 | 128 |
| workgroup barrier signal/wait pairs | 1 | 8 |
| LDS metadata bytes | 17,408 | 69,632 |
| v_readfirstlane_b32 / v_max_i64 / ds_load_b128_addtid | 0 / 0 / 0 | 0 / 0 / 0 |

Consumer ISA contains the **0x110 = 272 B** pitch, and both paths contain Group1 **0x07510000**. Additional compiler-inserted wait(0xa) instructions protect descriptor lifetimes; they do not replace explicit wait(0).

`check_tdm_isa_cpu.py` validates exact LDS sizes, padding config, padded pitch, instruction counts, completion waits, barrier pairs and absence of the prohibited markers. Final full dumps/logs and source SHA256 are in the unique node staging result directory reported by `host_compile_tdm2_cpu.sh`.

## CPU gates and reproduction

All verification uses public golden inputs only. No sealed files are accessed.

```bash
export QSA_GOLDEN_DIR=/path/to/public/golden_baseline
export QSA_BF16_K_LOAD=raw
G2=aiter/ops/flydsl/kernels/mqa_logits/_g2
python -B "$G2/test_contract_cpu.py"
python -B "$G2/audit_addressing_cpu.py"
python -B "$G2/test_q_lds_cpu.py"
QSA_TRACE_ROWS_PER_BLOCK=16 python -B "$G2/trace_check_cpu.py"
QSA_TRACE_ROWS_PER_BLOCK=32 python -B "$G2/trace_check_cpu.py"
python -B "$G2/test_k_tdm_cpu.py" -v
python -B "$G2/check_tdm_isa_cpu.py" --r16 r16.s --r32 r32.s
```

The expanded K suite has **28 passing tests**: every uint16 payload encoding (including NaN/subnormal bits), independent raw WMMA fragment reconstruction, skipped-padding sentinels/canaries, 1,024 aligned 16-B vector reads per tile, bounds/coverage, arbitrary producer orders, invalid page/request/width and visible/column tails, all four wave-private tiles/pass reuse, optimized scalar issue guards and synchronization/source bindings. Raw trace16 and trace32 each pass 16 fixtures; contract, address and Q-LDS gates pass.

`host_compile_tdm2_cpu.sh <absolute-source-archive>` makes fresh source/result directories, mounts source read-only, exposes no GPU devices, pins the exact image/version, compiles both raw and TDM modes and runs all CPU/ISA gates. Exclude __pycache__ from the archive; include the kernel/_g2 tree and the two public golden Python modules.

## Remaining GPU gate / risks

No percentage gain can be inferred from byte/ISA correctness. The old unpadded 233 us and 925 us paths need respectively **21.5%** and **28.6%** latency reductions merely to tie the 183 us and 660 us champion measurements; these are break-even calculations, not predictions.

The main new risk is r16's 69,632 B LDS footprint/occupancy and four fixed barrier rounds even when request dedup skips issue. SGPR pressure, scalar metadata overhead, direct-copy in-flight limits and actual TDM return/cache behavior may negate the bank-layout improvement. Sub-16-row calls still fall back to raw. No double buffering or fp8 extension was attempted.

Before any default switch: acquire the cooperative GPU lock; run a small full-r16 and full-r32 first-touch correctness ladder, then all contract fixtures (including mixed requests, partial rows, holes, invalid metadata, visibility/column tails and 16B-but-not-128B cache alignment). Only after these pass, run unprofiled A/A and raw-vs-TDM A/B plus held-out reruns on decode/draft and the prefill512/2880-row cases. Check actual dispatch names/LDS metadata; compare achieved read bandwidth against the measured 15.7 TB/s gate. If timing regresses, retain raw and profile under the existing profiling runbook rather than infer the cause from this CPU model.
