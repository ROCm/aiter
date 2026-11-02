# mHC projection experiments — GPU 2

These are isolated mHC experiments, not an accepted full-prefix replacement. All runs used physical GPU 2, WMMA and TDM tiled loads. Only second-round medians are reported. Existing full-layer numerical sensitivity requires the parent full-prefix validation before promotion.

## Updated matched A/B: exact-order RMS plus TDM residual/AN tail

Latest experiment copies the parent corrected tail: TDM stages full residual and AN early; residual RMS preserves original 16 sequential 1024-element reductions. All other geometry unchanged. Same 256 calls/graph,40 alternating samples,round2. This is the relevant final comparison:

| Variant | Reused weights us | 256 rotating weights us |
|---|---:|---:|
| Original projection | 9.062 | 9.490 |
| Whole-weight L2 hints | 8.818 | 9.543 |
| Producer-local L2 hints | 8.954 | 9.258 |
| Offline hi/lo BF16 packed | 8.993 | 9.333 |
| Packed + whole-weight L2 hints | 8.902 | 9.409 |

Producer-local L2 prefetch gives a small consistent win (1.2% reused,2.4% rotating). Whole-weight distributed hints hurt rotating weights; packed weights give no compelling benefit for extra layout complexity. Original projection geometry retained.

Updated64-case/960-output validation passes, maxrelativeRMS1.023e-7 (Sinkhorn reciprocal precision); parent owns full-prefix validation. Ordered final variants have zero spills. See ordered-timings.json, ordered-validation.json, ordered-*.amdgcn. Run ordered_bench.py then check_ordered.py; ordered.py and ordered_tail_snapshot.py freeze the corrected implementation.

The local hint maps 32 active lane addresses onto each producer's16rows x2 cachelines (256B perline), covering its8KiB tile; tail Nrows beyond24 and unusedlanes addressvalidW[0]. All tiled numerical loads still useTDM. Fullydistributedhint is issuedbyfirst48CTAs,128lines each,covering6144lines=1.5MiB. Each is a device-scope non-speculative hint; no output depends on completion.

## Findings

- Geometry: original BK128, BN16, 256 CTAs remains best for fixed 256-CTA grid. Raising BK reduces producer count but increases per-CTA work and register/LDS footprint; idle CTAs still pay the fixed grid barrier. BN32/BK128 with a 128-CTA standalone grid was modestly faster but does not improve the 256-CTA prefix.
- Original four-barrier tail: 14.776 us; best reduced standalone grid (BK256 BN32 64 CTAs) 14.328 us. Fixed-grid versions regressed.
- Producer square-sum (AITER approach) removes 24 complete residual scans: 11.475 us with original four-barrier tail. Superseded by the tail agent's reduced-barrier approach. Different RMS reduction order is NOT approved for full-prefix use.
- With the tail-agent one-barrier snapshot: original projection 6.872 us; BN32/BK128 fixed256 7.077 us; standalone128 6.616 us. Larger K tiles slower.
- AITER-inspired offline hi/lo BF16 weight packing preserves the exact two-term WMMA arithmetic and total 1.5 MiB weight bytes. Requires 48x16384 BF16 layout, first24 rows hi then24 lo. Standalone sweep suggested improvement, but final rotating comparison is small and variable.
- Explicit L2 prefetch tested both distributed whole-weight coverage and per-producer tile hints. Whole-weight prefetch helps reused-weight measurement but slightly hurts rotating copies. No consistent general win demonstrated.

## Final matched A/B with reciprocal tail snapshot

128 split K, BN16, 256 CTAs. One launch, one grid barrier. 256 calls per captured graph; 40 alternating-order samples per round. Rotating case uses 256 physically distinct 1.5 MiB weight copies (384 MiB); weights identical in value. Packed variants use equivalent separate copies. This tail snapshot is not the parent's later exact-order fix.

| Variant | Reused weights us | 256 rotating weights us |
|---|---:|---:|
| Original projection | 6.398 | 6.698 |
| Whole-weight L2 hints | 6.118 | 6.785 |
| Producer-local L2 hints | 6.144 | 6.560 |
| Offline hi/lo BF16 packed | 6.292 | 6.535 |
| Packed + whole-weight L2 hints | 6.221 | 6.593 |

## Validation and ISA

64 input cases (9 checkpoint embeddings plus55 varied-scale random residuals), 5 variants, 3 outputs checked =960 comparisons. mHC mix, raw stream mix and normalized X all pass relative RMS<0.001, max4.15e-6. This does not prove exact downstream FP8 quantization: parent observed rare threshold crossings from RMS summation-order differences and is correcting the tail. No standalone results should override full-prefix validation.

Final variants: zero spills; original uses274 VGPR,13008 bytes LDS. Saved final-*.amdgcn verifies tensor_load_to_lds, WMMA, and explicit device-scope global_prefetch_b8 when enabled. Baseline also has a compiler-generated scope:SE prefetch; distinguish it from experimental weight prefetch.

## Reproduce

Container vigokhal_dsv4_tp4, HIP_VISIBLE_DEVICES=2; directory /work/mhc-opt-projection. Run final_bench.py and check_final.py. Other sweeps: tune_projection.py, tune_sq.py, tune_tail.py, tune_packed.py. Raw JSON and logs retained. projection.py implements all variants; tail_snapshot.py and rcp_snapshot.py freeze tail-agent source used for each experiment. Final timed source is final_region.

Studied AITER gfx1250 fusions/mhc.py: WMMA hi/lo BF16 precision, prepacked weight option, staged TDM, per-producer projection plus square-sum. Main compatible finding is that reducing split count is not beneficial under the 256-CTA prefix topology.
