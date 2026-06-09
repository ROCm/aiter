# SonicMoE ROCprof Dataflow Plan

This note captures the first MI355X/gfx950 profiling pass for the
SonicMoE-compatible AITER wrapper. The current implementation still calls
AITER's existing two-stage `fused_moe` path; SonicMoE's original CUDA/QuACK
fast-path dataflow has not been ported.

## Saved Data

Host path:

```text
/shared/amdgpu/home/hyi_qle/yifehuan_temp/data/sonic_moe_mi355_latest/rocprof/rocprof_stages_20260609_152858_2a0cd1d53
```

Container path:

```text
/app/yifehuan_temp/data/sonic_moe_mi355_latest/rocprof/rocprof_stages_20260609_152858_2a0cd1d53
```

Important files:

- `raw/`: one rocprofv3 run per shape, stage, and counter.
- `run_status.csv`: command success matrix; all 48 profile runs completed.
- `commands.log`: exact rocprofv3 commands.
- `cases.json`: shapes, counters, commit, and driver metadata.
- `rocprof_stage_summary.csv`: normalized counter summary.
- `rocprof_stage_summary.md`: human-readable summary table and first-pass read.

Omniperf was not available in the container, so this pass uses rocprofv3 only.
Counters were collected one at a time to avoid hardware counter grouping limits.

## Current AITER Dataflow

Python call chain:

```text
aiter.fused_moe.fused_moe_
  -> moe_sorting
  -> fused_moe_2stages
      -> allocate A2 as [token, topk, inter_dim]
      -> metadata.stage1(...)
      -> optional A2 quantization/repacking
      -> metadata.stage2(...)
```

The CK path reaches:

```text
csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.cu
  -> ck_moe_stage1
  -> ck_moe_stage2
  -> ck_moe_stage1_gemm / ck_moe_stage2_gemm
  -> ck::tensor_operation::device::DeviceMoeGemm
```

Stage1 receives original `hidden_states`, `w1`, `sorted_token_ids`, and
`sorted_expert_ids`, then writes the intermediate activation A2. Stage2 receives
A2, `w2`, the same sorted metadata, and routed weights, then accumulates into
the final `[token, model_dim]` output.

For the measured shapes, logical A2 storage alone is:

| Shape | A2 MiB | A2 read+write MiB |
| --- | ---: | ---: |
| `32768,4096,512,128,8` | 256 | 512 |
| `32768,4096,1024,128,8` | 512 | 1024 |
| `32768,4096,1024,256,8` | 512 | 1024 |

That traffic is on top of normal input, weight, scale, routing, and output
traffic. The A2 allocation is also a hard stage boundary: stage1 must fully
materialize A2 before stage2 can consume it.

## ROCprof Read

The best three sweep rows were profiled with balanced routing and `block_m=128`.

| Shape | Stage | Bench ms | MFMA % | Mem stalled % | R+W MiB | Approx HBM GB/s | VGPR | LDS bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `32768,4096,512,128,8` | stage1 | 2.3563 | 40.27 | 0.01 | 2347.2 | 1044.5 | 196 | 32768 |
| `32768,4096,512,128,8` | stage2 | 3.2487 | 14.45 | 0.84 | 5385.8 | 1738.4 | 104 | 32768 |
| `32768,4096,1024,128,8` | stage1 | 4.6667 | 40.97 | 0.02 | 7365.6 | 1655.0 | 196 | 32768 |
| `32768,4096,1024,128,8` | stage2 | 3.7945 | 25.70 | 0.72 | 6668.1 | 1842.7 | 104 | 32768 |
| `32768,4096,1024,256,8` | stage1 | 4.7312 | 41.58 | 0.01 | 5742.4 | 1272.7 | 196 | 32768 |
| `32768,4096,1024,256,8` | stage2 | 4.0287 | 24.18 | 0.81 | 7200.7 | 1874.2 | 104 | 32768 |

Interpretation:

- Sorting and token padding are not the primary limiter on these rows. Balanced
  routing has tile efficiency close to 1.0, but end-to-end improvement over
  top-k routing is only about 3-6%.
- Stage1 has better MFMA utilization than stage2, but at about 40-42% it still
  has room to improve.
- Stage2 is more exposed to data movement: lower MFMA utilization, larger
  measured read/write volume, and higher memory-stall counters.
- LDS bank conflict is low, so the next target is not LDS banking; it is the
  global-memory dataflow around A2 and sorted routed rows.

## Prototype Direction

The shortest credible path toward SonicMoE-like performance is not a full new
MoE rewrite first. It is a sequence of increasingly invasive A2/dataflow
experiments that can be measured against this saved rocprof baseline.

### P1: A2 layout and stage2 read path

Goal: reduce stage2's gather-like A2 access cost without changing math.

Prototype:

- Add an experimental `a2_layout` switch to the benchmark/wrapper path.
- Compare current `[token, topk, inter_dim]` logical layout with a sorted-block
  layout that matches `sorted_token_ids` order more directly.
- Keep stage1 and stage2 as separate kernels, but make stage1 write the layout
  that stage2 consumes most efficiently.
- Validate on fixed `bf16`, `swiglu`, `topk=8`, `block_m=128`, no quant.

This is lower risk than full fusion because correctness remains close to the
current two-stage design. It should show whether stage2's low MFMA utilization
is mostly caused by the A2 read pattern.

### P2: Existing A2 quant/repack paths

Goal: reduce logical A2 read+write bytes before writing a new kernel family.

Prototype:

- Exercise current `fuse_quant`, fp8, or fp4 inter-stage paths where supported.
- Profile the same three shapes with the same counters.
- Separate numerical error, conversion overhead, and stage2 speedup.

If quantized A2 gives a large speedup with acceptable accuracy, it may be the
fastest MI355X path even before full SonicMoE dataflow fusion.

### P3: Tile-local stage1->stage2 fusion

Goal: remove most global A2 materialization.

Prototype:

- Start with one fixed kernel family: `bf16`, `swiglu`, `topk=8`, `block_m=128`,
  no quant, no expert parallelism.
- Work per expert and M block from the existing sorted metadata.
- Compute a tile of stage1 activation, apply activation, and feed it directly
  into the corresponding stage2 GEMM tile.
- Accumulate routed-weighted partial output into `[token, model_dim]`.

Main risk:

- With `H=4096` and `I=512/1024`, a naive full fusion cannot keep all A2 and
  all output accumulators on chip. It needs N-tiling and I-tiling, and may need
  atomic adds or split reductions for output accumulation.
- Reusing W1 and W2 efficiently at the same time is the hard part. A kernel that
  removes A2 but reloads weights too much can lose to the current two-stage CK
  path.

### P4: Kernel vehicle choice

Recommended order:

1. Extend the current CK route first, because AITER already dispatches CK
   `DeviceMoeGemm` for this workload and the benchmark/profiler can isolate
   stage1 and stage2.
2. Use FlyDSL/ASM only if CK cannot express the needed A2 layout or fused
   producer-consumer schedule cleanly.
3. Treat Opus as a candidate for a new grouped-MoE kernel only after P1/P2 show
   that CK layout and quant variants cannot close the gap.

The local checkout currently exposes the AITER CK wrappers and generated
instance dispatch, but the full external CK implementation headers are not
vendored in this tree. A CK prototype may require populating the CK source used
at build time, then changing both the AITER wrapper and the underlying
`DeviceMoeGemm`/gridwise dataflow.

## Next Measurement Gate

Before writing a full fused kernel, add a small sentinel-layout test and one
benchmark option:

- Verify exactly how `sorted_token_ids` maps stage1 output slots to stage2 input
  rows for a tiny deterministic shape.
- Add `a2_layout=current|sorted` for an experimental path.
- Re-run the same three saved shapes and compare stage2 MFMA, R+W MiB, and
  end-to-end `direct_fused_moe_ms`.

If P1 does not move stage2 MFMA or traffic, skip to P2/P3. If P1 helps
materially, keep the two-stage design and optimize layout before building a
larger fused kernel.
