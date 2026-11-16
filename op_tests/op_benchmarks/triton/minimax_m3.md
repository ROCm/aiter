# MiniMax-M3 Triton/Gluon capture

This is an opt-in, cross-repository regression suite, not a port of ATOM's
kernels into AITER. Tests call public wrappers and compare against independent
PyTorch oracles. The benchmark shares their input generators and uses AITER's
`RotatingBufferBench`; it does not run accuracy checks inside timed graphs.

Tracking: https://github.com/AMD-Triton/triton-tickets/issues/2198

## Provenance and scope

Shapes come from a MiniMax-M3 TP4 workload with FP8 KV, MXFP4 MoE and EAGLE3 K=3:
decode batch 32, four query tokens per request (128 rows), about 194K context;
prefill has 2,763 real query tokens, a 3,072-row bucket, 157,312 prefix tokens
and 160,075 total KV tokens. The 16-row decode and 2,048-row prefill cases are
secondary workload buckets. These are shape captures with synthetic values,
not model weights, full-model execution or replay of the original distributed
trace. CP=4 scoring and candidate exchange are local compute-only replays;
the benchmark excludes collectives.

Included implementations:

- ATOM `minimax_m3.index_topk.minimax_m3_index_topk`: Triton block scoring
  (`_index_block_score_kernel`) and packed top-k (`_topk_index_packed_kernel`).
  The capture passes no MXFP4 scorer and no AITER selection bounds, keeping
  both operations on the Triton path of the pinned ATOM wrapper.
- ATOM `minimax_m3.indexer_context_parallel.indexer_context_scores`: Triton
  `_context_score`, round-robin context partitioning with four index heads.
- ATOM `minimax_m3.indexer_candidate_exchange.local_candidate_keys` and
  `merge_candidate_keys`: Triton `_local_topk` and `_merge_topk`. The merge
  capture consumes the strided `[world, tokens, topk]` exchange output layout.
- ATOM `triton_fused_qkv_norm_rope_cache.triton_fused_norm_rope_cache`: Triton
  Gemma RMSNorm, NeoX RoPE and per-token FP8 SHUFFLE cache writes. Inputs are
  strided bands of a fused QKV projection, with 16 Q heads, one KV head, D=128.
- ATOM `base_attention.cp_mha_gather_cache`: Triton SHUFFLE cache gather and
  per-token dequantization. K and V have different shuffle layouts.
- ATOM `swiglu_oai.swiglu_oai_split`: Triton split-layout SwiGLU, I=3072,
  alpha=1.702, beta=1 and limit=7. This is not an interchangeable SiLU/multiply.
- AITER `gemm_a16w16(backend="gluon")`: native Gluon router-sized BF16 GEMM,
  N=128 and K=6144. Gluon is part of the Triton toolchain, with a distinct DSL.
  These shapes do not establish the backend of any generically named
  `kernel_gemm_a16w16_0` in a trace.

MXFP4 MoE, FP8 PTPC GEMM and FlyDSL paged attention are not Triton captures.
HIP `fusedQKNormIdxrQKNormKernel` and CK dense prefill FMHA are also excluded.
Backend attribution is specific to the pinned implementation and invocation;
newer dispatch paths may use different backends.

## Reproduce

Use an installed AITER build with its matching dependencies and an ATOM
checkout at `f6bf6f0aff2fd8267be7bff9deee51ce8863ad7c`. For example, put that
checkout on `PYTHONPATH` after the current AITER checkout. ATOM is optional for
normal AITER CI; its tests skip unless explicitly enabled. An enabled test with
a missing or incompatible ATOM installation fails rather than silently skips.

```bash
export AITER_TEST_MINIMAX_M3_ATOM=1
python -m pytest -q \
  op_tests/triton_tests/attention/test_minimax_m3.py \
  op_tests/triton_tests/rope/test_minimax_m3.py \
  op_tests/triton_tests/fusions/test_minimax_m3.py \
  op_tests/triton_tests/gemm/basic/test_minimax_m3.py
python -m op_tests.op_benchmarks.triton.bench_minimax_m3 \
  --output minimax-m3-baseline.json --profile-dir minimax-m3-profile
```

On `gfx1250-strict`, the upstream tree currently lacks the separate Gluon GEMM
config directory. Router tests skip there; use `--cases` to select the ATOM
cases for benchmarking, without substituting `gfx1250` configs or changing the
reported architecture. On a configured `gfx1250` system, router cases also
work without ATOM: leave the opt-in variable unset and select, for example,
`--cases router-16 router-128 router-2048 router-3072`.

Run on an otherwise idle device, using the machine's cooperative GPU lock.
JSON records device/software versions, input shapes, dtypes and strides,
rotating pool sizes, every timing sample, and source-file SHA256 hashes.
Optional profiler traces record the actual dispatched kernel names in a
separate untimed invocation. By default each reported median uses three
independent graph means with the standard rotating-buffer timing settings.

The historical trace lists ATOM `17a48053c` and AITER `ea8be1a18`. Those are
different from this suite's measured source versions. Results from the current
source must not be described as an exact reproduction of that historical
software baseline, nor as an optimization speedup.

## Recorded baseline

`baselines/minimax_m3_gfx1250_strict.json` records the first measured baseline:
256 CUs, 432 GiB HBM, PyTorch `2.13.0+rocm10.3.0a20261106`, Triton `3.9.0`,
FlyDSL `0.5.0`, AITER `3d3d56db3e3e32d823b109bfb60d061a1e0b5b82` and the ATOM
revision above. Source archives were used; their revision IDs are supplied by
the archive command and per-file SHA256 values by the benchmark. The checked-in
report removes machine identifiers, absolute paths and random tensor reprs.

Opt-in validation: **10 passed, 4 skipped, 2 xfailed**. Four router cases skip
because this target has no published Gluon config. The two gather cases
reproduce numerical mismatches on this exact PyTorch/Triton/strict stack and
have a strict xfail limited to those versions. A future unexpected pass is a
failure so the expectation must be reviewed. They are not valid performance
baselines and were not timed; the failure's root cause remains undetermined.

Twelve other benchmark shapes completed, with three timing samples each.
Representative medians are CP score 79.89 us, local top-k 4.31 us, merge top-k
4.35 us, prefill index score plus top-k 3871.42 us, SwiGLU M=128 3.20 us and
QKV norm/RoPE/cache M=128 5.22 us. The JSON has all samples and the observed
kernel names. These are local compute measurements, not distributed TP4
latency. The strict prefill dispatch also differs from the optimized plain
gfx1250 policy; the architecture identity and configs were not overridden.
