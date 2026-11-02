from pathlib import Path
import json,hashlib
r=Path(__file__).resolve().parent
load=lambda p:json.loads((r/p).read_text())
t=load('timings.json')[-1]['median_us'];v=load('full-validation.json')[-1];res=load('full-resources.json');cross=load('gpu2-crosscheck/timings.json')[-1]['median_us'];native=t['actual_atom_extracted_boundary'];fused=t['fused_single_launch'];staged=t['staged_prototype_control']
text=f'''# Full attention-layer monokernel: Flash C1, 1024 input /1024 output

Implemented one gfx1250 Gluon kernel from mHC pre through attention output residual. This is the layer0 sliding-window attention sublayer, not the MoE/FFN layer or compressed-attention layers. All GEMMs use WMMA; tiled loads use TDM; existing L2 prefetch remains enabled.

## Second-round microbenchmark

| Variant | Launches | GPU0 latency µs | Independent GPU2 latency µs |
|---|---:|---:|---:|
| Actual ATOM operators, extracted attention-only endpoint |16|{native:.5f}|{cross['actual_atom_extracted_boundary']:.5f}|
| Our staged prototype (unshuffle /prefix /output /post) |4|{staged:.5f}|{cross['staged_prototype_control']:.5f}|
| Full attention monokernel |1|{fused:.5f}|{cross['fused_single_launch']:.5f}|

Timing-only comparison: single launch is {native/fused:.3f}x faster than native operators ({100*(1-fused/native):.2f}% lower latency). However it is {100*(fused/staged-1):.2f}% slower than our four-launch staged prototype. Thus this result does not establish that maximum fusion is optimal. Numerical equivalence to ATOM still fails; no serving speedup or deployment readiness is claimed.

Both rounds and raw samples saved; only round2 reported.128 sequential calls per flat HIP graph, same buffer addresses in every repetition,5 warmups and40 alternating-order samples. Graph kernel counts2048/512/128 confirm16/4/1 launches per invocation. Poisoned graph outputs are restored and match each respective eager implementation bitwise. Standalone operator names saved in kernels.json and trace.json. FlatGraphBatch is the previously validated helper; no childgraph scheduling overhead.

## Scope and boundary qualification

Input: native KS64-shuffled BF16 residual[1,4,4096]. Output: plain BF16 updated residual[1,4,4096]. The monokernel performs input unshuffle internally, requiring no adapter launch. All variants use the same checkpoint layer0 weights, input, RoPE position1024, and synthetic BF16 KV-cache state. Validation also covers positions1536 and2047. KV visibility is128 sliding-window rows even for1024/1024 workload. Native buffer capacity16384 selects128 attention splits; prototype uses4, as in the preceding experiments.

The native reference calls installed ATOM/AITER/FlyDSL implementations. Its endpoint is deliberately extracted: actual serving normally fuses attention mHC post with the subsequent FFN pre. For this requested attention-only boundary we call ATOM's standalone hc_post, which unshuffles its residual and launches HIP post. Consequently16 launches are the native-operator attention-only baseline, NOT a claim that serving executes precisely these16 separate launches. Model-wide FFN fusion/side-stream overlap is outside this microbenchmark.

## Kernel flow

```text
256-CTA resident grid,4 waves/CTA
  64 CTAs: TDM input unshuffle -> plain residual scratch
        [device-scope grid barrier]
  mHC pre + corrected input RMSNorm
  input quant -> QA/KV -> norm/quant -> QB -> Q/KV norm/RoPE
  cache write +16 attention CTAs (4 head groups x4 splits)
  last arrival per head group reduces attention partials
        [device-scope grid barrier]
  256 CTAs: inverse RoPE + FP8 input quantization
        [device-scope grid barrier]
  256 CTAs: grouped OA projection (64 output tiles x4 K splits)
        [device-scope grid barrier]
   64 CTAs: OA partial reduction + BF16 rounding + FP8 quant
        [device-scope grid barrier]
  256 CTAs: OB projection (32 output tiles x8 K splits)
        [device-scope grid barrier]
   32 CTAs: OB partial reduction + BF16 rounding
        [device-scope grid barrier]
   32 CTAs: mHC post, each owns128 features across4 streams
  kernel completion publishes final residual
```

The prefix retains its documented internal publication/arrival barriers. Grid barriers use the established monotonically advancing atomic arrival counter plus device-scope load polling, bounded timeout health flag, and store-drain/CTA ordering. Full grid must be resident; run on an otherwise exclusive gfx1250 GPU. Do not overlap invocations sharing state. No new mailbox protocol was introduced.

Compiled full kernel: {res['vgpr']}VGPR, {res['lds']}bytesLDS, {res['spills']}spills. Source full_kernel.py; full.amdgcn/full.llir retained. Output projections use double-buffered TDM activation and weight tiles; post uses TDM residual/output tiles. Small coefficients/scales use normal loads.

## Correctness

137 distinct residual inputs x3 decode positions =411cases per round, repeated twice. Results below use round2:

- Fused versus staged prototype: {v['staged_nonexact']} non-bitwise cases out of411.
- Native final residual: max relative RMS {v['max_native_residual_rms']:.9f}; {v['native_failures']}/411 fail the unchanged0.001 gate.
- Native OB output before residual mixing: max relative RMS {v['max_native_ob_rms']:.9f}.
- Nonfinite outputs:{v['nonfinite']}; unhealthy mailboxes:{v['unhealthy']}.
-4096 additional single-kernel invocations with32 changing residuals passed poisoned-output, exact-staged and mailbox checks.

The mHC RMS denominator correction is included: compute squared norm from FP32 mixture while normalizing the BF16-rounded mixture. Its separate validation reduced maxXerror0.001061097 to0.000145058 and eliminated all mHC threshold failures (see sibling mhc-native-rms-fix/REPORT.md).

mHC post now matches native arithmetic bitwise in extended isolated tests: use native FMA association and prevent compiler folding of final FP32 FMA plus BF16 conversion into a single-rounding v_fma_mixlo_bf16. An opaque FP32 move preserves the separate FP32 then BF16 rounding.512 tests for both tile sizes plus64 prior plain-layout cases passed; see post-kernel/.

Remaining differences are upstream and within projections/attention. InverseRoPE+quant matches native bytes/scales in8 isolated cases. OA/OB region still fails native equivalence: maxOA≈0.000953, maxOB≈0.004666 over64inputs. Disabling OA/OB splitK does not resolve it, so splitK reassociation alone is not a sufficient explanation. Different GEMM arithmetic/rounding and amplification at the intermediate FP8 conversion remain to be reconciled. The previous QA/attention native differences also remain. No tolerance was relaxed.

## Isolated output region diagnostic

GPU1 second-round actual output operators inverseRoPE throughOB:66.1661µs. Fused output region:17.8403µs. Same arithmetic in5 separate prototype stages:14.5058µs. These separate region measurements are not additive predictions of full-layer latency; large weight/workspace footprints change cache behavior when combined. See output-kernel/REPORT.md.

## Reproduce and resume

Existing container vigokhal_dsv4_tp4; image rocm/fw-bringup-mlperf:gfx1250-atom-20261015. Host dsv4-monokernel mounted at /work. Installed ATOM/AITER/FlyDSL/PyTorch environment and model config under /data/DeepSeek-V4-Flash required. Layer0 output weights and all prior prefix fixtures are saved. Source dependencies in sibling atom-actual-microbench and mhc-optimized are included in the handoff archive.

```sh
docker exec -e HIP_VISIBLE_DEVICES=0 -e PYTHONPATH=/app/aiter:/app/ATOM vigokhal_dsv4_tp4 bash -c 'cd /work/full-attention-native && python bench_full.py'
docker exec -e HIP_VISIBLE_DEVICES=3 -e PYTHONPATH=/app/aiter:/app/ATOM vigokhal_dsv4_tp4 bash -c 'cd /work/full-attention-native && python validate_full.py'
docker exec -e HIP_VISIBLE_DEVICES=0 -e PYTHONPATH=/app/aiter:/app/ATOM vigokhal_dsv4_tp4 bash -c 'cd /work/full-attention-native && python stress.py'
```

Scripts overwrite their outputs; preserve a copy before rerun. Set OUTDIR for a separate benchmark directory. Native operator wrapper sources are unmodified arithmetic paths; TP1 distributed getters are stubbed only during standalone construction. No model-server integration, image push, Git push or GSM8K run was performed for this prototype. All current GPU experiments have completed.

Next work: establish native projection/attention numerical parity, then rebenchmark. The four-region version is already slightly faster than monolithic fusion, so retain both as candidates. Do not infer serving TPOT/TTFT/tokens-per-second from these isolated numbers.
'''
(r/'REPORT.md').write_text(text)
(r/'summary.json').write_text(json.dumps({'round':2,'latency_us':t,'resources':res,'numerically_accepted':False,'native_failures':v['native_failures'],'cases':v['count'],'max_residual_rms':v['max_native_residual_rms'],'staged_bitwise_all':v['staged_nonexact']==0,'stress_calls':4096},indent=2))
