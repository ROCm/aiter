from pathlib import Path
import json
r=Path(__file__).resolve().parent
t=json.loads((r/'timings.json').read_text())[-1]['median_us'];v=json.loads((r/'validation.json').read_text());res=json.loads((r/'resources.json').read_text())
s='''# Experimental Flash C1 full-layer monokernel

The uncompressed layer-0 kernel now includes attention mHC pre/RMS, input quant, QA/KV, norm/quant/QB/RoPE, KV update, split attention and reduction, inverse RoPE/quant/OA/quant/OB, attention post, FFN mHC pre/RMS, both input quantizations, router GEMM and stock C1 top-6 routing, shared gate/up/activation/quant/down, routed FP4 gate/up/activation/requant/down/combine, shared add, and FFN mHC post. All learned GEMMs use WMMA; tiled reads use TDM; existing QA/QB L2 prefetch remains.

This is a complete layer kernel, not a whole-model kernel. Serving currently enables it only for Flash layer 0 at C1 with BF16 cache and a compatible gfx1250/TP1 shape. Other layers, compressed attention, and prefill use ATOM native operators. The prototype is not accuracy-qualified and remains default-off.

## Full-layer microbenchmark

Flash 1K input / 1K output C1 shapes. Actual ATOM implementations and layer-0 checkpoint weights. 51 validation cases span 17 inputs and positions 1024,1536,2047, with a full 128-token SWA window. Second-round GPU-event medians; 16 stable-address repetitions per flat graph, five warmups, 40 alternating samples. Both rounds retained. Native is an extracted full-layer reference with explicit attention and FFN post boundaries; serving normally folds post into the next pre, so its launch count and time are not a prediction of serving cost.

| Path | Launches per layer | Round 2 us |
|---|---:|---:|
'''
for n,label in [('native','ATOM native extracted reference'),('two_fused_regions','Attention + FFN, two fused regions'),('single_full_layer','Single fully fused layer')]:s+=f"| {label} | {res['graph_nodes'][n]//16} | {t[n]:.5f} |\n"
s+='''
The single launch is slower than the two fused regions. It uses 492 VGPR, 133088 bytes LDS and zero spills, with 256 CTAs and four waves per CTA. This is a resource-heavy resident kernel; removal of a launch alone does not guarantee improvement.

## Correctness

One-launch and two-region outputs are bitwise identical in all 51 cases; all outputs finite and health flags clear. They are NOT sufficiently close to native: worst relative RMS error is %.8f. The worst case changes one selected expert (91 to 44). Native FFN fed the fused attention output agrees within 4.77e-8 RMS on that case, isolating the large error to inherited attention differences changing routing. FFN-only validation over 32 changing random residuals has matching routing IDs in all cases and maximum final RMS 3.11396e-5. See NUMERICS.md, validation.json, ffn-validation.json and diagnosis.json. No GSM8K or model-accuracy acceptance is claimed.

The serving adapter matches standalone outputs and KV updates at positions 1024/1536/2047, including a 194 GB allocation with cache row indices above 189 million. Testing exposed and fixed a 32-bit cache-address overflow: cache tile bases and final writeback now use 64-bit address arithmetic. Graph-capture integration also required an opaque custom operator so dummy-run guards would not erase the decode path during torch.compile.

## Serving integration

ATOM atom/models/deepseek_v4.py has guarded preparation and a layer-0 dispatch hook. atom/model_ops/dsv4_full_layer.py loads the experimental bundle. The custom operator reads live capture metadata, stages the active 128-row SWA window, runs the complete layer and writes the current KV row back to ATOM's cache in the same launch. Full FFN post is performed inside the kernel. An identity pending post preserves ATOM's existing next-layer post/pre interface. Prefill falls back to native computation with residual materialization to satisfy the same custom-op output contract.

The successful serving runs use one server lifetime per variant and two client invocations, reporting only the second. Eight requests per invocation, 1024 input / 1024 output, concurrency 1, ignore EOS, seed 0, prefix caching disabled, graph capture [1], original Flash environment. Baseline GPU2, experimental GPU3. Earlier failed integration startups were debugging attempts with no accepted fused benchmark results; the successful pair is not mixed across server starts. The earlier baseline value near 148 tok/s is total tokens, not output-only throughput.

'''%max(x['native_rms'] for x in v)
paths=[r/'baseline-client-round2.json',r/'fused-client-round2.json']
if all(p.exists() for p in paths):
 s+='| Serving path | Median TTFT ms | Median TPOT ms | Output tok/s | Total tok/s |\n|---|---:|---:|---:|---:|\n'
 for name,p in zip(['Baseline','Experimental layer-0 full kernel'],paths):
  d=json.loads(p.read_text());s+=f"| {name} | {d['median_ttft_ms']:.5f} | {d['median_tpot_ms']:.5f} | {d['output_throughput']:.5f} | {d['total_token_throughput']:.5f} |\n"
else:s+='Serving second-round results are pending.\n'
s+='''
These serving numbers are performance measurements of a known numerically divergent experimental path, not validated model speedups. Other layers retain native kernels, so a model-wide fused-layer benefit cannot be inferred.

## Reproduce and resume

Compatible environment: image rocm/fw-bringup-mlperf:gfx1250-atom-20261015, running container vigokhal_dsv4_tp4; ATOM f19778df4054e94d64d6f6bf6dd68f5681bd5945 and AITER 7cddd8e632e8d07eebdf9e1cab7e7a9f1e8c65fe with existing session modifications. Checkpoint /data/DeepSeek-V4-Flash and installed software remain external dependencies; checkpoint weights are not bundled in full.

Extract the artifact into the directory mounted as /work. It includes full-layer-native, FFN C1 ablations, full-attention source/fixtures and supporting dependencies. Run install.py inside the compatible container to apply the narrow ATOM hook (idempotent on this installed version); atom-integration.patch records only this task's model-file changes, preserving the prior session changes. dsv4_full_layer.py must also be installed in atom/model_ops (install.py does this). The shim defaults ATOM_DSV4_FULL_LAYER_ROOT to /work/full-layer-native. No commit, push or image publication is performed by this task.

```sh
# Isolated full-layer validation + two-round benchmark, on an idle GPU:
docker exec -e HIP_VISIBLE_DEVICES=0 -e PYTHONPATH=/app/aiter:/app/ATOM vigokhal_dsv4_tp4 bash -c 'cd /work/full-layer-native && python bench.py'

# Experimental server (default-off integration explicitly enabled by this script):
HIP_VISIBLE_DEVICES=0 HTTP_PORT=8000 PORT=8001 bash /app/scripts/run_dsv4_flash_full_layer_c1.sh
# Start server once; clients.sh performs both client rounds:
bash /work/full-layer-native/clients.sh fused 8000
```

For baseline use ATOM_DSV4_FULL_LAYER_EXPERIMENTAL=0 with the same script. To return to normal serving, leave that variable unset or zero. The run script is also saved in this bundle. Raw client results, profiler trace, exact kernel symbols, generated assembly, source patch and numerical diagnostics are preserved. Next required work is native-equivalent attention arithmetic before promoting this experimental integration or expanding coverage to other layers.
'''
(r/'REPORT.md').write_text(s)
