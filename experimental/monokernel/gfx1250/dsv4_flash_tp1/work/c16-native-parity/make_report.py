from pathlib import Path
import json,hashlib,datetime
r=Path(__file__).resolve().parent
regions={}
for n in ['qkv_cache','output_projection']:
 p=r/'native-c16/balanced_regions'/f'{n}.json'
 x=json.loads(p.read_text());regions[n]=x
full={}
for ratio in [0,4,128]:
 x=json.loads((r/f'balanced-final-micro-16-{ratio}.json').read_text())[0];full[str(ratio)]=x
large={str(i):json.loads((r/f'selected-final-micro-16-{i}.json').read_text())[0] for i in [0,4,128]}
checks={}
for folder in ['selected_stress','advantage_stress','advantage_subgroup_stress','selected_stress_final']:
 for p in (r/'native-c16'/folder).glob('stress_*.json'):
  x=json.loads(p.read_text());checks[str(p.relative_to(r))]=dict(cases=len(x),max_native_rrms=max(v['relative_rms'] for v in x),all_replays_exact=all(v['replay_exact'] for v in x),all_previous_exact=all(v.get('exact_previous',True) for v in x),all_health_zero=all(not v['health'] for v in x))
out=dict(regions=regions,full_layers=full,large_qb_full_layers=large,stress=checks)
(r/'RESULTS.json').write_text(json.dumps(out,indent=2))
lines=['# C16 native projection adaptation and monokernel advantage set','',
'Completed October 21, 2026 Central time (artifact date October 22 UTC). Flash TP1, independent workers on all four GPUs. C16 decode shapes at position 1023. **These are microbenchmarks, not serving throughput. C16 serving remains disabled.**','',
'The output region improves substantially; QKV still trails native. The complete advantage-set recipe is `advantage_set.env`. It keeps early TDM, bounded L2 prefetch, group-scoped synchronization, local/elected ownership, and an increased QB buffer count. `fastest_measured.env` is a comparison-only ablation of explicit cross-region hints. `large_qb.env` spends more LDS on larger QB tiles; it is not the default because its standalone gain did not carry over consistently to whole layers.','',
'## Paired isolated regions','',
'32 flat HIP-graph repetitions, 5 warmups, 40 alternating samples per round, two rounds; only the second round is reported. Native is the installed ATOM operator chain with its normal AITER dispatch, using the same inputs as the monokernel. Prior is the frozen selected implementation from `c16-latency-tuning`.','',
'| Region | Native ATOM µs | Prior monokernel µs | Advantage-set monokernel µs | Change versus prior |',
'|---|---:|---:|---:|---:|']
for key,label in [('qkv_cache','Input quant → QKVA → norms/quant → QB → Q norm/RoPE + KV cache'),('output_projection','Inverse RoPE/quant → OA → quant → OB')]:
 t=regions[key]['rounds'][1]['median_us'];lines.append(f"| {label} | {t['native']:.2f} | {t['previous']:.2f} | {t['fused']:.2f} | {(t['fused']/t['previous']-1)*100:+.1f}% |")
lines+=['','## Paired full-layer micros','',
'Synthetic full-layer fixtures, including attention and MoE, with 16 graph repetitions and 40 alternating samples. Second round only. This is integration into the experimental full-layer Gluon kernel, not modelwide ATOM serving.','',
'| Layer | Prior µs | Advantage set µs | Change | VGPRs | LDS bytes |', '|---|---:|---:|---:|---:|---:|']
for k,label in [('0','SWA'),('4','HCA (ratio 4)'),('128','CSA (ratio 128)')]:
 a,b=full[k]['resources'];lines.append(f"| {label} | {a['second_us']:.2f} | {b['second_us']:.2f} | {(b['second_us']/a['second_us']-1)*100:+.1f}% | {b['vgpr']} | {b['lds']} |")
lines+=['','All 13 intermediate tensors match the prior monokernel exactly for each full-layer fixture. Repeated graph replay is exact, health flags are zero, and all variants in the table have zero spills.','',
'## What changed and which techniques survived','',
'| Boundary or region | Implementation and reason |',
'|---|---|',
'| QKVA | Implemented the native N32/K1024 no-split-K pipeline, including four buffers. It passed initial numerical checks but made the combined region slower (~20 µs), so the default retains the prior N128/split-16 QKVA. Split 8/4 and redundant local input quantization were also tested and rejected. |',
'| QB + norm/RoPE | Packed-weight WMMA/TDM pipeline adapted from native. One of 64 CTAs owns all 512 columns of a head and all 16 rows, then normalizes/rotates its own result. No QB-output global store/read or cross-CTA norm barrier is needed. Three K128 buffers replace native two-buffer staging for this adapted geometry. The prior norm reduction tree is preserved exactly. |',
'| Early QB data | The first QB weight tile is issued after the QA projection work, before the QA completion wait and RMSNorm. This avoids draining that transfer in an earlier QA TDM wait. Weight prefetch covers only valid mapped addresses. |',
'| Inverse RoPE/quant | AITER-style 32 values/thread, four threads per 128-element quant group; one CTA processes a row/span instead of serially walking 16 rows. Static RoPE coefficients load while the input TDM transfer is pending. |',
'| Inverse quant → OA | Eight independent groups of 32 producer/consumer CTAs. Each OA group waits only for its own input group; no 256-CTA join. |',
'| OA | Packed N32/K1024 no-split-K GEMM, tuned to M16 and three buffers. Removes split-K partial writes and reduction. All 256 OA CTAs participate. |',
'| OA → quant | Four OA CTAs produce each 128-column quantization group. Their last producer, elected by an atomic ticket, quantizes all 16 rows of that group immediately. Other groups can still be computing; there is no spin wait or global barrier for this quantization handoff. |',
'| Quant → OB | 64 elected quantization producers publish completion; only the 64 OB consumers poll. The remaining CTAs can advance to the next full-layer boundary. Payload transfers explicitly use device-scope TDM and publication waits for their completion. |',
'| Early OB data | Bounded L2 prefetch precedes OA. The first OB weight tile is issued before OA computation, so it overlaps OA and its quantization epilogue. OB retains a four-buffer K512 rolling pipeline. |',
'| Larger LDS | Tested more buffers and larger tiles separately. The default QB uses three K128 buffers. The larger two-K256-buffer variant grows full-layer LDS from about 204 KB to 271 KB and improves isolated QKV, but its SWA layer regressed. Saved as an optional experiment. |',
'','Remaining whole-grid boundaries include QKVA completion/QA norm input and QA-norm completion before QB, plus the unchanged enclosing layer boundaries. This work does not claim every monokernel boundary is optimal.','',
'## Validation and rejected experiments','',
'- Output projection: 64 changing-input cases exactly match native, with exact repeated replay. The final earlier-OB-transfer variant also passed all 64 cases.',
'- QKV: 64 changing-input cases exactly match the prior monokernel, including the final larger-tile experiment. Native QKV relative RMS difference reaches 4.45e-5 on these cases; this is inherited from the prior path, not introduced here. The original fixture is much closer (4.35e-8). This does not qualify model accuracy.',
'- The first complete-head norm used a different reduction order. It differed on the full-layer fixture; that variant was rejected. Restoring the prior reduction tree made all 13 full-layer checkpoints exact.',
'- Generic per-K-tile counter polling plus cache invalidation was slower. Some experimental QKV polling variants also raised synchronization health diagnostics. They are not part of the selected recipe. Do not enable `COHERENT=1` or arbitrary `ADV` combinations as production settings.',
'- Early TDM originally copied its first tile through registers into the ring. The current implementation uses that tile directly as ring slot zero.',
'- Early standalone QA/QB/OB sub-benchmarks in `parts/` used the wrong activation-scale stride and are invalid for comparison. `parts_fixed/` uses the native K-major scale layout and passes exact checks. Preserve those failed logs as diagnostics, not results.',
'- No C16 modelwide E2E, TTFT/TPOT/tokens/s, or GSM8K was run in this pass. No Git or Docker push was performed. Prior C1–C8 serving results remain in the parent archive.','',
'## Reproduce','',
'Container: `vigokhal_dsv4_tp4`. Host directory `/home/vigokhal/dsv4-monokernel` is mounted at `/work`. Dependencies are preserved in the nested complete preceding archive.','',
'```bash',
'docker exec -e HIP_VISIBLE_DEVICES=0 vigokhal_dsv4_tp4 bash /work/c16-native-parity/run.sh qkv',
'docker exec -e HIP_VISIBLE_DEVICES=1 vigokhal_dsv4_tp4 bash /work/c16-native-parity/run.sh output',
'docker exec -e HIP_VISIBLE_DEVICES=2 vigokhal_dsv4_tp4 bash /work/c16-native-parity/run.sh swa',
'docker exec -e HIP_VISIBLE_DEVICES=3 vigokhal_dsv4_tp4 bash /work/c16-native-parity/run.sh hca',
'# Once one GPU is free: run.sh csa, stress-q, stress-o',
'```','',
'Key sources: `native_tiles.py`, `native_quant.py`, `native_norm.py`, `qa_deferred.py`, `qkv_candidate.py`, `output_batch.py`, `model_layer_batch.py`. `previous/` is frozen. Weights are packed offline by `packing.py`; `paired_*.py` passes raw weights to the old kernel and packed QB/OA/OB weights to the new kernel. Serving integration must use that same distinction.','',
'Raw timing samples, native kernel names/profiler traces, emitted assembly, resource counts, failed experiment logs and full-layer output tensors are included. `RESULTS.json` consolidates the final results; `SOURCE_SHA256.json` identifies the final code.','',
'ISA cache-scope reasoning was checked against AMD CDNA5 ISA §4.1.1: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna5-instruction-set-architecture.pdf . Emitted TDM payload loads/stores show `scope:SCOPE_DEV`; flags also use device-coherent loads.','',
'## Next work','',
'1. QKV remains materially behind native: profile QA/norm and complete-head QB separately within the fused launch, preserving the exact reduction order.',
'2. Explore more overlap without adding a blocking TDM wait to the predecessor. Preserve bounded non-speculative prefetch.',
'3. Keep group ownership/elected epilogues; investigate the rejected generic polling variants only in isolated stress harnesses.',
'4. Before enabling C16 serving: implement packed-weight/runtime dispatch, progressive checkpoint checks with actual model inputs, accuracy testing, and baseline/fused E2E (one server startup, two clients; use the second client).','']
(r/'REPORT.md').write_text('\n'.join(lines))
hashes={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(r.rglob('*')) if p.is_file() and (p.suffix in ['.py','.sh','.env']) and '__pycache__' not in p.parts}
(r/'SOURCE_SHA256.json').write_text(json.dumps(hashes,indent=2))
print('report written',len(hashes),'source files')
