from pathlib import Path
import json,hashlib,datetime
p=Path(__file__).resolve().parent
r=json.loads((p/'timings.json').read_text())[1]['results'];rot=json.loads((p/'rotating-timings.json').read_text())[1]['results']
tune=[a for a in json.loads((p/'tuning.json').read_text()) if a['round']==2]
val=json.loads((p/'validation.json').read_text());chain=json.loads((p/'chain-validation.json').read_text());stress=json.loads((p/'stress.json').read_text());traces=json.loads((p/'trace-verification.json').read_text())
assert len(val)==1152 and all(a['passed'] for a in val)
assert all(max(a['q_error'],a['kv_error'],a['attention_error'])<.001 and a['health'] for a in chain)
assert len(stress)==6 and all(a['passed'] and a['checked_invocations']==10240 for a in stress)
assert [a['count'] for a in traces]==[2,1]
isa={}
for name in ['aiter_s1','aiter_s4','qkv_attention_fused_s1','qkv_attention_fused_s4']:
 asm=(p/f'{name}.amdgcn').read_text();stores=[l.strip() for l in asm.splitlines() if 'buffer_store_' in l]
 assert stores and all('scope:SCOPE_DEV' in l for l in stores),(name,[l for l in stores if 'scope:SCOPE_DEV' not in l])
 assert 'tensor_load_to_lds' in asm and 'v_wmma' in asm and 'global_prefetch_b8' in asm
 isa[name]=dict(buffer_stores=len(stores),all_buffer_stores_scope_dev=True,tdm_instructions=asm.count('tensor_load_to_lds'),wmma_instructions=asm.count('v_wmma'),prefetch_instructions=asm.count('global_prefetch_b8'))
(p/'isa-verification.json').write_text(json.dumps(isa,indent=2))
up=dict(url='https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/_gluon_kernels/gfx1250/attention/pa_decode_sparse.py',snapshot_sha256=hashlib.sha256((p/'upstream/pa_decode_sparse.py').read_bytes()).hexdigest(),license='MIT',license_file='upstream/LICENSE',local_container_aiter_commit='7cddd8e632e8d07eebdf9e1cab7e7a9f1e8c65fe',note='Adapted remote source snapshot, not assumed identical to the installed checkout.')
(p/'UPSTREAM.json').write_text(json.dumps(up,indent=2))
table='| QKV + cache update + attention | Reused weights (us) | 16 rotating QKV copies (us) |\n|---|---:|---:|\n'
for title,n,m in [('Old vector attention, two launches','qkv_vector_two','old_vector_two'),('AITER-derived attention, two launches','qkv_attention_two_s1','aiter_two'),('AITER-derived attention, one launch','qkv_attention_fused_s1','aiter_fused')]:table+=f"| {title} | {r[n]['median_us']:.6f} | {rot[m]['median_us']:.6f} |\n"
st='| Context | Heads/CTA | KV splits | Time (us) |\n|---|---:|---:|---:|\n'
for row in tune:
 for s in [1,2,4]:st+=f"| {row['tokens']} | 16 | {s} | {row['results'][f'h16_s{s}']['median_us']:.6f} |\n"
(p/'REPORT.md').write_text('''# AITER-derived attention integrated into the QKV monokernel

Implemented an adaptation of the requested [AITER gfx1250 sparse decode kernel](https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/_gluon_kernels/gfx1250/attention/pa_decode_sparse.py). Exact downloaded source and MIT license are preserved in upstream/, with SHA256/provenance in UPSTREAM.json. No AITER runtime dependency is needed by the adapted kernel.

## What is integrated

- AITER's grouped-head BF16 WMMA QK and PV, online softmax, sparse slot handling, two-deep TDM slot/KV gather pipeline, and split-KV softmax merge.
- Q tiles also use TDM. Scalar/small-vector accesses remain direct. Sparse KV indices can be -1; valid indices address a unified BF16 cache whose rows supply both K and V.
- KV cache update is performed byCTA0 and published with a device-scope grid barrier before attention reads. Non-speculative L2 prefetch covers requested KV rows, using safe addresses for masked indices. Existing all-weight TDM and non-speculative QB L2 prefetch remain enabled in the QKV stage.
- Split partials m/l/acc are published with **device-scope buffer stores**, drained before an arrival counter. The last CTA for a head group invokes AITER's TDM partial reducer inside the same kernel and resets the group counter. The original upstream split/reduce kernel boundary is therefore removed. No polling is needed for this last-arriver merge; the initial cache publication still uses a grid barrier.
- The sink is included once: inside unsplit attention or in the final split merge. All-invalid tiles are guarded against -inf-minus--inf NaNs. The API requires each configured split to own a tile and validates index bounds at construction.
- Query scaling happens after QK accumulation in FP32 rather than rounding a scaled query to BF16. PV uses high and low BF16 probability terms in two WMMA operations to preserve the existing0.001 relative-RMS gate. This is an adaptation, not a claim of bitwise identity to unmodified upstream.

## Launch integration

`AttentionRegion`: one launch combining cache update, attention and (when split) merge.

`QKVAttentionRegion(fused=True)`: one256-CTA launch combining our entire inputquant→QA/KV→norm→latentquant→QB→norm→RoPE region, cache update and attention. A grid publication barrier after QKV precedes the cache-update/publication barrier. There are no child kernel launches. At128tokens, default BH16/split1 gives4 active attention CTAs, each handling16 heads. Split4 gives16 active attention CTAs. All256 CTAs participate in the QKV/cache publication barriers.

`QKVAttentionRegion(fused=False)`: the same QKV and attention operations in two region launches. Profiler traces in trace-verification.json confirm exactly1 versus2 kernels.

Default BH16/BK32. Auto split selects1 below1024 entries and4 at/above1024, based on tested128/1024 endpoints; other lengths are not claimed to have an optimal split. Manual1/2/4 overrides are supported when all segments are active. Wider BH32/64 configurations were swept but did not beat BH16 in this C1 test. Two-region integration is currently faster than full fusion; both paths are available, with fused=True retained as the explicit monokernel API default.

## Performance: second round only

'''+table+f'''
Isolated cache-update+attention at128tokens: old vector implementation **{r['vector']['median_us']:.6f}us**, adapted AITER split1 **{r['aiter_s1']['median_us']:.6f}us**. The original historical14.56us number used a different profiler harness; the matched remeasurement above is the valid control. The old control retains its vector attention math under the current cache-update/barrier wrapper.

Split sweep (separate interleaved experiment, so do not subtract its timings from the chain experiment):

'''+st+'''
At128tokens, merging partials costs more than splitting saves. At1024tokens, split4 substantially improves over split1. The default split heuristic is based on these endpoints, not a full production sequence-length tuning table.

All performance figures use round2 only,128 calls per captured graph,5 warmup replays and40 alternating-order samples per round. Rotating mode cycles through16 identical QKV weight copies at distinct addresses,608MiB total, while the attention cache remains a controlled128-token window. It is a larger-weight-working-set proxy, not actual model-layer serving. Original round1 and all samples are retained. No TTFT/TPOT/token-throughput claim.

## Correctness and assembly

'''+f'''{len(val)} attention checks:127/128/256/1024 entries,32 cases per length,3 split choices,3 replays. Cases include random order, -1 masks, entirely invalid slots, varied query/cache magnitudes and sinks. Reference is independent FP64 attention with BF16 output. Max relativeRMS={max(a['error'] for a in val):.9g}; threshold remains0.001.

{len(chain)} full-chain cases check Q/KV against the previously validated QKV control, and attention against FP64. Six stress configurations each check10240 changing-input graph invocations: attention split1/2/4; two-region split1; fused split1 andsplit4. All pass with clean health/counters. Max stress relativeRMS={max(a['max_error'] for a in stress):.9g}. These checks do not replace serving/GSM8K evaluation.

'''+'''Assembly and LLVM IR are saved for measured paths. ISA verification checks TDM, WMMA, prefetch instructions and explicit device scope on buffer stores. Selected paths have zero spills. Exact register/LDS counts are in resources.json. Publication scope was corrected during integration; earlier exploratory results remain clearly labeled initial-* and are not the final comparison.

## Scope and limitations

SingleGPU FlashC1,64 heads,512 dimensions, unified uncompressed BF16 K/V cache. Sparse slot indexing is supported, but this does not implement the full Flash compressed-cache/index-selection machinery or quantized KV formats. Current-token destination is fixed per object for capture; callers rebuild/select an object for a different slot. No TP/EP, no MoE, no inverse-RoPE/output-projection integration, no ATOM serving integration, no git/image push. Fixture uses checkpoint layer0 projection weights and real sink, with controlled cache histories; stress additionally uses synthetic queries/cache. No full checkpoint required to replay the saved artifact.

Read README.md for reproduction. GPU idle after final checks. All sources, fixtures, provenance/license, traces, assembly, raw timings, validation and environment are included in the archive.
''')
(p/'STATUS.json').write_text(json.dumps(dict(integrated=True,scope='Flash C1 uncompressed BF16 QKV + cache update + attention',single_launch_verified=True,tdm=True,l2_prefetch=True,serving_integrated=False,default_fused=True,fastest_measured_chain='two regions',second_round_chain_reused_us=r['qkv_attention_fused_s1']['median_us'],second_round_chain_rotating_us=rot['aiter_fused']['median_us']),indent=2))
print('report and ISA verification complete',flush=True)
