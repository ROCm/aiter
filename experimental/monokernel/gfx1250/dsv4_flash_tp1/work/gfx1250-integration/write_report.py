from pathlib import Path
import ast
import collections
import hashlib
import json
import re

r=Path(__file__).parent
rows=[]
for mode in ('flash_compiled','flash_eager','pro_eager_unsplit','pro_compiled_unsplit'):
 p=r/f'{mode}_client2.json'
 if not p.exists():
  rows.append(f'{mode}: pending')
  continue
 a=json.loads(p.read_text())
 assert {int(k):len(v) for k,v in a.items()} == {c:c for c in (1,8,16,32,64,128)}
 assert all(x['usage']['completion_tokens']==24 for v in a.values() for x in v)
 rows.append(f'{mode}: client2 completed249 requests, C1/8/16/32/64/128,24 generated tokens/request; client1 preserved separately.')

s=['gfx1250 DeepSeek V4 upstream-style hybrid integration','',
'Upstream: ROCm/ATOM main53e4fb52403707ab07b69ee909501ea3cc25ca5c. Audit: UPSTREAM_AUDIT.txt. No gfx1250-specific mono or DeepseekV4ForCausalLM mono installer was found in this main snapshot; remote branches/unmerged PRs were not exhaustively searched.','',
'Checkout: /home/vigokhal/ATOM-gfx1250-integration; branch gfx1250-dsv4-hybrid. Code: atom/models/deepseek_v4_mono. Uses the shared post-load installer registry, deployment refusals, LazyRunner, TP bind/prepare agreement, per-width refusal, graph-safe warmup and native fallback. The native inner compiled model remains available. Views share loaded weights and bind only after KV cache addresses are fixed. Hash-routing IDs, attention/FFN stream ordering and TP reductions are preserved.','',
'Activation: ATOM_DSV4_MONO=1 and ATOM_MONO_ENABLE=1. The new model-specific flag defaults0. Launch scripts/serve_dsv4_mono_gfx1250.sh flash or pro; also installed in the test container at /app/scripts/serve_dsv4_mono_gfx1250.sh. Baseline: ATOM_DSV4_MONO=0. Public AITER mHC quant wrapper required; dependency patch bundled.','',
'Flash TP1: decode1..16 rows; mHC reduce/apply+RMSNorm epilogue adds FP8 quant/E8M0 scales, with separate tuned attention/shared-expert GEMMs at1/8 rows.','Pro TP4: decode1..64 rows; same mHC/quant fusion, separate unsplit QKV GEMMs at captured widths through64 and multicast attention/indexer projections at64. Missing tuning keys retain native GEMM. Prefill,C128,and unsupported deployments remain native. Routed MoE stays native for both profiles. This is regional hybrid fusion, not a new full-layer monokernel or a port of gfx950 MFMA code.','',
'CORRECTNESS FIX: live Pro C8 on rank3 exposed1/16384 BF16 QKV output elements differing under split-K (absolute0.00390625). That configuration and its first wheel are rejected. All shipped tables now use NUM_KSPLIT=1; a regression test enforces it. The older published TP4 image is unchanged; set ATOM_DSV4_HYBRID_GEMM=0 to avoid its known split-K QKV path. Earlier Pro timings describe the older configuration and must not be attributed to this corrected integration.','',
'Validation:91 focused CPU regression tests passed; black/ruff and git diff --check passed. Complete patch applies to a clean checkout of the pinned upstream commit. Wheel build passed and includes both tuning JSON files; wheel-verification.json and installed-source-check.json record fingerprints. The local tests namespace needed a temporary tests/__init__.py marker because of an unrelated preinstalled package; it was removed.','']+rows
s+=['','Exact eager regional checks (not Python assertions during graph replay):']
for mode,log in [('flash','flash-eager-1-server.log'),('pro','pro-eager-1-server.log')]:
 p=r/log
 if not p.exists():continue
 raw=p.read_text();keys=[ast.literal_eval(x) for x in re.findall(r'exact checks=16 key=(\([^\n]+\))',raw)]
 out={'exact_keys_reaching_16':len(set(keys)),'regions':dict(collections.Counter('mhc' if k[0]=='mhc' else k[1] for k in keys)),'errors':len(re.findall('Traceback|AssertionError|RuntimeError',raw))}
 (r/f'{mode}-final-eager-summary.json').write_text(json.dumps(out,indent=2)+'\n')
 s.append(f'{mode}: {out}')
s+=['',
'No new TPOT/TTFT/throughput claims are made from these short correctness clients. Report second full clients when rebenchmarking the corrected integration. Flash graph-vs-eager generated text varied in18/249 requests, including17 atC128 where both use native execution; this comparison is not a proof of full-model equivalence. Exact regional comparisons passed; full task-accuracy evaluation is not claimed.','',
'Artifacts include complete source, patch, current wheel under wheels/, its AITER dependency patch, rejected experiments/logs under explicit rejected names, all client1/client2 results and reproduction scripts. Do not install wheels-rejected-splitk. Model weights remain under/data and are not bundled. No new Docker image was published.']
(r/'REPORT.txt').write_text('\n'.join(s)+'\n')
