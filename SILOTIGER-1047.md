**<a href="https://amd.atlassian.net/browse/SILOTIGER-1040" id="parent_issue_summary">Qwen4-preview (Qwen3.8-Flash-Next) kernels</a>** <span style="font-size: 9px">(<a href="https://amd.atlassian.net/browse/SILOTIGER-1040" id="parent_issue_key">SILOTIGER-1040</a>)</span>

### <img src="https://amd.atlassian.net/images/icons/link_out_bot.gif" width="16" height="16" /> \[SILOTIGER-1047\] [\[Qwen4-preview QSA\] FlyDSL QSA indexer scorer + sparse GQA](https://amd.atlassian.net/browse/SILOTIGER-1047) <span class="subText"> Created: 08/Sep/26  Updated: 15/Sep/26 </span>

**Status:**

Opened

**Project:**

[Silo Tiger](https://amd.atlassian.net/secure/BrowseProject.jspa?id=13053)

**Components:**

[FlyDSL](https://amd.atlassian.net/issues/?jql=project%3D13053%20AND%20%22component%22%3D31844%20ORDER%20BY%20priority%20ASC "FlyDSL - All tasks related to FlyDSL"), [Kernels](https://amd.atlassian.net/issues/?jql=project%3D13053%20AND%20%22component%22%3D25143%20ORDER%20BY%20priority%20ASC "Kernels")

**Affects versions:**

None

**Fix versions:**

None

**Parent:**

[Qwen4-preview (Qwen3.8-Flash-Next) kernels](https://amd.atlassian.net/browse/SILOTIGER-1040)

  

<table class="grid" data-cellpadding="0" data-cellspacing="0" data-border="0" width="100%">
<tbody>
<tr>
<td data-bgcolor="#f0f0f0" data-valign="top" width="20%"><strong>Type:</strong></td>
<td data-bgcolor="#ffffff" data-valign="top" width="30%">Story</td>
<td data-bgcolor="#f0f0f0"><strong>Priority:</strong></td>
<td data-bgcolor="#ffffff" data-valign="top" data-nowrap="">Undefined</td>
</tr>
<tr>
<td data-bgcolor="#f0f0f0" data-valign="top" width="20%"><strong>Reporter:</strong></td>
<td data-bgcolor="#ffffff" data-valign="top" width="30%"><a href="https://amd.atlassian.net/secure/ViewProfile.jspa?accountId=5a01f2bcd3afb36093f28514" id="word_reporter_5a01f2bcd3afb36093f28514" class="user-hover" rel="5a01f2bcd3afb36093f28514">Remes, Sami</a></td>
<td data-bgcolor="#f0f0f0" width="20%"><strong>Assignee:</strong></td>
<td data-bgcolor="#ffffff" data-valign="top" data-nowrap="" width="30%"><a href="https://amd.atlassian.net/secure/ViewProfile.jspa?accountId=619cec3af241500072a2aeb6" id="word_assignee_619cec3af241500072a2aeb6" class="user-hover" rel="619cec3af241500072a2aeb6">Aario, Sami</a></td>
</tr>
<tr>
<td data-bgcolor="#f0f0f0" width="20%"><strong>Resolution:</strong></td>
<td data-bgcolor="#ffffff" data-valign="top" width="30%" data-nowrap="">Unresolved</td>
<td data-bgcolor="#f0f0f0" width="20%"><strong>Votes:</strong></td>
<td data-bgcolor="#ffffff" data-valign="top" width="30%" data-nowrap="">0</td>
</tr>
<tr>
<td data-bgcolor="#f0f0f0" width="20%"><strong>Labels:</strong></td>
<td colspan="3" id="labels-6982593-value" class="value" data-bgcolor="#ffffff" data-valign="top" data-nowrap="">None</td>
</tr>
<tr>
<td data-bgcolor="#f0f0f0" width="20%"><strong>Remaining Estimate:</strong></td>
<td colspan="3" data-bgcolor="#ffffff" data-valign="top" data-nowrap="" width="80%">Not Specified</td>
</tr>
<tr>
<td data-bgcolor="#f0f0f0" width="20%"><strong>Time Spent:</strong></td>
<td colspan="3" data-bgcolor="#ffffff" data-valign="top" data-nowrap="" width="80%">Not Specified</td>
</tr>
<tr>
<td data-bgcolor="#f0f0f0" width="20%"><strong>Original estimate:</strong></td>
<td colspan="3" data-bgcolor="#ffffff" data-valign="top" data-nowrap="" width="80%">Not Specified</td>
</tr>
</tbody>
</table>

  

<table class="grid" data-cellpadding="0" data-cellspacing="0" data-border="0" width="100%">
<colgroup>
<col style="width: 50%" />
<col style="width: 50%" />
</colgroup>
<tbody>
<tr>
<td data-bgcolor="#f0f0f0" width="20%" data-valign="top"><strong>Severity:</strong></td>
<td id="customfield_10141-6982593-value" class="value" data-bgcolor="#ffffff" width="80%">Medium</td>
</tr>
<tr>
<td data-bgcolor="#f0f0f0" width="20%" data-valign="top"><strong>Epic Link:</strong></td>
<td id="customfield_10014-6982593-value" class="value" data-bgcolor="#ffffff" width="80%"><a href="https://amd.atlassian.net/browse/SILOTIGER-1040" class="aui-label ghx-label-13">Qwen4-preview (Qwen3.8-Flash-Next) kernels</a></td>
</tr>
<tr>
<td data-bgcolor="#f0f0f0" width="20%" data-valign="top"><strong>Sprint:</strong></td>
<td id="customfield_10020-6982593-value" class="value" data-bgcolor="#ffffff" width="80%">FlyDSL Sprint 5</td>
</tr>
<tr>
<td data-bgcolor="#f0f0f0" width="20%" data-valign="top"><strong>Team:</strong></td>
<td id="customfield_10001-6982593-value" class="value" data-bgcolor="#ffffff" width="80%"><div id="customfield_10001-field" class="shorten">
<span></span>
</div></td>
</tr>
</tbody>
</table>

  

|                   |     |
|:-----------------:|-----|
|  **Description**  |     |

<table data-cellpadding="0" data-cellspacing="0" data-border="0" width="100%">
<colgroup>
<col style="width: 100%" />
</colgroup>
<tbody>
<tr>
<td id="descriptionArea"><h2 id="tldr"><span id="TL%3BDR"></span>TL;DR</h2>
<p>Ship <strong>FlyDSL</strong> kernels for the Qwen Sparse Attention (QSA) block in Qwen3.8-Flash-Next / Qwen4-preview (<code>qwen4_exp</code>): (1) paged indexer <strong>scorer</strong> plus fused top-k, (2) <strong>sparse GQA</strong> attend on the selected tokens.</p>
<p><strong>Primary bar is whatever AMD serving actually launches today</strong>, not the unmerged AITER PR. That is vLLM-vendored Triton plus HIP top-k in <code>qwen4_exp/amd/ops/qsa.py</code> (vLLM PR 53896). Beat that end-to-end on one QSA layer (<code>indexer + select + attend</code>) before claiming a win.</p>
<p><strong>Secondary bar</strong> is the unmerged AITER Triton/Gluon stack in <a href="https://github.com/ROCm/aiter/pull/4882" class="external-link" rel="nofollow noreferrer">ROCm/aiter#4882</a>: beat its portable Triton path, and beat its gfx950 Gluon path <strong>on the shapes Gluon actually dispatches</strong>. #4882 is not on <code>main</code> and is not the vLLM default; treat it as a competitor, not as production.</p>
<p>QSA is <strong>not</strong> every layer: hybrid is <code>GGGQ</code> x 12, so <strong>12 of 48</strong> layers (plus MTP, which can reuse indices).</p>
<p>Parent: <a href="https://amd.atlassian.net/browse/SILOTIGER-1040" class="external-link" rel="nofollow noreferrer">SILOTIGER-1040</a>. Sibling GR work is <a href="https://amd.atlassian.net/browse/SILOTIGER-1042" class="external-link" rel="nofollow noreferrer">SILOTIGER-1042</a> / <a href="https://amd.atlassian.net/browse/SILOTIGER-1041" class="external-link" rel="nofollow noreferrer">SILOTIGER-1041</a>; do not mix GR into this ticket.</p>
<h2 id="why-a-new-kernel"><span id="Whyanewkernel"></span>Why a new kernel</h2>
<p>At long context a QSA layer is two jobs with different scaling:</p>
<ol>
<li><strong>Indexer</strong> scores every <strong>complete 4-token block</strong> (<code>O(L/4)</code> keys), top-512, expand plus tail to at most 2051 token ids. Cost grows with <code>L</code>.</li>
<li><strong>Sparse GQA</strong> attends those ids on uncompressed K/V. Cost is almost independent of <code>L</code> once the budget is full (<code>K = 2048</code> tokens, <code>r = 4</code> blocks).</li>
</ol>
<p>The architecture report's 7.6x prefill / 4.9x decode at 1M is vs <strong>dense</strong> GQA and includes both. AMD serving today is vLLM-vendored Triton (scalar MQA loop, sparse GQA <code>num_stages=1</code> even on gfx950). #4882 is AITER Triton plus optional gfx950 Gluon; it is <strong>not on `main`</strong>, not wired as the vLLM default, and Gluon sparse GQA does <strong>not</strong> match the released Flash-Next GQA shape.</p>
<p>Do not retarget GLM/DeepSeek DSA: that scorer is <code>H=32</code> FP8 with per-head <code>w_h</code> over <strong>tokens</strong>. QSA is <code>H=4</code> BF16, no <code>w_h</code>, over <strong>mean-pooled blocks</strong>. Do not retarget sparse MLA prefill or SWA.</p>
<h2 id="two-shape-families-both-gated"><span id="Twoshapefamilies%28bothgated%29"></span>Two shape families (both gated)</h2>
<h3 id="a-flash-next-production-must-win"><span id="A%E2%80%94FlashNextproduction%28mustwin%29"></span>A — Flash-Next production (must win)</h3>
<p>Checkpoint <code>Qwen/Qwen3.8-Flash-Next</code> / FP8 twin. Config: <code>full_attention_interval=4</code>.</p>
<p>Indexer:</p>
<ul>
<li><code>indexer_n_heads = 4</code>, <code>indexer_kv_heads = 1</code>, <code>indexer_head_dim = 128</code></li>
<li><code>indexer_compress_ratio = 4</code>, <code>indexer_budget = 2048</code> so <code>K_B = 512</code> blocks</li>
<li>Score: <code>I_ib = sum_h ReLU(dot(q</code><span class="error"><code>[h]</code></span><code>, k_bar</code><span class="error"><code>[b]</code></span><code>))</code> for complete blocks only (<code>p_b + r - 1 &lt;= i</code>). No learned per-head weight. Serving may apply <code>1/sqrt(128)</code>; that cannot change top-k argmax.</li>
<li>Then <code>TopK</code> of 512, expand each block to 4 tokens, union the incomplete tail (0..3 tokens). Output width at most 2051.</li>
</ul>
<p>Sparse GQA:</p>
<ul>
<li>24 Q heads, 2 KV heads, so group size 12, <code>head_dim = 256</code>, partial RoPE 64, sigmoid output gate</li>
<li>Attend uncompressed paged K/V at the expanded token indices</li>
<li>BF16 activations (owner requires BF16 main QSA cache, not FP8 KV)</li>
</ul>
<h3 id="b-4882-gluon-validated-shapes-parity-competitor"><span id="B%E2%80%94%234882Gluonvalidatedshapes%28parity%2Fcompetitor%29"></span>B — #4882 Gluon-validated shapes (parity / competitor)</h3>
<p>Keep these so FlyDSL is measured on the same points Gluon was tuned for, not only the checkpoint.</p>
<p>Indexer (Gluon auto-dispatch in #4882): gfx950, Triton <code>&gt;= 3.6</code>, <code>H</code> is 4 or 8, <code>D = 128</code>. Their published bench: <code>rows=32</code>, <code>heads=4</code>, <code>head_dim=128</code>, <code>page_size=8</code>, <code>pages=512</code>. Reported p50: Triton 0.0184 ms, Gluon 0.0176 ms (<strong>1.047x</strong>).</p>
<p>Sparse GQA (Gluon auto-dispatch): <code>head_dim = 128</code>, GQA <strong>group size 5</strong>, <code>selection_width = 2051</code>. Their published bench: <code>num_tokens=16</code>, <code>num_query_heads=10</code>, <code>head_dim=128</code>, width 2051. Ordered: Triton 0.116 ms, Gluon 0.092 ms (<strong>1.25x</strong>). Randomized indices similar.</p>
<p><strong>Released Flash-Next GQA is group 12 and D=256, so #4882 Gluon will not auto-dispatch there.</strong> Family A vs #4882 is Triton-vs-FlyDSL. Family B vs #4882 is Triton-and-Gluon-vs-FlyDSL.</p>
<h2 id="kernel-structure-flydsl"><span id="Kernelstructure%28FlyDSL%29"></span>Kernel structure (FlyDSL)</h2>
<p>Two kernels, one op surface under <code>aiter/ops/flydsl/</code>.</p>
<p><strong>K1 — scorer plus fused top-k.</strong> Stream paged compressed index-K tiles, compute the ReLU-sum score, keep a <strong>local</strong> top-512 (or local top-k for family B), merge to a global 512 (or k) without writing <span class="error"><code>[rows, n_blocks]</code></span> FP32 scores. Same "score-plus-top-k" idea as DSA fused indexer work; different ABI (<code>H</code>, no <code>w_h</code>, block keys, <code>k=512</code>). Expand-plus-tail can live in K1's epilogue or K2's prologue.</p>
<p>Existing FlyDSL <code>fp8_mqa_logits</code> is the wrong kernel: it needs <code>H % 16 == 0</code>, is dense-only, and uses weighted ReLU. Pad-to-16 is a prototype only.</p>
<p><strong>K2 — sparse GQA.</strong> 24x2 (family A) or 10x2 / group 5 (family B). Selected positions are 512 runs of 4 plus a short tail on A, or a 2051-wide index list on B. Split-K as needed. Compile gfx942 and gfx950 separately if LDS/VGPR models differ; gfx950 should use the extra LDS (#4882 and vLLM both left <code>num_stages=1</code> on the vLLM AMD path).</p>
<p>Prefetch K/V for the selected runs. Do not union decode GEMV and prefill MFMA in one instantiation if that costs occupancy.</p>
<h2 id="baselines-live-path-first"><span id="Baselines%28livepathfirst%29"></span>Baselines (live path first)</h2>
<p>Report each named backend separately. Do not hide a loss to vLLM behind a win vs #4882, or the reverse.</p>
<h3 id="live-amd-serving-must-beat"><span id="LiveAMDserving%28mustbeat%29"></span>Live AMD serving (must beat)</h3>
<p>This is the active path in <code>qwen4_exp/amd/</code> today:</p>
<div class="table-wrap">
<table class="confluenceTable">
<tbody>
<tr>
<th class="confluenceTh">Step</th>
<th class="confluenceTh">What runs</th>
</tr>
&#10;<tr>
<td class="confluenceTd">Indexer Q/K GEMM</td>
<td class="confluenceTd">vLLM unquant linear: <code>wvSplitK</code> (tokens 1–5), hole at 6–9, else <code>F.linear</code> / hipBLASLt; gfx950 may add <code>wvSplitKrc</code> / AITER tgemm if those linears are on</td>
</tr>
<tr>
<td class="confluenceTd"><code>RMSNorm + partial MRoPE</code></td>
<td class="confluenceTd">unfused <code>GemmaRMSNorm + triton_mrope</code> (no <code>qsa_pre_indexer.py</code> on AMD)</td>
</tr>
<tr>
<td class="confluenceTd"><code>Compress r=4 + paged store</code></td>
<td class="confluenceTd">Triton <code>qsa_compress_groups_with_ratio</code> / <code>qsa_store_cache_rows</code></td>
</tr>
<tr>
<td class="confluenceTd">Paged MQA scores</td>
<td class="confluenceTd">Triton <code>_qsa_mqa_paged_kernel</code> — <strong>scalar per-head loop, no `tl.dot`</strong></td>
</tr>
<tr>
<td class="confluenceTd">Top-k (512 blocks)</td>
<td class="confluenceTd">HIP <code>top_k_per_row_decode</code> in <code>csrc/libtorch_stable/sampler.cu</code></td>
</tr>
<tr>
<td class="confluenceTd"><code>Expand + tail</code></td>
<td class="confluenceTd">Triton <code>_expand_qsa_indices_kernel</code></td>
</tr>
<tr>
<td class="confluenceTd">Sparse GQA</td>
<td class="confluenceTd">Triton <code>_qsa_sparse_paged_gqa_splitk_kernel</code>, <strong>`num_stages=1`</strong> (comment assumes 64 KiB LDS on gfx942 and gfx950)</td>
</tr>
</tbody>
</table>
</div>
<p>Files: <code>vllm/models/qwen4_exp/amd/ops/qsa.py</code> plus the Linear / HIP top-k call sites (vLLM PR 53896). Same Triton on gfx942 and gfx950 except skinny GEMM extras on gfx950.</p>
<p>End-to-end gate is this <strong>whole chain</strong> (launches, bytes, and fused K1/K2), not a single kernel vs its Triton twin in isolation.</p>
<h3 id="not-live-on-amd-reference-competitor-only"><span id="NotliveonAMD%28reference%2Fcompetitoronly%29"></span>Not live on AMD (reference / competitor only)</h3>
<ul>
<li>AITER #4882 Triton — <code>aiter/ops/triton/_triton_kernels/attention/qsa_paged_mqa_logits.py</code> and <code>qsa_sparse_paged_gqa.py</code>. Opt-in AITER; SGLang can use it eager-only. Competitor on family A GQA, where Gluon does not dispatch.</li>
<li>AITER #4882 Gluon — <code>aiter/ops/triton/_gluon_kernels/gfx950/attention/</code>. gfx950, Triton <code>&gt;= 3.6</code>. Indexer: <code>H</code> is 4 or 8, <code>D=128</code>. Sparse GQA: <code>D=128</code>, group size 5, width 2051. Forced <code>gluon</code> on a miss must error; <code>auto</code> falls back to Triton. If #4882 merges, retarget to <code>main</code>. Until then pin a PR head (runtime-tested parent <code>2462d5b64</code>; later heads may be docs-only).</li>
<li>vLLM NVIDIA QSA Triton — <code>qwen4_exp/nvidia/ops/qsa.py</code>: tensor-core MQA (<code>tl.dot</code>), sparse GQA <code>num_stages=2</code>, fused <code>qsa_pre_indexer</code>, CUDA <code>persistent_topk</code> / <code>cooperative_topk</code>. This is not the AMD acceptance bar. Keep it as an optional column in the result table so a FlyDSL win vs AMD Triton is not confused with catching NVIDIA.</li>
</ul>
<p>AITER fused MoE, tgemm, and vision FA may be on in the same process; they are <strong>not</strong> QSA baselines. Do not compare against FlashInfer TRT-LLM QSA (NVIDIA recipe even passes <code>--no-enable-flashinfer-autotune</code>).</p>
<h2 id="interface-family-a"><span id="Interface%28familyA%29"></span>Interface (family A)</h2>
<p>Indexer in: BF16 <code>q</code> <span class="error"><code>[M, 4, 128]</code></span>, paged compressed <code>k</code> (one KV head, D=128), block table, per-row causal complete-block bound, <code>eps</code> unused. Optional scale <code>1/sqrt(128)</code>.</p>
<p>Indexer out: <code>block_ids</code> <span class="error"><code>[M, 512]</code></span>, then token <code>indices</code> <span class="error"><code>[M, &lt;=2051]</code></span> after <code>expand+tail</code>.</p>
<p>GQA in: BF16 <code>q</code> <span class="error"><code>[M, 24, 256]</code></span>, paged <code>k</code>/<code>v</code> <span class="error"><code>[..., 2, 256]</code></span>, <code>indices</code>, softmax scale, sigmoid gate weights if fused into the epilogue.</p>
<p>GQA out: BF16 <code>o</code> <span class="error"><code>[M, 24, 256]</code></span> (pre-<code>o_proj</code>).</p>
<p><code>M</code> is flattened tokens. Decode <code>1..8</code> and prefill 512 / 2048 / 8192 plus at least one long-context length (32k or 128k) so indexer scaling is visible. gfx942 and gfx950.</p>
<h2 id="correctness"><span id="Correctness"></span>Correctness</h2>
<p>Independent fp32 oracle: block-causal ReLU-sum scores, exact top-512 (tie-break documented), <code>expand+tail</code>, then standard GQA on those positions. Cross-check vs vLLM Triton <code>qsa.py</code> and vs #4882 on shared shapes. Authoritative math: Qwen3.8-Next tech report §2.1 / QSA, <a href="https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf" class="external-link" rel="nofollow noreferrer">https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf</a> . Config: <a href="https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/config.json" class="external-link" rel="nofollow noreferrer">https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/config.json</a></p>
<p>Fusing top-k may change fp32 score order vs materializing full logits; gate K1 on the oracle with a documented tolerance, and require <strong>set equality</strong> of selected blocks (or a fixed tie policy).</p>
<h2 id="phases"><span id="Phases"></span>Phases</h2>
<ol>
<li>Harness: pin <strong>vLLM AMD live path</strong>, #4882 Triton, #4882 Gluon (where it dispatches), and a fp32 oracle. Report family A and family B separately. rocprof one real QSA layer (indexer through GQA) at short and long <code>L</code>, including HIP graph replay at decode.</li>
<li>FlyDSL K1 on family A (<code>H=4</code>) and family B (<code>H</code> is 4 or 8). Gate: beat live vLLM AMD (<code>MQA Triton + HIP top-k</code>) and beat #4882 Triton; beat Gluon on gfx950 where Gluon dispatches. No full score matrix.</li>
<li>FlyDSL K2 on family A (<code>D=256</code>, group 12) vs live vLLM AMD sparse GQA and #4882 Triton. Then family B (<code>D=128</code>, group 5) vs #4882 Triton <strong>and</strong> Gluon.</li>
<li>Wire <code>aiter/ops/flydsl/</code> plus vLLM <code>qwen4_exp</code> opt-in, same three-way backend idea as #4882 (<code>auto</code> / FlyDSL / Triton).</li>
<li>Optional: fuse <code>expand+tail</code> into K2; fuse <code>qsa_pre_indexer</code> (<code>Gemma RMSNorm + partial MRoPE + compress</code>) only after K1/K2 beat the bar.</li>
</ol>
<p>Warm up by duration. Interleave paired rounds. Do not claim a GQA win from the group-5 D=128 Gluon bench, and do not claim an indexer win from DSA <code>H=32</code> FP8 numbers.</p>
<h2 id="related-code"><span id="Relatedcode"></span>Related code</h2>
<ul>
<li>Baseline serving: <code>vllm/models/qwen4_exp/amd/ops/qsa.py</code>, NVIDIA twin <code>qwen4_exp/nvidia/ops/qsa.py</code> (tensor-core MQA, <code>num_stages=2</code>; not the AMD bar).</li>
<li>Competitor: <a href="https://github.com/ROCm/aiter/pull/4882" class="external-link" rel="nofollow noreferrer">https://github.com/ROCm/aiter/pull/4882</a> (<code>aiter/ops/triton/attention/qsa.py</code>).</li>
<li>Wrong ops: FlyDSL <code>fp8_mqa_logits</code> (<code>H % 16 == 0</code>), MLA sparse decode, SWA, AITER mHC, GR tickets above.</li>
<li>Neighbors: MoE tune CSVs <a href="https://github.com/ROCm/aiter/pull/5213" class="external-link" rel="nofollow noreferrer">https://github.com/ROCm/aiter/pull/5213</a> ; GDN is a different 36-layer problem.</li>
</ul>
<h2 id="open-questions"><span id="Openquestions"></span>Open questions</h2>
<ul>
<li>Where indexer vs GQA dominates on AMD at 8k / 32k / 128k / 1M; that sets which kernel to land first after the harness.</li>
<li>Whether FlyDSL K1 should emit block ids or already-expanded token ids.</li>
<li>Packed vs padded <code>M</code> (varlen) from a real vLLM prefill trace.</li>
<li>MTP IndexShare (<code>indexer.skip_topk</code>): GQA-only launch, no scorer.</li>
<li>#4882 merge timing: keep competing even if it lands; do not wait on it to start the harness.</li>
</ul>
<br />
</td>
</tr>
</tbody>
</table>

Generated at Tue Sep 15 11:07:39 UTC 2026 by Aario, Sami using Jira 1001.0.0-SNAPSHOT#100294-rev:b0380760584d59a844a637560bce3ca4d5aec84d.
