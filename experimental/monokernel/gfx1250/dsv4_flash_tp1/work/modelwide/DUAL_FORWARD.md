# Native-led dual forward diagnosis

Current focus: Flash TP1, C1, 1024/1024 shapes. `dual.py` keeps native outputs authoritative and runs candidate regions on identical inputs. Candidate attention caches and compressor states are private. `dual-server.sh` uses `--level 0 --enforce-eager`; original experimental monokernel flags are disabled.

Set `ATOM_DSV4_DUAL_REGION` to `qkv`, `mhc`, `attention_output_ffn`, or `compressor`. Empty disables diagnostics. All 43 blocks are hooked. Binding is deferred until first forward, after the complete parent-first model weight-preparation traversal.

## Confirmed results

`dual-qkv.jsonl` is the late-binding run: 860 layer/token observations. `dual-qkv-late-responses.json` matches all three untouched-native responses in `dual-native-level0-responses.json` exactly. QA projection: 860/860 exact; QA norm/FP8: 859/860 exact; QB projection: 859/860 exact. Worst final query relative RMS 0.000732152 (0.0732152%), layer13 position1027. Maximum KV relative RMS 0.000425224. This localizes one discrepancy to QA RMSNorm/quantization; it does not establish full model fused correctness.

The earlier layer38 position8 snapshot has one FP8 mismatch: native -144 versus candidate -128, scale2^-11. Candidate normalized value was exactly -0.06640625 (FP8 midpoint after scaling); CPU PyTorch calculation was -0.0664062649011612. It demonstrates threshold sensitivity but comes from the early-binding diagnostic run, not the accepted native-led input sequence.

## Excluded runs / diagnostic fixes

Initial diagnostic binding called `_triton_views_of_flydsl_weights` in the parent's weight-processing hook, before expert children finished preparation. This cached stale views and changed native generation even when candidate outputs were discarded. Deferring binding fixed all three tested generation responses. Earlier `dual-mhc`, `dual-attention_output_ffn`, `dual-qkv-initial`, and `*-early-binding` metrics are diagnostic only, not acceptance against an untouched native trajectory. These have now been rerun with late binding; see completed checks below.

Compressor diagnostic bugs: cache stride metadata was off by one; FP8 cache rows have a padded tail; state clones require their own strides rather than the original strided state-view strides. Initial compressor numerical errors are invalid. `*-late` and `*-raw` compressor runs crashed with illegal access before the clone-stride correction. Using raw checkpoint projection weights alone did not fix that crash. `*-clone` is the corrected-state-stride rerun. Do not interpret failed runs as kernel accuracy measurements.

## Files and next steps

- `summarize_dual.py` produces `dual-summary.json`.
- `dual_late_clients.py`, `dual_clone_clients.py`: three deterministic correctness requests (short prompt;1024 input IDs/8 outputs;1151 IDs/4 outputs). These are not performance benchmarks.
- `dual.py` saves both QA/QB intermediate paths and weights whenever QA quantization differs. JSONL rows record layer, position, and stage error.
- Verify compressor-clone responses against native; then rerun remaining regions with late binding.
- Once comparisons are qualified, substitute one region at a time while preserving native reference comparisons. Investigate runtime.py's early LayerWeights binding too; do not claim existing model-wide rollout correct.
- Performance still requires one server lifetime, two clients, only client2 reported. No new accepted serving performance from diagnostics; earlier faster full-model run failed generation quality.

## Late-binding checks completed

All three deterministic responses match untouched native for each of QKV, mHC, and attention/output/FFN diagnostic runs. Each stage has 860 layer/token observations. Worst relative RMS: mHC attention input0.0001850205; mHC FFN input0.0003254845; attention0.00003253554; output projection0.0001435562; FFN0.0004173414. These are local same-input comparisons, not independent full-fused quality validation.

Compressor clone-stride correction completed1240 state comparisons: all states exact, all health counters0; all three responses match native. Main cache122/125 exact, worst relative RMS0.000195798. Indexer scale105/105 exact. Initial indexer cache comparisons used the tensor's apparent132-element row stride, but AITER writes packed128-element rows before a per-block scale tail. Thus those cache comparisons are invalid. Fixed both diagnostic reference read and candidate serving write address to use `token * head_dim` for per-row FP8. The speculative uint8-store branch did not resolve the mismatch and is not the established root cause. Latest packed-cache rerun is separate.

Packed-cache rerun completed: indexer cache104/105 exact, worst relative RMS0.0000135239134; scales105/105 exact. Main cache122/125 exact, worst0.000195798. All1240 state comparisons exact. All three responses match untouched native. See current `dual-summary.json` and `dual-compressor-packed-responses.json`. Native-leading diagnostics are now validated for all implemented groups; remaining nonzero local numerical differences and full-fused rollout still need qualification.
