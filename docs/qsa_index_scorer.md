# QSA index scorer — `aiter.ops.qsa.qsa_paged_mqa_logits` (op contract v1)

`aiter.ops.qsa.qsa_paged_mqa_logits` scores compressed key groups for the Qwen sparse-attention
(QSA) indexer: paged MQA logits. It is the one public entry (the neutral op of #6469); this
(QSA) indexer: paged MQA logits. It is the one public entry (the neutral op of #6469), routed
through `aiter.plugin` as op `qsa_paged_mqa_logits` v1: an enabled out-of-tree provider may serve
a call; otherwise aiter's native implementation does. Existing callers and the default behaviour
are unchanged (nothing is enabled by default).

```
visible[t]   = clamp(min((pos[t] + 1) // ratio, ctx[req(t)] // ratio), min=0)   (0 if req(t) invalid)
logits[t][c] = sum_h relu(q[t,h,:] . k[c]) / divisor     if c < visible[t] and page(c) is valid
             = -inf                                       otherwise
k[c]         = cache[page_table[req, c // P], c % P, 0, :]                    (unquantized)
             = cache_e4m3[...] * scale[page_table[req, c // P], c % P, 0]     (fp8 + fp32 row scale)
columns      = max(1, min(width * P, max_columns))        (width * P when max_columns is None)
```

## Contract sheet (op `qsa_paged_mqa_logits`, version 1)

| field | contract |
|---|---|
| op name / version | `qsa_paged_mqa_logits`, version 1 (`@route(op="qsa_paged_mqa_logits", version=1)` in `aiter/ops/qsa.py`; baseline entry in `tests/plugin_contracts.json`, checked by the CPU `Kernel Provider Interface` gate). Backend-neutral. |
| public signature | `qsa_paged_mqa_logits(q, compressed_k_cache, page_table, token_to_request, query_positions, context_lens, compress_ratio=4, score_divisor=None, max_columns=None, row_starts=None, allow_flydsl=False, compressed_k_scale=None, stacked_decode=None, *, logits=None, visible_groups=None) -> tuple[Tensor, Tensor]`. The 12 leading parameters are the frozen ATOM scorer's; `stacked_decode` is main's Triton scheduling hint, default `None` = aiter decides (main's default was `False`); `logits` / `visible_groups` are optional caller-owned output buffers. A provider receives the 13 positional parameters as a tuple (defaults applied) and the two keyword-only ones as a mapping. |
| dtypes | `q` bf16 / fp16 / fp32; `compressed_k_cache` unquantized in q's dtype, FP8 E4M3 + `compressed_k_scale` fp32 `[pages, P, 1]`, or MXFP4 + E8M0 scale (Triton); metadata int32 / int64. |
| shapes / layouts | `q [rows, heads, head_dim]`; cache `[pages, P, 1, head_dim]`; `page_table [requests, width]`; `token_to_request`, `query_positions [rows]`, `context_lens [requests]`. |
| outputs / mutation | `logits` fp32 `[rows, columns]`, `visible_groups` int32 `[rows]` (the caller's buffers when supplied, filled in place); `row_starts`, if given, zero-filled. Inputs are never written. |
| tolerance / determinism | Reference: the frozen torch golden. Pass = `rtol = atol = 1e-4` over `valid_page & c < visible`, `visible_groups` exact, no NaN, every element outside the mask `-inf`, `row_starts == 0`. Implementations agree within the tolerance, not bitwise. |
| graph capture | The op is capture-safe when every implementation it reaches is: aiter's native kernels are; a provider must decide a call class before capture (the framework's eager warm-up) and must not bind per-call tensors, synchronize or JIT-compile inside capture. A captured call keeps whatever served it at capture (replay runs no Python; `AITER_PLUGINS=0` does not reach a captured graph). |
| error behaviour | Never raises for a valid call. Contract violations raise `ValueError` (Triton's own where Triton validates). A provider decline / no provider runs native; an exception inside a provider's `run()` propagates (no native retry). |
| coverage (native) | Triton (any arch it supports), FlyDSL page-16 prefill (`allow_flydsl`, > 256 rows, bf16, gfx9 only, see below), pure torch for every other valid call on any torch device (CPU included). |
| non-goals | top-k selection / index expansion (`qsa_select_paged_tokens`); tuned arch-specific kernels (providers). |

## Routing

```
aiter.ops.qsa.qsa_paged_mqa_logits(...)            # routed: aiter.plugin dispatch
  provider enabled (configure(source=..., enable=[...])) and its select() returns a candidate
      -> provider run()                            # e.g. the fused FlyDSL gfx1250 scorer
  otherwise (nothing enabled, AITER_PLUGINS=0, decline, select error)
      -> native: FlyDSL page-16 (gfx9, allow_flydsl) | Triton | pure torch
```

`aiter.ops.qsa.native_route(q, compressed_k_cache, page_table, allow_flydsl, compressed_k_scale)`
names the native implementation of a call (`native.flydsl_page16.v1` / `native.triton.v1` /
`native.torch.v1`); `aiter.plugin.explain("qsa_paged_mqa_logits", ...)` names the route
(provider / native and why). Nothing enabled (the default): the routed call costs one config
check before the native dispatch and records nothing.

**Tuned implementation.** The fused FlyDSL gfx1250 scorer (index page 4 / 8 / 16, bf16 and fp8 +
fp32 row scale; golden-verified profile; per-call-class decisions; warm-up / AOT) is the
out-of-tree provider `qsa-index-scorer-provider` (`coderfeli/plugin_poc`,
`plugins/aiter_qsa`). It is used only when a framework enables it, e.g.
`aiter.plugin.configure(source="atom ...", enable=["qsa-index-scorer-provider"])`; installing
it enables nothing.

## Native implementation

**FlyDSL page-16 prefill arch gate.** `native.flydsl_page16.v1`
(`aiter/ops/flydsl/kernels/mqa_logits/qsa_flydsl_layout.py`, the same kernel body as ATOM's
`qsa_flydsl_layout`) is a bf16 MFMA 16x16x16 kernel with wave64 shuffles. On gfx1250 its compile
aborts the whole process (SIGABRT, `LLVM ERROR: Cannot select: intrinsic
llvm.amdgcn.permlane32.swap`; reproduced compile-only, no device, with FlyDSL 0.3.4.1 and 0.4.0;
gfx950 compiles with both). The native dispatch therefore takes it only on devices whose arch
starts with `aiter.ops.qsa.FLYDSL_PAGE16_ARCHS` (`("gfx9",)`, read once per device index);
`allow_flydsl` calls on other devices (gfx1250, ...) run the Triton scorer.
`FlydslPage16ArchGate.test_gate_matches_compile_outcome` re-checks the abort in a child process
wherever FlyDSL is installed.

**Stacked decode (`stacked_decode`).** A Triton scheduling hint; results agree within the
tolerance either way. `None` (the default; ATOM passes only the 12 leading parameters) lets
aiter decide: a call served by `native.triton.v1` asks Triton for main's stacked decode scorer
(heads stacked into the MFMA M dimension, a request's keys read once per call) exactly when ATOM's
own scorer stacks heads with `ATOM_QSA_STACKED_DECODE_SCORER` at its default (on): at most 256
rows, `q`, cache and page table contiguous in their last dimension, power-of-two `heads`, and a
power-of-two `head_dim >= 16`; Triton then also needs the arch's `stacked_decode` tile
(gfx942 / gfx950 / gfx1250) and `BMT * heads >= 16`, as ATOM's scorer does, and otherwise scores
the call unstacked. Explicit `True` / `False` are passed to Triton as given
(`qsa_select_paged_tokens` keeps its own default `False` and passes it explicitly). The pure
torch path ignores the hint; a provider may ignore it (it is a Triton scheduling hint).

## Tests

- `op_tests/test_qsa_paged_mqa_logits_route_cpu.py` (CPU, no build): contract signature and
  baseline entry, routing through the real `aiter.plugin` with a fake provider (native by
  default, served, declined, `run()` error propagates, kill switch), native torch numerics vs
  a float64 reference, `stacked_decode` auto, page-16 arch gate.
- `tests/test_plugin_contracts.py` (the CPU `Kernel Provider Interface` gate) checks the
  routed signature against `tests/plugin_contracts.json`.
