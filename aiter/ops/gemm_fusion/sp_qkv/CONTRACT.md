# `sp_qkv_fused` v1 — AITER contract sheet

Status: **contract + reference native.** AITER owns this sheet and the public op
signature. The tuned kernel implementation, profile/config data and GPU evidence live in
the out-of-tree plugin_poc provider branch `feat/sp-qkv-aiter-provider`. This tree keeps a
slow pure-PyTorch reference so the op remains usable when no provider is installed; the
provider is the fast path.

## 1. Ownership

| party | owns |
|---|---|
| AITER | the operation name/version, this signature, routing, the `aiter.plugin` interface, the routed-op test, the reference native |
| provider | the tuned kernel, profile/config data, readiness/lifecycle, coverage and qualification evidence |
| framework / user | enablement: `aiter.plugin.configure(source=..., enable=[...])`; `AITER_PLUGINS=0` can only turn routing off |

Installing a provider makes it available, never active. A provider is asked per call; a
provider that declines is normal and the reference native serves the call. The reference
is correct but deliberately slow; production performance requires the provider.

This PR relies on `aiter.plugin` as merged on main (#7465 / #7466), including
`API_VERSION`, `route`, `KernelCandidate`, `configure`, `explain`, `status` and
`_reset_for_testing`. It adds no interface files; the op registers against the stock
main interface.

## 2. Public signature (exact)

```python
sp_qkv_fused(
    q, x, x_scale, w, w_scale, q_norm_w, k_norm_w, cos_cache, sin_cache,
    positions, slot_mapping, k_cache, v_cache, k_scale, v_scale,
    *,
    mode,
    k=None, v=None, index_q=None,
    index_q_norm_w=None, index_k_norm_w=None, index_cache=None,
    eps=1e-6, group=None, stream=None,
) -> None
```

The routed wrapper preserves this signature; the 15 required parameters arrive at a
provider as a positional tuple with defaults applied, and the keyword-only tail as a
read-only mapping. No `*args`/`**kwargs` in the public contract.

| parameter | meaning |
|---|---|
| `q` | bf16 `[T, 16, 128]`, in-place output |
| `x` | fp8 e4m3fn `[M, 6144]` activation |
| `x_scale` | fp32 `[M]` or `[M, 1]` |
| `w` | fp8 e4m3fn `(16,16)`-preshuffled `[rows >= N, 6144]` |
| `w_scale` | fp32 `[N]` or `[N, 1]` |
| `q_norm_w`, `k_norm_w` | fp32 `[128]` RMSNorm weights (Gemma: `1 + w`) |
| `cos_cache`, `sin_cache` | bf16 rotary tables, 32 values per position (rotary dim 64) |
| `positions`, `slot_mapping` | int64 `[>= T]`; pad rows are position 0 / slot `-1` |
| `k_cache`, `v_cache` | fp8 golden-shuffle KV caches; page 128 dense, page 16 indexer/reuse |
| `k_scale`, `v_scale` | fp32 KV cache scales |
| `mode` | keyword-only required: `"dense"` \| `"indexer"` \| `"reuse"` |
| `k`, `v` | bf16 `[T, 1, 128]` outputs, required for dense/reuse |
| `index_q` | bf16 `[T, 1, 128]` output, required for indexer |
| `index_q_norm_w`, `index_k_norm_w` | fp32 `[128]`, required for indexer |
| `index_cache` | fp8 flat page-128 index cache, required for indexer |
| `eps` | Gemma-RMSNorm epsilon, default `1e-6` |
| `group` | torch.distributed ProcessGroup of the SP group (execution context, world 4) |
| `stream` | current stream (execution context; the op never creates one) |

`M = x.shape[0]`; `T = 4*M` for the SP world 4 contract; `N = 9216` for dense, `9856`
for indexer/reuse; `K = 6144`.

## 3. Native reference semantics (fallback, not tuned)

When no provider serves, the native body performs the same contract operations with
plain PyTorch:

1. dequantise `x` by its per-row `x_scale`; dequantise `w` by `w_scale`; undo the
   `(16,16)` `shuffle_weight` permutation; compute the fp32 GEMM and round the scaled
   epilogue to bf16;
2. treat `T = 4*M` output rows as four **source token blocks**: `t // M` selects the
   source block whose token row this is, `t % M` the local token index in `x` used to
   compute it. Every block carries the **local rank's** head shard: q heads
   `rank*16..rank*16+15`, the local rank's k/v/iq head, and a replicated ik head. The
   reference does not exchange; with byte-identical inputs across ranks all four blocks
   are equal (value-neutral), which is the invariant the GPU e2e compares;
3. per 128-wide head: Gemma RMSNorm (`x * rsqrt(mean(x^2)+eps) * (1+weight)`), then
   partial NeoX RoPE on dims 0..63 using `cos_cache.stride(0)` and the position row;
4. write `q`, and `k`/`v` or `index_q` in place (bf16). KV cache values are amax-scaled
   fp8 in the golden-shuffle layout; index cache is flat `slot*128 + d` fp8. Cache
   writes are skipped for `slot_mapping < 0`; `q` is still written for every row.

This is a reference: it does not perform the SP exchange, is not graph-capture aware and
must not be used for performance measurements. GPU numerics/perf and multi-rank
behaviour belong to the provider repo.

## 4. Provider obligations

- Implement the `aiter.plugin.KernelProvider` protocol (`name`, `api_versions`,
  `op_versions`, `select`).
- `select(op, args, kwargs)` is pure, cheap, deterministic and never raises; return a
  `KernelCandidate` to serve or `None` to decline. Declining is normal operation.
- The candidate identifies the provider/kernel/artifact and supplies a `run` callable
  accepting the canonicalized contract shape.
- A provider that declines is skipped; if none serves, AITER runs the reference native.
- Once a provider's `run()` starts, exceptions propagate; AITER does **not** retry
  native (side effects must not repeat).
- A candidate whose `run` cannot accept the routed contract shape is rejected by the
  interface (contract mismatch) and treated like a provider failure.
- Coverage, kernel evidence and GPU correctness are provider responsibilities; AITER
  records `KernelCandidate.qualification` as self-reported provenance, not a
  certification.

## 5. Error behaviour

| condition | behaviour |
|---|---|
| `mode` not in `{"dense","indexer","reuse"}` | `ValueError` from the public op before dispatch |
| mode-specific required tensors missing | `ValueError` from the public op |
| tensor dtype/shape/layout violation in the native reference | `ValueError` naming the tensor (before any write) |
| valid contract but unsupported native case (e.g. non-4 SP world) | `NotImplementedError` naming the case |
| no enabled provider serves the call (none installed, all decline, or `AITER_PLUGINS=0`) | reference native runs; returns `None` |
| provider `select` raises / returns an invalid candidate / contract mismatch | that provider is rejected; next enabled provider is asked; otherwise reference native runs |
| provider `run` raises after starting | exception propagates; no native retry |
| `import aiter` with no provider installed | works; `aiter.sp_qkv_fused` is a lazy export |

`AITER_PLUGINS=0` forces the reference native; it cannot enable a provider. Enablement
has exactly one source, `aiter.plugin.configure(...)`.

## 6. What is not here

This branch deliberately contains no tuned `gemm_fusion` framework, no
`csrc/gemm_fusion` kernels, no profile JSONs and no tuner/bench scripts. Those were
removed from AITER and moved to the plugin_poc provider. The native reference above is
the only in-tree implementation; GPU validation (byte-exactness, multi-rank, graph
replay) belongs to the provider repo. The checks in this PR are CPU-only
contract/routing/reference tests.
