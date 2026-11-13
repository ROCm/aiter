# `a2a_gemm_fused` v1 — AITER contract sheet

Status: **contract + reference native.** AITER owns this sheet and the public op
signature. The tuned kernel implementation, profile/config data and GPU evidence live in
the out-of-tree plugin_poc provider (`a2a-gemm-provider`). This tree keeps a slow
pure-PyTorch reference so the op remains usable when no provider is installed; the
provider is the fast path.

Fingerprint: `388d9816496c260e` = `sha256(SIGNATURE)[:16]`, with `SIGNATURE` the exact
string in §2. `tests/plugin_contracts.json` pins it and the test gate
`tests/test_plugin_contracts.py` stops it changing silently.

## 1. Ownership

| party | owns |
|---|---|
| AITER | the operation name/version, this signature, routing, the `aiter.plugin` interface, the routed-op test, the reference native |
| provider | the tuned kernel, profile/config data, readiness/lifecycle, coverage and qualification evidence, multi-rank exchange and graph-capture behaviour |
| framework / user | enablement: `aiter.plugin.configure(source=..., enable=[...])`; `AITER_PLUGINS=0` can only turn routing off |

Installing a provider makes it available, never active. A provider is asked per call; a
provider that declines is normal and the reference native serves the call. The reference
is correct within the limitation in §3, but deliberately slow; production performance
requires the provider. This PR relies on `aiter.plugin` as merged on main and adds no
interface files; the op registers against the stock interface.

## 2. Public signature (exact)

```text
x:positional, x_scale:positional, w:positional, w_scale:positional, out:positional,
total_tokens:keyword-only, bucket_m:keyword-only=None, registered:keyword-only=False,
group:keyword-only=None, stream:keyword-only=None, -> None
```

```python
a2a_gemm_fused(
    x, x_scale, w, w_scale, out,
    *,
    total_tokens,
    bucket_m=None,
    registered=False,
    group=None,
    stream=None,
) -> None
```

The routed wrapper preserves this signature; the five required parameters arrive at a
provider as a positional tuple, and the keyword-only tail as a read-only mapping with the
signature defaults applied (`bucket_m=None`, `registered=False`, `group=None`,
`stream=None`); resolving `bucket_m=None` to `M_real` is the provider's job, exactly as
in the reference native. No `*args`/`**kwargs` in the public contract.

| parameter | meaning |
|---|---|
| `x` | fp8_e4m3fn `[4*M_real, 2048]`, this rank's already-quantized **send** buffer |
| `x_scale` | fp32 `[M_real]`, one scale per token, shared over **all 8192** channels |
| `w` | fp8_e4m3fn `[6144, 8192]`, replicated weight, AITER `(16,16)` preshuffled |
| `w_scale` | fp32 `[6144]`, one scale per output channel |
| `out` | bf16 `[bucket_m, 6144]`, caller-owned output; rows `[M_real:bucket_m]` untouched |
| `total_tokens` | keyword-only required, must equal `4*M_real` = `x.shape[0]`; `x` is never padded |
| `bucket_m` | keyword-only, output-capacity stride only; defaults to `M_real`; `M_real <= bucket_m <= 16384`; sizes `out`, never a compute shape |
| `registered` | keyword-only producer-owned-pointer hint, provider policy only, never a semantic argument |
| `group` | execution context: the SP4 process group (world 4); `None` for the single-process reference |
| `stream` | execution context: current stream; the op never creates one |

`M_real` is the local real token count: `x.shape[0] = 4*M_real`. `K = 8192`, `N = 6144`,
per-source K quarter = `2048`. Domain: gfx950, SP=4, TP=1, single node, logical
`M_real = 1..16384` including Pad M.

## 3. Semantics (v1)

All-to-all → GEMM, after quantization. The quantization (amax exchange, per-token scale
computation, fp8 cast) is **outside** the op: `x` arrives already quantized.

1. The full GEMM is `A[M_real, 8192] @ W[6144, 8192].T`, FP32 accumulation, no bias,
   BF16 RNE output. `W` is replicated on every rank; `w` is stored in AITER's `(16,16)`
   `shuffle_weight` layout, so the logical weight is recovered with
   `w.view(N/16, K/32, 2, 16, 16).permute(0, 3, 1, 2, 4).reshape(N, K)`.
2. `x` is the SP4 **send** buffer: rank `s` holds destination blocks `d = 0..3`, and rows
   `[d*M_real : (d+1)*M_real]` are destination `d`'s K quarter from source `s`. Rank `d`
   receives, from every source `s`, that source's block `d`. The assembled operand is
   therefore
   `A = concat_{s=0..3} x_s[d*M_real : (d+1)*M_real, :]` along K, in source order:
   source `s` contributes columns `[s*2048 : (s+1)*2048)`.
3. `A` has **one** fp32 scale per token (`x_scale[m]`), not one per 2048-channel shard:
   the token's scale is computed over all 8192 channels before shard quantization and is
   applied once to the completed 8192-wide row. `w_scale[n]` is per output channel.
4. Epilogue: `out[m, n] = bf16_rne( (sum_k A_fp32[m, k] * W_fp32[n, k]) *
   x_scale[m] * w_scale[n] )`, with both scales applied in FP32 before the single BF16
   rounding.
5. `out` is the caller's buffer. Only rows `[0 : M_real)` are written; rows
   `[M_real : bucket_m)` are never read or written. A row whose `x_scale` is zero yields
   exact BF16 zero.

**`total_tokens == x.shape[0]` is part of the contract**: it is the exact, unpadded row
count of `x` (`= 4*M_real`), and the native rejects any other value. `x` itself is never
padded — Pad M is an output-capacity notion only. `bucket_m` solely strides the
caller-owned `out` (`out.shape[0] = bucket_m`); it is not a compute shape and never
changes which rows are computed. A provider that stages per-destination segments must use
source offsets of real M and destination/window offsets of bucket M; the routed op itself
only sees the real-M source layout.

## 4. `registered` ownership rule

`registered=True` says "`x` is the producer-owned registered-pool pointer for this call",
which lets a provider skip its staging copy. It is honoured only for the provider's
frozen in-band class (local `M_real` in `2048..8192` with its producer view); for any
other class the provider uses its staged ingress or declines. It is **never** a semantic
argument: the result is identical for `True` and `False`, and the reference native
ignores it after type-checking it is a `bool`.

## 5. Decline-safe native reference (fallback, not tuned)

When no provider serves, `aiter.ops.gemm_fusion.a2a_gemm.op.a2a_gemm_fused` runs a
deliberately slow pure-torch reference of the same v1 arithmetic:

1. undo the `(16,16)` preshuffle with the same `view/permute` math as the
   `sp_qkv_fused` native, decode fp8 to FP32 exactly;
2. **simulate** the SP4 exchange. A lone process cannot exchange, so the four source
   buffers are assumed byte-identical to the local send buffer (the same rank-uniform
   stand-in the `sp_qkv_fused` reference makes); destination block `d` is taken from each
   simulated source and concatenated along K into `[M_real, 8192]`;
3. FP32 reference GEMM, apply `x_scale` per row and `w_scale` per column in FP32, round
   once to BF16 RNE, write only `out[:M_real]`.

The native is a correctness fallback for a decline, **not** a performance path. It does
not perform a real collective and is not a distributed oracle: with rank-distinct inputs
the simulated exchange is a layout/arithmetic reference only. Real multi-rank exchange,
graph capture/replay, bucket dispatch and all performance belong to the provider.

## 6. Provider obligations

- Implement the `aiter.plugin.KernelProvider` protocol (`name`, `api_versions`,
  `op_versions`, `select`).
- `select(op, args, kwargs)` is pure, cheap, deterministic and never raises; return a
  `KernelCandidate` to serve or `None` to decline. Declining is normal operation.
- The candidate identifies the provider/kernel/artifact and supplies a `run` callable
  accepting the canonicalized contract shape (5 positional + keyword-only mapping).
- A provider that declines is skipped; if none serves, AITER runs the reference native.
- Once a provider's `run()` starts, exceptions propagate; AITER does **not** retry native
  (side effects must not repeat).
- A candidate whose `run` cannot accept the routed contract shape is rejected by the
  interface (contract mismatch) and treated like a provider failure.
- Coverage, kernel evidence, multi-rank correctness, lifecycle (`warmup`/`prepare`), and
  graph-capture readiness are provider responsibilities. AITER records
  `KernelCandidate.qualification` as self-reported provenance, not a certification.

## 7. Error behaviour

| condition | behaviour |
|---|---|
| `x` not fp8_e4m3fn / not `[4*M_real, 2048]` / not contiguous | `ValueError` from the native before any write |
| `x.shape[0]` not divisible by 4, or `M_real` outside `1..16384` | `ValueError` naming the constraint |
| `total_tokens != 4*M_real` or not an `int` | `ValueError` before any write |
| `bucket_m` not `None`/`int`, or outside `M_real..16384` | `ValueError` before any write |
| `registered` not a `bool` | `ValueError` before any write |
| `x_scale`/`w`/`w_scale`/`out` shape or dtype violation | `ValueError` naming the tensor, before any write |
| non-4 SP world in `group` | `NotImplementedError` naming the case |
| `group` rank outside `0..3` | `ValueError` |
| no enabled provider serves the call (none installed, all decline, or `AITER_PLUGINS=0`) | reference native runs; returns `None` |
| provider `select` raises / invalid candidate / contract mismatch | that provider is rejected; next enabled provider is asked; otherwise reference native runs |
| provider `run` raises after starting | exception propagates; no native retry |
| `import aiter` with no provider installed | works; `aiter.a2a_gemm_fused` is a lazy export |

`AITER_PLUGINS=0` forces the reference native; it cannot enable a provider. Enablement
has exactly one source, `aiter.plugin.configure(...)`.

## 8. What is not here

No tuned kernel, no device-side `csrc` code, no profile JSON, no tuner or bench scripts,
no multi-rank transport: those live in the out-of-tree provider. The native reference
above is the only in-tree implementation and it is CPU-testable only. The checks in this
PR are CPU-only contract/routing/reference tests; GPU numerics, byte-exact transport,
capture/replay and performance evidence belong to the provider repo. The provider is the
only party that may claim coverage; AITER records what a provider declares and answers
`explain()`/`status()` about routing only.
