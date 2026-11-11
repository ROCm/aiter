# DeepSeek-V4.1 decode layer — AITER contract sheet (v1)

Two routed ops, one decoder layer of a DeepSeek-V4.1 decode step:

| op | version | code |
|---|---|---|
| `deepseek_v41_decode_attn_pre` | 1 | `aiter/ops/deepseek_v41_decode.py` |
| `deepseek_v41_decode_layer_post` | 1 | same file |

The native implementation is `aiter/ops/deepseek_v41_decode_ref.py` (pure torch,
correctness only). A provider serves a call by implementing exactly what follows,
or declines it. The split is the one ATOM's mono decode uses (K1 / K2 = K2a + K2b),
so a fused provider runs each op as one persistent launch; nothing in the
contract depends on ATOM.

## 1. Order of calls, and what the framework still owns

A layer is `attn_pre` then `layer_post`, on one stream, every TP rank in the same
order. Between them the framework runs what the ops do **not** cover:

- the KV compressor (FULL / REINDEX layers read `normed_out`, the q latent),
- the indexer's scoring and top-k (it reads `index_q_out` / `index_q_scale_out` /
  `index_weights_out`) — its selection becomes `topk_rows` / `topk_count`,
- Engram (before `attn_pre`, settling the state, so that call does not fold).

`layer_index` (keyword-only, both ops) is the layer's position in the step's layer
sequence, 0-based. **A call with `layer_index == 0` to `attn_pre` begins a new
step**: within one step on one TP group, each `layer_index` is used once per op and
the calls go in ascending order. (A fused provider tags its in-kernel hand-offs by
it and clears them at a step's start.)

Notation: `S` rows of this step (1 ≤ S; a provider may cover fewer, see §8),
`HC` mHC streams (4), `H` hidden (5120), `Hl` this rank's query heads
(64 / TP), `D` head dim (512), `QR` q-latent rank (1280), `G` this rank's `wo_a`
groups (8 / TP), `OR` o rank (1024), `E` routed experts (384), `I` this rank's
routed intermediate width padded to 128 (`ceil128(2304 / TP)`), `SI` this rank's
shared-expert width (2304 / TP), `IH`/`ID` indexer heads/dim (32 / 128).

## 2. Formats

- **bf16** activations; **fp32** gates and scales.
- **FP8**: `torch.float8_e4m3fn` (OCP E4M3).
- **E8M0** scales: `torch.float8_e8m0fnu` (or their `uint8` bytes); value
  `2^(code − 127)`.
- **Dense FP8 weight** `[N, K]`: E4M3 bytes in aiter's `shuffle_weight(w, (16, 16))`
  order, with **32×32 block** E8M0 scales `[N/32, K/32]`, row-major, not shuffled.
  Used by `wqkv_a`, `wq_b`, `index_wq_b`, `wo_a`, `wo_b`, `shared_w13`, `shared_w2`.
- **Routed experts (MXFP4)**: `w13` `[E, 2I, H/2]`, `w2` `[E, H, I/2]` E2M1 pairs
  (`torch.float4_e2m1fn_x2` or `uint8`, low nibble = even element), in aiter's
  `shuffle_weight(w, (16, 16), is_guinterleave=True, gate_up=…)` order (gfx950's
  `moe_shuffle_weight`); the unshuffled `w13` is gate rows `[0, I)` then up rows
  `[I, 2I)`. Scales 1×32 E8M0 as aiter `shuffle_scale(s2d, E, is_guinterleave=True,
  gate_up=…)`: `w13_scale` `[E·2I, ceil8(H/32)]`, `w2_scale` `[E·H, ceil8(I/32)]`.
  Padded intermediate rows/columns are zero.
- **KV cache** (`kv_cache`), 2-D, row `r` is key = value of one cached token:
  bf16 `[R, D]`, or MXFP8 `uint8 [R, D + D/32]` (D E4M3 bytes then D/32 E8M0).
- **CEIL rule** (MXFP8 activation quant): per 32, `amax = max(|x|, 1e-10)`,
  `code = ceil_pow2(amax / 448)` as a biased exponent, values
  `e4m3(clamp(x · 2^−(code−127), ±448))`.
- **ROUND rule** (the q latent only): `p2 = (bits(amax) + 0x200000) & 0x7F800000`,
  scale `2^(log2(p2) − 8)` (exponent clamped ±127), values `e4m3(x / scale)`.
- **RoPE**: GPT-J interleaved pairs on the **last 64 dims** of a head; tables
  `cos_cache`/`sin_cache` `[P, 32]` (bf16), indexed by `positions` (int64 `[S]`).

## 3. Communication, streams, capture

- `group`: the TP group (`torch.distributed.ProcessGroup`, or a framework
  coordinator exposing `.device_group` and `.cpu_group`). `None` means TP 1.
  Execution context, not policy: it fixes the TP size the weights are sharded for.
  Both all-reduces of `layer_post` happen **inside** the op. Their value is the
  fp32 sum of every rank's bf16 partial in rank order, rounded to bf16 once.
- `stream`: the caller's stream; `None` means the current stream. The op never
  creates a stream.
- A provider owns any symmetric / peer memory it needs and sets it up
  collectively on its first call per group (all ranks). The interface has no
  prepare hook: **call each shape once outside graph capture before capturing**
  (dispatch does no capture check). The native path allocates only stream-ordered
  temporaries.
- The op allocates no caller-visible workspace. Every output is caller-allocated.

## 4. `deepseek_v41_decode_attn_pre` (K1)

```
deepseek_v41_decode_attn_pre(
    residual, pre_mix, hc_fn, hc_scale, hc_base, attn_norm_weight,
    wqkv_a, wqkv_a_scale, q_norm_weight, kv_norm_weight, wq_b, wq_b_scale,
    cos_cache, sin_cache, positions, kv_cache, kv_slots,
    pre_out, post_out, comb_out, q_out,
    *, layer_index,
    sublayer_output=None, post_mix=None, comb_mix=None, residual_out=None,
    normed_out=None, q_latent_out=None, q_latent_scale_out=None,
    index_wq_b=None, index_wq_b_scale=None, index_weights_proj=None,
    index_q_out=None, index_q_scale_out=None, index_weights_out=None,
    aux_out=None, eps=1e-20, hc_eps=1e-6, group=None, stream=None) -> None
```

| argument | shape | dtype | role |
|---|---|---|---|
| `residual` | `[S, HC, H]` | bf16 | mHC streams entering the seam |
| `pre_mix` | `[S, HC]` | fp32 | the incoming collapse weights |
| `hc_fn` / `hc_scale` / `hc_base` | `[HC(HC+2), HC·H]` / `[3]` / `[HC(HC+2)]` | fp32 | attention seam's mix |
| `attn_norm_weight` | `[H]` | bf16 | |
| `wqkv_a`, `_scale` | `[QR + D, H]`, `[(QR+D)/32, H/32]` | FP8, E8M0 | dense FP8 |
| `q_norm_weight` / `kv_norm_weight` | `[QR]` / `[D]` | bf16 | |
| `wq_b`, `_scale` | `[Hl·D, QR]`, `[Hl·D/32, QR/32]` | FP8, E8M0 | dense FP8 |
| `kv_slots` | `[S]` | int32 | cache row for the token's window row; **−1 = no write** |
| `pre_out`/`post_out`/`comb_out` | `[S, HC]`/`[S, HC]`/`[S, HC, HC]` | fp32 | out: this seam's gates |
| `q_out` | `[S, Hl, D]` | bf16 | out: rotated query |
| `sublayer_output`, `post_mix`, `comb_mix` | `[S, H]`, `[S, HC]`, `[S, HC, HC]` | bf16, fp32 | the owed fold (all or none) |
| `residual_out` | `[S, HC, H]` | bf16 | out, **iff folding**; must not alias `residual` |
| index group (all or none) | | | FULL / REINDEX layers |
| `normed_out` | `[S, H]` | bf16 | out: the normed seam row |
| `q_latent_out`, `q_latent_scale_out` | `[S, QR]`, `[S, QR/32]` | FP8, E8M0 | out: ROUND-rule q latent |
| `index_wq_b`, `_scale` | `[IH·ID, QR]`, `[IH·ID/32, QR/32]` | FP8, E8M0 | dense FP8 |
| `index_weights_proj` | `[IH, H]` | bf16 | |
| `index_q_out`, `index_q_scale_out` | `[S, IH, ID]`, `[S, IH]` | FP8, fp32 | out: per-(row, head) scaled FP8 index query |
| `index_weights_out` | `[S, IH]` | fp32 | out: scaled head weights |
| `aux_out` | `[S, H]` | bf16 | out: `bf16(mean_h R[h])` (a drafter's tap) |

Semantics, in order:

1. **Seam** (mHC, `pre_delayed`). Fold if given:
   `R[j] = bf16(post_mix[j]·y + Σ_i comb_mix[i, j]·residual[i])` (i = source
   stream, j = destination); else `R = residual`. `x = R` flattened (fp32);
   `mix = (x @ hc_fnᵀ)·rsqrt(mean(x²) + eps)`;
   `pre = σ(mix[0:4]·s0 + b[0:4]) + hc_eps`; `post = 2·σ(mix[4:8]·s1 + b[4:8])`;
   `C = mix[8:24]·s2 + b[8:24]` as `[4, 4]` → row softmax `+ hc_eps`, column
   normalize, then 19 × (row, column) normalize, each with `+ hc_eps`.
   `collapsed = bf16(Σ_h pre_mix[h]·R[h])`; `normed = bf16(collapsed·rsqrt(mean
   (collapsed²) + eps)·attn_norm_weight)`. Writes `residual_out = R` (fold only),
   `pre_out`, `post_out`, `comb_out = C`.
2. `qkv = bf16(CEIL(normed) @ wqkv_aᵀ)`; `q_lora = qkv[:QR]`, `kv = qkv[QR:]`.
3. `qr = ROUND(rmsnorm(q_lora)·q_norm_weight)` (fp32, not rounded before quant);
   `kv = bf16(rmsnorm(kv)·kv_norm_weight)`.
4. `q_out = bf16(RoPE(bf16(qr @ wq_bᵀ)))` per head.
5. Window row: `CEIL(bf16(RoPE(kv)))` written at `kv_cache[kv_slots[t]]` for
   `kv_slots[t] ≥ 0`: bf16 cache — the dequantized (QAT) row; MXFP8 cache — the
   values then the codes. No other cache row changes.
6. Index group: `normed_out = normed`, `q_latent_out/scale = qr`;
   `iq = bf16(RoPE(bf16(qr @ index_wq_bᵀ)))` per index head;
   `s = max(amax(iq), 1e-10)/448` per (row, head), `index_q_out = e4m3(clamp(iq/s,
   ±448))`, `index_q_scale_out = s`; `w = bf16(normed @ index_weights_projᵀ)`,
   `index_weights_out = (w·s)·ID^−½·IH^−½` (fp32).
7. `aux_out = bf16(mean over HC of R)`.

`eps` is the RMS epsilon (1e-20 for V4.1), `hc_eps` the mHC epsilon (1e-6).

## 5. `deepseek_v41_decode_layer_post` (K2 = K2a + K2b)

```
deepseek_v41_decode_layer_post(
    q, kv_cache, topk_rows, topk_count, window_meta, attn_sink,
    positions, cos_cache, sin_cache, wo_a, wo_a_scale, wo_b, wo_b_scale,
    residual, pre_mix, post_mix, comb_mix, hc_fn, hc_scale, hc_base, ffn_norm_weight,
    gate_weight, gate_bias, w13, w13_scale, w2, w2_scale,
    shared_w13, shared_w13_scale, shared_w2, shared_w2_scale,
    residual_out, pre_out, post_out, comb_out, ffn_out,
    *, softmax_scale, ring_slots, layer_index, ffn_normed_out=None,
    top_k=6, routed_scaling_factor=1.5, swiglu_limit=10.0,
    eps=1e-20, hc_eps=1e-6, group=None, stream=None) -> None
```

| argument | shape | dtype | role |
|---|---|---|---|
| `q` | `[S, Hl, D]` | bf16 | `attn_pre`'s `q_out` |
| `topk_rows` | `[S, T]`, 1 ≤ T ≤ 512 | int32 | cache rows of the selected compressed keys |
| `topk_count` | `[S]` | int32 | valid entries of a row of `topk_rows` (0..T) |
| `window_meta` | `[S, 4]` | int32 | `(first, count, reserved, base)` of the window |
| `attn_sink` | `[Hl]` | fp32 | |
| `wo_a`, `_scale` | `[G·OR, (Hl/G)·D]`, `[G·OR/32, (Hl/G)·D/32]` | FP8, E8M0 | dense FP8, group-major rows |
| `wo_b`, `_scale` | `[H, G·OR]`, `[H/32, G·OR/32]` | FP8, E8M0 | dense FP8 |
| `residual`, `pre_mix`, `post_mix`, `comb_mix` | as `attn_pre` | | the attention seam's `R` and gates (its outputs) |
| `hc_fn`, `hc_scale`, `hc_base`, `ffn_norm_weight` | as `attn_pre` | | FFN seam's |
| `gate_weight`, `gate_bias` | `[E, H]`, `[E]` | bf16, fp32 | router; bias = selection only |
| `w13`, `w13_scale`, `w2`, `w2_scale` | §2 | MXFP4 | routed experts (TP shard of every expert) |
| `shared_w13`, `_scale` / `shared_w2`, `_scale` | `[2SI, H]` / `[H, SI]` | FP8 | gate rows then up rows |
| `residual_out`, `pre_out`, `post_out`, `comb_out` | as `attn_pre` | | out: FFN seam |
| `ffn_out` | `[S, H]` | bf16 | out: MoE output, reduced over `group` |
| `ffn_normed_out` | `[S, H]` | bf16 | out, optional: the MoE input |

Keys of row `t`, in this order: `topk_rows[t, :topk_count[t]]`, then the window
`base + (first + j) mod ring_slots` for `j < count` (oldest first). `base` already
includes the layer's ring start. Semantics:

1. **Attention** (MQA, K = V = the cache row, dequantized for an MXFP8 cache;
   q CEIL-quantized against an MXFP8 cache):
   `s = (q·k)·f32(softmax_scale·log2 e)`, `m = max(max s, sink·log2 e)`,
   `p = 2^(s − m)`, `o = Σ bf16(p)·v / max(Σ p + 2^(sink·log2 e − m), 1e−30)`, bf16.
   A row without keys gives zeros.
2. Inverse RoPE on the last 64 dims (fp32), reshape to `[S, G, (Hl/G)·D]` (group g
   = heads `g·Hl/G ..`), CEIL; `y_g = bf16(x_g @ wo_a[g]ᵀ)`; `CEIL(y) @ wo_bᵀ` → bf16
   partial; **all-reduce** → `attn_out`.
3. **FFN seam**: §4 step 1 with the FFN weights on `residual`/`pre_mix`, folding
   `attn_out` with `post_mix`/`comb_mix`. Writes `residual_out`, `pre_out`,
   `post_out`, `comb_out`, `ffn_normed_out`.
4. **MoE** on `x = normed`: `logits = bf16(x @ gate_weightᵀ)`;
   `sc = sqrt(softplus(logits))` (softplus = identity above 20; NaN → 0);
   experts = top-`top_k` of `sc + gate_bias` (ties: lower index);
   `w_k = sc_k·routed_scaling_factor / max(Σ sc_k, 1e−20)`.
   Routed: `g, u = CEIL(x) @ w13_eᵀ`; `h = bf16(min(g, L)·σ(min(g, L))·clamp(u, ±L))`,
   `contrib = bf16((CEIL(h) @ w2_eᵀ)·w_k)`. Shared: `g, u = bf16(CEIL(x) @
   shared_w13ᵀ)`; `m = bf16(silu(min(g, L))·clamp(u, ±L))`; `bf16(CEIL(m) @
   shared_w2ᵀ)`. Partial = `bf16(Σ_k contrib_k + shared)`; **all-reduce** → `ffn_out`.
   `L = swiglu_limit`.

`ffn_out`, `residual_out` and the FFN gates are the next layer's `attn_pre`
`sublayer_output`, `residual`, `pre_mix`/`post_mix`/`comb_mix`.

## 6. Mutation and aliasing

Every `*_out` is fully overwritten; `kv_cache` rows named by `kv_slots ≥ 0` are the
only cache writes. Outputs must not alias inputs or each other. Inputs are not
modified.

## 7. Tolerance and errors

Implementations agree up to summation order (GEMM split-K, attention split
reduce, the mix projection) and the rounding points named above; compare in the
units of the last bf16 / FP8 rounding, per stage. Bit equality is not part of the
contract. A call outside the contract raises `ValueError`
(`ContractError` in the native path) before any write. A provider never raises
from `select()`; once its `run` starts, errors propagate.

## 8. Coverage

The native path covers every call this sheet allows (any S, any TP, CPU or GPU),
slowly. A provider states its own coverage in its `select()`; ATOM's fused kernels,
served out of tree, cover gfx950, TP 2 / 4 / 8, 1 ≤ S ≤ 48, bf16 or MXFP8 cache,
the V4.1-Flash shapes, and at most 1024 keys a row (`topk_count + window count`).

## 9. Non-goals (v1)

- The indexer's scoring / top-k and the compressor (a fused provider can run the
  indexer inside `layer_post`; that is a v2 contract with the index plane, its
  tables and the candidate blocks as arguments).
- FP4 index queries (`index_q_out` is FP8 in v1).
- DP attention / expert parallel (TP only), token-parallel sharding of the rows.
- The DSpark draft's layers (different MoE width; same contract shape, not
  covered by the reference test sweep).
