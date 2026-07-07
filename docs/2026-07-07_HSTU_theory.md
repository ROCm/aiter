# HSTU Attention — Theory & Kernel Derivation

*Date: 2026-07-07*
*Scope: math behind HSTU attention, with emphasis on the **backward** kernel that we will implement in FlyDSL. The forward kernel already exists elsewhere.*

Paper: *Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations* (Zhai et al., 2024), [arXiv:2402.17152v2](https://arxiv.org/pdf/2402.17152v2).

---

## 0. How to read this document

This is written for someone comfortable with ML but not necessarily with the way research papers compress notation. So:

- Every symbol is defined the first time it appears, and there is a glossary in §2.
- The backward pass is derived **step by step** from scratch (chain rule, one tensor at a time), not just stated.
- Wherever the math maps onto an actual kernel decision (tiling, recompute, atomics), that is called out in a **> Kernel note** block.

The reference we are matching numerically is the PyTorch implementation in
`meta/aiter/op_tests/triton_tests/utils/hstu_attention_ref.py` (`torch_hstu_attention`),
and the existing Triton kernels in
`meta/aiter/aiter/ops/triton/_triton_kernels/attention/hstu_attention.py`
(`_hstu_attn_fwd`, `_hstu_attn_bwd`). Our FlyDSL work reimplements the **backward** of those.

---

## 1. Where this attention lives (context in one page)

HSTU (Hierarchical Sequential Transduction Unit) is the attention-like block Meta uses for
*generative recommenders*. You do not need the recommender background to implement the kernel,
but two facts explain the unusual design:

1. **It is not softmax attention.** Classic attention normalizes each query's scores into a
   probability distribution over keys (softmax → they sum to 1). HSTU deliberately does **not**
   normalize across the sequence. Instead it applies a **pointwise** nonlinearity (SiLU) to each
   score independently. The paper's motivation: the *number* and *magnitude* of related past
   events is itself a signal (intensity of user preference), and softmax throws that magnitude
   away by forcing scores to sum to 1. This also behaves better on non-stationary, streaming
   vocabularies.

2. **Sequences are jagged (ragged).** A batch is many user histories of *different lengths*,
   concatenated into one long tensor along the token axis, with a `seq_offsets` array marking
   where each sequence starts/ends. There is no padding in memory. This is why the kernels loop
   per-sequence using offsets rather than over a rectangular `[B, N]` grid.

The full HSTU layer (paper Eq. 1–3) is:

$$
U, V, Q, K = \text{Split}\big(\phi_1(f_1(X))\big)
$$
$$
A(X)\,V(X) = \phi_2\!\big(Q K^\top + \text{rab}^{p,t}\big)\,V
$$
$$
Y(X) = f_2\big(\text{Norm}(A(X)V(X)) \odot U(X)\big)
$$

where $f_1, f_2$ are linear layers, $\phi_1,\phi_2$ are SiLU, $\odot$ is elementwise product,
and $\text{rab}^{p,t}$ is a relative attention bias (positional + temporal).

**The kernel we care about is only the middle line**, the *spatial aggregation*
$A(X)V(X) = \phi_2(QK^\top + \text{rab})\,V$ — this is "HSTU attention." Everything else
($f_1$, split, gating $\odot U$, norm, $f_2$) is done by other ops. In the aiter/reference
implementation the relative attention bias term is dropped (ablated), the nonlinearity $\phi_2$
is SiLU, and there is an extra scaling by $\alpha$ and $1/N$. So the concrete function our kernel
computes is the next section.

---

## 2. Notation / glossary

| Symbol | Meaning |
|---|---|
| $b, h$ | batch (sequence) index, head index. The math below is written for a *single* $(b,h)$ — the kernel loops/grids over these. |
| $n$ | length of the current sequence (jagged: varies per $b$). |
| $N$ = `MAX_SEQ_LEN` | the maximum sequence length; used as a **constant** normalizer $1/N$. Not summed over. |
| $d_{qk}$ = `BLOCK_D_Q` (Triton) = `head_dim` (FlyDSL) | head dimension of queries and keys. |
| $d_v$ = `BLOCK_D_V` (Triton) = `hidden_dim` (FlyDSL) | head dimension of values (may differ from $d_{qk}$). |
| $Q \in \mathbb{R}^{n\times d_{qk}}$ | queries for this sequence/head. Row $i$ is query token $i$. |
| $K \in \mathbb{R}^{n\times d_{qk}}$ | keys. Row $j$ is key token $j$. |
| $V \in \mathbb{R}^{n\times d_v}$ | values. Row $j$ is value token $j$. |
| $\alpha$ | scalar scaling applied to the raw scores (`alpha`, e.g. $1/\sqrt{d_{qk}}$ or a tuned constant). |
| $S \in \mathbb{R}^{n\times n}$ | raw scaled scores, $S = \alpha\, Q K^\top$. $S_{ij}$ = query $i$ vs key $j$. |
| $\sigma(x)$ | logistic sigmoid $1/(1+e^{-x})$. |
| $\text{silu}(x)$ | SiLU / swish: $\text{silu}(x) = x\,\sigma(x)$. |
| $M \in \{0,1\}^{n\times n}$ | validity mask (causal / target / contextual / windowed). $M_{ij}=1$ means query $i$ may attend to key $j$. |
| $A \in \mathbb{R}^{n\times n}$ | the (unnormalized) attention weights after SiLU, scaling and masking. |
| $O \in \mathbb{R}^{n\times d_v}$ | output, $O = A V$. |
| $\mathrm{d}X$ | gradient of the scalar loss $\mathcal{L}$ w.r.t. tensor $X$, i.e. $\partial \mathcal{L}/\partial X$, same shape as $X$. Given from upstream: $\mathrm{d}O$. Wanted: $\mathrm{d}Q, \mathrm{d}K, \mathrm{d}V$. |

> **Important convention:** unlike softmax attention there is **no row-wise normalization and no
> `logsumexp`**. That single fact is what makes the HSTU backward *much* simpler than FlashAttention's
> backward — there is no $D_i = \sum_j p_{ij}(\dots)$ correction term to carry around. Each entry
> $A_{ij}$ depends only on its own score $S_{ij}$, not on the other entries in row $i$.

---

## 3. The forward pass (recap, since backward recomputes it)

For a single sequence/head, the forward computation is:

$$
S_{ij} = \alpha \sum_{c=1}^{d_{qk}} Q_{ic} K_{jc}
\qquad\Longleftrightarrow\qquad
S = \alpha\, Q K^\top
$$

$$
A_{ij} = M_{ij}\cdot \frac{\text{silu}(S_{ij})}{N}
       = M_{ij}\cdot \frac{S_{ij}\,\sigma(S_{ij})}{N}
$$

$$
O_{iv} = \sum_{j=1}^{n} A_{ij} V_{jv}
\qquad\Longleftrightarrow\qquad
O = A V
$$

That is exactly the reference:

```150:167:meta/aiter/op_tests/triton_tests/utils/hstu_attention_ref.py
    qk_attn = torch.einsum("bhxa,bhya->bhxy", q, k) * alpha
    qk_attn = F.silu(qk_attn) / max_seq_len
    valid_attn_mask = _get_valid_attn_mask(...)
    qk_attn = qk_attn * valid_attn_mask.unsqueeze(1)
    ...
    attn_dense = torch.einsum("bhxd,bhdv->bhxv", qk_attn, v)  # [B, H, N, V]
```

Reading it back into words:
1. `qk_attn = einsum(q,k)*alpha` → $S=\alpha QK^\top$.
2. `F.silu(qk_attn)/max_seq_len` → $\text{silu}(S)/N$.
3. `* valid_attn_mask` → apply $M$.
4. `einsum(qk_attn, v)` → $O = A V$.

The Triton forward (`_hstu_attn_fwd_one_block`) fuses these so the full $n\times n$ matrix $A$ is
never written to HBM: it is produced tile-by-tile in registers/SRAM and immediately consumed by
$AV$. This is the FlashAttention "fuse the two GEMMs" idea, minus the softmax bookkeeping.

> **Kernel note (forward, for orientation).** A program instance owns a block of query rows
> `[start_m : start_m+BLOCK_M]`. It streams over key/value blocks `start_n`, and for each:
> loads $K$ block, computes `qk = q·k * alpha`, computes `silu = silu(qk)/N`, applies the mask,
> then accumulates `acc += silu·v`. The causal structure lets it skip key blocks that are entirely
> masked out (the `low/high` bounds), which is where the sparsity speedup comes from.

### 3.1 The mask $M$ in detail

$M$ is not just lower-triangular causal. `_get_valid_attn_mask` composes several rules. You need
these because the backward must apply the **same** mask (the gradient of a masked-out entry is 0).
Let `row = i`, `col = j` (token positions), and define `dist = row - col`.

- **Self / diagonal:** the diagonal $i=j$ is always valid (`torch.eye`).
- **Causal:** if `causal`, valid when `dist > 0` (strictly past) — i.e. query attends to strictly
  earlier keys plus itself. If not causal, it uses `|dist| > 0` (both directions).
- **Targets (`num_targets`):** the last `num_targets` tokens of a sequence are "candidate targets."
  Positions are clamped at `max_ids = seq_len - num_targets` so all target tokens share the same
  effective position (they must not attend to each other, and history attends to them uniformly).
- **`max_attn_len` (sliding window):** valid only when `dist <= max_attn_len` — a finite look-back
  window. (Optionally combined with `min_full_attn_seq_len` for a full-attention tail.)
- **`contextual_seq_len`:** the first `contextual_seq_len` tokens are a global "context" prefix;
  positions are shifted by `-contextual_seq_len + 1` and clamped at 0, and row 0 is allowed to
  attend to all non-target history.

For the first implementation pass, the important thing is: **$M$ is an elementwise 0/1 gate that
depends only on token indices** (not on values), so it factors cleanly through the derivatives.
You can implement plain `causal` first and add the variants incrementally.

---

## 4. The backward pass — full derivation

We are given $\mathrm{d}O = \partial\mathcal L/\partial O \in \mathbb{R}^{n\times d_v}$ from
upstream. We want $\mathrm{d}Q, \mathrm{d}K, \mathrm{d}V$. We go backwards through the three
forward steps: $O=AV$, then $A=\text{mask}\circ\text{silu}(S)/N$, then $S=\alpha QK^\top$.

The tool throughout is the chain rule for matrix/elementwise ops. I will derive each gradient
from index notation once, then give the compact matrix form.

### 4.1 Gradient w.r.t. $V$ — from $O = AV$

$O_{iv} = \sum_j A_{ij}V_{jv}$. We want $\mathrm{d}V_{jv} = \sum_{i}\frac{\partial \mathcal L}{\partial O_{iv}}\frac{\partial O_{iv}}{\partial V_{jv}}$.
Since $\partial O_{iv}/\partial V_{jv} = A_{ij}$:

$$
\mathrm{d}V_{jv} = \sum_{i} A_{ij}\,\mathrm{d}O_{iv}
\qquad\Longleftrightarrow\qquad
\boxed{\;\mathrm{d}V = A^\top\, \mathrm{d}O\;}
$$

Shapes: $A^\top$ is $n\times n$, $\mathrm{d}O$ is $n\times d_v$, so $\mathrm{d}V$ is $n\times d_v$. ✓

### 4.2 Gradient w.r.t. $A$ — from $O = AV$

Similarly $\mathrm{d}A_{ij} = \sum_v \frac{\partial\mathcal L}{\partial O_{iv}}\frac{\partial O_{iv}}{\partial A_{ij}} = \sum_v \mathrm{d}O_{iv} V_{jv}$:

$$
\boxed{\;\mathrm{d}A = \mathrm{d}O\, V^\top\;} \qquad (n\times n)
$$

This is the "gradient flowing into the attention matrix."

### 4.3 Gradient w.r.t. $S$ — through mask + SiLU + $1/N$ (the only nonlinearity)

Because $A_{ij} = M_{ij}\,\text{silu}(S_{ij})/N$ is a **pointwise** function of $S_{ij}$ (each entry
independent — no softmax coupling!), the chain rule is elementwise:

$$
\mathrm{d}S_{ij} = \mathrm{d}A_{ij}\cdot \frac{\partial A_{ij}}{\partial S_{ij}}
= \mathrm{d}A_{ij}\cdot M_{ij}\cdot \frac{\text{silu}'(S_{ij})}{N}.
$$

We need the SiLU derivative. With $\text{silu}(s) = s\,\sigma(s)$ and $\sigma'(s)=\sigma(s)(1-\sigma(s))$:

$$
\text{silu}'(s) = \sigma(s) + s\,\sigma'(s)
= \sigma(s) + s\,\sigma(s)\big(1-\sigma(s)\big)
= \sigma(s)\Big(1 + s\big(1-\sigma(s)\big)\Big).
$$

So, writing $\sigma_{ij} = \sigma(S_{ij})$:

$$
\boxed{\;
\mathrm{d}S_{ij} = M_{ij}\cdot \mathrm{d}A_{ij}\cdot \frac{\sigma_{ij}\big(1 + S_{ij}(1-\sigma_{ij})\big)}{N}
\;}
$$

Two things to note:
- The mask $M_{ij}$ reappears exactly as in the forward: gradients through masked-out entries are 0.
- $\sigma_{ij}$ and $S_{ij}$ are needed here. Rather than store them from the forward, the kernel
  **recomputes** $S$ (hence $\sigma$) in the backward — see §5.1.

This matches the Triton backward line-for-line (variables are transposed there; see §5):

```495:500:meta/aiter/aiter/ops/triton/_triton_kernels/attention/hstu_attention.py
    dqk_trans = tl.dot(v, tl.trans(do), allow_tf32=ALLOW_TF32)
    dqk_trans = (
        dqk_trans * sig_trans * (1 + qk_trans * (1 - sig_trans)) * (1.0 / MAX_SEQ_LEN)
    )
    dqk_trans = tl.where(invalid_mask_trans, dqk_trans, 0)
```

Here `dqk_trans` starts as $\mathrm{d}A$ (computed as $V\,\mathrm{d}O^\top$, the transpose of
$\mathrm{d}O\,V^\top$), then is multiplied by $\sigma(1+S(1-\sigma))/N$, then masked. That is exactly
the boxed $\mathrm{d}S$.

### 4.4 Gradients w.r.t. $Q$ and $K$ — from $S = \alpha QK^\top$

$S_{ij} = \alpha\sum_c Q_{ic}K_{jc}$. Standard matmul backprop:

$$
\mathrm{d}Q_{ic} = \alpha\sum_j \mathrm{d}S_{ij} K_{jc}
\quad\Longleftrightarrow\quad
\boxed{\;\mathrm{d}Q = \alpha\,\mathrm{d}S\,K\;}\quad (n\times d_{qk})
$$

$$
\mathrm{d}K_{jc} = \alpha\sum_i \mathrm{d}S_{ij} Q_{ic}
\quad\Longleftrightarrow\quad
\boxed{\;\mathrm{d}K = \alpha\,\mathrm{d}S^\top\,Q\;}\quad (n\times d_{qk})
$$

### 4.5 Summary of the backward (the whole thing)

Given $\mathrm{d}O$, recompute $S=\alpha QK^\top$, $\sigma=\sigma(S)$, then:

$$
\begin{aligned}
\mathrm{d}A &= \mathrm{d}O\,V^\top \\
\mathrm{d}S_{ij} &= M_{ij}\,\mathrm{d}A_{ij}\,\tfrac{1}{N}\,\sigma_{ij}\big(1+S_{ij}(1-\sigma_{ij})\big) \\
\mathrm{d}V &= A^\top \mathrm{d}O \qquad\text{(reuses } A=M\circ\text{silu}(S)/N\text{)}\\
\mathrm{d}Q &= \alpha\,\mathrm{d}S\,K \\
\mathrm{d}K &= \alpha\,\mathrm{d}S^\top Q
\end{aligned}
$$

That is **five matmuls** plus one elementwise stage — no softmax normalization, no `logsumexp`, no
row-sum correction term. Compared to FlashAttention backward, this is genuinely simpler; the
complexity is entirely in (a) doing it tiled/fused to stay memory-bound, and (b) getting the
jagged + masking bookkeeping right.

> **Sanity check — dimensional & symmetry.** $\mathrm{d}V=A^\top\mathrm dO$ pairs each value token
> $j$ with how much every query used it. $\mathrm dQ$ and $\mathrm dK$ are transposes of each other's
> structure (as expected from $S$ being bilinear in $Q,K$), both scaled by $\alpha$. If you set
> $\text{silu}=\text{identity}$ and $M=1$, $N=1$, $\alpha=1$, these collapse to the plain bilinear
> attention gradients $\mathrm dQ=\mathrm dA\,K$, $\mathrm dK=\mathrm dA^\top Q$, $\mathrm dV=A^\top\mathrm dO$ — a good unit test.

---

## 5. Mapping the math to a real kernel (what FlyDSL must do)

The reference Triton backward is the blueprint. Here is how the boxed equations become a tiled GPU
kernel, and the decisions you must replicate.

### 5.1 Recompute instead of store

The forward never wrote $S$, $A$, or the $n\times n$ intermediates to HBM (they are $O(n^2)$). So the
backward **recomputes** them from $Q,K$ on the fly:

```468:471:meta/aiter/aiter/ops/triton/_triton_kernels/attention/hstu_attention.py
    qk_trans = tl.dot(k, q_trans, allow_tf32=ALLOW_TF32) * alpha
    sig_trans = fast_dividef(1.0, 1.0 + tl.exp(-qk_trans))
    silu_trans = qk_trans * sig_trans * (1.0 / MAX_SEQ_LEN)
```

This is $S$, $\sigma$, and $A$ (pre-mask) for the current tile. Recompute is cheaper than the HBM
traffic of storing/reloading the full score matrix — the kernel is memory-bound, so we trade the
(abundant) FLOPs for (scarce) bandwidth.

### 5.2 Tiling scheme: parallelize over keys, loop over queries

The Triton backward is organized **per key/value block** (column block `start_n`), and inside it
loops over query row blocks `start_m`. This is the standard FlashAttention-2 backward layout:

- Each program owns a block of $K,V$ rows: $K_{[n_0:n_0+B_N]}$, $V_{[n_0:n_0+B_N]}$, kept resident.
- It accumulates $\mathrm{d}K$, $\mathrm{d}V$ for those rows across all contributing query blocks.
- For each query block it also produces a **partial** contribution to $\mathrm{d}Q$ for those query
  rows and adds it into the global $\mathrm{d}Q$.

Why this orientation? $\mathrm{d}K=\alpha\,\mathrm dS^\top Q$ and $\mathrm{d}V=A^\top\mathrm dO$ are
**reductions over the query index $i$**. Fixing a $K/V$ block and looping queries lets each program
own a private $\mathrm{d}K,\mathrm{d}V$ accumulator (no cross-program contention for those). The
transposed variables (`_trans`) in the code are because it computes $\mathrm dS^\top$ directly
(shape `[BLOCK_N, BLOCK_M]`) to feed these reductions.

Per inner block (`_hstu_attn_bwd_one_block`), in transposed form:

| Math | Code (transposed) |
|---|---|
| recompute $S,\sigma,A$ for tile | `qk_trans`, `sig_trans`, `silu_trans` |
| $\mathrm dV \mathrel{+}= A^\top \mathrm dO$ | `dv = tl.dot(silu_trans, do, acc=dv)` |
| $\mathrm dA^\top = V\,\mathrm dO^\top$ | `dqk_trans = tl.dot(v, tl.trans(do))` |
| $\mathrm dS^\top = M\circ \tfrac1N \sigma(1+S(1-\sigma))\circ \mathrm dA^\top$ | `dqk_trans = dqk_trans * sig_trans*(1+qk_trans*(1-sig_trans))/N`, then masked |
| $\mathrm dK \mathrel{+}= \mathrm dS^\top Q$ (×$\alpha$ later) | `dk = tl.dot(dqk_trans, tl.trans(q_trans), acc=dk)` |
| $\mathrm dQ \mathrel{+}= \alpha\,K^\top \mathrm dS^\top{}^\top$ | `dq_trans += tl.dot(tl.trans(k), dqk_trans) * alpha` |

Note `dk` is accumulated *without* $\alpha$ and multiplied once at the very end
(`dk = dk * alpha` before store) to save flops — a small but worth-copying optimization.

### 5.3 The $\mathrm{d}Q$ accumulation hazard (locks / atomics / sequence-parallel)

$\mathrm dQ_{[m]}$ receives contributions from **every** key block that query block $m$ attends to.
If multiple programs (different `start_n`) run in parallel and all add into the same $\mathrm dQ$
rows, that is a race. The kernel offers two modes:

- **`SEQUENCE_PARALLEL = False`:** one program per $(b,h)$ loops over all `start_n` sequentially, so
  $\mathrm dQ$ is updated by a single writer — no atomics needed. Simpler, less parallelism.
- **`SEQUENCE_PARALLEL = True`:** key blocks run as separate programs (grid dim `program_id(1)`), and
  $\mathrm dQ$ updates are guarded. The code uses a **spin-lock per query block** (`LOCK`,
  `atomic_cas`/`atomic_xchg`) around the read-modify-write of `dq_trans`:

```504:526:meta/aiter/aiter/ops/triton/_triton_kernels/attention/hstu_attention.py
    if ATOMIC_ADD:
        lock_id = start_m // BLOCK_M
        stride_lock = tl.cdiv(MAX_SEQ_LEN, BLOCK_M)
        lock = LOCK + tl.program_id(0) * stride_lock + lock_id
        tl.debug_barrier()  # add a barrier to force sync
        while tl.atomic_cas(lock, 0, 1) == 1:
            pass
    dq_trans = tl.load(dq_ptrs_trans + start_m * stride_dqm, ...)
    dq_trans += tl.dot(tl.trans(k), dqk_trans, ...) * alpha
    ...
    tl.store(dq_ptrs_trans + start_m * stride_dqm, dq_trans, ...)
    if ATOMIC_ADD:
        tl.atomic_xchg(lock, 0)
```

> **Kernel note for FlyDSL.** This lock-based read-add-write is the trickiest part to port and the
> most GPU-specific. Recommended staging:
> 1. First implement the **non-sequence-parallel** path (single writer per $(b,h)$): correct and
>    lock-free. Validate numerically against the PyTorch reference.
> 2. Then add sequence-parallel with either (a) the same spin-lock pattern, or (b) a global
>    `atomic_add` into $\mathrm dQ$ (if FlyDSL/ROCm exposes float atomic-add efficiently), or
>    (c) writing per-key-block partial $\mathrm dQ$ to scratch and reducing in a second pass.
>    Choose based on what FlyDSL supports well on the target `gfx94x/gfx95x`.

### 5.4 Skipping masked blocks (jagged + causal bounds)

Because $M$ is index-based, whole tiles are provably all-zero and are skipped via `low`/`high`
loop bounds in `_hstu_attn_bwd_one_col_block` (e.g. causal ⇒ only query blocks with `start_m >=
start_n` contribute; `max_attn_len` bounds `high`; `contextual_seq_len` forces a separate leading
block loop). This is both a correctness requirement (don't add masked contributions) and the source
of the sparsity speedup. Get the dense causal bounds right first; add the target/contextual/window
variants one at a time, each guarded by its `HAS_*` compile-time flag exactly as the forward does.

### 5.5 Jagged indexing

Every program first resolves its sequence via `seq_offsets`:

```776:790:meta/aiter/aiter/ops/triton/_triton_kernels/attention/hstu_attention.py
    seq_start = tl.load(seq_offsets + off_z).to(tl.int64)
    seq_end = tl.load(seq_offsets + off_z + 1)
    seq_len = (seq_end - seq_start).to(tl.int32)
    ...
    Q = Q + seq_start * stride_qm + off_h * stride_qh
    K = K + seq_start * stride_kn + off_h * stride_kh
    ...
```

i.e. base pointers are advanced to this sequence's slice; all row indices are local to `[0, seq_len)`
and masked against `seq_len`. FlyDSL will need the same offset-then-mask pattern (its `Tensor`/layout
+ `mark_layout_dynamic` machinery), rather than a rectangular `[B,N,H,D]` view.

---

## 6. Suggested implementation & validation plan for FlyDSL

1. **Reproduce the forward once in FlyDSL** (even if a version exists elsewhere) purely as a
   numerical oracle scaffold, or reuse the existing forward. You need $O$ to feed a synthetic
   $\mathrm dO$.
2. **Reference gradients via autograd.** Build tiny random $Q,K,V$ (single sequence, single head,
   small $n,d$), run `torch_hstu_attention`, backprop a random scalar, and capture
   $\mathrm dQ,\mathrm dK,\mathrm dV$. This is your ground truth (`torch.autograd.grad`).
3. **Implement the backward dense + causal-only** in FlyDSL, non-sequence-parallel (single writer
   for $\mathrm dQ$). Compare to step 2 with `allclose` (loosen tolerance for bf16/tf32).
4. **Add masking variants** (`num_targets`, `max_attn_len`, `contextual_seq_len`) one flag at a
   time, each with its own reference config.
5. **Add tiling / block sizes**, then the sequence-parallel + $\mathrm dQ$ synchronization path.
6. **Benchmark** against the Triton backward (`bench_hstu_attn.py`) on the target arch.

### Correctness gotchas to watch

- **Apply the exact same mask** in backward as forward, including the diagonal-always-valid rule and
  the position clamping for targets/contextual. A mismatch shows up as wrong $\mathrm dQ/\mathrm dK$
  near sequence boundaries only.
- **The $1/N$ factor is a constant** (`MAX_SEQ_LEN`), not $1/n$ (the current sequence length). Easy
  to conflate in jagged code.
- **`alpha` appears in $S$ (forward) and again in $\mathrm dQ,\mathrm dK$ (backward)** but *not* in
  $\mathrm dV$ (which depends on $A$, where $\alpha$ is already baked into $S$ before SiLU). Don't
  double-apply it.
- **SiLU derivative**: use $\sigma(1+S(1-\sigma))$, computed from the recomputed $S,\sigma$ — do not
  approximate.
- **bf16/tf32 accumulation**: accumulate matmuls in fp32 (as the Triton code does) and only cast on
  store, or your gradient checks will fail tolerance.

---

## 7. Alignment with the existing FlyDSL forward kernel

We now have a real FlyDSL HSTU **forward** kernel at
`meta/robin_aiter/aiter/ops/flydsl/hstu_attention_kernels.py` (host API) and
`meta/robin_aiter/aiter/ops/flydsl/kernels/hstu_attention_fwd.py` (device kernel). The math in
§3–§6 matches it exactly; this section records the **conventions and numerical choices** the
backward must mirror so its gradients agree with this forward and pass numerical checks.

### 7.1 Supported configuration (narrower than the reference)

The FlyDSL forward is deliberately more restricted than the PyTorch reference:

- **Causal only.** `validate_hstu_attention_fwd` raises on `causal=False`. So the non-causal
  `|dist|>0` mask from §3.1 exists in the *reference* but not on the FlyDSL path — the backward is
  causal-only too (at least initially).
- **dtype ∈ {f16, bf16}** (accumulation in fp32).
- **Shape constraints:** `head_dim % 16 == 0`, `hidden_dim % 16 == 0`, `(batch*num_heads) % 8 == 0`
  (MFMA 16×16×16 tiling + group-major grid). Arch must be `gfx942` (CDNA3) or `gfx950` (CDNA4).
- **Layout:** q/k/v are native rank-3 `(L, H, dim)`, contiguous, jagged along `L` with
  `seq_offsets` (Z+1) and `num_targets` (Z), *no* host flatten to `(L, H*dim)`. Same convention the
  backward should take.
- Masking variants `causal / max_attn_len (window) / contextual_seq_len / num_targets` are all
  present and implement exactly the `to_id` order from §3.1 (**contextual shift, then target-tail
  clamp**), with the diagonal always kept.

### 7.2 Numerical recipe the backward must reproduce bit-for-bit

The recompute in the backward (§5.1) must match the forward's *exact* arithmetic, not just the
mathematical SiLU, or gradient `allclose` checks will drift:

- **`alpha` is folded into the score before SiLU**, and **`1/N` is hoisted out to the epilogue**
  (`silu_scale_batch` computes `silu(alpha·s)`; the `1/N` is applied once at store via `c_inv_n`).
  In the backward, be deliberate: `alpha` enters $S$ (and reappears in $\mathrm dQ,\mathrm dK$),
  while `1/N` multiplies $\mathrm dS$ — matching where the forward puts them.
- **SiLU is evaluated with fast/unsafe FP intrinsics**, not a library sigmoid:
  $\sigma(x)=\text{rcp}(1+\text{exp2}(x\cdot(-\log_2 e)))$ via `llvm.amdgcn.exp2.f32` and
  `llvm.amdgcn.rcp.f32`, under `fast_fp_math=True, unsafe_fp_math=True` (denormals flushed,
  no-nans). The build sets `-unsafe-fp-math`, `no-nans-fp-math`. **Implication:** validate the
  backward against the PyTorch reference with *relaxed* tolerances (bf16 + non-IEEE), and, if you
  need tight agreement, reuse the same `exp2/rcp` formulation for $\sigma$ inside $\mathrm dS$.
- **Nothing is stashed by the forward** — no $S$, $\sigma$, $A$, or mask is written to HBM. This
  confirms §5.1: the backward *must* recompute $S=\alpha QK^\top$ and $\sigma$ from $Q,K$.

### 7.3 FlyDSL idioms to reuse (instead of the Triton blueprint)

§5 describes the algorithm using the Triton backward as a blueprint. When writing the FlyDSL
backward, translate those steps into the same idioms this forward already uses:

- **Host plumbing:** mirror the `validate_hstu_attention_*` / `build_hstu_attention_*` split, the
  `@functools.lru_cache` `_compile_launcher`, the tuned-CSV → `_get_tuned_config` →
  `_get_default_config` → `custom_config` override chain, and `kernels/tensor_shim._run_compiled` +
  `get_dtype_str`. Add a `flydsl_hstu_attention_bwd` entry point shaped like
  `flydsl_hstu_attention_fwd`.
- **Device idioms:** `SmemAllocator`/`SmemPtr` for LDS tiles, `fx.make_mma_atom(fx.rocdl.MFMA(16,16,16,·))`
  for the GEMMs, arch-conditional DMA + K LDS XOR swizzle (`_arch_dma_params`, `k_swz_col`),
  register-prefetch + counted `s_waitcnt vmcnt` overlap, and carried-accumulator loops
  (`for ... init=...: ... yield`).
- **Grid orientation flips.** The forward grid is `num_q_tiles * batch * num_heads`, group-major
  (`NUM_GRID_GROUPS=8`), **parallel over query tiles** (each block owns Q rows, streams K/V). The
  backward instead wants to be **parallel over KV tiles** (§5.2), because $\mathrm dK,\mathrm dV$
  reduce over the query index. Reframe the $\mathrm dQ$-accumulation hazard (§5.3) in FlyDSL terms
  (single-writer loop first; then atomics or a scratch-and-reduce pass) rather than porting Triton's
  spin-lock literally.

### 7.4 Net effect on this document

The math (§3–§4) and the algorithm/kernel-mapping (§5–§6) required **no corrections** — the FlyDSL
forward is consistent with them. This section only *adds* the FlyDSL-specific conventions the
backward has to honor. Concrete follow-ups for the backward that this forward makes clear:

1. Match the fast-math SiLU (`exp2`/`rcp`) and the `alpha`-in / `1/N`-in-epilogue placement.
2. Keep it causal-only and reuse the `to_id` masking order.
3. Build the host wrapper + config plumbing as a sibling of the forward's.
4. Choose the $\mathrm dQ$ synchronization strategy in FlyDSL primitives.

---

## 8. Quick reference card

Forward (per $(b,h)$, sequence length $n$, constant $N$):

$$
S=\alpha QK^\top,\quad A = M\circ \tfrac1N\,\text{silu}(S),\quad O = AV.
$$

Backward (given $\mathrm dO$; recompute $S,\sigma$):

$$
\mathrm dV = A^\top \mathrm dO,\qquad
\mathrm dS = M\circ \tfrac1N\,\sigma(1+S(1-\sigma))\circ(\mathrm dO\,V^\top),
$$
$$
\mathrm dQ = \alpha\,\mathrm dS\,K,\qquad
\mathrm dK = \alpha\,\mathrm dS^\top Q.
$$

SiLU: $\text{silu}(s)=s\sigma(s)$, $\ \text{silu}'(s)=\sigma(s)\big(1+s(1-\sigma(s))\big)$.

---

### File references

- Reference (PyTorch oracle): `meta/aiter/op_tests/triton_tests/utils/hstu_attention_ref.py`
- Triton kernels (fwd + bwd blueprint): `meta/aiter/aiter/ops/triton/_triton_kernels/attention/hstu_attention.py`
- FlyDSL forward (host API): `meta/robin_aiter/aiter/ops/flydsl/hstu_attention_kernels.py`
- FlyDSL forward (device kernel): `meta/robin_aiter/aiter/ops/flydsl/kernels/hstu_attention_fwd.py`
- Host wrappers: `meta/aiter/aiter/ops/triton/attention/hstu_attention.py`
- Tests / bench: `meta/aiter/op_tests/triton_tests/attention/test_hstu_attn.py`, `meta/aiter/op_tests/op_benchmarks/triton/bench_hstu_attn.py`
- Paper: [arXiv:2402.17152v2](https://arxiv.org/pdf/2402.17152v2), §3.1 (pointwise aggregated attention), Eq. 1–3.
