"""
Minimal repro for AITER MLA decode causal mask bug under CP round-robin sharding + qlen>1.

Environment: ROCm + aiter (gfx950 tested).

Prerequisite:
    python -c "import aiter, torch; print(torch.version.hip)"

What it does:
    1. Creates a single request with global KV length S=13.
    2. Query length qlen=4 (e.g. MTP verify with 3 speculative tokens).
    3. KV is round-robin sharded across W=4 ranks.
    4. Runs AITER mla_decode_fwd per-rank with is_causal=True.
    5. Merges per-rank outputs via online-softmax and compares to golden.

Bug manifestation:
    - All ranks show numerical mismatch vs cp-aware reference.
    - Empty shard: lse does not return -inf.

Expected after fix:
    - All ranks match cp-aware reference within bf16 tolerance (~1e-2).
    - Empty shard returns out=0, lse=-inf.
"""

import math
import torch

from aiter.mla import mla_decode_fwd
from aiter import get_mla_metadata_v1, get_mla_metadata_info_v1

torch.manual_seed(0)
DEV = "cuda"
DT = torch.bfloat16
L = 512  # kv_lora_rank
R = 64  # qk_rope_head_dim
D = L + R  # total key/query dim
NHEAD = 16  # gqa=16
NKV = 1
SCALE = 1.0 / math.sqrt(D)
NSPLIT = 32


# --------------------------------------------------------------------------- #
# Reference decode in pure torch (fp32). This is the ground truth.
# --------------------------------------------------------------------------- #
def ref_decode(q, kv, vis):
    """q[T,H,D], kv[S,D], vis[T,S] bool -> out[T,H,L], lse[T,H] (fp32)."""
    q32, kv32 = q.float(), kv.float()
    scores = torch.einsum("thd,sd->ths", q32, kv32) * SCALE
    neg = torch.finfo(torch.float32).min
    scores = scores.masked_fill(~vis[:, None, :], neg)
    lse = torch.logsumexp(scores, dim=-1)
    p = torch.softmax(scores, dim=-1)
    out = torch.einsum("ths,sl->thl", p, kv32[:, :L])
    allmasked = ~vis.any(dim=-1)
    out[allmasked] = 0.0
    lse[allmasked] = float("-inf")
    return out, lse


def merge(outs, lses):
    """Online-softmax merge across CP ranks."""
    LS = torch.stack(lses, 0)  # [W,T,H]
    glse = torch.logsumexp(LS, 0)  # [T,H]
    w = torch.exp(LS - glse).nan_to_num_(0.0)
    out = sum(w[r][..., None] * outs[r] for r in range(len(outs)))
    return out


# --------------------------------------------------------------------------- #
# Per-rank AITER kernel call (aligned with production calling convention).
# --------------------------------------------------------------------------- #
def aiter_decode(q_local, kv_local, qlen, cp_world_size=1, cp_rank=0, s_global=None):
    """q_local[qlen,H,D], kv_local[Lloc,D] -> (out[qlen,H,L], lse[qlen,H]).

    Round-robin CP: local kv index j maps to global position j*cp_world_size+cp_rank;
    g_kv_indptr=[0, s_global] gives the global KV length for the causal bound.
    """
    Lloc = kv_local.shape[0]
    o = torch.zeros(qlen, NHEAD, L, dtype=DT, device=DEV)
    lse = torch.zeros(qlen, NHEAD, dtype=torch.float32, device=DEV)

    kv_buffer = (
        kv_local if Lloc > 0 else torch.zeros(1, D, dtype=DT, device=DEV)
    ).reshape(-1, 1, 1, D)
    print(f"kv_buffer.shape: {kv_buffer.shape}")
    print(f"q_local.shape: {q_local.shape}")
    qo_indptr = torch.tensor([0, qlen], dtype=torch.int32, device=DEV)
    kv_indptr = torch.tensor([0, Lloc], dtype=torch.int32, device=DEV)
    kv_indices = torch.arange(max(Lloc, 1), dtype=torch.int32, device=DEV)
    kv_last_len = torch.ones(1, dtype=torch.int32, device=DEV)

    # round-robin CP: GLOBAL kv_indptr = [0, s_global] (per-request global KV len)
    if s_global is None:
        s_global = Lloc * cp_world_size + cp_rank
    g_kv_indptr = torch.tensor([0, s_global], dtype=torch.int32, device=DEV)

    is_causal = qlen > 1
    nreqs = 1

    info = get_mla_metadata_info_v1(
        nreqs,
        qlen,
        NHEAD,
        DT,
        DT,
        is_sparse=False,
        fast_mode=True,
        num_kv_splits=NSPLIT,
        intra_batch_mode=False,
    )

    def _alloc(sz, ty):
        return torch.empty(sz, dtype=ty, device=DEV)

    work_meta_data = _alloc(*info[0])
    work_indptr = _alloc(*info[1])
    work_info_set = _alloc(*info[2])
    reduce_indptr = _alloc(*info[3])
    reduce_final_map = _alloc(*info[4])
    reduce_partial_map = _alloc(*info[5])

    get_mla_metadata_v1(
        qo_indptr,
        kv_indptr,
        kv_last_len,
        NHEAD // NKV,
        NKV,
        is_causal,
        work_meta_data,
        work_info_set,
        work_indptr,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        page_size=1,
        kv_granularity=16,
        max_seqlen_qo=qlen,
        uni_seqlen_qo=qlen,
        fast_mode=True,
        max_split_per_batch=NSPLIT,
        intra_batch_mode=False,
        dtype_q=DT,
        dtype_kv=DT,
    )

    _, final_lse = mla_decode_fwd(
        q_local,
        kv_buffer,
        o,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_len,
        max_seqlen_q=qlen,
        page_size=1,
        nhead_kv=NKV,
        sm_scale=SCALE,
        logit_cap=0.0,
        num_kv_splits=NSPLIT,
        work_meta_data=work_meta_data,
        work_indptr=work_indptr,
        work_info_set=work_info_set,
        reduce_indptr=reduce_indptr,
        reduce_final_map=reduce_final_map,
        reduce_partial_map=reduce_partial_map,
        intra_batch_mode=False,
        return_lse=True,
        g_kv_indptr=g_kv_indptr,
        cp_world_size=cp_world_size,
        cp_rank=cp_rank,
    )
    return o.float(), final_lse.float()


def sanitize(out, lse):
    """Post-process NaN/+inf lse -> large negative, NaN out -> 0."""
    lse = torch.where(
        torch.isnan(lse) | torch.isinf(lse), torch.full_like(lse, -1e30), lse
    )
    out = torch.where(torch.isnan(out), torch.zeros_like(out), out)
    return out, lse


# --------------------------------------------------------------------------- #
# Problem setup: 1 request, S=13, qlen=4, W=4 round-robin CP.
# --------------------------------------------------------------------------- #
S, qlen, W = 13, 4, 4
kv_g = (torch.randn(S, D, device=DEV) * 0.5).to(DT)
q = (torch.randn(qlen, NHEAD, D, device=DEV) * 0.5).to(DT)
bound = torch.tensor([S - qlen + i for i in range(qlen)], device=DEV)

# Golden: full KV, no sharding
gpos = torch.arange(S, device=DEV)
vis_full = gpos[None, :] <= bound[:, None]
print(f"bound: {bound}")
print(f"gpos: {gpos}")
print(f"vis_full: {vis_full}")
gold_out, _ = ref_decode(q, kv_g, vis_full)

shard_pos = [gpos[gpos % W == r] for r in range(W)]
print(f"shard_pos: {shard_pos}")
print(
    f"Setup: S={S} qlen={qlen} W={W}  query global positions={[S - qlen + i for i in range(qlen)]}"
)
for r in range(W):
    print(
        f"  rank{r}: global positions {shard_pos[r].tolist()}  (count={len(shard_pos[r])})"
    )

# --------------------------------------------------------------------------- #
# SANITY: cp-aware reference (correct mask) per-rank -> merge == golden
# --------------------------------------------------------------------------- #
cpa_outs, cpa_lses = [], []
for r in range(W):
    pos = shard_pos[r]
    kv_r = kv_g[pos]
    vis_r = pos[None, :] <= bound[:, None]
    o_r, l_r = ref_decode(q, kv_r, vis_r)
    cpa_outs.append(o_r)
    cpa_lses.append(l_r)
cpa_merged = merge(cpa_outs, cpa_lses)
cpa_err = (cpa_merged - gold_out).abs().max().item()
print(
    f"\n[SANITY] cp-aware merge vs golden: max|diff| = {cpa_err:.2e}  {'PASS' if cpa_err < 1e-4 else 'FAIL'}"
)

# --------------------------------------------------------------------------- #
# REPRO 1+2: per-rank AITER kernel with is_causal=True
# --------------------------------------------------------------------------- #
print("\n[REPRO 1/2] AITER mla_decode_fwd per rank (is_causal=True, qlen=4)")
ait_outs, ait_lses = [], []
for r in range(W):
    pos = shard_pos[r]
    kv_r = kv_g[pos].contiguous()
    a_out, a_lse = aiter_decode(
        q.contiguous(), kv_r, qlen, cp_world_size=W, cp_rank=r, s_global=S
    )
    vis_r = pos[None, :] <= bound[:, None]
    ref_out, _ = ref_decode(q, kv_r, vis_r)
    nan_rows = torch.isnan(a_out).flatten(1).any(dim=1).tolist()
    finite = ~torch.tensor(nan_rows, device=DEV)
    diff = (
        (a_out - ref_out).abs()[finite].max().item() if finite.any() else float("nan")
    )
    print(
        f"  rank{r} (count={len(pos)}): has_NaN={bool(torch.isnan(a_out).any())} "
        f"NaN_rows={nan_rows}  max|diff vs cp-aware ref|={diff:.3e}"
    )
    ait_outs.append(a_out)
    ait_lses.append(a_lse)

# --------------------------------------------------------------------------- #
# REPRO 3: empty shard must return (out=0, lse=-inf), not NaN
# --------------------------------------------------------------------------- #
print("\n[REPRO 3] Empty local shard (0 KV entries), qlen=4")
empty_out, empty_lse = aiter_decode(
    q.contiguous(),
    torch.zeros(0, D, dtype=DT, device=DEV),
    qlen,
    cp_world_size=W,
    cp_rank=0,
    s_global=S,
)
out_ok = (empty_out == 0).all().item()
lse_ok = torch.isinf(empty_lse).all().item() and (empty_lse < 0).all().item()
print(f"  out: NaN={bool(torch.isnan(empty_out).any())}  all_zero={out_ok}")
print(f"  lse: NaN={bool(torch.isnan(empty_lse).any())}  all_neg_inf={lse_ok}")
print(f"  Expected: out==0 and lse==-inf -> {'PASS' if out_ok and lse_ok else 'FAIL'}")

# --------------------------------------------------------------------------- #
# End-to-end: sanitized AITER per-rank -> merge vs golden
# --------------------------------------------------------------------------- #
san = [sanitize(o, l) for o, l in zip(ait_outs, ait_lses)]
ait_merged = merge([o for o, _ in san], [l for _, l in san])
ait_err = (ait_merged - gold_out).abs().max().item()
print("\n[END-TO-END] AITER (sanitized) merge vs golden:")
print(f"  max|diff| = {ait_err:.3e}   (cp-aware ref diff = {cpa_err:.2e})")
print(f"  -> {'BUG REPRODUCED' if ait_err > 1e-2 else 'OK'}")
