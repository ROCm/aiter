# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL dense absorb MLA prefill vs production baselines.

Candidates (aiter-op-test style):
  flydsl     — this kernel (absorb 576→512, any H, causal or not)
  asm        — aiter.mla.mla_prefill_fwd (absorb; H in {16,128}; causal semantics)
  triton     — aiter.ops.triton.attention.mla.mla_prefill_fwd (absorb; causal only)
  fmha_decomp — decompress-shaped FMHA qh192/vh128 (ticket competitor attention)

Torch absorb is the correctness reference only (not timed).
"""

import argparse
import itertools
import math

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import flydsl_mla_prefill_fwd, flydsl_mla_prefill_supported
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]
KV_LORA = 512
QK_ROPE = 64
QK_DIM = KV_LORA + QK_ROPE  # 576
V_DIM = KV_LORA  # 512
# Ticket decompress path (AiterFlashAttnPrefillBackend context)
QK_DECOMP = 128 + QK_ROPE  # 192
V_DECOMP = 128
ASM_HEADS = (16, 128)


def _cos_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x64, y64 = x.double(), y.double()
    return 1 - 2 * (x64 * y64).sum().item() / max(
        (x64 * x64 + y64 * y64).sum().item(), 1e-12
    )


def run_torch_absorb(q, kv_buffer, qo_indptr, kv_indptr, kv_indices, sm_scale, is_causal):
    page_size = kv_buffer.shape[1]
    kv_flat = kv_buffer.reshape(-1, QK_DIM)
    batch = qo_indptr.numel() - 1
    outs = []
    for b in range(batch):
        qs, qe = int(qo_indptr[b]), int(qo_indptr[b + 1])
        ks, ke = int(kv_indptr[b]), int(kv_indptr[b + 1])
        q_b = q[qs:qe]
        rows = [
            kv_flat[
                int(kv_indices[ks + t // page_size]) * page_size + (t % page_size)
            ]
            for t in range(ke - ks)
        ]
        kv = torch.stack(rows, dim=0)
        h = q_b.shape[1]
        k = kv[:, None, :].expand(-1, h, -1)
        v = kv[:, None, :V_DIM].expand(-1, h, -1)
        attn = torch.einsum("qhd,khd->hqk", q_b.float(), k.float()) * sm_scale
        if is_causal:
            sq, sk = q_b.shape[0], kv.shape[0]
            mask = torch.ones(sq, sk, dtype=torch.bool, device=q.device).tril(
                diagonal=sk - sq
            )
            attn = attn.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        outs.append(torch.einsum("hqk,khd->qhd", attn, v.float()).to(q.dtype))
    return torch.cat(outs, dim=0)


def _make_inputs(batch, sq, skv, nhead, seed=0):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    page_size = 1
    qo_indptr = torch.arange(batch + 1, dtype=torch.int32, device="cuda") * sq
    kv_indptr = torch.arange(batch + 1, dtype=torch.int32, device="cuda") * skv
    total_q = int(qo_indptr[-1])
    total_kv = int(kv_indptr[-1])
    kv_indices = torch.arange(total_kv, dtype=torch.int32, device="cuda")
    q = torch.randn(
        total_q, nhead, QK_DIM, dtype=dtypes.bf16, device="cuda", generator=g
    )
    kv_buffer = torch.randn(
        total_kv + 8, page_size, 1, QK_DIM, dtype=dtypes.bf16, device="cuda", generator=g
    )
    sm_scale = 1.0 / math.sqrt(QK_DIM)
    return q, kv_buffer, qo_indptr, kv_indptr, kv_indices, sm_scale


def _pad_heads(q: torch.Tensor, nhead_pad: int) -> torch.Tensor:
    total_q, nhead, d = q.shape
    if nhead == nhead_pad:
        return q
    out = torch.zeros(total_q, nhead_pad, d, dtype=q.dtype, device=q.device)
    out[:, :nhead].copy_(q)
    return out


def _kv_as_paged(kv_buffer, page_size=1):
    # [num_page, page, 1, 576]
    if kv_buffer.ndim == 4:
        return kv_buffer
    raise ValueError(kv_buffer.shape)


def _triton_block_tables(kv_indptr, kv_indices, skv):
    batch = kv_indptr.numel() - 1
    # page_size=1: one block id per token
    bt = torch.zeros(batch, skv, dtype=torch.int32, device=kv_indices.device)
    for b in range(batch):
        ks, ke = int(kv_indptr[b]), int(kv_indptr[b + 1])
        bt[b, : ke - ks] = kv_indices[ks:ke]
    seqused_k = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)
    return bt, seqused_k


def _record(ret, name, us, flops, nbytes, cos, err):
    ret[f"{name} us"] = us
    ret[f"{name} TFLOPS"] = flops / us / 1e6 if us > 0 else 0.0
    ret[f"{name} TB/s"] = nbytes / us / 1e6 if us > 0 else 0.0
    ret[f"{name} cos"] = cos
    ret[f"{name} err"] = float(err) if err is not None else 0.0


@benchmark()
def test_flydsl_mla_prefill(batch, sq, skv, nhead, is_causal, dtype):
    ret = {"gfx": get_gfx()}
    if dtype != dtypes.bf16 or not flydsl_mla_prefill_supported():
        return ret

    q, kv_buffer, qo_indptr, kv_indptr, kv_indices, sm_scale = _make_inputs(
        batch, sq, skv, nhead
    )
    total_q = q.shape[0]
    o = torch.empty(total_q, nhead, V_DIM, dtype=dtypes.bf16, device="cuda")
    ref = run_torch_absorb(
        q, kv_buffer, qo_indptr, kv_indptr, kv_indices, sm_scale, is_causal
    )

    flops_absorb = 2.0 * batch * nhead * sq * skv * (QK_DIM + V_DIM)
    nbytes_absorb = (
        batch * nhead * sq * QK_DIM + batch * skv * QK_DIM + batch * nhead * sq * V_DIM
    ) * 2
    flops_decomp = 2.0 * batch * nhead * sq * skv * (QK_DECOMP + V_DECOMP)
    nbytes_decomp = (
        batch * nhead * sq * QK_DECOMP
        + batch * nhead * skv * QK_DECOMP
        + batch * nhead * skv * V_DECOMP
        + batch * nhead * sq * V_DECOMP
    ) * 2

    # ---- flydsl (always) ----
    o.zero_()
    out_fd, us_fd = run_perftest(
        lambda: flydsl_mla_prefill_fwd(
            q,
            kv_buffer,
            o,
            qo_indptr,
            kv_indptr,
            kv_indices,
            sm_scale,
            is_causal=is_causal,
        )
    )
    err = checkAllclose(
        ref.to(dtypes.fp32), out_fd.to(dtypes.fp32), rtol=2e-2, atol=2e-2, printLog=False
    )
    cos = _cos_diff(ref, out_fd)
    _record(ret, "flydsl", us_fd, flops_absorb, nbytes_absorb, cos, err)
    assert cos < 1e-4, f"flydsl cos={cos}"

    # ---- asm absorb (H pad to 16 if needed; causal-correct only) ----
    asm_h = 16 if nhead <= 16 else (128 if nhead <= 128 else None)
    if asm_h is not None and asm_h in ASM_HEADS:
        q_asm = _pad_heads(q, asm_h)
        o_asm = torch.empty(total_q, asm_h, V_DIM, dtype=dtypes.bf16, device="cuda")
        kv_pages = kv_buffer.shape[0]
        kv_4d = kv_buffer  # already [P,1,1,576]

        def _asm():
            return aiter.mla.mla_prefill_fwd(
                q_asm,
                kv_4d,
                o_asm,
                qo_indptr,
                kv_indptr,
                kv_indices,
                torch.ones(batch, dtype=torch.int32, device="cuda"),
                sq,
                sm_scale,
            )[0]

        out_asm, us_asm = run_perftest(_asm)
        out_asm_h = out_asm[:, :nhead]
        cos_asm = _cos_diff(ref, out_asm_h)
        err_asm = checkAllclose(
            ref.to(dtypes.fp32),
            out_asm_h.to(dtypes.fp32),
            rtol=2e-2,
            atol=2e-2,
            printLog=False,
        )
        _record(ret, "asm", us_asm, flops_absorb, nbytes_absorb, cos_asm, err_asm)
        # ASM shipped kernels match causal; non-causal is expected to diverge.
        if is_causal:
            assert cos_asm < 1e-4, f"asm causal cos={cos_asm}"

    # ---- triton absorb (causal only) ----
    if is_causal:
        try:
            from aiter.ops.triton.attention.mla import mla_prefill_fwd as triton_mla_pfl

            block_tables, seqused_k = _triton_block_tables(kv_indptr, kv_indices, skv)
            # Triton wants [blocks, block_size, kv_heads, dim]
            kv_tr = kv_buffer  # [P,1,1,576]
            o_tr = torch.empty_like(o)

            def _triton():
                return triton_mla_pfl(
                    q,
                    kv_tr,
                    o_tr,
                    qo_indptr,
                    seqused_k,
                    skv,
                    block_tables,
                    sm_scale,
                    KV_LORA,
                    QK_ROPE,
                    True,
                    None,
                    None,
                )

            out_tr, us_tr = run_perftest(_triton)
            cos_tr = _cos_diff(ref, out_tr if out_tr is not None else o_tr)
            out_cmp = out_tr if isinstance(out_tr, torch.Tensor) else o_tr
            err_tr = checkAllclose(
                ref.to(dtypes.fp32),
                out_cmp.to(dtypes.fp32),
                rtol=2e-2,
                atol=2e-2,
                printLog=False,
            )
            _record(ret, "triton", us_tr, flops_absorb, nbytes_absorb, cos_tr, err_tr)
        except Exception as exc:  # noqa: BLE001 — baseline optional
            ret["triton skip"] = str(exc)[:80]

    # ---- decompress-shaped FMHA (ticket attention competitor) ----
    # Not the same math as absorb; times the FMHA the serving stack runs after
    # decompress. Fair for "can we beat the production attention kernel".
    g = torch.Generator(device="cuda")
    g.manual_seed(1)
    q_d = torch.randn(
        total_q, nhead, QK_DECOMP, dtype=dtypes.bf16, device="cuda", generator=g
    )
    k_d = torch.randn(
        batch * skv, nhead, QK_DECOMP, dtype=dtypes.bf16, device="cuda", generator=g
    )
    v_d = torch.randn(
        batch * skv, nhead, V_DECOMP, dtype=dtypes.bf16, device="cuda", generator=g
    )
    sm_d = 1.0 / math.sqrt(QK_DECOMP)

    def _fmha():
        return aiter.flash_attn_varlen_func(
            q_d,
            k_d,
            v_d,
            qo_indptr,
            kv_indptr,
            sq,
            skv,
            softmax_scale=sm_d,
            causal=is_causal,
        )

    try:
        _out_fmha, us_fmha = run_perftest(_fmha)
        _record(ret, "fmha_decomp", us_fmha, flops_decomp, nbytes_decomp, 0.0, 0.0)
        if us_fd > 0 and us_fmha > 0:
            ret["flydsl/fmha"] = us_fmha / us_fd
    except Exception as exc:  # noqa: BLE001
        ret["fmha_decomp skip"] = str(exc)[:80]

    if "asm us" in ret and ret["flydsl us"] > 0:
        ret["flydsl/asm"] = ret["asm us"] / ret["flydsl us"]
    if "triton us" in ret and ret["flydsl us"] > 0:
        ret["flydsl/triton"] = ret["triton us"] / ret["flydsl us"]

    return ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dtype", nargs="*", default=["bf16"])
    parser.add_argument("-b", "--batch", type=int, nargs="*", default=[1])
    parser.add_argument("--sq", type=int, nargs="*", default=[64, 256])
    parser.add_argument("--skv", type=int, nargs="*", default=[256, 1024])
    parser.add_argument("--nhead", type=int, nargs="*", default=[12, 16])
    parser.add_argument(
        "--causal",
        type=int,
        nargs="*",
        default=[0, 1],
        help="0 non-causal (ticket), 1 causal (asm/triton)",
    )
    args = parser.parse_args()

    if get_gfx() not in SUPPORTED_GFX:
        print(f"skip: gfx={get_gfx()}")
        return

    dtype_map = {"bf16": dtypes.bf16}
    rows = []
    for batch, sq, skv, nhead, causal, dt in itertools.product(
        args.batch, args.sq, args.skv, args.nhead, args.causal, args.dtype
    ):
        rows.append(
            test_flydsl_mla_prefill(batch, sq, skv, nhead, bool(causal), dtype_map[dt])
        )
    df = pd.DataFrame(rows)
    aiter.logger.info("dense MLA prefill A/B:\n%s", df.to_markdown(index=False))


if __name__ == "__main__":
    main()
