import argparse
import csv
import math
from dataclasses import dataclass

import torch

import aiter
from aiter import dtypes


NHEAD = 32
NHEAD_KV = 1
KV_LORA = 512
QK_ROPE = 64
QK_HEAD_DIM = KV_LORA + QK_ROPE
V_HEAD_DIM = KV_LORA
PAGE_SIZE = 1


@dataclass
class Shape:
    batch: int
    ctx: int
    qlen: int

    @property
    def total_q(self) -> int:
        return self.batch * self.qlen


def parse_shapes(value: str) -> list[Shape]:
    shapes = []
    for item in value.split(","):
        if not item:
            continue
        b, ctx = item.split("x")
        shapes.append(Shape(int(b), int(ctx), 4))
    return shapes


def cosine_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x_f = x.float().flatten()
    y_f = y.float().flatten()
    denom = torch.clamp((x_f * x_f + y_f * y_f).sum(), min=1e-12)
    return float(1 - 2 * (x_f * y_f).sum() / denom)


def error_metrics(ref: torch.Tensor, out: torch.Tensor) -> dict[str, float]:
    diff = (ref.float() - out.float()).abs()
    return {
        "cos": cosine_diff(ref, out),
        "rmse": float((diff * diff).mean().sqrt()),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
    }


def make_metadata(shape: Shape):
    cu_num = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    max_split_per_batch = min((cu_num + shape.batch - 1) // shape.batch, 32)
    sizes = aiter.get_mla_metadata_info_v1(
        shape.batch,
        shape.qlen,
        NHEAD,
        dtypes.fp8,
        dtypes.fp8,
        is_sparse=False,
        fast_mode=True,
        num_kv_splits=max_split_per_batch,
        intra_batch_mode=False,
    )
    ((wms, wmt), (wis, wit), (wss, wst), (ris, rit), (rfms, rfmt), (rpms, rpmt)) = sizes
    work_meta = torch.empty(wms, dtype=wmt, device="cuda")
    work_indptr = torch.empty(wis, dtype=wit, device="cuda")
    work_info = torch.empty(wss, dtype=wst, device="cuda")
    reduce_indptr = torch.empty(ris, dtype=rit, device="cuda")
    reduce_final_map = torch.empty(rfms, dtype=rfmt, device="cuda")
    reduce_partial_map = torch.empty(rpms, dtype=rpmt, device="cuda")

    qo_indptr = torch.arange(
        0, shape.total_q + 1, shape.qlen, dtype=torch.int32, device="cuda"
    )
    kv_indptr = torch.arange(
        0, shape.batch * shape.ctx + 1, shape.ctx, dtype=torch.int32, device="cuda"
    )
    kv_last_page_lens = torch.ones((shape.batch,), dtype=torch.int32, device="cuda")
    aiter.get_mla_metadata_v1(
        qo_indptr,
        kv_indptr,
        kv_last_page_lens,
        NHEAD,
        NHEAD_KV,
        False,
        work_meta,
        work_info,
        work_indptr,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        page_size=PAGE_SIZE,
        kv_granularity=16,
        max_seqlen_qo=shape.qlen,
        uni_seqlen_qo=shape.qlen,
        fast_mode=True,
        max_split_per_batch=max_split_per_batch,
        dtype_q=dtypes.fp8,
        dtype_kv=dtypes.fp8,
    )
    return (
        qo_indptr,
        kv_indptr,
        kv_last_page_lens,
        torch.arange(shape.batch * shape.ctx, dtype=torch.int32, device="cuda"),
        work_meta,
        work_indptr,
        work_info,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
    )


def alloc_outputs(shape: Shape, reduce_partial_map: torch.Tensor):
    split = torch.empty(
        (reduce_partial_map.size(0) * shape.qlen, 1, NHEAD, V_HEAD_DIM),
        dtype=torch.float32,
        device="cuda",
    )
    lse = torch.empty(
        (reduce_partial_map.size(0) * shape.qlen, 1, NHEAD, 1),
        dtype=torch.float32,
        device="cuda",
    )
    out = torch.empty(
        (shape.total_q, NHEAD, V_HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    )
    return split, lse, out


def reference_mla(q_fp8: torch.Tensor, kv_fp8: torch.Tensor, shape: Shape) -> torch.Tensor:
    q = q_fp8.float().view(shape.batch, shape.qlen, NHEAD, QK_HEAD_DIM)
    kv = kv_fp8.float().view(shape.batch, shape.ctx, QK_HEAD_DIM)
    v = kv[:, :, :V_HEAD_DIM]
    scores = torch.einsum("bqhd,bcd->bqhc", q, kv) / math.sqrt(QK_HEAD_DIM)
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bqhc,bcd->bqhd", probs, v)
    return out.reshape(shape.total_q, NHEAD, V_HEAD_DIM).to(torch.bfloat16)


def causal_reference_mla(
    q_fp8: torch.Tensor, kv_fp8: torch.Tensor, shape: Shape
) -> torch.Tensor:
    q = q_fp8.float().view(shape.batch, shape.qlen, NHEAD, QK_HEAD_DIM)
    kv = kv_fp8.float().view(shape.batch, shape.ctx, QK_HEAD_DIM)
    v = kv[:, :, :V_HEAD_DIM]
    outs = []
    for q_idx in range(shape.qlen):
        kv_end = max(1, shape.ctx - (shape.qlen - 1 - q_idx))
        scores = (
            torch.einsum("bhd,bcd->bhc", q[:, q_idx], kv[:, :kv_end])
            / math.sqrt(QK_HEAD_DIM)
        )
        probs = torch.softmax(scores, dim=-1)
        outs.append(torch.einsum("bhc,bcd->bhd", probs, v[:, :kv_end]))
    out = torch.stack(outs, dim=1)
    return out.reshape(shape.total_q, NHEAD, V_HEAD_DIM).to(torch.bfloat16)


def run_one(shape: Shape, use_long_kernel: bool = False) -> dict[str, object]:
    torch.manual_seed(1234 + shape.batch * 100000 + shape.ctx)
    q = torch.randn(
        (shape.total_q, NHEAD, QK_HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    ).to(dtypes.fp8)
    kv = torch.randn(
        (shape.batch * shape.ctx, PAGE_SIZE, NHEAD_KV, QK_HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
    ).to(dtypes.fp8)
    (
        qo_indptr,
        kv_indptr,
        kv_last_page_lens,
        kv_indices,
        work_meta,
        work_indptr,
        work_info,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
    ) = make_metadata(shape)

    sm_scale = 1.0 / math.sqrt(QK_HEAD_DIM)
    q_scale = torch.ones((1,), dtype=torch.float32, device="cuda")
    kv_scale = torch.ones((1,), dtype=torch.float32, device="cuda")

    asm_split, asm_lse, asm_out = alloc_outputs(shape, reduce_partial_map)
    hk_split, hk_lse, hk_out = alloc_outputs(shape, reduce_partial_map)

    aiter.mla_decode_stage1_asm_fwd(
        q,
        kv,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        None,
        work_meta,
        work_indptr,
        work_info,
        shape.qlen,
        PAGE_SIZE,
        NHEAD_KV,
        sm_scale,
        asm_split,
        asm_lse,
        asm_out,
        None,
        q_scale,
        kv_scale,
    )
    aiter.mla_reduce_v1(
        asm_split,
        asm_lse,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        shape.qlen,
        asm_out,
        None,
    )

    hk_decode = aiter.hk_mla_decode_fwd_long if use_long_kernel else aiter.hk_mla_decode_fwd
    hk_decode(
        q,
        kv,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        work_indptr,
        work_info,
        shape.qlen,
        sm_scale,
        hk_split,
        hk_lse,
        hk_out,
    )
    aiter.mla_reduce_v1(
        hk_split,
        hk_lse,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        shape.qlen,
        hk_out,
        None,
    )

    ref = reference_mla(q, kv, shape)
    causal_ref = causal_reference_mla(q, kv, shape)
    torch.cuda.synchronize()

    asm_err = error_metrics(ref, asm_out)
    hk_err = error_metrics(ref, hk_out)
    asm_causal_err = error_metrics(causal_ref, asm_out)
    hk_causal_err = error_metrics(causal_ref, hk_out)
    hk_vs_asm = error_metrics(asm_out, hk_out)
    row: dict[str, object] = {
        "batch": shape.batch,
        "ctx": shape.ctx,
        "total_kv": shape.batch * shape.ctx,
    }
    for prefix, metrics in (
        ("asm_ref", asm_err),
        ("hk_ref", hk_err),
        ("asm_causal_ref", asm_causal_err),
        ("hk_causal_ref", hk_causal_err),
        ("hk_asm", hk_vs_asm),
    ):
        for key, value in metrics.items():
            row[f"{prefix}_{key}"] = value
    print(
        f"B={shape.batch} ctx={shape.ctx} total={shape.batch * shape.ctx} | "
        f"ASM/ref cos={asm_err['cos']:.6f} rmse={asm_err['rmse']:.6f} "
        f"HK/ref cos={hk_err['cos']:.6f} rmse={hk_err['rmse']:.6f} "
        f"ASM/causal cos={asm_causal_err['cos']:.6f} "
        f"HK/causal cos={hk_causal_err['cos']:.6f} "
        f"HK/ASM cos={hk_vs_asm['cos']:.6f}",
        flush=True,
    )
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shapes",
        type=parse_shapes,
        default=parse_shapes("1x512,1x2048,1x8192,2x2048,4x2048,8x1024,16x512"),
        help="Comma-separated BxCTX list, e.g. 1x512,2x2048",
    )
    parser.add_argument("--csv", default="")
    parser.add_argument("--hk-long-kernel", action="store_true")
    args = parser.parse_args()

    torch.set_default_device("cuda")
    rows = [run_one(shape, args.hk_long_kernel) for shape in args.shapes]
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
