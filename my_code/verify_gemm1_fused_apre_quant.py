#!/usr/bin/env python3
"""Compare fused GEMM1 quant output with the original standalone producer.

The grouped-MoE helper already performs a normal production launch followed by
an untimed stage-capture launch.  Installing ``quant_output_capture`` records
GEMM1's fused payload/ScaleA from the first launch and the standalone quant
payload/ScaleA from the second launch, using identical inputs and routing.

Only routed rows are compared because expert padding and the sentinel tail are
intentionally undefined in both implementations.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# These switches are read while the AITER modules are imported.
os.environ.setdefault("ENABLE_CK", "0")
os.environ.setdefault("AITER_MOE_EXPERT_BALANCE", "true")
os.environ.setdefault("AITER_LOG_MORE", "1")
os.environ.setdefault("AITER_USE_GROUPED_GEMM", "1")
os.environ.setdefault("AITER_GROUPED_DEBUG", "0")
os.environ.setdefault("AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE", "1")

# Import only after the environment above is complete; these modules read the
# switches during initialization.
torch = importlib.import_module("torch")
ActivationType = importlib.import_module("aiter").ActivationType
grouped = importlib.import_module("aiter.ops.flydsl.grouped_moe_gfx1250")
moe_test = importlib.import_module("my_code.test_flydsl_grouped_gemm_gfx1250")


def _hash128(tensor: torch.Tensor) -> str:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(
        tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    )
    return digest.hexdigest()


def _decode_payload_rows(
    payload: torch.Tensor,
    rows: torch.Tensor,
    *,
    feature_dim: int,
) -> torch.Tensor:
    """Undo ``shuffle_weight_f4`` row/MX-block ordering for selected rows."""
    payload_bytes_per_row = feature_dim // 2
    mx_blocks_per_row = feature_dim // 32
    flat = payload.detach().view(torch.uint8).reshape(-1).cpu()
    rows = rows.to(torch.long).cpu()
    decoded = torch.empty(
        (rows.numel(), payload_bytes_per_row), dtype=torch.uint8, device="cpu"
    )
    byte_lane = torch.arange(16, dtype=torch.long, device="cpu")
    row_tile16 = torch.div(rows, 16, rounding_mode="floor")
    row_in_tile16 = rows - row_tile16 * 16
    for mx_block in range(mx_blocks_per_row):
        offsets = (
            row_tile16 * (payload_bytes_per_row * 16)
            + mx_block * (16 * 16)
            + row_in_tile16 * 16
        )
        decoded[:, mx_block * 16 : (mx_block + 1) * 16] = flat[
            offsets[:, None] + byte_lane[None, :]
        ]
    return decoded


def _decode_scale_rows(
    scale: torch.Tensor,
    rows: torch.Tensor,
    *,
    feature_dim: int,
    wmma_rep: int,
) -> torch.Tensor:
    """Undo ``shuffle_scale_f4(..., wmma_rep)`` for selected rows."""
    scale_bytes_per_row = feature_dim // 32
    scale_dwords_per_row = scale_bytes_per_row // 4
    rows_per_tile = wmma_rep * 16
    flat = scale.detach().view(torch.uint8).reshape(-1).cpu()
    rows = rows.to(torch.long).cpu()
    decoded = torch.empty(
        (rows.numel(), scale_bytes_per_row), dtype=torch.uint8, device="cpu"
    )
    row_tile = torch.div(rows, rows_per_tile, rounding_mode="floor")
    row_in_tile = rows - row_tile * rows_per_tile
    wmma_row = torch.div(row_in_tile, 16, rounding_mode="floor")
    scale_lane = row_in_tile - wmma_row * 16
    for mx_block in range(scale_bytes_per_row):
        scale_dword = mx_block // 4
        byte_in_dword = mx_block & 3
        offsets = (
            (
                (
                    (row_tile * scale_dwords_per_row + scale_dword) * wmma_rep
                    + wmma_row
                )
                * 16
                + scale_lane
            )
            * 4
            + byte_in_dword
        )
        decoded[:, mx_block] = flat[offsets]
    return decoded


def _layout_self_test() -> None:
    rows = 192
    feature_dim = 2048
    wmma_rep = 6
    payload_bytes = feature_dim // 2
    scale_bytes = feature_dim // 32

    logical_payload = torch.arange(
        rows * payload_bytes, dtype=torch.int64, device="cpu"
    )
    logical_payload = (logical_payload & 0xFF).to(torch.uint8).view(rows, -1)
    shuffled_payload = torch.empty_like(logical_payload).reshape(-1)
    for row in range(rows):
        for mx_block in range(feature_dim // 32):
            src = logical_payload[row, mx_block * 16 : (mx_block + 1) * 16]
            dst = (
                (row // 16) * payload_bytes * 16
                + mx_block * 16 * 16
                + (row % 16) * 16
            )
            shuffled_payload[dst : dst + 16] = src

    logical_scale = torch.arange(
        rows * scale_bytes, dtype=torch.int64, device="cpu"
    )
    logical_scale = ((logical_scale * 17 + 3) & 0xFF).to(torch.uint8).view(rows, -1)
    shuffled_scale = torch.empty_like(logical_scale).reshape(-1)
    scale_dwords = scale_bytes // 4
    rows_per_tile = wmma_rep * 16
    for row in range(rows):
        row_tile = row // rows_per_tile
        row_in_tile = row % rows_per_tile
        wmma_row = row_in_tile // 16
        lane = row_in_tile % 16
        for mx_block in range(scale_bytes):
            dst = (
                (
                    (
                        (row_tile * scale_dwords + mx_block // 4) * wmma_rep
                        + wmma_row
                    )
                    * 16
                    + lane
                )
                * 4
                + (mx_block & 3)
            )
            shuffled_scale[dst] = logical_scale[row, mx_block]

    selected = torch.tensor(
        [0, 1, 15, 16, 31, 32, 95, 96, 191], device="cpu"
    )
    assert torch.equal(
        _decode_payload_rows(
            shuffled_payload, selected, feature_dim=feature_dim
        ),
        logical_payload[selected],
    )
    assert torch.equal(
        _decode_scale_rows(
            shuffled_scale,
            selected,
            feature_dim=feature_dim,
            wmma_rep=wmma_rep,
        ),
        logical_scale[selected],
    )


def _report(name: str, fused: torch.Tensor, reference: torch.Tensor) -> int:
    mismatch = fused != reference
    count = int(mismatch.sum())
    print(
        f"[{name}] bytes={fused.numel()} mismatch={count} "
        f"fused_hash128={_hash128(fused)} "
        f"reference_hash128={_hash128(reference)}",
        flush=True,
    )
    if count:
        first = mismatch.nonzero(as_tuple=False)[:16]
        for row, col in first.tolist():
            print(
                f"[{name}] row={row} byte={col} "
                f"fused=0x{int(fused[row, col]):02x} "
                f"reference=0x{int(reference[row, col]):02x}",
                flush=True,
            )
    return count


def _report_payload_permutation(
    fused_payload: torch.Tensor, reference_payload: torch.Tensor
) -> None:
    """Report the strongest reference-position match for each fused nibble."""

    def unpack(payload: torch.Tensor) -> torch.Tensor:
        low = payload & 0x0F
        high = payload >> 4
        return torch.stack((low, high), dim=-1).reshape(payload.shape[0], -1, 32)

    fused = unpack(fused_payload).reshape(-1, 32)[:4096]
    reference = unpack(reference_payload).reshape(-1, 32)[:4096]
    scores = (fused[:, :, None] == reference[:, None, :]).float().mean(dim=0)
    best_score, best_reference_pos = scores.max(dim=1)
    mapping = ", ".join(
        f"{fused_pos}->{int(best_reference_pos[fused_pos])}"
        f"({float(best_score[fused_pos]) * 100:.1f}%)"
        for fused_pos in range(32)
    )
    print(f"[payload permutation] {mapping}", flush=True)


def _report_fp4_semantic_mismatch(
    fused_payload: torch.Tensor, reference_payload: torch.Tensor
) -> int:
    """Compare FP4 nibbles after canonicalizing negative zero to positive zero."""

    def unpack(payload: torch.Tensor) -> torch.Tensor:
        low = payload & 0x0F
        high = payload >> 4
        return torch.stack((low, high), dim=-1).reshape(payload.shape[0], -1)

    fused = unpack(fused_payload)
    reference = unpack(reference_payload)
    fused = torch.where(fused == 0x8, 0, fused)
    reference = torch.where(reference == 0x8, 0, reference)
    mismatch = fused != reference
    count = int(mismatch.sum())
    print(
        f"[payload semantic] fp4_values={fused.numel()} "
        f"mismatch_after_zero_sign_normalization={count}",
        flush=True,
    )
    return count


def _report_tensor(name: str, fused: torch.Tensor, reference: torch.Tensor) -> None:
    fused = fused.detach().cpu()
    reference = reference.detach().cpu()
    bitwise_mismatch = int(
        (
            fused.contiguous().view(torch.uint8)
            != reference.contiguous().view(torch.uint8)
        ).sum()
    )
    diff = fused.float() - reference.float()
    rel_l2 = float(diff.norm() / reference.float().norm().clamp(min=1e-12))
    print(
        f"[{name}] bitwise_mismatch={bitwise_mismatch} "
        f"max_abs={float(diff.abs().max()):.6e} rel_l2={rel_l2:.6e} "
        f"fused_hash128={_hash128(fused)} "
        f"reference_hash128={_hash128(reference)}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--tokens", type=int, default=1536)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--model-dim", type=int, default=7168)
    parser.add_argument("--inter-dim", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--const-init", type=float, default=None)
    args = parser.parse_args()

    _layout_self_test()
    capture: dict[str, object] = {}
    if grouped.quant_output_capture is not None:
        raise RuntimeError("another quant-output capture is already active")
    grouped.quant_output_capture = capture
    try:
        out, reference, *_ = moe_test._run_grouped_via_fused_moe(
            experts=args.experts,
            tokens=args.tokens,
            topk=args.topk,
            model_dim=args.model_dim,
            inter_dim=args.inter_dim,
            data_format="a4w4",
            activation=ActivationType.Silu,
            use_bias=False,
            bench=False,
            kernel_bench=False,
            seed=args.seed,
            warmup=0,
            iters=1,
            const_init=args.const_init,
        )
    finally:
        grouped.quant_output_capture = None
    logits_diff = moe_test._logits_diff(out, reference)
    rel_l2 = moe_test._rel_l2(out, reference)
    passed = logits_diff < moe_test.LOGITS_DIFF_TOL

    required = {
        "fused_payload",
        "fused_scale",
        "reference_payload",
        "reference_scale",
        "fused_topids_to_rows",
        "reference_topids_to_rows",
        "contiguous_m",
        "inter_dim",
        "wmma_rep",
        "gemm1_a_preshuffle",
        "gemm2_a_preshuffle",
        "fused_gemm2_grouped_out",
        "reference_gemm2_grouped_out",
        "fused_moe_out",
        "reference_moe_out",
    }
    missing = required.difference(capture)
    if missing:
        raise RuntimeError(f"quant-output capture is incomplete: {sorted(missing)}")

    fused_route_rows = capture["fused_topids_to_rows"]
    reference_route_rows = capture["reference_topids_to_rows"]
    if fused_route_rows is None or reference_route_rows is None:
        raise RuntimeError("the diagnostic requires the non-EP routed-row layout")
    contiguous_m = int(capture["contiguous_m"])
    feature_dim = int(capture["inter_dim"])
    wmma_rep = int(capture["wmma_rep"])
    fused_route_rows = fused_route_rows.detach().reshape(-1).to(torch.long).cpu()
    reference_route_rows = (
        reference_route_rows.detach().reshape(-1).to(torch.long).cpu()
    )
    if not bool(
        ((fused_route_rows >= 0) & (fused_route_rows < contiguous_m)).all()
    ):
        raise RuntimeError("fused route map contains an invalid grouped row")
    if not bool(
        ((reference_route_rows >= 0) & (reference_route_rows < contiguous_m)).all()
    ):
        raise RuntimeError("reference route map contains an invalid grouped row")

    fused_payload = _decode_payload_rows(
        capture["fused_payload"], fused_route_rows, feature_dim=feature_dim
    )
    reference_payload = _decode_payload_rows(
        capture["reference_payload"], reference_route_rows, feature_dim=feature_dim
    )
    fused_scale = _decode_scale_rows(
        capture["fused_scale"],
        fused_route_rows,
        feature_dim=feature_dim,
        wmma_rep=wmma_rep,
    )
    reference_scale = _decode_scale_rows(
        capture["reference_scale"],
        reference_route_rows,
        feature_dim=feature_dim,
        wmma_rep=wmma_rep,
    )

    payload_mismatch = _report("payload", fused_payload, reference_payload)
    semantic_payload_mismatch = _report_fp4_semantic_mismatch(
        fused_payload, reference_payload
    )
    if payload_mismatch:
        _report_payload_permutation(fused_payload, reference_payload)
    scale_mismatch = _report("scale", fused_scale, reference_scale)
    fused_gemm2 = capture["fused_gemm2_grouped_out"].reshape(
        -1, args.model_dim
    ).index_select(
        0, fused_route_rows.to(capture["fused_gemm2_grouped_out"].device)
    )
    reference_gemm2 = capture["reference_gemm2_grouped_out"].reshape(
        -1, args.model_dim
    ).index_select(
        0,
        reference_route_rows.to(capture["reference_gemm2_grouped_out"].device),
    )
    _report_tensor("gemm2 valid rows", fused_gemm2, reference_gemm2)
    _report_tensor("moe output", capture["fused_moe_out"], capture["reference_moe_out"])
    print(
        "[layout] "
        f"gemm1_a_preshuffle={capture['gemm1_a_preshuffle']} "
        f"gemm2_a_preshuffle={capture['gemm2_a_preshuffle']} "
        f"wmma_rep={wmma_rep}",
        flush=True,
    )
    print(
        f"[moe] pass={passed} logits_diff={logits_diff:.4e} rel_l2={rel_l2:.4e}",
        flush=True,
    )
    return (
        0
        if passed and semantic_payload_mismatch == 0 and scale_mismatch == 0
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
