# SPDX-License-Identifier: MIT
"""Single-GPU GMM1 physical-row invariance diagnostic for MI355.

This test deliberately bypasses Stage1 dispatch, CCO, LSA, queues and Stage2.
It constructs several BM32 physical tiles containing byte-identical A4 rows,
identical preshuffled E8M0 input scales, identity ``tile_row_input`` and one
fixed expert.  The production standalone A4W4 GMM1 is launched twice on the
same buffers.  Correct row-independent GEMM semantics require:

* every physical row's H1 A4 payload to be bitwise identical;
* every physical row's inverse-preshuffled H1 scale row to be identical;
* the complete H1 payload and scale outputs to match across both launches.

The script is intentionally standalone rather than a pytest test so it can be
run directly in the MI355 container without distributed initialization::

    PYTHONPATH=/home/hzm/aiter python3 \
      op_tests/multigpu_tests/test_megamoe_tile_gemm1_physical_layout_invariance.py \
      --device 0 --physical-tiles 4
"""

from __future__ import annotations

import argparse
import hashlib
import json

import torch


HIDDEN = 7168
INTER = 3072
BM = 32
BN = 256
BK = 256


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--physical-tiles", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260819)
    return parser.parse_args()


def _scale_byte_indices(rows: int, scale_bytes: int, device: torch.device) -> torch.Tensor:
    """Map canonical [row, scale-byte] to the GMM consumer's BM32 ABI.

    One 32-row tile is laid out as ``(ku, kl, nl, ik, im)`` where ``ku``
    selects an eight-scale K group, ``ik`` its lower/upper group of four,
    ``kl`` the byte within that group, and ``(im, nl)`` identify the row.
    This is intentionally not the older row-major/interleaved-dword formula.
    """

    if rows <= 0 or rows % BM:
        raise ValueError("rows must be a positive multiple of BM32")
    if scale_bytes <= 0 or scale_bytes % 4:
        raise ValueError("scale bytes per row must be a positive multiple of four")
    row = torch.arange(rows, dtype=torch.int64, device=device).view(rows, 1)
    byte = torch.arange(scale_bytes, dtype=torch.int64, device=device).view(1, scale_bytes)
    physical_tile = torch.div(row, BM, rounding_mode="floor")
    row_in_tile = row - physical_tile * BM
    im = torch.div(row_in_tile, 16, rounding_mode="floor")
    nl = row_in_tile - im * 16
    ku = torch.div(byte, 8, rounding_mode="floor")
    byte_in_ku = byte - ku * 8
    ik = torch.div(byte_in_ku, 4, rounding_mode="floor")
    kl = byte_in_ku - ik * 4
    tile_dwords = scale_bytes * BM // 4
    dword = physical_tile * tile_dwords + ku * 64 + kl * 16 + nl
    return dword * 4 + ik * 2 + im


def _preshuffle_scale_rows(raw: torch.Tensor) -> torch.Tensor:
    if raw.ndim != 2 or raw.dtype != torch.uint8 or not raw.is_cuda:
        raise ValueError("raw scales must be a CUDA uint8 [rows, scale_bytes] tensor")
    rows, scale_bytes = (int(raw.shape[0]), int(raw.shape[1]))
    indices = _scale_byte_indices(rows, scale_bytes, raw.device)
    flat = torch.empty(rows * scale_bytes, dtype=torch.uint8, device=raw.device)
    flat[indices.reshape(-1)] = raw.reshape(-1)
    return flat.contiguous()


def _inverse_preshuffle_scale_rows(
    shuffled: torch.Tensor, rows: int, scale_bytes: int
) -> torch.Tensor:
    if shuffled.dtype != torch.uint8 or not shuffled.is_cuda:
        raise ValueError("shuffled scales must be CUDA uint8")
    if shuffled.numel() < rows * scale_bytes:
        raise ValueError("shuffled scale allocation is too small")
    indices = _scale_byte_indices(rows, scale_bytes, shuffled.device)
    return shuffled.reshape(-1)[indices].contiguous()


def _sha256(tensor: torch.Tensor) -> str:
    payload = tensor.detach().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _row_mismatches(rows: torch.Tensor) -> int:
    if rows.ndim != 2 or rows.shape[0] == 0:
        raise ValueError("expected a non-empty rank-two row tensor")
    return int(torch.count_nonzero(torch.any(rows != rows[0:1], dim=1)).item())


def _rows_equal_to_row0(rows: torch.Tensor) -> list[int]:
    if rows.ndim != 2 or rows.shape[0] == 0:
        raise ValueError("expected a non-empty rank-two row tensor")
    equal = torch.all(rows == rows[0:1], dim=1)
    return [int(value) for value in torch.nonzero(equal).reshape(-1).cpu().tolist()]


def _element_mismatches(left: torch.Tensor, right: torch.Tensor) -> int:
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError("mismatch operands must have the same shape and dtype")
    return int(torch.count_nonzero(left != right).item())


@torch.no_grad()
def main() -> int:
    args = _parse_args()
    if args.physical_tiles <= 1:
        raise ValueError("--physical-tiles must be greater than one")
    if not torch.cuda.is_available():
        raise RuntimeError("a ROCm CUDA device is required")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    properties = torch.cuda.get_device_properties(device)
    rows = args.physical_tiles * BM

    from aiter.ops.flydsl.kernels.megamoe_tile.gemm1 import (
        compile_gemm1_a4w4_port,
    )
    from aiter.ops.quant import per_1x32_f4_quant
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility.fp4_utils import e8m0_shuffle

    generator = torch.Generator(device=device).manual_seed(args.seed)

    # One fixed BF16 activation is quantized once, then copied byte-for-byte to
    # every physical row. Its raw E8M0 scales are independently preshuffled by
    # the exact BM32 layout consumed by GMM1.
    x = torch.randn((1, HIDDEN), dtype=torch.bfloat16, device=device, generator=generator)
    x.mul_(HIDDEN**-0.25)
    aq_one, ascale_one = per_1x32_f4_quant(x, shuffle=False)
    # PyTorch intentionally implements very few tensor transforms for the
    # packed Float4 dtype. GMM1 consumes bytes, so repeat its byte view without
    # changing the underlying A4 representation.
    aq_one = (
        aq_one.reshape(1, HIDDEN // 2)
        .contiguous()
        .view(torch.uint8)
    )
    ascale_one = ascale_one.reshape(1, HIDDEN // 32).view(torch.uint8).contiguous()
    aq = aq_one.repeat(rows, 1).contiguous()
    raw_ascale = ascale_one.repeat(rows, 1).contiguous()
    ascale = _preshuffle_scale_rows(raw_ascale)

    # Native separated GGUU A4W4 W1 layout, matching the two-kernel operator.
    w1 = torch.randn(
        (1, 2 * INTER, HIDDEN),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    w1.mul_(HIDDEN**-0.25)
    w1q, w1scale = per_1x32_f4_quant(w1, shuffle=False)
    del w1
    w1q = shuffle_weight(w1q, layout=(16, 16)).contiguous()
    w1scale = e8m0_shuffle(w1scale).contiguous()
    torch.cuda.empty_cache()

    tile_expert = torch.zeros(
        args.physical_tiles, dtype=torch.int32, device=device
    )
    num_valid = torch.tensor([rows], dtype=torch.int32, device=device)
    tile_row_input = torch.arange(rows, dtype=torch.int32, device=device)
    h1_scale_bytes = INTER // 32
    out_q = [
        torch.full(
            (rows, INTER // 2),
            0xA5,
            dtype=torch.uint8,
            device=device,
        )
        for _ in range(2)
    ]
    out_scale = [
        torch.full(
            (rows * h1_scale_bytes,),
            0xA5,
            dtype=torch.uint8,
            device=device,
        )
        for _ in range(2)
    ]
    hidden_dummy = torch.empty(1, dtype=torch.bfloat16, device=device)

    launcher = compile_gemm1_a4w4_port(
        BM=BM,
        use_nt=True,
        inline_quant=False,
        D_HIDDEN=HIDDEN,
        D_INTER=INTER,
        NE=1,
        TOPK=16,
        BN=BN,
        BK=BK,
        interleave=False,
        act="silu",
        persistent=False,
    )
    jobs = args.physical_tiles * ((2 * INTER) // BN)
    stream = torch.cuda.current_stream(device)
    for run in range(2):
        launcher(
            aq.data_ptr(),
            ascale.data_ptr(),
            w1q.data_ptr(),
            w1scale.data_ptr(),
            tile_expert.data_ptr(),
            num_valid.data_ptr(),
            tile_row_input.data_ptr(),
            rows,
            jobs,
            out_q[run].data_ptr(),
            out_scale[run].data_ptr(),
            hidden_dummy.data_ptr(),
            stream=stream,
        )
    torch.cuda.synchronize(device)

    canonical_scale = [
        _inverse_preshuffle_scale_rows(value, rows, h1_scale_bytes)
        for value in out_scale
    ]
    input_scale_roundtrip = _inverse_preshuffle_scale_rows(
        ascale, rows, HIDDEN // 32
    )
    summary = {
        "device": str(properties.name),
        "arch": str(getattr(properties, "gcnArchName", "unknown")),
        "shape": {
            "hidden": HIDDEN,
            "inter": INTER,
            "bm": BM,
            "bn": BN,
            "bk": BK,
            "physical_tiles": args.physical_tiles,
            "rows": rows,
            "jobs": jobs,
            "expert": 0,
        },
        "input": {
            "q_sha256": _sha256(aq),
            "canonical_scale_sha256": _sha256(raw_ascale),
            "preshuffled_scale_sha256": _sha256(ascale),
            "q_rows_different_from_row0": _row_mismatches(aq),
            "scale_rows_different_from_row0": _row_mismatches(raw_ascale),
            "scale_roundtrip_element_mismatches": _element_mismatches(
                raw_ascale, input_scale_roundtrip
            ),
            "identity_row_map": bool(
                torch.equal(
                    tile_row_input,
                    torch.arange(rows, dtype=torch.int32, device=device),
                )
            ),
        },
        "run1": {
            "q_sha256": _sha256(out_q[0]),
            "scale_sha256": _sha256(canonical_scale[0]),
            "q_rows_different_from_row0": _row_mismatches(out_q[0]),
            "scale_rows_different_from_row0": _row_mismatches(canonical_scale[0]),
            "q_rows_equal_to_row0": _rows_equal_to_row0(out_q[0]),
            "scale_rows_equal_to_row0": _rows_equal_to_row0(canonical_scale[0]),
        },
        "run2": {
            "q_sha256": _sha256(out_q[1]),
            "scale_sha256": _sha256(canonical_scale[1]),
            "q_rows_different_from_row0": _row_mismatches(out_q[1]),
            "scale_rows_different_from_row0": _row_mismatches(canonical_scale[1]),
            "q_rows_equal_to_row0": _rows_equal_to_row0(out_q[1]),
            "scale_rows_equal_to_row0": _rows_equal_to_row0(canonical_scale[1]),
        },
        "cross_run": {
            "q_element_mismatches": _element_mismatches(out_q[0], out_q[1]),
            "scale_element_mismatches": _element_mismatches(
                canonical_scale[0], canonical_scale[1]
            ),
        },
    }
    passed = (
        summary["input"]["q_rows_different_from_row0"] == 0
        and summary["input"]["scale_rows_different_from_row0"] == 0
        and summary["input"]["scale_roundtrip_element_mismatches"] == 0
        and summary["run1"]["q_rows_different_from_row0"] == 0
        and summary["run1"]["scale_rows_different_from_row0"] == 0
        and summary["run2"]["q_rows_different_from_row0"] == 0
        and summary["run2"]["scale_rows_different_from_row0"] == 0
        and summary["cross_run"]["q_element_mismatches"] == 0
        and summary["cross_run"]["scale_element_mismatches"] == 0
    )
    summary["passed"] = passed
    print(
        "MEGAMOE_GMM1_PHYSICAL_LAYOUT_INVARIANCE "
        + json.dumps(summary, sort_keys=True),
        flush=True,
    )
    print(
        "MEGAMOE_GMM1_PHYSICAL_LAYOUT_INVARIANCE_"
        + ("PASS" if passed else "FAIL"),
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
