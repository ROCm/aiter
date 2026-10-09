#!/usr/bin/env python3
"""Run AITER's full Q256 test and verify dispatch and contribution storage."""

import argparse
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aiter-root", type=Path, required=True)
    parser.add_argument("--layout", choices=("sorted", "compact"), default="compact")
    parser.add_argument("--tokens", type=int, choices=(8192, 32768), default=8192)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--compare-producer", action="store_true",
                        help="Compare compact contributions exactly with the installed sorted producer")
    args = parser.parse_args()
    root = args.aiter_root.resolve()
    harness = root / "op_tests/test_moe_q256_prefill.py"
    if not harness.is_file():
        parser.error(f"AITER Q256 test is missing: {harness}")
    if not os.environ.get("AITER_CONFIG_FMOE"):
        parser.error("Set AITER_CONFIG_FMOE to the Q256 selector CSV before running")
    if not os.environ.get("AITER_JIT_DIR"):
        parser.error("Set AITER_JIT_DIR to a fresh cache directory before running")
    if args.compare_producer and args.layout != "compact":
        parser.error("--compare-producer requires --layout compact")
    if Path(os.environ.setdefault("AITER_META_DIR", str(root))).resolve() != root:
        parser.error("AITER_META_DIR must name the tested AITER checkout")
    sys.path.insert(0, str(root))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    import aiter
    import torch
    from aiter.ops import moe_op

    if Path(aiter.__file__).resolve().parent != root / "aiter":
        raise RuntimeError(f"Imported a different AITER checkout: {aiter.__file__}")

    native = moe_op.fmoe_q256
    storage = {}

    def verify_q256(*native_args, **native_kwargs):
        out, partials, _, _, _, sorted_ids = native_args[:6]
        tokens, model = out.shape
        topk = native_args[-1]
        assert (tokens, model, topk) == (args.tokens, 8192, 11), (
            tokens, model, topk
        )
        expected = (
            (tokens, topk, model)
            if args.layout == "compact"
            else (sorted_ids.numel(), model)
        )
        assert tuple(partials.shape) == expected, (
            f"Expected {args.layout} Q256 contributions {expected}, "
            f"received {tuple(partials.shape)}. Check the applied patches."
        )
        first_call = not storage
        if first_call:
            partial_bytes = partials.numel() * partials.element_size()
            sorted_bytes = sorted_ids.numel() * model * partials.element_size()
            storage.update(
                layout=args.layout,
                partial_shape=list(partials.shape),
                sorted_capacity=sorted_ids.numel(),
                partial_bytes=partial_bytes,
                sorted_partial_bytes=sorted_bytes,
                contribution_saving_gib=(sorted_bytes - partial_bytes) / 2**30,
            )
            print("Q256_STORAGE " + json.dumps(storage), flush=True)
        result = native(*native_args, **native_kwargs)
        if first_call and args.compare_producer:
            sorted_partials = torch.empty(
                (sorted_ids.numel(), model), dtype=partials.dtype, device=partials.device
            )
            baseline_args = list(native_args)
            baseline_args[0] = torch.empty_like(out)
            baseline_args[1] = sorted_partials
            moe_op.fmoe_q256_producer(*baseline_args, **native_kwargs)
            route_rows = partials.view(-1, model)
            # AITER reserves worst-case sorter capacity; the unused suffix
            # has unspecified IDs and is outside the producer's work count.
            valid_sorted_rows = int(native_args[8][0].item())
            assert 0 < valid_sorted_rows <= sorted_ids.numel()
            storage["valid_sorted_rows"] = valid_sorted_rows
            for start in range(0, valid_sorted_rows, 1024):
                stop = min(start + 1024, valid_sorted_rows)
                packed = sorted_ids[start:stop].to(torch.int64)
                token = packed & 0xFFFFFF
                valid = token < tokens
                logical = token[valid] * topk + (packed[valid] >> 24)
                expected_values = sorted_partials[start:stop][valid]
                actual_values = route_rows[logical]
                assert torch.equal(expected_values, actual_values), (
                    f"Compact producer contributions differ from hybrid at sorted rows {start}"
                )
            assert bool(torch.isfinite(partials).all())
            storage["exact_hybrid_contributions_equal"] = True
            print("Q256_EXACT_HYBRID_CONTRIBUTIONS_PASS", flush=True)
        return result

    moe_op.fmoe_q256 = verify_q256
    previous_argv = sys.argv
    sys.argv = [
        str(harness), "--tokens", str(args.tokens), "--model", "8192",
        "--reference", "--output", str(args.output),
    ]
    if args.profile:
        sys.argv.append("--profile")
    try:
        runpy.run_path(str(harness), run_name="__main__")
    finally:
        moe_op.fmoe_q256 = native
        sys.argv = previous_argv
    if not storage:
        raise RuntimeError(
            "The test did not dispatch Q256. Check AITER_CONFIG_FMOE and its shape keys."
        )
    result = json.loads(args.output.read_text())
    result["q256_storage"] = storage
    result["aiter_checkout"] = str(root)
    result["selector"] = os.environ["AITER_CONFIG_FMOE"]
    result["jit_dir"] = os.environ["AITER_JIT_DIR"]
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Q256_AITER_PASS " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
