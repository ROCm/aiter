from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import triton
import triton.language as tl


def assembly(swap: bool) -> tuple[str, str]:
    lines = [f"v_mov_b32 v{i}, 0" for i in range(16)]
    lines += [f"v_mov_b32 v{i}, 0x3c003c00" for i in range(64, 72)]
    lines += ["v_mov_b32 v78, 0.5", "v_mov_b32 v79, 1.0", "s_mov_b32 s4, 80"]
    lines += [
        ".Lmfma_${:uid}:",
        "v_mfma_f32_32x32x16_f16 v[0:15], v[64:67], v[68:71], v[0:15]",
    ]
    lines += ["s_nop 15"] * 4
    for i in range(0, 16, 2):
        pair = f"v[{i}:{i + 1}]"
        sources = (
            f"v[78:79], {pair} op_sel:[1,0]"
            if swap
            else (f"{pair}, v[78:79] op_sel:[0,1]")
        )
        lines.append(f"v_pk_mul_f32 {pair}, {sources}")
    lines += ["s_nop 15"] * 4
    lines += [
        "s_sub_u32 s4, s4, 1",
        "s_cmp_gt_u32 s4, 0",
        "s_cbranch_scc1 .Lmfma_${:uid}",
    ]
    lines += [f"v_mov_b32 ${i}, v{i}" for i in range(16)]
    scratch = [*range(16), *range(64, 72), 78, 79]
    constraints = ["=&v"] * 16 + ["v"]
    constraints += [f"~{{v{i}}}" for i in scratch] + ["~{s4}", "~{scc}"]
    return "\n".join(lines), ",".join(constraints)


@triton.jit
def reproduce(Out, ASM: tl.constexpr, CONSTRAINTS: tl.constexpr):
    lane = tl.arange(0, 256)
    values = tl.inline_asm_elementwise(
        ASM,
        constraints=CONSTRAINTS,
        args=[lane],
        dtype=(tl.float32,) * 16,
        is_pure=False,
        pack=1,
    )
    index = (tl.program_id(0) * 256 + lane) * 16
    for i in tl.static_range(16):
        tl.store(Out + index + i, values[i])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--swap", action="store_true")
    parser.add_argument("--dump", type=Path)
    args = parser.parse_args()
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip" or target.arch != "gfx950":
        parser.error("This reproducer requires a gfx950 GPU")
    asm, constraints = assembly(args.swap)
    output = torch.empty(2048 * 256 * 16, device="cuda", dtype=torch.float32)
    kernel = reproduce[(2048,)](output, asm, constraints, num_warps=4)
    torch.cuda.synchronize()
    if args.dump:
        args.dump.write_text(kernel.asm["amdgcn"])
    mismatches = (output != 1280.0).sum().item()
    print(
        json.dumps(
            {
                "swap": args.swap,
                "mismatches": mismatches,
                "elements": output.numel(),
                "expected": 1280.0,
                "max_error": (output - 1280.0).abs().max().item(),
            }
        )
    )
    return int(mismatches != 0)


if __name__ == "__main__":
    raise SystemExit(main())
