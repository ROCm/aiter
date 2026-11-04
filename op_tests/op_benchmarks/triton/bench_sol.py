# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Sol-Attn vs keep-or-drop sparse over the SAME preset selection, swept by sparsity.

Routing makes a Sol-Attn run's density a property of the data and beta, so timing a
routed run against a dense one mostly measures whichever density beta happened to pick.
Pinning the selection instead isolates what actually differs between mode 1 and mode 2:
the approximate pass over the pooled K/V. Both kernels here are handed the identical
mask, at every Sol-Attn recipe and geometry the manifest ships, optionally with the
Jensen term and KV ranges on the Sol-Attn side.

Two caveats on reading the delta. The kernels are separately compiled ASM objects, so at
matched density they can differ by more than the extra work -- a mode-2 row measuring
FASTER than its mode-1 counterpart means the two binaries schedule the same exact pass
differently, not that the sweep is free. And the cleanest number in the table is the
sparsity 1.0 row, where no block is exact and the difference is the sweep alone.

The mask is random rather than routed on purpose. Density is the only property of it
that the kernels' work depends on -- neither has data-dependent control flow, and the
pooled VALUES cannot change the instruction count -- so a random mask at a density times
like a routed one at that density while being something we can dial. Accuracy is
deliberately not reported: a random selection has nothing to do with the one routing
would choose, and the accuracy of the real thing is what op_tests/test_mha_v4_sparse.py
pins.

Shape is Wan 2.2 A14B self-attention (dim 5120, 40 heads, head_dim 128, patch (1,2,2),
VAE stride (4,8,8)), which is MHA rather than GQA. Ulysses shards heads inside
attention and the sequence outside it, so degree U keeps the full token count and leaves
40/U heads on each device.

Usage:
    python bench_sol.py                           # fp8, 720p, Ulysses 1 and 8
    python bench_sol.py --resolution 480p --ulysses 8
    python bench_sol.py --recipe f6f8 mxfp6_fp6p --block-tile all --jensen
    python bench_sol.py --recipe all --sparsity 0.5 0.8 1.0 --markdown
"""

import argparse
import functools
import sys
from collections.abc import Callable
from typing import NamedTuple

import torch
import triton

from aiter.ops.mha_v4 import (
    MHA_V4_SOL_MODE,
    MHA_V4_SPARSE_MODE,
    AttentionFormat,
    AttentionPack,
    AttentionScaleMode,
    mha_v4_block_tile,
    mha_v4_block_tiles,
    mha_v4_kv_tile_for_q_tile,
    mha_v4_operands,
    mha_v4_packed,
    mha_v4_q_multiplier,
    mxfp4_k_view,
    mxfp4_v_view,
    mxfp6_k_view,
    native_fp8_format,
    quantize_fp8,
    quantize_fp8_rotated,
    quantize_int8,
    quantize_mxfp4_k,
    quantize_mxfp4_q,
    quantize_mxfp6_k,
    quantize_mxfp6_q,
    quantize_mxfp8_k,
    quantize_mxfp8_q,
    quantize_v_fp8,
    quantize_v_mxfp4_fp6_p,
    quantize_v_mxfp6_fp6_p,
)
from aiter.ops.triton.attention.utils import (
    block_attn_mask_to_ragged_lut,
    sol_prepare,
)

# Wan 2.2 A14B transformer.
WAN_HEADS = 40
WAN_HEAD_DIM = 128
WAN_PATCH = (1, 2, 2)
WAN_VAE_STRIDE = (4, 8, 8)
SOFTMAX_SCALE = WAN_HEAD_DIM**-0.5

# (width, height, frames) for the two sampling resolutions Wan 2.2 ships.
RESOLUTIONS = {"720p": (1280, 720, 81), "480p": (832, 480, 81)}

# A KV range has to cover a whole number of this many KV tiles.
KV_RANGE_TILES = 32


def wan_self_attn_tokens(resolution: str) -> int:
    """Video tokens entering self-attention: VAE downsample, then 3D patchify.

    Self-attention in Wan runs over the video tokens alone; the text conditioning
    arrives through the separate cross-attention, so its 512 tokens do not belong in
    this sequence length.
    """
    width, height, frames = RESOLUTIONS[resolution]
    stride_t, stride_h, stride_w = WAN_VAE_STRIDE
    patch_t, patch_h, patch_w = WAN_PATCH
    latent_t = (frames - 1) // stride_t + 1
    return (
        (latent_t // patch_t)
        * (height // stride_h // patch_h)
        * (width // stride_w // patch_w)
    )


class Recipe(NamedTuple):
    """One Sol-Attn manifest row's operands: formats, scale modes and production quantizers."""

    qk_format: AttentionFormat
    qk_scale_mode: AttentionScaleMode
    quantize_q: Callable
    quantize_k: Callable
    quantize_v: Callable
    v_format: AttentionFormat
    v_scale_mode: AttentionScaleMode
    # Set when an operand's stored codes are sub-byte and permuted: sol_prepare cannot pool
    # them, so it pools from the BF16 source and quantizes the result again.
    packed_format: str | None = None
    v_packed_format: str | None = None
    v_pack: AttentionPack = AttentionPack.DEFAULT

    @property
    def formats(self):
        return self.qk_format, self.qk_format, self.v_format

    @property
    def scale_modes(self):
        return self.qk_scale_mode, self.qk_scale_mode, self.v_scale_mode

    def manifest_operands(self):
        return mha_v4_operands(*self.formats, *self.scale_modes, self.v_pack)


def _passthrough(t):
    # The NONE scale mode makes the kernel ignore the descale, so any tensor holds its slot.
    return t, t


def _mx_quantize_q(quantize):
    # The MX quantizers fold softmax_scale * log2(e) into Q.
    return lambda q: quantize(q, mha_v4_q_multiplier(SOFTMAX_SCALE))


def _mxfp4_quantize_k(k):
    raw, scale = quantize_mxfp4_k(k)
    return mxfp4_k_view(raw, scale), scale


def _mxfp4_quantize_v(v):
    raw, scale = quantize_v_mxfp4_fp6_p(v)
    return mxfp4_v_view(raw, scale, v.shape[1]), scale


def _mxfp6_quantize_k(k):
    raw, scale = quantize_mxfp6_k(k)
    return mxfp6_k_view(raw, scale, *k.shape[:3])


def build_recipes():
    """Every Sol-Attn recipe, keyed as op_tests/test_mha_v4_sparse.py names them."""
    fp8 = native_fp8_format()
    bf16 = AttentionFormat.BF16
    mxfp4 = AttentionFormat.MXFP4
    mxfp6 = AttentionFormat.MXFP6
    none = AttentionScaleMode.NONE
    per_tensor = AttentionScaleMode.F32_PER_TENSOR
    e8m0 = AttentionScaleMode.E8M0_PER_1X32
    fp6_p = AttentionPack.V_FOR_FP6_P
    mxfp4_q = _mx_quantize_q(quantize_mxfp4_q)
    mxfp6_q = _mx_quantize_q(quantize_mxfp6_q)
    fp8_qk = {
        "qk_format": fp8,
        "qk_scale_mode": per_tensor,
        "quantize_q": quantize_fp8_rotated,
        "quantize_k": quantize_fp8_rotated,
    }
    bf16_qk = {
        "qk_format": bf16,
        "qk_scale_mode": none,
        "quantize_q": _passthrough,
        "quantize_k": _passthrough,
    }
    mxfp6_qk = {
        "qk_format": mxfp6,
        "qk_scale_mode": e8m0,
        "quantize_q": mxfp6_q,
        "quantize_k": _mxfp6_quantize_k,
        "packed_format": "mxfp6",
    }
    fp8_v = {"quantize_v": quantize_fp8, "v_format": fp8, "v_scale_mode": per_tensor}
    mxfp4_v = {
        "quantize_v": _mxfp4_quantize_v,
        "v_format": mxfp4,
        "v_scale_mode": e8m0,
        "v_packed_format": "mxfp4_fp6_p",
        "v_pack": fp6_p,
    }
    mxfp6_v = {
        "quantize_v": quantize_v_mxfp6_fp6_p,
        "v_format": mxfp6,
        "v_scale_mode": e8m0,
        "v_packed_format": "mxfp6_fp6_p",
        "v_pack": fp6_p,
    }
    return {
        "fp8": Recipe(**fp8_qk, **fp8_v),
        "bf16": Recipe(
            **bf16_qk, quantize_v=_passthrough, v_format=bf16, v_scale_mode=none
        ),
        "bf16fp8": Recipe(**bf16_qk, **fp8_v),
        "i8fp8": Recipe(
            qk_format=AttentionFormat.INT8,
            qk_scale_mode=per_tensor,
            quantize_q=quantize_int8,
            quantize_k=quantize_int8,
            **fp8_v,
        ),
        "mxfp8": Recipe(
            qk_format=fp8,
            qk_scale_mode=e8m0,
            quantize_q=_mx_quantize_q(quantize_mxfp8_q),
            quantize_k=quantize_mxfp8_k,
            **fp8_v,
        ),
        "mxfp4_fp6p": Recipe(
            qk_format=mxfp4,
            qk_scale_mode=e8m0,
            quantize_q=mxfp4_q,
            quantize_k=_mxfp4_quantize_k,
            packed_format="mxfp4",
            **mxfp4_v,
        ),
        "mxfp6_fp6p": Recipe(**mxfp6_qk, **mxfp6_v),
        "f8f6_fp6p": Recipe(**fp8_qk, **mxfp6_v),
        "f6f4_fp6p": Recipe(**mxfp6_qk, **mxfp4_v),
        "f6f8": Recipe(
            **mxfp6_qk,
            quantize_v=quantize_v_fp8,
            v_format=fp8,
            v_scale_mode=AttentionScaleMode.F32_PER_CHANNEL,
        ),
    }


def recipe_block_tiles(recipe: Recipe, requested: list[str]) -> list[tuple[int, int]]:
    """The geometries to time: the recipe's default, every Sol-Attn one, or those named.

    Each must have both a Sol-Attn row and a sparse row for this recipe, since the point is
    to time the two over one mask cut to one tile.
    """
    operands = recipe.manifest_operands()
    shipped = mha_v4_block_tiles(operands, MHA_V4_SOL_MODE)
    tiles = []
    for name in requested:
        if name == "all":
            tiles.extend(shipped)
        elif name == "default":
            tiles.append(mha_v4_block_tile(operands, MHA_V4_SOL_MODE))
        else:
            q_tile, kv_tile = (int(x) for x in name.lower().split("x"))
            tiles.append((q_tile, kv_tile))
    tiles = list(dict.fromkeys(tiles))
    for q_tile, kv_tile in tiles:
        sparse_kv_tile = mha_v4_kv_tile_for_q_tile(q_tile, operands, MHA_V4_SPARSE_MODE)
        if (q_tile, kv_tile) not in shipped or sparse_kv_tile != kv_tile:
            raise ValueError(
                f"no Sol-Attn and sparse row pair at {q_tile}x{kv_tile} for this recipe; "
                f"Sol-Attn ships {list(shipped)}"
            )
    return tiles


def build_operands(seqlen: int, heads: int, recipe: Recipe, device="cuda"):
    """Quantize one set of BF16 operands through the production path for this recipe."""
    torch.manual_seed(0)
    shape = (1, seqlen, heads, WAN_HEAD_DIM)
    q = torch.randn(*shape, device=device, dtype=torch.bfloat16)
    k = torch.randn(*shape, device=device, dtype=torch.bfloat16)
    v = torch.randn(*shape, device=device, dtype=torch.bfloat16)

    q_quant, q_descale = recipe.quantize_q(q)
    k_quant, k_descale = recipe.quantize_k(k)
    v_quant, v_descale = recipe.quantize_v(v)
    packed_qk = recipe.packed_format is not None
    packed_v = recipe.v_packed_format is not None

    prepare_kwargs = {
        "k_packed_format": recipe.packed_format,
        "v_packed_format": recipe.v_packed_format,
        # A packed operand pools from its source; an addressable E8M0 one pools in dequantized
        # space and needs its scale. A per-tensor or per-channel descale survives pooling as is.
        "k_source": k if packed_qk else None,
        "v_source": v if packed_v else None,
        "k_scale": (
            k_descale
            if recipe.qk_scale_mode == AttentionScaleMode.E8M0_PER_1X32
            and not packed_qk
            else None
        ),
        "v_scale": (
            v_descale
            if recipe.v_scale_mode == AttentionScaleMode.E8M0_PER_1X32 and not packed_v
            else None
        ),
    }
    # Routing never runs over a supplied mask, but a packed Q is not even d wide, so hand
    # sol_prepare the source for its shape checks.
    routing_q = q if packed_qk else q_quant
    if not packed_qk:
        del q
    if not (packed_qk or packed_v):
        del k, v
    torch.cuda.empty_cache()
    return {
        "tensors": (q_quant, k_quant, v_quant, q_descale, k_descale, v_descale),
        "routing_q": routing_q,
        "prepare_kwargs": prepare_kwargs,
        "recipe": recipe,
    }


def _launch_args(operands):
    recipe = operands["recipe"]
    return (*operands["tensors"], *recipe.formats, *recipe.scale_modes)


def time_point(operands, mask, block_tile, jensen, kv_range_tokens, warmup, rep):
    """Time the sparse and Sol-Attn launches over one selection.

    sol_prepare owns the mask it returns -- it forces a short tail block onto the
    exact pass, which the approximate branch's constant full-block factor cannot
    represent -- so the sparse LUT is built from THAT mask, not the one handed in.
    Otherwise the two kernels would be timed over selections differing by one block.
    """
    recipe = operands["recipe"]
    _, k_quant, v_quant = operands["tensors"][:3]
    plan = sol_prepare(
        operands["routing_q"],
        k_quant,
        v_quant,
        BLOCK_M=block_tile[0],
        BLOCK_N=block_tile[1],
        block_attn_mask=mask,
        k_variance=jensen,
        **operands["prepare_kwargs"],
    )
    used = plan["block_attn_mask"]
    kv_idx, lut_start, lut_count = block_attn_mask_to_ragged_lut(used)
    lut_start, lut_count = lut_start.to(torch.int32), lut_count.to(torch.int32)

    launch = functools.partial(
        mha_v4_packed,
        *_launch_args(operands),
        v_pack=recipe.v_pack,
        softmax_scale=SOFTMAX_SCALE,
        block_tile=block_tile,
    )
    sparse = functools.partial(
        launch, kv_block_indices=kv_idx, lut_start=lut_start, lut_count=lut_count
    )
    sol_attn = functools.partial(
        launch,
        kv_block_indices=plan["kv_block_indices"],
        lut_start=plan["lut_start"],
        lut_count=plan["lut_count"],
        mean_k=plan["mean_k"],
        mean_v=plan["mean_v"],
        block_bitmap=plan["block_bitmap"],
        mean_k_scale=plan["mean_k_scale"],
        mean_v_scale=plan["mean_v_scale"],
        kv_range_tokens=kv_range_tokens,
    )

    result = {
        "density": used.float().mean().item(),
        "sparse_ms": triton.testing.do_bench(sparse, warmup=warmup, rep=rep),
        "sol_ms": triton.testing.do_bench(sol_attn, warmup=warmup, rep=rep),
    }
    if jensen:
        sol_jensen = functools.partial(
            sol_attn,
            mean_k_var=plan["mean_k_var"],
            mean_k_var_scale=plan["mean_k_var_scale"],
        )
        result["jensen_ms"] = triton.testing.do_bench(
            sol_jensen, warmup=warmup, rep=rep
        )
    del sparse, sol_attn, plan, kv_idx, lut_start, lut_count
    torch.cuda.empty_cache()
    return result


def _time_dense(operands, warmup, rep):
    def dense():
        return mha_v4_packed(
            *_launch_args(operands),
            v_pack=operands["recipe"].v_pack,
            softmax_scale=SOFTMAX_SCALE,
        )

    return triton.testing.do_bench(dense, warmup=warmup, rep=rep)


def run_recipe(name, recipe, degree, tokens, args):
    heads = WAN_HEADS // degree
    tiles = recipe_block_tiles(recipe, args.block_tile)
    # Mode 1 requires a key length that is a whole number of KV tiles, and Wan's token
    # counts are not (75600 leaves 80 over a 128 tile, 32760 leaves 120). Padding up is
    # what a deployment does regardless, and it also keeps the two kernels honest: a
    # short tail block is forced onto Sol-Attn's exact pass and would otherwise be one
    # block of selection the sparse row never saw. Padded to the coarsest tile timed,
    # so every geometry sees the same sequence.
    pad = max(kv_tile for _, kv_tile in tiles)
    seqlen = -(-tokens // pad) * pad
    dense_flops = 2.0 * heads * seqlen * seqlen * 2 * WAN_HEAD_DIM

    operands = build_operands(seqlen, heads, recipe)
    dense_ms = _time_dense(operands, args.warmup, args.rep)
    print(
        f"\n=== Wan 2.2 self-attn, Ulysses {degree}: "
        f"{heads} heads x {seqlen} tokens x d{WAN_HEAD_DIM}, {name} ===\n"
        f"{tokens} real tokens padded to {seqlen}, {dense_flops / 1e12:.1f} TFLOP dense, "
        f"dense (mode 0) {dense_ms:.4f} ms"
    )

    rows = []
    for block_tile in tiles:
        q_tile, kv_tile = block_tile
        num_q_tiles = -(-seqlen // q_tile)
        num_kv_blocks = seqlen // kv_tile
        if args.kv_range_tokens % (KV_RANGE_TILES * kv_tile):
            raise ValueError(
                f"--kv-range-tokens must be a multiple of {KV_RANGE_TILES} KV tiles "
                f"({KV_RANGE_TILES * kv_tile} tokens at {q_tile}x{kv_tile})"
            )
        header = (
            f"{'sparsity':>9} {'density':>8} {'sparse(ms)':>11} {'sol-attn(ms)':>13}"
            f" {'delta':>8}"
        )
        if args.jensen:
            header += f" {'+jensen(ms)':>12} {'delta':>8}"
        header += f" {'sol vs dense':>13}"
        print(
            f"\n{q_tile}x{kv_tile}: {num_q_tiles} query tiles x {num_kv_blocks} KV blocks"
            + (
                f", KV ranges of {args.kv_range_tokens} tokens"
                if args.kv_range_tokens
                else ""
            )
        )
        print(header)
        print("-" * len(header))
        for sparsity in args.sparsity:
            torch.manual_seed(int(sparsity * 1000))
            mask = (
                torch.rand(1, heads, num_q_tiles, num_kv_blocks, device="cuda")
                > sparsity
            )
            point = time_point(
                operands,
                mask,
                block_tile,
                args.jensen,
                args.kv_range_tokens,
                args.warmup,
                args.rep,
            )
            del mask
            torch.cuda.empty_cache()
            sparse_ms, sol_ms = point["sparse_ms"], point["sol_ms"]
            line = (
                f"{sparsity:>9.1f} {100 * point['density']:>7.1f}% {sparse_ms:>11.4f}"
                f" {sol_ms:>13.4f} {(sol_ms / sparse_ms - 1) * 100:>+7.1f}%"
            )
            if args.jensen:
                line += (
                    f" {point['jensen_ms']:>12.4f}"
                    f" {(point['jensen_ms'] / sparse_ms - 1) * 100:>+7.1f}%"
                )
            line += f" {dense_ms / sol_ms:>12.2f}x"
            print(line)
            rows.append(
                {
                    "recipe": name,
                    "degree": degree,
                    "heads": heads,
                    "seqlen": seqlen,
                    "tile": f"{q_tile}x{kv_tile}",
                    "sparsity": sparsity,
                    "dense_ms": dense_ms,
                    **point,
                }
            )

    del operands
    torch.cuda.empty_cache()
    return rows


def print_markdown(rows, jensen):
    """One table over every point, for pasting into a PR description."""
    columns = [
        "recipe",
        "tile",
        "U",
        "sparsity",
        "density",
        "sparse ms",
        "Sol ms",
        "Sol vs sparse",
    ]
    if jensen:
        columns += ["Sol+Jensen ms", "Sol+Jensen vs sparse"]
    columns += ["dense ms", "Sol vs dense"]
    print("\n| " + " | ".join(columns) + " |")
    print("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        cells = [
            row["recipe"],
            row["tile"],
            str(row["degree"]),
            f"{row['sparsity']:.1f}",
            f"{100 * row['density']:.1f}%",
            f"{row['sparse_ms']:.3f}",
            f"{row['sol_ms']:.3f}",
            f"{(row['sol_ms'] / row['sparse_ms'] - 1) * 100:+.1f}%",
        ]
        if jensen:
            cells += [
                f"{row['jensen_ms']:.3f}",
                f"{(row['jensen_ms'] / row['sparse_ms'] - 1) * 100:+.1f}%",
            ]
        cells += [f"{row['dense_ms']:.3f}", f"{row['dense_ms'] / row['sol_ms']:.2f}x"]
        print("| " + " | ".join(cells) + " |")


def main() -> int:
    recipes = build_recipes()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--resolution",
        default="720p",
        choices=sorted(RESOLUTIONS),
        help="Wan sampling size",
    )
    parser.add_argument(
        "--ulysses",
        type=int,
        nargs="+",
        default=[1, 8],
        help="Ulysses degrees; each leaves 40/degree heads on a device",
    )
    parser.add_argument(
        "--recipe",
        nargs="+",
        default=["fp8"],
        choices=[*recipes, "all"],
        help="Sol-Attn recipes to time; 'all' is every recipe this GPU has a Sol-Attn row for",
    )
    parser.add_argument(
        "--block-tile",
        nargs="+",
        default=["default"],
        metavar="QxKV",
        help="geometries to time, e.g. 64x64; 'default' is the recipe's default and 'all' "
        "every Sol-Attn geometry it ships",
    )
    parser.add_argument(
        "--jensen",
        action="store_true",
        help="also time Sol-Attn with the Jensen (K variance) term",
    )
    parser.add_argument(
        "--kv-range-tokens",
        type=int,
        default=0,
        help=f"Sol-Attn KV range length, a multiple of {KV_RANGE_TILES} KV tiles; "
        "0 is one range",
    )
    parser.add_argument("--warmup", type=int, default=25, help="do_bench warmup ms")
    parser.add_argument("--rep", type=int, default=100, help="do_bench rep ms")
    parser.add_argument(
        "--sparsity",
        type=float,
        nargs="+",
        default=[round(0.1 * i, 1) for i in range(1, 11)],
        help="block sparsity levels; exact density is 1 - sparsity",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="finish with one markdown table over every point",
    )
    args = parser.parse_args()

    if "all" in args.recipe:
        names = [
            name
            for name, recipe in recipes.items()
            if mha_v4_block_tiles(recipe.manifest_operands(), MHA_V4_SOL_MODE)
        ]
    else:
        names = list(dict.fromkeys(args.recipe))
    tokens = wan_self_attn_tokens(args.resolution)
    rows = []
    for degree in args.ulysses:
        if WAN_HEADS % degree:
            raise ValueError(
                f"Ulysses degree {degree} does not divide {WAN_HEADS} heads"
            )
        for name in names:
            rows += run_recipe(name, recipes[name], degree, tokens, args)
    if args.markdown:
        print_markdown(rows, args.jensen)
    return 0


if __name__ == "__main__":
    sys.exit(main())
