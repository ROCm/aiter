"""Test flydsl_moe_stage1 fuse_fp4_quant + verify non-quant precision unchanged.

Reference path:  stage1(bf16) -> TORCH_QUANT -> fp4 + e8m0_scale
Fused path:      stage1(fuse_fp4_quant=True) -> fp4 packed bytes

Also re-runs the non-quant correctness check (same as bench_stage1_async.py)
to confirm fuse_fp4_quant changes don't affect the normal code path.

Usage:
    python bench_fuse_quant.py
"""

import argparse
import torch
import aiter
from aiter import dtypes, QuantType, ActivationType
from aiter.test_common import checkAllclose
from aiter.fused_moe import fused_topk, moe_sorting, torch_moe_stage1
from aiter.ops.shuffle import shuffle_weight
from aiter.utility.fp4_utils import e8m0_shuffle, moe_mxfp4_sort
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1

torch.set_default_device("cuda")

Q_TYPE = QuantType.per_1x32
Q_DTYPE_A = dtypes.fp4x2
TORCH_QUANT = aiter.get_torch_quant(Q_TYPE)

CASES = [
    dict(token=1,  model_dim=7168, inter_dim=256,  expert=8,   topk=2, block_m=32),
    dict(token=2,  model_dim=7168, inter_dim=256,  expert=8,   topk=2, block_m=32),
    dict(token=4,  model_dim=7168, inter_dim=256,  expert=8,   topk=2, block_m=32),
    dict(token=8,  model_dim=7168, inter_dim=2048, expert=256, topk=8, block_m=16),
    dict(token=16, model_dim=7168, inter_dim=2048, expert=256, topk=8, block_m=16),
]


def setup_data(token, model_dim, inter_dim, E, topk, block_m,
               dtype=torch.bfloat16):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    inp = torch.randn((token, model_dim), dtype=dtype) / 10
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype) / 10

    score = torch.zeros((token, E), dtype=dtype)
    start_col, end_col = 0, topk
    for tid in range(token):
        score[tid, start_col:end_col] = 1.0
        start_col = end_col % E
        end_col = start_col + topk
    topk_weights, topk_ids = fused_topk(inp, score, topk, True)

    w1_qt, w1_scale = TORCH_QUANT(w1, quant_dtype=Q_DTYPE_A)
    w1_qt = w1_qt.view(E, inter_dim * 2, model_dim // 2)
    a1_qt, a1_scale = TORCH_QUANT(inp, quant_dtype=Q_DTYPE_A)

    sort_block_m = max(32, block_m)
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = \
        moe_sorting(topk_ids, topk_weights, E, model_dim, dtype, sort_block_m)

    needed = sorted_expert_ids.shape[0] * sort_block_m
    if sorted_ids.shape[0] < needed:
        pad = torch.full((needed - sorted_ids.shape[0],), token,
                         dtype=sorted_ids.dtype, device=sorted_ids.device)
        sorted_ids = torch.cat([sorted_ids, pad])
        sorted_weights = torch.cat([sorted_weights,
            torch.zeros(pad.shape[0], dtype=sorted_weights.dtype,
                        device=sorted_weights.device)])

    w1_qt_shuf = shuffle_weight(w1_qt, (16, 16))
    w1_scale_shuf = e8m0_shuffle(w1_scale)
    a1_scale_sort = moe_mxfp4_sort(
        a1_scale[:token, :].view(token, 1, -1),
        sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
        token_num=token, block_size=max(32, block_m),
    )

    w2_ref = torch.empty((E, model_dim, inter_dim // 2), dtype=w1_qt.dtype,
                          device=w1_qt.device)

    return dict(
        inp=inp, a1_qt=a1_qt, a1_scale=a1_scale,
        w1_qt=w1_qt, w1_scale=w1_scale,
        w1_qt_shuf=w1_qt_shuf, w1_scale_shuf=w1_scale_shuf,
        a1_scale_sort=a1_scale_sort,
        sorted_ids=sorted_ids, sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids, w2_ref=w2_ref,
        topk_weights=topk_weights, topk_ids=topk_ids,
    )


def call_stage1(d, topk, block_m, tile_n=128):
    return flydsl_moe_stage1(
        a=d["a1_qt"], w1=d["w1_qt_shuf"],
        sorted_token_ids=d["sorted_ids"],
        sorted_expert_ids=d["sorted_expert_ids"],
        num_valid_ids=d["num_valid_ids"],
        topk=topk, tile_m=block_m, tile_n=tile_n, tile_k=256,
        a_dtype="fp4", b_dtype="fp4", out_dtype="bf16",
        w1_scale=d["w1_scale_shuf"], a1_scale=d["a1_scale_sort"],
        use_async_copy=True,
    )


def call_stage1_fq(d, topk, block_m, tile_n=128, fuse_sort_scale=False):
    return flydsl_moe_stage1(
        a=d["a1_qt"], w1=d["w1_qt_shuf"],
        sorted_token_ids=d["sorted_ids"],
        sorted_expert_ids=d["sorted_expert_ids"],
        num_valid_ids=d["num_valid_ids"],
        topk=topk, tile_m=block_m, tile_n=tile_n, tile_k=256,
        a_dtype="fp4", b_dtype="fp4", out_dtype="bf16",
        w1_scale=d["w1_scale_shuf"], a1_scale=d["a1_scale_sort"],
        use_async_copy=True,
        fuse_fp4_quant=True,
        fuse_sort_scale=fuse_sort_scale,
    )


def extract_fp4_bytes(fq_tensor, token_num, topk, inter_dim):
    """Extract packed fp4 bytes from the kernel output buffer.

    Kernel writes inter_dim//2 bytes per row, rows indexed by t*topk+s.
    """
    raw = fq_tensor.contiguous().view(torch.uint8).reshape(-1)
    row_bytes = inter_dim // 2
    total_rows = token_num * topk
    return raw[:total_rows * row_bytes].view(total_rows, row_bytes)


def nibble_match_rate(a, b):
    a_flat, b_flat = a.view(-1), b.view(-1)
    lo = ((a_flat & 0x0F) == (b_flat & 0x0F)).float()
    hi = ((a_flat >> 4) == (b_flat >> 4)).float()
    return (lo.sum() + hi.sum()).item() / (2 * a_flat.shape[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--atol", type=float, default=0.01)
    parser.add_argument("--rtol", type=float, default=0.01)
    args = parser.parse_args()

    results = []
    print("\n" + "=" * 100)
    print("Part 1: Non-quant precision (verify no regression)")
    print("=" * 100)

    for c in CASES:
        token, block_m, topk = c["token"], c["block_m"], c["topk"]
        model_dim, inter_dim = c["model_dim"], c["inter_dim"]
        d = setup_data(token, model_dim, inter_dim, c["expert"], topk, block_m)

        ref_out = torch_moe_stage1(
            d["a1_qt"], d["w1_qt"], d["w2_ref"],
            d["topk_weights"], d["topk_ids"],
            dtype=torch.bfloat16,
            activation=ActivationType.Silu,
            quant_type=Q_TYPE,
            a1_scale=d["a1_scale"], w1_scale=d["w1_scale"],
        )
        torch.cuda.synchronize()

        for tn in [128, 64, 32]:
            fly_out = call_stage1(d, topk, block_m, tile_n=tn)
            torch.cuda.synchronize()
            err = checkAllclose(
                ref_out, fly_out,
                rtol=args.rtol, atol=args.atol,
                msg=f"  [t={token},tn={tn}] ",
            )
            tag = "PASS" if err == 0 else ("WARN" if err <= 0.05 else "FAIL")
            results.append(dict(token=token, tn=tn, inter=inter_dim,
                                test="bf16", result=tag))

    print("\n" + "=" * 100)
    print("Part 2: fuse_fp4_quant precision")
    print("=" * 100)

    for c in CASES:
        token, block_m, topk = c["token"], c["block_m"], c["topk"]
        model_dim, inter_dim = c["model_dim"], c["inter_dim"]

        print(f"\n  token={token}  model_dim={model_dim}  inter_dim={inter_dim}")
        d = setup_data(token, model_dim, inter_dim, c["expert"], topk, block_m)

        for tn in [128, 64, 32]:
            # Reference: stage1 bf16 → separate quant
            ref_bf16 = call_stage1(d, topk, block_m, tile_n=tn)
            torch.cuda.synchronize()
            ref_fp4, ref_scale = TORCH_QUANT(ref_bf16, quant_dtype=Q_DTYPE_A)
            torch.cuda.synchronize()

            total_rows = token * topk
            row_bytes = inter_dim // 2
            ref_bytes = ref_fp4.contiguous().view(torch.uint8).view(
                total_rows, row_bytes)

            # Fused quant
            try:
                fq_out = call_stage1_fq(d, topk, block_m, tile_n=tn,
                                        fuse_sort_scale=False)
                torch.cuda.synchronize()
                fq_tensor = fq_out[0] if isinstance(fq_out, tuple) else fq_out
                fq_bytes = extract_fp4_bytes(fq_tensor, token, topk, inter_dim)

                byte_match = (fq_bytes == ref_bytes).float().mean().item()
                nib_match = nibble_match_rate(fq_bytes, ref_bytes)

                tag = "PASS" if nib_match > 0.80 else (
                    "WARN" if nib_match > 0.60 else "FAIL")
                print(f"    tn={tn:>3d} fq:  nibble={nib_match*100:.1f}%  "
                      f"byte={byte_match*100:.1f}%  {tag}")
            except Exception as e:
                print(f"    tn={tn:>3d} fq:  ERROR: {e}")
                tag = "ERR"
                import traceback; traceback.print_exc()

            results.append(dict(token=token, tn=tn, inter=inter_dim,
                                test="fq", result=tag))

            # Fused quant + sort scale
            try:
                fqs_out = call_stage1_fq(d, topk, block_m, tile_n=tn,
                                         fuse_sort_scale=True)
                torch.cuda.synchronize()
                if isinstance(fqs_out, tuple):
                    fqs_tensor, fqs_scale = fqs_out
                    fqs_bytes = extract_fp4_bytes(fqs_tensor, token, topk,
                                                  inter_dim)
                    nib_s = nibble_match_rate(fqs_bytes, ref_bytes)
                    tag_s = "PASS" if nib_s > 0.80 else (
                        "WARN" if nib_s > 0.60 else "FAIL")
                    print(f"    tn={tn:>3d} fqs: nibble={nib_s*100:.1f}%  "
                          f"scale_shape={list(fqs_scale.shape)}  {tag_s}")
                else:
                    tag_s = "SKIP"
                    print(f"    tn={tn:>3d} fqs: non-tuple return")
            except Exception as e:
                print(f"    tn={tn:>3d} fqs: ERROR: {e}")
                tag_s = "ERR"
                import traceback; traceback.print_exc()

            results.append(dict(token=token, tn=tn, inter=inter_dim,
                                test="fqs", result=tag_s))

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"  {'token':>5s}  {'tn':>4s}  {'inter':>6s}  {'test':>5s}  {'result':>6s}")
    print(f"  {'-'*5}  {'-'*4}  {'-'*6}  {'-'*5}  {'-'*6}")
    for r in results:
        print(f"  {r['token']:>5d}  {r['tn']:>4d}  {r['inter']:>6d}  "
              f"{r['test']:>5s}  {r['result']:>6s}")


if __name__ == "__main__":
    main()
