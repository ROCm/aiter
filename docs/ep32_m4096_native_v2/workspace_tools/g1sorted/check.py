"""Actual bench input: sorted-copy oracle and frozen GEMM1 comparison."""
import functools
import importlib.util
import json
import sys
from pathlib import Path
import torch
from aiter.ops.flydsl import mxfp4_gemm1_kernels as impl
from aiter.ops.flydsl.kernels.mxfp4_gemm1 import gemm1_grid
from aiter.ops.flydsl.kernels.mxfp4_gemm_common import kas_per_chunk_dw_for
from aiter.ops.flydsl.moe_kernels import _run_compiled
from aiter.ops.flydsl.ep32_sorted_a import prepare


ROOT = Path(__file__).resolve().parent
output = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT.parent.parent / "amd_moe_bench_share"))
import bench_moe
spec = importlib.util.spec_from_file_location("aiter.ops.flydsl.kernels._cx_g1_sorted_ref", ROOT / "source/baseline_kernel.py")
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)
original = impl.flydsl_mxfp4_gemm1
done = False


@functools.wraps(original)
def check(**kw):
    global done
    if done:
        return original(**kw)
    done = True
    assert (kw["BM"], kw["n_tokens"], kw["NE"], kw["D_HIDDEN"], kw["D_INTER"]) == (80, 4096, 28, 3584, 3072)
    rows, pad = kw["cumsum_tensor"].cpu().tolist()
    mind = kw["m_indices"][:rows].long()
    live = (mind >= 0) & (mind < kw["n_tokens"])
    expected = torch.zeros((rows, 3584), device=mind.device, dtype=torch.uint8)
    expected[live] = kw["a_quant"].view(torch.uint8)[mind[live]]
    packed = prepare(kw["a_quant"], kw["m_indices"], kw["cumsum_tensor"], kw["n_tokens"])
    assert torch.equal(packed[:rows], expected), "sorted copy differs"
    ref = torch.empty_like(kw["inter_sorted_quant"])
    ref.view(torch.uint8).fill_(127)
    scale = torch.empty_like(kw["inter_sorted_shuffled_scale"])
    scale.view(torch.uint8).fill_(255)
    params = {k: kw[k] for k in ("BM", "use_nt", "inline_quant", "D_HIDDEN", "D_INTER", "NE", "BN", "BK", "interleave", "a_dtype", "out_dtype", "act", "situ_beta", "situ_linear_beta", "swiglu_limit")}
    fn = baseline.compile_gemm1_ep32(**params)
    args = [kw[k].data_ptr() for k in ("a_quant", "a_scale_sorted_shuffled", "w1_u8", "w1_scale_u8", "sorted_expert_ids", "cumsum_tensor", "m_indices")]
    args += [kw["n_tokens"], gemm1_grid(4096, 80, NE=28, TOPK=kw["topk"], INTER=3072),
        ref.data_ptr(), scale.data_ptr(), kw["hidden_states"].data_ptr(), 0, torch.cuda.current_stream()]
    _run_compiled(fn, tuple(args))
    scale_bytes = ((rows + 31) // 32) * kas_per_chunk_dw_for(3072) * 4
    for replay in range(3):
        kw["inter_sorted_quant"].view(torch.uint8).fill_(127)
        kw["inter_sorted_shuffled_scale"].view(torch.uint8).fill_(255)
        original(**kw)
        torch.cuda.synchronize()
        assert torch.equal(kw["inter_sorted_quant"].view(torch.uint8)[:rows], ref.view(torch.uint8)[:rows]), replay
        assert torch.equal(kw["inter_sorted_shuffled_scale"].view(torch.uint8).flatten()[:scale_bytes], scale.view(torch.uint8).flatten()[:scale_bytes]), replay
        assert torch.isfinite(kw["inter_sorted_quant"][:rows].float()).all(), "unwritten output"
    result = {"routes": int(live.sum()), "sorted_rows": rows, "pad_token": pad,
        "replays": 3, "sorted_copy_bit_exact": True, "all_written_payload_bit_exact": True,
        "all_written_scale_bit_exact": True, "all_written_output_finite": True}
    (output / "check_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


impl.flydsl_mxfp4_gemm1 = check
args = bench_moe.build_parser().parse_args(["run", "--output-dir", str(output / "input"),
    "--batch-sizes", "4096", "--ep", "32", "--rank", "0", "--warmup", "1", "--samples", "1"])
bench_moe.run(args)
assert done, "target GEMM1 path was not exercised"
