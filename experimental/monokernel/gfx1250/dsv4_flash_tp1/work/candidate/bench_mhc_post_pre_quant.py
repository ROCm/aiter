import json
from pathlib import Path
import torch, aiter
from triton.testing import do_bench_cudagraph
from aiter.ops.quant import per_group_quant_hip
from aiter.ops.triton.fusions.mhc import mhc_post_pre_quant_gfx1250

rows = []
for m in (1, 8, 16, 32, 64, 128):
    d = torch.load(f"/work/golden/m{m}_c7168_shuffle1.pt", weights_only=True)
    args = tuple(t.cuda() for t in d["args"])
    norm = d["norm"].cuda()

    def baseline():
        out = aiter.mhc_fused_post_pre(
            *args,
            norm_weight=norm,
            res_preshuffle=True,
            w_preshuffle_bf16=True,
            hc_post_mult_value=2.0,
            backend="gluon",
        )
        return (
            *out,
            *per_group_quant_hip(
                out[2],
                quant_dtype=torch.float8_e4m3fn,
                group_size=128,
                transpose_scale=True,
                scale_type=torch.float8_e8m0fnu,
            ),
        )

    def hybrid():
        return mhc_post_pre_quant_gfx1250(*args, norm_weight=norm, res_preshuffle=True)

    baseline()
    hybrid()
    torch.cuda.synchronize()
    # Alternate order to reduce clock/thermal ordering effects.
    bs = []
    hs = []
    for i in range(5):
        for label, fn in (
            [("b", baseline), ("h", hybrid)]
            if i % 2 == 0
            else [("h", hybrid), ("b", baseline)]
        ):
            val = do_bench_cudagraph(fn, rep=100) * 1000
            (bs if label == "b" else hs).append(val)
    import statistics

    rows.append(
        dict(
            m=m,
            baseline_us=statistics.median(bs),
            hybrid_us=statistics.median(hs),
            baseline_samples=bs,
            hybrid_samples=hs,
        )
    )
Path("/work/results/mhc_microbench.json").write_text(json.dumps(rows, indent=2))
