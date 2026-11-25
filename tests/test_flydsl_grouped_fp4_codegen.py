"""Offline codegen and AOT selection for gfx1250-strict grouped FP4 MoE.

Device correctness remains in the public fused_moe grouped op test. These
checks compile both GEMM stages without launching on a GPU and require real
matrix instructions in the result, so a hollow A0 object cannot pass.
"""

import inspect
from pathlib import Path

import pytest

flyc = pytest.importorskip("flydsl.compiler")
import flydsl.expr as fx

from aiter.ops.flydsl.kernels import mxfp4_preshuffle_gfx1250_tdm as kernel


@pytest.mark.parametrize("stage1,k", [(1, 4096), (0, 512)])
@pytest.mark.parametrize("interleaved", [0, 1])
@pytest.mark.parametrize("preshuffled", [0, 1])
def test_strict_fp4_codegen_has_supported_mma(
    tmp_path, monkeypatch, stage1, k, interleaved, preshuffled
):
    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_GPU_ARCH", "gfx1250-strict")
    monkeypatch.setenv("FLYDSL_DUMP_IR", "1")
    monkeypatch.setenv("FLYDSL_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setattr(kernel, "get_gfx", lambda: "gfx1250-strict")

    signature = inspect.signature(kernel.launch_gemm_a8w4_tdm.func)
    args = {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    # Both model GEMMs use the same tile: K=4096 for gate/up, K=512 for down.
    # Pointer storage has no device allocation and is never dereferenced.
    pointer = flyc.from_c_void_p(fx.Int32, 0)
    args.update(
        {name: pointer for name in signature.parameters if name.startswith("arg_")}
    )
    args.update(
        i32_m=64,
        stream=fx.Stream(None),
        N=1024 if stage1 else 4096,
        K=k,
        tile_m=64,
        tile_n=256,
        tile_k=128,
        m_warp=1,
        n_warp=4,
        out_is_f16=0,
        num_buffers=3,
        a_is_fp4=1,
        n_experts=256,
        stage1_act=stage1,
        has_bias=0,
        f32_swiglu_limit=7.0,
        stage1_quant_out=stage1,
        a_preshuffled=preshuffled,
        lds_soa_load_interleave=interleaved,
    )
    flyc.compile(kernel.launch_gemm_a8w4_tdm, **args)
    isa_files = list(Path(tmp_path).rglob("*final_isa.s"))
    assert isa_files, "the compiler must emit an ISA dump, not reuse a cached object"
    for path in isa_files:
        isa = path.read_text()
        assert "v_wmma_scale_f32_16x16x128_f8f6f4" in isa
        assert "v_wmma_scale_f32_32x16x128_f4" not in isa


@pytest.mark.parametrize("entry", ["cli", "driver"])
def test_strict_moe_aot_keeps_grouped_fp4_only(tmp_path, monkeypatch, entry):
    from aiter.aot.flydsl import moe

    jobs = []
    for name in ("grouped_gemm_a4w4", "flydsl_moe1_a4w4"):
        job = dict.fromkeys(moe._ROW_KEY, "")
        job.update(
            gfx="gfx1250",
            cu_num="384",
            kernelName1=name,
            q_dtype_a="torch.float4_e2m1fn_x2",
        )
        jobs.append(job)
    seen = []
    csv_path = tmp_path / "grouped.csv"
    csv_path.write_text("gfx\n")
    monkeypatch.setattr(moe, "collect_aot_jobs", lambda *a, **kw: jobs)
    monkeypatch.setattr(moe, "_replay_arch", lambda kept, arch: seen.extend(kept) or 0)
    if entry == "cli":
        monkeypatch.setattr(
            moe.sys,
            "argv",
            ["moe", "--arch", "gfx1250-strict", "--csv", str(csv_path)],
        )
        with pytest.raises(SystemExit) as result:
            moe.main()
        assert result.value.code == 0
        assert [(job["gfx"], job["kernelName1"]) for job in seen] == [
            ("gfx1250-strict", "grouped_gemm_a4w4")
        ]
    else:
        monkeypatch.setenv("GPU_ARCHS", "gfx1250-strict")
        monkeypatch.setattr(moe, "get_max_workers", lambda count: 1)
        monkeypatch.setattr(moe.subprocess, "run", lambda *a, **kw: None)
        commands = []

        class Process:
            def __init__(self, command, **kwargs):
                commands.append(command)

            def wait(self):
                return 0

        monkeypatch.setattr(moe.subprocess, "Popen", Process)
        assert moe.run([str(csv_path)], edges=False) == 0
        assert len(commands) == 1
        assert commands[0][commands[0].index("--arch") + 1] == "gfx1250-strict"
