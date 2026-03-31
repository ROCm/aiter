import torch
import warnings
import argparse
import itertools
from dataclasses import dataclass
from typing import Callable
import triton
from aiter.ops.triton.attention.mha import (
    flash_attn_func,
    flash_attn_varlen_func,
    mha_set_use_fused_bwd_kernel,
    mha_set_impl,
)
from aiter.ops.triton.attention.mha_v3 import (
    flash_attn_fp8_func,
    flash_attn_varlen_fp8_func,
)
from aiter.test_mha_common import (
    generate_random_padding_mask,
    generate_qkv,
)
from op_tests.op_benchmarks.triton.utils.argparse import get_parser
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_model_configs,
    print_vgpr,
    get_caller_name_no_ext,
)


@dataclass
class BenchRun:
    configs: list["BenchConfig"]
    torch_dtype: torch.dtype
    unit: str  # "ms", "TFLOPS", "GB/s"
    plot_name: str
    sink: bool
    equal_seqlens: bool
    save_path: str | None
    profile_dir: str | None
    print_vgpr: bool
    bench_torch: bool


@dataclass(frozen=True)
class BenchConfig:
    batch: int
    hq: int
    hk: int
    sq: int
    sk: int
    d_head: int
    d_head_v: int
    causal: bool
    layout: str  # "bshd" or "thd"
    mode: str  # "fwd" or "bwd"
    dtype_str: str  # "bf16", "fp16", "fp32", "fp8"
    impl: str = "default"  # "default" or "dao_ai"
    fused: bool = False
    model: str | None = None

    def __str__(self) -> str:
        label = self.model or "custom"
        return (
            f"{label} B={self.batch} HQ={self.hq} HK={self.hk} "
            f"sq={self.sq} sk={self.sk} d={self.d_head} "
            f"{self.layout} {self.mode} {self.dtype_str} causal={self.causal}"
        )

    def to_tuple(self) -> tuple:
        return (
            self.model,
            self.batch,
            self.hq,
            self.hk,
            self.sq,
            self.sk,
            self.d_head,
            self.d_head_v,
            self.causal,
            self.layout,
            self.mode,
            self.dtype_str,
            self.impl,
            self.fused,
        )


def _make_bf16_fn(q, k, v, **kw):
    mha_set_use_fused_bwd_kernel(False)
    return lambda: flash_attn_func(
        q,
        k,
        v,
        dropout_p=kw["dropout"],
        softmax_scale=kw["sm_scale"],
        causal=kw["causal"],
        return_lse=kw["return_lse"],
        return_attn_probs=kw["return_attn_probs"],
        sink=kw["sink"],
    )


def _make_bf16_varlen_fn(q, k, v, **kw):
    mha_set_use_fused_bwd_kernel(False)
    return lambda: flash_attn_varlen_func(
        q,
        k,
        v,
        kw["cu_seqlens_q"],
        kw["cu_seqlens_k"],
        kw["max_seqlen_q"],
        kw["max_seqlen_k"],
        dropout_p=kw["dropout"],
        softmax_scale=kw["sm_scale"],
        causal=kw["causal"],
        return_lse=kw["return_lse"],
        return_attn_probs=kw["return_attn_probs"],
        sink=kw["sink"],
    )


def _make_bf16_fused_fn(q, k, v, **kw):
    if kw.get("has_pe") or kw.get("has_sink"):
        warnings.warn("Skipping: PE or sink not supported for this provider.")
        return None
    mha_set_use_fused_bwd_kernel(True)
    return lambda: flash_attn_func(
        q,
        k,
        v,
        dropout_p=kw["dropout"],
        softmax_scale=kw["sm_scale"],
        causal=kw["causal"],
        return_lse=kw["return_lse"],
        return_attn_probs=kw["return_attn_probs"],
        sink=kw["sink"],
    )


def _make_bf16_fused_varlen_fn(q, k, v, **kw):
    if kw.get("has_pe") or kw.get("has_sink"):
        warnings.warn("Skipping: PE or sink not supported for this provider.")
        return None
    mha_set_use_fused_bwd_kernel(True)
    return lambda: flash_attn_varlen_func(
        q,
        k,
        v,
        kw["cu_seqlens_q"],
        kw["cu_seqlens_k"],
        kw["max_seqlen_q"],
        kw["max_seqlen_k"],
        dropout_p=kw["dropout"],
        softmax_scale=kw["sm_scale"],
        causal=kw["causal"],
        return_lse=kw["return_lse"],
        return_attn_probs=kw["return_attn_probs"],
        sink=kw["sink"],
    )


def _make_fp8_fn(q, k, v, **kw):
    if kw.get("has_pe") or kw.get("has_sink"):
        warnings.warn("Skipping: PE or sink not supported for this provider.")
        return None
    mha_set_use_fused_bwd_kernel(False)
    return lambda: flash_attn_fp8_func(
        q,
        k,
        v,
        softmax_scale=kw["sm_scale"],
        causal=kw["causal"],
    )


def _make_fp8_varlen_fn(q, k, v, **kw):
    if kw.get("has_pe") or kw.get("has_sink"):
        warnings.warn("Skipping: PE or sink not supported for this provider.")
        return None
    mha_set_use_fused_bwd_kernel(False)
    return lambda: flash_attn_varlen_fp8_func(
        q,
        k,
        v,
        kw["cu_seqlens_q"],
        kw["cu_seqlens_k"],
        kw["max_seqlen_q"],
        kw["max_seqlen_k"],
        softmax_scale=kw["sm_scale"],
        causal=kw["causal"],
    )


_MAKE_FN = {
    ("bf16", "bshd"): _make_bf16_fn,
    ("bf16", "thd"): _make_bf16_varlen_fn,
    ("fp16", "bshd"): _make_bf16_fn,
    ("fp16", "thd"): _make_bf16_varlen_fn,
    ("fp32", "bshd"): _make_bf16_fn,
    ("fp32", "thd"): _make_bf16_varlen_fn,
    ("fp8", "bshd"): _make_fp8_fn,
    ("fp8", "thd"): _make_fp8_varlen_fn,
    ("bf16_fused", "bshd"): _make_bf16_fused_fn,
    ("bf16_fused", "thd"): _make_bf16_fused_varlen_fn,
    ("fp16_fused", "bshd"): _make_bf16_fused_fn,
    ("fp16_fused", "thd"): _make_bf16_fused_varlen_fn,
    ("fp32_fused", "bshd"): _make_bf16_fused_fn,
    ("fp32_fused", "thd"): _make_bf16_fused_varlen_fn,
}


def get_make_fn(dtype: str, layout: str, fused: bool = False) -> Callable:
    key = (f"{dtype}_fused" if fused else dtype, layout)
    return _MAKE_FN[key]


def edge_case_configs(
    dtypes: list[str],
    modes: list[str],
    impl: str,
    fused: bool,
) -> list[BenchConfig]:
    """Configs covering gaps in model benchmarks: decode, non-causal,
    mismatched sq/sk, non-power-of-2 seqlens, batched, bshd layout."""
    d_head = 128
    configs: list[BenchConfig] = []

    # Decode: sq=1, large KV cache
    for (hq, hk), sk, b, m, d in itertools.product(
        [(32, 8), (64, 8)],
        [4096, 8192],
        [1, 8],
        modes,
        dtypes,
    ):
        configs.append(
            BenchConfig(
                batch=b,
                hq=hq,
                hk=hk,
                sq=1,
                sk=sk,
                d_head=d_head,
                d_head_v=d_head,
                causal=True,
                layout="thd",
                mode=m,
                dtype_str=d,
                impl=impl,
                fused=fused and d != "fp8",
            )
        )

    # Non-causal (cross-attention): sq != sk
    for (hq, hk), (sq, sk), m, d in itertools.product(
        [(32, 32), (64, 8)],
        [(512, 1024), (1024, 4096)],
        modes,
        dtypes,
    ):
        configs.append(
            BenchConfig(
                batch=4,
                hq=hq,
                hk=hk,
                sq=sq,
                sk=sk,
                d_head=d_head,
                d_head_v=d_head,
                causal=False,
                layout="bshd",
                mode=m,
                dtype_str=d,
                impl=impl,
                fused=fused and d != "fp8",
            )
        )

    # Non-power-of-2 seqlens
    for (hq, hk), (sq, sk), m, d in itertools.product(
        [(32, 8)],
        [(163, 163), (1000, 2000)],
        modes,
        dtypes,
    ):
        configs.append(
            BenchConfig(
                batch=4,
                hq=hq,
                hk=hk,
                sq=sq,
                sk=sk,
                d_head=d_head,
                d_head_v=d_head,
                causal=True,
                layout="thd",
                mode=m,
                dtype_str=d,
                impl=impl,
                fused=fused and d != "fp8",
            )
        )

    # Batched bshd (training-like)
    for (hq, hk), sq, b, m, d in itertools.product(
        [(32, 8), (64, 8)],
        [1024, 4096],
        [4, 16],
        modes,
        dtypes,
    ):
        configs.append(
            BenchConfig(
                batch=b,
                hq=hq,
                hk=hk,
                sq=sq,
                sk=sq,
                d_head=d_head,
                d_head_v=d_head,
                causal=True,
                layout="bshd",
                mode=m,
                dtype_str=d,
                impl=impl,
                fused=fused and d != "fp8",
            )
        )

    return configs


def model_benchmark_configs(
    args,
    *,
    dtypes: list[str],
    modes: list[str],
    impl: str,
    fused: bool,
    layout: str | None = None,
    model: str | None = None,
) -> list[BenchConfig]:
    config_file = args.model_configs
    configs = get_model_configs(config_path=config_file, models=model or "all")
    layouts = [layout] if layout else ["thd", "bshd"]
    fa_configs: list[BenchConfig] = []
    causals = [args.causal] if args.causal is not None else [True, False]

    for model_name, config in configs.items():
        HQ = config["num_attention_heads"]
        HK = (
            HQ
            if config["num_key_value_heads"] is None
            else config["num_key_value_heads"]
        )
        HEAD_DIM = config["hidden_size"] // HQ
        if args.sq:
            b = args.b if args.b else 1
            scenarios = [(b, args.sq, args.sk if args.sk else args.sq)]
        else:
            # (batch, sq, sk) triplets for realistic serving scenarios
            # Prefill: few concurrent prefills, sq == sk
            prefill = [(4, s, s) for s in [128, 256, 512, 1024, 2048, 4096, 8192]]
            # Decode: many concurrent decodes, sq=1
            decode = [(32, 1, s) for s in [256, 512, 1024, 2048, 4096, 8192]]
            scenarios = prefill + decode
            if args.b:
                scenarios = [(args.b, sq, sk) for (_, sq, sk) in scenarios]
        for (b, sq, sk), lay, c, m, d in itertools.product(
            scenarios, layouts, causals, modes, dtypes
        ):
            fa_configs.append(
                BenchConfig(
                    model=model_name,
                    batch=b,
                    hq=HQ,
                    hk=HK,
                    sq=sq,
                    sk=sk,
                    d_head=HEAD_DIM,
                    d_head_v=HEAD_DIM,
                    causal=c,
                    layout=lay,
                    mode=m,
                    dtype_str=d,
                    impl=impl,
                    fused=fused and d != "fp8",
                )
            )

    return fa_configs


def pad_rearrange_dropout_mask(
    S_dmask,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    seqlen_q,
    seqlen_k,
    num_q_heads,
):
    batch_size = cu_seqlens_q.numel() - 1

    padded_dropout_mask = torch.ones(
        (batch_size, num_q_heads, seqlen_q, seqlen_k), device="cuda"
    )
    for b in range(batch_size):
        start_q = cu_seqlens_q[b].item()
        end_q = cu_seqlens_q[b + 1].item()
        start_k = cu_seqlens_k[b].item()
        end_k = cu_seqlens_k[b + 1].item()

        seqlen_q = end_q - start_q
        seqlen_k = end_k - start_k
        for h in range(S_dmask.shape[1]):
            padded_dropout_mask[b, h, :max_seqlen_q, :max_seqlen_k] = S_dmask[
                b, h, :, :
            ]

    return padded_dropout_mask


def _make_triton_benchmark(run: BenchRun) -> list:
    x_names = [
        "model",
        "BATCH",
        "HQ",
        "HK",
        "N_CTX_Q",
        "N_CTX_K",
        "D_HEAD",
        "D_HEAD_V",
        "causal",
        "layout",
        "mode",
        "dtype",
        "impl",
        "fused",
    ]
    return [
        triton.testing.Benchmark(
            x_names=x_names,
            x_vals=[c.to_tuple() for c in run.configs],
            line_arg="provider",
            line_vals=[run.unit],
            line_names=[run.unit],
            styles=[("red", "-")],
            ylabel=run.unit,
            plot_name=run.plot_name,
            args={
                "torch_dtype": run.torch_dtype,
                "unit": run.unit,
            },
        )
    ]


def run_benchmark(run: BenchRun):
    torch.manual_seed(20)
    total = len(run.configs)
    counter = 0

    @triton.testing.perf_report(_make_triton_benchmark(run))
    def bench_mha(
        model,
        BATCH,
        HQ,
        HK,
        N_CTX_Q,
        N_CTX_K,
        D_HEAD,
        D_HEAD_V,
        causal,
        layout,
        mode,
        dtype,
        impl,
        fused,
        torch_dtype,
        unit,
        provider,
        dropout=0.0,
        sm_scale=None,
        device="cuda",
    ):
        nonlocal counter
        counter += 1
        label = model or "custom"
        print(
            f"[{counter}/{total}] {label} B={BATCH} HQ={HQ} HK={HK} "
            f"sq={N_CTX_Q} sk={N_CTX_K} d={D_HEAD} {layout} {mode} {dtype} causal={causal}",
            flush=True,
        )
        assert dropout <= 0.0, "Dropout not supported in this benchmark."
        requires_grad = mode == "bwd"
        return_lse = True
        return_attn_probs = False
        varlen = layout == "thd"
        has_pe = D_HEAD > D_HEAD_V
        if impl != "default":
            mha_set_impl(impl)
        make_fn = get_make_fn(dtype, layout, fused)

        # Default softmax scale to match standard attention
        if sm_scale is None:
            sm_scale = 1.0 / (D_HEAD**0.5)

        # Generate base inputs
        q = torch.randn(
            (BATCH, N_CTX_Q, HQ, D_HEAD),
            device=device,
            dtype=torch_dtype,
            requires_grad=requires_grad,
        )
        k = torch.randn(
            (BATCH, N_CTX_K, HK, D_HEAD),
            device=device,
            dtype=torch_dtype,
            requires_grad=requires_grad,
        )
        v = torch.randn(
            (BATCH, N_CTX_K, HK, D_HEAD_V),
            device=device,
            dtype=torch_dtype,
            requires_grad=requires_grad,
        )
        sink = (
            torch.randn((HQ,), device=device, dtype=dtype, requires_grad=requires_grad)
            if run.sink
            else None
        )

        # FLOPS calculation variables
        total_flops = 0.0

        # Input preparation
        if varlen:
            query_padding_mask = generate_random_padding_mask(
                N_CTX_Q, BATCH, device, mode="full" if run.equal_seqlens else "random"
            )
            key_padding_mask = generate_random_padding_mask(
                N_CTX_K, BATCH, device, mode="full" if run.equal_seqlens else "random"
            )
            (
                q_unpad,
                k_unpad,
                v_unpad,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                q,
                k,
                v,
                _,
                _,
                _,
            ) = generate_qkv(
                q, k, v, query_padding_mask, key_padding_mask, kvpacked=False
            )
            q_unpad.requires_grad = requires_grad
            k_unpad.requires_grad = requires_grad
            v_unpad.requires_grad = requires_grad

            q_input, k_input, v_input = q_unpad, k_unpad, v_unpad

            num_contexts = len(cu_seqlens_q) - 1
            for i in range(num_contexts):
                seqlen_q = (cu_seqlens_q[i + 1] - cu_seqlens_q[i]).item()
                seqlen_k = (cu_seqlens_k[i + 1] - cu_seqlens_k[i]).item()
                if causal:
                    valid_out_elements = (
                        ((seqlen_k**2 + seqlen_k) / 2)
                        if seqlen_q > seqlen_k
                        else (seqlen_q * seqlen_k - ((seqlen_q**2 - seqlen_q) / 2))
                    )
                    total_flops += valid_out_elements * HQ * (D_HEAD + D_HEAD_V) * 2.0
                else:
                    total_flops += seqlen_q * seqlen_k * HQ * (D_HEAD + D_HEAD_V) * 2.0
        else:
            q_input, k_input, v_input = q, k, v

            if causal:
                valid_out_elements = (
                    ((N_CTX_K**2 + N_CTX_K) / 2)
                    if N_CTX_Q > N_CTX_K
                    else (N_CTX_Q * N_CTX_K - ((N_CTX_Q**2 - N_CTX_Q) / 2))
                )
                total_flops += (
                    2.0 * BATCH * HQ * valid_out_elements * (D_HEAD + D_HEAD_V)
                )
            else:
                total_flops += (
                    2.0 * BATCH * HQ * N_CTX_Q * N_CTX_K * (D_HEAD + D_HEAD_V)
                )

        # Build fn from provider
        fn_kwargs = dict(
            sm_scale=sm_scale,
            causal=causal,
            dropout=dropout,
            return_lse=return_lse,
            return_attn_probs=return_attn_probs,
            sink=sink,
            has_pe=has_pe,
            has_sink=run.sink,
        )
        if varlen:
            fn_kwargs.update(
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
            )
        fn = make_fn(q_input, k_input, v_input, **fn_kwargs)
        if fn is None:
            return 0

        if mode == "bwd":
            with torch.enable_grad():
                triton_out = fn()[0]
                d_out = torch.randn_like(triton_out)

                grad_inputs = (q_input, k_input, v_input)
                if sink is not None:
                    grad_inputs += (sink,)

                def fn():
                    grads = torch.autograd.grad(
                        triton_out,
                        grad_inputs,
                        d_out,
                        retain_graph=True,
                    )
                    return grads

        if run.profile_dir is not None:
            import os

            # Warmup
            for _ in range(3):
                fn()
                torch.cuda.synchronize()

            prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                with_stack=True,
            )
            prof.start()
            for _ in range(5):
                fn()
                torch.cuda.synchronize()
            prof.stop()

            shape_str = (
                f"{model}_B{BATCH}_HQ{HQ}_HK{HK}_SQ{N_CTX_Q}_SK{N_CTX_K}_D{D_HEAD}"
            )
            print(f"\n--- Profile: {mode} {shape_str} ---")
            print(
                prof.key_averages().table(
                    sort_by="self_cuda_time_total",
                    row_limit=30,
                )
            )
            trace_dir = os.path.join(run.profile_dir, f"{mode}_{shape_str}")
            os.makedirs(trace_dir, exist_ok=True)
            prof.export_chrome_trace(os.path.join(trace_dir, "trace.json"))
            return 0

        ms = triton.testing.do_bench(fn)

        if mode == "bwd":
            total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)

        if varlen:
            total_num_tokens_q = cu_seqlens_q[-1].item()
            total_num_tokens_k = cu_seqlens_k[-1].item()
        else:
            total_num_tokens_q = BATCH * N_CTX_Q
            total_num_tokens_k = BATCH * N_CTX_K
        q_size = total_num_tokens_q * HQ * D_HEAD * q.element_size()
        k_size = total_num_tokens_k * HK * D_HEAD * k.element_size()
        v_size = total_num_tokens_k * HK * D_HEAD_V * v.element_size()
        o_size = total_num_tokens_q * HQ * D_HEAD_V * q.element_size()
        if mode == "fwd":
            # read q, k, v
            mem_read = q_size + k_size + v_size
            # write o
            mem_write = o_size
        else:
            # read q, k, v, do
            mem_read = q_size + k_size + v_size + o_size
            # write dq, dk, dv
            mem_write = q_size + k_size + v_size
        mem = mem_read + mem_write

        if unit == "ms":
            return ms
        elif unit == "TFLOPS":
            return total_flops / ms * 1e-9
        else:  # GB/s
            return mem / ms * 1e-6

    try:
        bench_mha.run(save_path=run.save_path, print_data=True)
    except Exception as e:
        print(f"\n[WARN] benchmark failed: {e}", flush=True)


def supported_layouts():
    layouts = (
        "bshd: Q, K, V are individual tensors of [batch, seqlen_q/k, num_heads, head_size]. "
        "thd: Q, K, V are individual tensors of [total_q/k, num_heads, head_size]. "
    )
    return layouts


# argparse lacks support for boolean argument type (sigh...)
def str2bool(v):
    if isinstance(v, bool) or v is None:
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


VALID_DTYPES = {"fp16", "bf16", "fp32", "fp8"}

arg_to_torch_dtype = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def parse_args(args: list[str] | None = None) -> BenchRun:
    parser = get_parser(kernel_name="FlashAttention")
    parser.add_argument(
        "-mode", type=str, default=None, help="fwd, bwd, or omit for both"
    )
    parser.add_argument("-b", type=int, default=0)
    parser.add_argument("-hq", type=int, default=0)
    parser.add_argument("-hk", type=int, default=0)
    parser.add_argument("-sq", type=int, default=0)
    parser.add_argument("-sk", type=int, default=0)
    parser.add_argument(
        "-equal_seqlens",
        action="store_true",
        default=False,
        help="If specified, uses equal sequence lengths with thd layout, i.e t = b * sq",
    )
    parser.add_argument(
        "-d",
        type=int,
        default=0,
        help="Q and K head size, if -dv is absent then -d specifies V head size too",
    )
    parser.add_argument("-dv", type=int, default=0, help="optional V head size")
    parser.add_argument("-causal", type=str2bool, default=None)
    parser.add_argument("-quantize_p", action="store_true", default=False)
    parser.add_argument(
        "--dtype",
        default="bf16",
        help="Comma-separated compute types to benchmark: bf16, fp16, fp32, fp8 (e.g. --dtype bf16,fp8)",
    )
    parser.add_argument("-bench_torch", action="store_true", default=False)
    parser.add_argument("-fused_bwd", action="store_true", default=False)
    parser.add_argument("-print_vgpr", action="store_true", default=False)
    parser.add_argument("--layout", type=str, default=None, help=supported_layouts())
    parser.add_argument(
        "-metric",
        nargs="?",
        const="throughput",
        choices=["time", "throughput", "bandwidth"],
        default=None,
        help="Metrics for the kernel benchmark.",
    )
    parser.add_argument(
        "-persistent",
        nargs="?",
        const="fixed",
        choices=["fixed", "dynamic"],
        default=None,
        help="Enable persistent kernels. Use '-persistent dynamic' for dynamic scheduling of the tiles.",
    )
    parser.add_argument(
        "-o",
        type=str,
        default=None,
        metavar="DIR",
        help="Write performance results to CSV in DIR",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        metavar="DIR",
        help="Enable torch.profiler and write chrome traces to DIR.",
    )
    parser.add_argument(
        "-sink", action="store_true", default=False, help="use attention sink"
    )
    parser.add_argument(
        "-impl",
        type=str,
        default="default",
        choices=["default", "dao_ai"],
        help="MHA forward implementation: default (_attn_fwd) or dao_ai (flash_attn_triton_amd)",
    )
    parsed = parser.parse_args(args=args)

    # Validate dtypes
    dtypes = [d.strip() for d in parsed.dtype.split(",")]
    for d in dtypes:
        assert (
            d in VALID_DTYPES
        ), f"Unknown dtype '{d}'. Supported: {sorted(VALID_DTYPES)}"
    tensor_dtype_str = next((d for d in dtypes if d != "fp8"), "bf16")
    torch_dtype = arg_to_torch_dtype[tensor_dtype_str]

    assert parsed.layout in (
        None,
        "bshd",
        "thd",
    ), f"{parsed.layout} is not a supported layout. Use 'bshd' or 'thd'."

    custom = bool(parsed.hq or parsed.hk or parsed.d or parsed.dv)
    if custom:
        if not parsed.dv:
            parsed.dv = parsed.d
        assert (
            parsed.b and parsed.hq and parsed.sq and parsed.d and parsed.dv
        ), "Custom config requires: -b, -hq, -sq, -d (and optionally -dv)."
    if parsed.model:
        assert (
            not custom
        ), "--model sets hq, hk, d from the config. Do not provide them."

    if parsed.layout in ("thd", None) and parsed.equal_seqlens:
        warnings.warn(
            "Using 'thd' layout with equal_seqlen=True incurs an extra sequence length lookup cost "
            "compared to 'bshd' layout. Consider using 'bshd' for better performance.",
            category=RuntimeWarning,
        )

    # Resolve metric/unit
    metric = parsed.metric or "throughput"
    unit_map = {"throughput": "TFLOPS", "time": "ms", "bandwidth": "GB/s"}
    unit = unit_map[metric]

    # Build configs
    impl = parsed.impl
    fused = parsed.fused_bwd
    modes = [parsed.mode] if parsed.mode else ["fwd", "bwd"]
    d_head = parsed.d if parsed.d else 128
    d_head_v = parsed.dv if parsed.dv else d_head

    if custom:
        hk = parsed.hk if parsed.hk else parsed.hq
        sk = parsed.sk if parsed.sk else parsed.sq
        causals = [parsed.causal] if parsed.causal is not None else [False, True]
        custom_layout = parsed.layout or "bshd"
        configs = [
            BenchConfig(
                model=f"custom_B{parsed.b}_HQ{parsed.hq}_HK{hk}",
                batch=parsed.b,
                hq=parsed.hq,
                hk=hk,
                sq=parsed.sq,
                sk=sk,
                d_head=d_head,
                d_head_v=d_head_v,
                causal=c,
                layout=custom_layout,
                mode=m,
                dtype_str=d,
                impl=impl,
                fused=fused and d != "fp8",
            )
            for c, m, d in itertools.product(causals, modes, dtypes)
        ]
    elif parsed.model:
        configs = model_benchmark_configs(
            parsed,
            dtypes=dtypes,
            modes=modes,
            impl=impl,
            fused=fused,
            model=parsed.model,
            layout=parsed.layout,
        )
    else:
        # Default: all models, both layouts, both modes
        configs = model_benchmark_configs(
            parsed,
            dtypes=dtypes,
            modes=modes,
            impl=impl,
            fused=fused,
        )

    return BenchRun(
        configs=configs,
        torch_dtype=torch_dtype,
        unit=unit,
        plot_name=get_caller_name_no_ext(),
        sink=parsed.sink,
        equal_seqlens=parsed.equal_seqlens,
        save_path=parsed.o,
        profile_dir=parsed.profile,
        print_vgpr=parsed.print_vgpr,
        bench_torch=parsed.bench_torch,
    )


def main(args: list[str] | None = None) -> None:
    run = parse_args(args=args)

    if run.print_vgpr:
        assert not run.bench_torch, "Do not use -bench_torch with -print_vgpr."
        print("Retrieving VGPR usage for Triton kernels...")

        def fun():
            return run_benchmark(run)

        print_vgpr(fun, get_caller_name_no_ext())
        return

    run_benchmark(run)


if __name__ == "__main__":
    main()
