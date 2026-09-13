"""Current AITER per-call varlen correctness, graph replay and warm timing."""

import json
import statistics

import torch
from aiter.ops.triton.attention import mha


def main():
    assert mha._USE_INT64_STRIDES is True
    print(
        json.dumps(
            {
                "source": mha.__file__,
                "torch": torch.__version__,
                "hip": torch.version.hip,
            }
        ),
        flush=True,
    )
    torch.manual_seed(20260913)
    original_kernel = mha._attn_fwd
    selected_widths = []

    class RecordStrideWidth:
        def __getitem__(self, grid):
            launch = original_kernel[grid]

            def wrapped(*args, **kwargs):
                selected_widths.append(kwargs["USE_INT64_STRIDES"])
                return launch(*args, **kwargs)

            return wrapped

    def timing(call):
        for _ in range(3):
            call()
        torch.cuda.synchronize()
        values = []
        for _ in range(9):
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            call()
            end.record()
            end.synchronize()
            values.append(start.elapsed_time(end))
        return statistics.median(values)

    with torch.inference_mode():
        for length, heads in ((129, 4), (1024, 4), (65536, 16)):
            tensors = [
                torch.randn((length, heads, 128), device="cuda", dtype=torch.bfloat16)
                for _ in range(3)
            ]
            cu = torch.tensor([0, length], device="cuda", dtype=torch.int32)

            def invoke(prefer):
                return mha.flash_attn_varlen_func(
                    *tensors, cu, cu, length, length, prefer_int32_strides=prefer
                )

            mha._attn_fwd = RecordStrideWidth()
            selected_widths.clear()
            control, candidate = invoke(False), invoke(True)
            mha._attn_fwd = original_kernel
            assert selected_widths == [True, False], selected_widths
            torch.testing.assert_close(control, candidate, rtol=0, atol=0)
            if length == 129:
                q, k, v = [t.transpose(0, 1).float() for t in tensors]
                reference = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v
                ).transpose(0, 1)
                torch.testing.assert_close(
                    candidate.float(), reference, rtol=0.02, atol=0.01
                )
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    invoke(True)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = invoke(True)
            for _ in range(5):
                graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(control, captured, rtol=0, atol=0)
            assert mha._USE_INT64_STRIDES is True
            a, b = timing(lambda: invoke(False)), timing(lambda: invoke(True))
            print(
                json.dumps(
                    {
                        "length": length,
                        "heads": heads,
                        "bitwise_parity": True,
                        "graph_pass": True,
                        "int64_ms": a,
                        "guarded_int32_ms": b,
                        "speedup": a / b,
                    }
                ),
                flush=True,
            )
    print("H3_STRIDE_GPU_PREFLIGHT_PASS_NO_VIDEO_RERUN", flush=True)


if __name__ == "__main__":
    main()
