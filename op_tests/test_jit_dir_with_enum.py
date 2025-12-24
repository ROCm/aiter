import contextlib
import os
import tempfile
import torch


@contextlib.contextmanager
def TemporaryEnvironmentVariable(var_name, value):
    original_value = os.environ.get(var_name)
    os.environ[var_name] = value
    yield
    if original_value is not None:
        os.environ[var_name] = original_value
    elif var_name in os.environ:
        del os.environ[var_name]


def test_aiter_jit_dir_with_enum():
    # Create a temporary directory for AITER_JIT_DIR
    with tempfile.TemporaryDirectory() as temp_dir, TemporaryEnvironmentVariable(
        "AITER_JIT_DIR", temp_dir
    ):
        # Import aiter only after we set AITER_JIT_DIR
        from aiter import ActivationType, QuantType

        # Using moe_stage1_g1u1 as an example of a compiled function with enum types in its signature
        from aiter.ops.moe_op import moe_stage1_g1u1
        from aiter.utility import dtypes
        from aiter.fused_moe_bf16_asm import moe_sorting_ck

        # Create dummy tensors for testing
        torch.set_default_device("cuda")
        fp8_dtype = dtypes.fp8

        # Setup parameters
        num_tokens = 4
        model_dim = 128
        inter_dim = 256  # Must be divisible by tile_n (64 or 128)
        num_experts = 2
        topk = 2

        hidden_states = torch.randn(
            num_tokens, model_dim, dtype=torch.bfloat16, device="cuda"
        )
        hidden_states_fp8 = hidden_states.to(fp8_dtype)

        w1 = torch.randn(
            num_experts, inter_dim * 2, model_dim, dtype=torch.bfloat16, device="cuda"
        )
        w1_fp8 = w1.to(fp8_dtype)

        w2 = torch.randn(
            num_experts, model_dim, inter_dim, dtype=torch.bfloat16, device="cuda"
        )
        w2_fp8 = w2.to(fp8_dtype)

        # Create topk_ids and topk_weights for sorting
        topk_ids = torch.randint(
            0, num_experts, (num_tokens, topk), dtype=torch.int32, device="cuda"
        )
        topk_weights = torch.rand(num_tokens, topk, dtype=torch.float32, device="cuda")

        # Use moe_sorting_ck to prepare sorted data (required by the kernel)
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = (
            moe_sorting_ck(
                topk_ids,
                topk_weights,
                num_experts,
                model_dim,
                torch.bfloat16,
                block_size=32,
                expert_mask=None,
            )
        )

        # Create output tensor
        out = torch.empty(
            (num_tokens, topk, inter_dim * 2), dtype=torch.bfloat16, device="cuda"
        )

        a1_scale = torch.rand(num_tokens, 1, dtype=torch.float32, device="cuda")
        w1_scale = torch.rand(
            num_experts, 1, inter_dim, dtype=torch.float32, device="cuda"
        )

        moe_stage1_g1u1(
            hidden_states_fp8,
            w1_fp8,
            w2_fp8,
            sorted_ids,
            sorted_expert_ids,
            num_valid_ids,
            out,
            inter_dim=inter_dim,
            kernelName="",
            block_m=32,
            activation=ActivationType.Silu.value,
            quant_type=QuantType.per_Token.value,
            a1_scale=a1_scale,
            w1_scale=w1_scale,
        )

        torch.cuda.synchronize()

        out_cpu = out.cpu()
        assert out_cpu is not None, "moe_stage1_g1u1 should have written to out"
        assert out_cpu.numel() > 0, "Output tensor should not be empty"
        assert not torch.all(
            out_cpu == 0
        ), "Output tensor should contain non-zero values (kernel should have computed results)"


if __name__ == "__main__":
    test_aiter_jit_dir_with_enum()
