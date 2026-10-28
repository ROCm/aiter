import torch
import triton

from aiter.ops.triton._gluon_kernels.gfx1250.fusions.fused_qk_norm_mrope import (
    HEAD_DIM,
    ROPE_HALF,
    fused_qk_norm_mrope_gfx1250,
    get_fused_qk_norm_mrope_layouts,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.config_utils import (
    load_config_json,
    resolve_config_dir,
    select_leq_config,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()
_CONFIG_NAME = "FUSED_QK_NORM_MROPE"


def _get_config(num_tokens: int) -> dict:
    config_dir = resolve_config_dir("fusions", _CONFIG_NAME, backend="gluon")
    configs = load_config_json(f"{config_dir}/DEFAULT.json")
    return select_leq_config(configs, num_tokens, prefix="T_LEQ_")


def fused_qk_norm_mrope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    positions: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: list[int],
):
    """Q/K Gemma RMSNorm and interleaved MRoPE (Qwen3.5 full attention).

    Each head of 256 values gets x / rms(x) * (1 + weight). Values [0, 64) then get
    Neox RoPE: values i and i + 32 rotate with cos/sin[pos, i]. pos is the H position
    when i % 3 == 1 and i < 3 * mrope_section[1], the W position when i % 3 == 2 and
    i < 3 * mrope_section[2], else the T position.

    Args:
        q: [T, Hq * 256] bf16. k: [T, Hk * 256] bf16. Rows can be strided.
        q_weight, k_weight: [256] Gemma RMSNorm weights.
        eps: RMSNorm epsilon.
        positions: [3, T] T, H and W positions. Each must be in [0, max_pos).
        cos, sin: [max_pos, 32] fp32.
        mrope_section: T, H and W section sizes.

    Returns:
        q_out [T, Hq * 256] and k_out [T, Hk * 256], contiguous.
    """
    assert arch_info.get_arch() == "gfx1250"
    num_tokens = q.shape[0]
    q_heads = q.shape[1] // HEAD_DIM.value
    k_heads = k.shape[1] // HEAD_DIM.value
    assert q.shape[1] == q_heads * HEAD_DIM.value
    assert k.shape == (num_tokens, k_heads * HEAD_DIM.value)
    assert q.stride(1) == 1 and k.stride(1) == 1
    assert positions.shape == (3, num_tokens) and positions.stride(1) == 1
    assert cos.shape[1] == ROPE_HALF.value and cos.stride(1) == 1
    assert sin.stride() == cos.stride()
    config = _get_config(num_tokens)

    q_out = torch.empty((num_tokens, q.shape[1]), dtype=q.dtype, device=q.device)
    k_out = torch.empty((num_tokens, k.shape[1]), dtype=k.dtype, device=k.device)
    _LOGGER.info("FUSED_QK_NORM_MROPE: T=%d Hq=%d Hk=%d", num_tokens, q_heads, k_heads)

    # A program does one head row per wave. The rows of token t are its Q heads, then its K heads.
    num_rows = num_tokens * (q_heads + k_heads)
    fused_qk_norm_mrope_gfx1250[(triton.cdiv(num_rows, config["num_warps"]),)](
        q,
        k,
        q_weight,
        k_weight,
        positions,
        cos,
        sin,
        q_out,
        k_out,
        num_rows,
        q.stride(0),
        k.stride(0),
        positions.stride(0),
        cos.stride(0),
        EPS=eps,
        Q_HEADS=q_heads,
        K_HEADS=k_heads,
        SECTION_H=mrope_section[1],
        SECTION_W=mrope_section[2],
        ROWS=config["num_warps"],
        num_warps=config["num_warps"],
        **get_fused_qk_norm_mrope_layouts(config["num_warps"]),
    )
    return q_out, k_out
