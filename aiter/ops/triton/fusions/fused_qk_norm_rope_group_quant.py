import torch

from aiter.ops.triton._gluon_kernels.gfx1250.fusions.fused_qk_norm_rope_group_quant import (
    HEAD_DIM,
    ROPE_DIM,
    fused_qk_norm_rope_group_quant_gfx1250,
    get_fused_qk_norm_rope_group_quant_layouts,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.config_utils import (
    load_config_json,
    resolve_config_dir,
    select_leq_config,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils.types import e4m3_dtype

_LOGGER = AiterTritonLogger()
_CONFIG_NAME = "FUSED_QK_NORM_ROPE_GROUP_QUANT"


def _get_config(num_tokens: int) -> dict:
    config_dir = resolve_config_dir("fusions", _CONFIG_NAME, backend="gluon")
    configs = load_config_json(f"{config_dir}/DEFAULT.json")
    return select_leq_config(configs, num_tokens, prefix="T_LEQ_")


def fused_qk_norm_rope_group_quant(
    q: torch.Tensor,
    kv: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    swa_nope: torch.Tensor,
    swa_rope: torch.Tensor,
    swa_dest_rows: torch.Tensor,
    batch_ids: torch.Tensor,
):
    """Q/K RMSNorm, RoPE, fp8 group quant and SWA cache write.

    Each row of 512 values gets an RMSNorm. K also uses ``k_weight``. Values
    [0, 448) go to fp8 e4m3 with one e8m0 scale for each 64 values. Values
    [448, 512) get RoPE on adjacent value pairs with ``cos/sin[positions]``.
    The K row of token t also goes to row ``swa_dest_rows[t]`` of the SWA cache.

    Args:
        q: [T, H, 512] contiguous bf16. H must be a multiple of HEADS_PER_PROGRAM in
            the config.
        kv: [T, 512] bf16.
        k_weight: [512] RMSNorm weight of K.
        positions: [T] int. Each position must be in [0, max_pos).
        cos, sin: [max_pos, 32] contiguous bf16.
        eps: RMSNorm epsilon.
        swa_nope, swa_rope: contiguous SWA cache, [rows, 512] fp8 and [rows, 64] bf16.
        swa_dest_rows: [T] int32 SWA row of each token. A negative row skips the write.
        batch_ids: [T] int32. A negative batch id skips the SWA write.

    Returns:
        q_nope [T, H, 512] fp8, q_rope [T, H, 64] bf16, k_nope [T, 512] fp8 and
        k_rope [T, 64] bf16. Bytes [448, 462) of a row hold the 7 scales, each
        written twice. The kernel does not write bytes [462, 512).
    """
    assert arch_info.get_arch() == "gfx1250"
    num_tokens, num_heads, head_dim = q.shape
    config = _get_config(num_tokens)
    assert head_dim == HEAD_DIM and num_heads % config["HEADS_PER_PROGRAM"] == 0
    assert kv.shape == (num_tokens, head_dim)
    assert cos.shape[1] == ROPE_DIM // 2 and cos.is_contiguous() and sin.is_contiguous()
    assert q.is_contiguous() and swa_nope.is_contiguous() and swa_rope.is_contiguous()
    assert kv.stride(1) == 1 and kv.stride(0) * kv.element_size() % 16 == 0
    assert q.data_ptr() % 16 == 0 and kv.data_ptr() % 16 == 0

    q_nope = torch.empty(
        (num_tokens, num_heads, head_dim), dtype=e4m3_dtype, device=q.device
    )
    q_rope = torch.empty(
        (num_tokens, num_heads, ROPE_DIM), dtype=q.dtype, device=q.device
    )
    k_nope = torch.empty((num_tokens, head_dim), dtype=e4m3_dtype, device=q.device)
    k_rope = torch.empty((num_tokens, ROPE_DIM), dtype=q.dtype, device=q.device)
    _LOGGER.info("FUSED_QK_NORM_ROPE_GROUP_QUANT: T=%d H=%d", num_tokens, num_heads)

    # Each token has one program for its K row, then its Q programs.
    num_q_programs = num_tokens * num_heads // config["HEADS_PER_PROGRAM"]
    grid = (num_tokens + num_q_programs,)
    fused_qk_norm_rope_group_quant_gfx1250[grid](
        q,
        positions,
        cos,
        sin,
        q_nope,
        q_rope,
        kv,
        k_weight,
        k_nope,
        k_rope,
        kv.stride(0),
        swa_dest_rows,
        batch_ids,
        swa_nope,
        swa_rope,
        EPS=eps,
        NUM_HEADS=num_heads,
        **config,
        **get_fused_qk_norm_rope_group_quant_layouts(config["num_warps"]),
    )
    return q_nope, q_rope, k_nope, k_rope
