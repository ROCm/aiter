"""FlyDSL MOE kernel builders (stage1, stage2, reduction)."""

from .kimi_k3_attnres import (  # noqa: F401
    attnres_combine_asm,
    attnres_score_asm,
    is_kimi_k3_attnres_asm_supported,
)
