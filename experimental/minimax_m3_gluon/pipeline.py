# SPDX-License-Identifier: MIT
"""Source-only launch contract for the MiniMax M3 MoE hybrid.

No runtime imports: host tests can assert order and direct tensor ownership
without initializing Torch, Triton or HIP.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Shape:
    tokens: int
    tp: int = 4
    grid: int = 256

    def __post_init__(self):
        if not 1 <= self.tokens <= 16 or self.tp != 4 or self.grid != 256:
            raise ValueError(
                "MiniMax M3 Gluon hybrid supports 1..16 tokens, TP4, 256 CTAs"
            )


def use_hybrid(tokens: int) -> bool:
    """ATOM's measured change to expert grouping starts at five tokens.

    This is only a candidate switch: ported Gluon versus AITER timings have not
    been measured. A serving integration must keep the established path until
    its numerical and performance checks pass.
    """
    return 5 <= tokens <= 16


def launch(shape, *, front, experts, tail, front_args, expert_args, tail_args):
    """One Gluon front, the selected two-stage MoE, one Gluon tail.

    All calls must enqueue on one stream. `experts` returns the actual MoE2
    output allocation; the tail consumes it directly without a copy.
    """
    if not isinstance(shape, Shape):
        raise TypeError("shape must be a validated Shape")
    front(*front_args)
    routed = experts(*expert_args)
    tail(routed, *tail_args)
    return routed
