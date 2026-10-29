# SPDX-License-Identifier: MIT
"""Plain-Python copy of upstream's MoE route shape (``stages/moe_shape.py``
``RouteShape``, ``stages/dims.py``, ``stages/moe_plan.py``; vLLM 8321d11): the
task counts, placements and LDS route-table offsets the Gluon stages are built
with. A copy so CPU compile probes need no aiter import; the GPU harness asserts
every value equals upstream's (``check_against_upstream``)."""

from dataclasses import dataclass

BLOCKS, WAVES = 256, 8
HIDDEN = 5120
ROWS = 16
TILE = 16  # gfx950
ROUTER_PARTS = 4
ROUTER0 = 8
MASK_WORDS = 2
MOE_INTER = SHARED_INTER = 2304
MOE_PAD = 128
UG_PART = 32
SORT_NETS = {
    6: ((0, 1), (2, 3), (4, 5), (0, 2), (1, 4), (3, 5), (0, 1), (2, 3), (4, 5),
        (1, 2), (3, 4), (2, 3)),
    2: ((0, 1),),
}  # fmt: skip


def cdiv(a, b):
    return (a + b - 1) // b


def padded(n, a):
    return cdiv(n, a) * a


def token_tiles(s):
    return [(t0, min(TILE, s - t0)) for t0 in range(0, s, TILE)]


@dataclass(frozen=True)
class Route:
    tokens: int
    tp: int = 4
    experts: int = 384
    topk: int = 6
    ug_groups: int = 2

    # ---- dims
    @property
    def inter_real(self):
        return MOE_INTER // self.tp

    @property
    def inter(self):
        return padded(self.inter_real, MOE_PAD)

    @property
    def sh_inter(self):
        return SHARED_INTER // self.tp

    @property
    def shared_tasks(self):
        return self.sh_inter // ROWS

    @property
    def mid_words(self):
        return self.inter // 4

    @property
    def down_scale_cols(self):
        return padded(self.inter // 32, 8)

    # ---- route shape
    @property
    def ug_parts(self):
        return cdiv(self.inter_real // UG_PART, self.ug_groups)

    @property
    def picks(self):
        return self.tokens * self.topk

    @property
    def max_u(self):
        return min(self.picks, self.experts)

    @property
    def router_groups(self):
        per_group = self.experts // ROWS * ROUTER_PARTS
        groups = max(1, min(len(token_tiles(self.tokens)), BLOCKS // per_group))
        tokens = -(-self.tokens // groups)
        return groups, tokens

    @property
    def router_tasks(self):
        return self.router_groups[0] * self.experts // ROWS * ROUTER_PARTS

    @property
    def router0(self):
        return max(ROUTER0, self.tokens)

    @property
    def route0(self):
        return self.router0 + self.router_tasks

    @property
    def shared0(self):
        return self.route0 + max(8, self.tokens)

    @property
    def shared_units(self):
        return self.shared_tasks * len(token_tiles(self.tokens))

    @property
    def ug0(self):
        return self.shared0 + self.shared_units

    @property
    def multi(self):
        return self.tokens > TILE

    @property
    def slot_tiles(self):
        return len(token_tiles(self.tokens))

    @property
    def max_tiles(self):
        return min(self.picks, self.max_u + self.picks // TILE)

    @property
    def expert_slot0(self):
        return 3 * self.picks + 1 + self.max_u

    @property
    def order0(self):
        return self.expert_slot0 + self.experts

    @property
    def first0(self):
        return self.order0 + self.picks

    @property
    def mask0(self):
        return self.first0 + self.max_u + 1

    @property
    def queue0(self):
        return self.mask0 + MASK_WORDS * self.experts

    @property
    def tile_first0(self):
        return self.queue0 + 1

    @property
    def tile_slot0(self):
        return self.tile_first0 + self.max_u + 1

    @property
    def tab_words(self):
        return self.tile_slot0 + self.max_tiles if self.multi else self.queue0 + 1

    @property
    def sort_net(self):
        return SORT_NETS[self.experts // 64]


FIELDS = (
    "inter_real", "inter", "sh_inter", "shared_tasks", "mid_words", "down_scale_cols",
    "ug_parts", "picks", "max_u", "router_groups", "router_tasks", "router0", "route0",
    "shared0", "shared_units", "ug0", "multi", "slot_tiles", "max_tiles",
    "expert_slot0", "order0", "first0", "mask0", "queue0", "tile_first0",
    "tile_slot0", "tab_words",
)  # fmt: skip


def check_against_upstream(s: int, tp: int) -> None:
    """Every value equals upstream's RouteShape / Dims (GPU image only: the
    upstream modules import aiter)."""
    from vllm.models.deepseek_v41.amd.mono.stages.moe_shape import MoeBuild, route_shape

    up = route_shape(MoeBuild(tokens=s, tp=tp))
    mine = Route(s, tp)
    for f in FIELDS:
        theirs = getattr(up.dims, f) if hasattr(up.dims, f) else getattr(up, f)
        assert getattr(mine, f) == theirs, (f, s, tp, getattr(mine, f), theirs)
    assert tuple(map(tuple, up.sort_net)) == mine.sort_net
