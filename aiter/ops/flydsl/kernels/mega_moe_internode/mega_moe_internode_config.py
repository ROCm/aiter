# SPDX-License-Identifier: MIT
"""Paired Stage1/Stage2 tuning for MegaMoEInternode.

Load ``aiter/configs/mega_moe_internode_tuned.csv``, or the file selected by
``AITER_CONFIG_MEGAMOE_INTERNODE``. One row configures both stages. Missing
shapes use built-in rules; a row also chooses the split path (``split_local``),
so Stage1 and Stage2 always follow the same row. There are no per-field
environment overrides.

The key follows the MegaMoE TP table naming: ``gfx, cu_num, model_dim,
inter_dim, experts, topk, tokens, quant, world_size, gpus_per_node``. Legacy rows
without ``quant`` use A4W4; without topology columns they use EP16 on two
eight-GPU nodes. A8W4 never selects an A4W4 row.
``tokens`` is the per-rank input capacity,
not tokens*topk. ``experts`` is the routed-expert count per EP rank. Internal
lookup functions still accept GEMM rows as ``token`` and divide by topk.

Stage1 parameters:
* ``tile_group``: G adjacent 32-row arena tiles form one GMM1 M tile of 32*G.
  The arena's 32-row granularity stays fixed; G also controls map allocation.
* ``split_local``: split local/remote-source groups and start local GMM1 early.
  Without a row the shape rule selects split at tokens>=1024. Each path derives its
  associated scheduling, sealing, and padding switches from this setting.
* ``fanout_ctas_per_dest``: Stage1 fanout-role CTA slots per destination GPU.
  Eight destinations reserve 8*N slots. T0 occupies destination 0's first slot
  for CCO/credits/sealing; rail posting slots also skip token copies. These
  CTAs can join GMM1 after their communication work. Unsplit accepts 2..31,
  defaulting to 16; split accepts 17..31, defaulting to 17. Missing or invalid
  values fall back without discarding other valid fields.

Stage2 parameters:
* ``gemm2_bm``: M tile. 0 selects the path-dependent default: 64 for xGMI,
  128 for SDMA, or 32 when remote-first fusion/tile-group alignment prevents
  the larger tile. SDMA rows retain 0 so a missing SDMA queue can safely fall
  back to the xGMI tile. An explicit tile above 32 may also be used without
  the Stage1 local/remote split; it must divide 32*tile_group.
* ``gemm2_bn``: GEMM2 N tile, 128 or 256; model_dim / gemm2_bn must be even
  (the 8-wave kernel1 runs n_block pairs).
* ``gemm2_bk``: GEMM2 K tile; 0 or empty selects 256 (one E8M0 word per K tile).
* ``gemm2_cu``: persistent grid; 0 uses the operator's worker_blocks.
* ``return_chunk_tokens``: token granularity of rail chunks/readiness signals.
* ``intra_comm``: xgmi or sdma; SDMA requires the fused remote-first split path.

``us`` is informational and never affects selection. Measured rows record
full-forward GPU event latency for their quant, including quantization and communication,
for EPLB: warmup 10, 40 samples per rank, mean of all 16 per-rank minima.
No redundant configuration tag is stored: the parameter columns already
describe the chosen configuration.
"""

from __future__ import annotations

import csv
import logging
import os
import threading

logger = logging.getLogger(__name__)

_SHAPE_FIELDS = ("gfx", "cu_num", "model_dim", "inter_dim", "experts", "topk", "tokens")


def _quant_key(quant: str, where: str) -> str:
    quant = str(quant).strip().lower()
    if quant not in ("a4w4", "a8w4"):
        raise ValueError(f"{where}: quant must be a4w4 or a8w4 (got {quant!r})")
    return quant


class _TuneTable:
    """One CSV tune table: ``$<env>`` or ``aiter/configs/<default_name>``, loaded
    once per path, rows validated at load time (so an error names the CSV row)."""

    def __init__(
        self,
        env: str,
        default_name: str,
        value_fields: tuple,
        validate,
        optional_fields: tuple = (),
    ):
        self.env, self.default_name = env, default_name
        self.value_fields, self.validate = value_fields, validate
        # (name, default): a table without the column (or with it empty) gets
        # the default; the value takes the default's type (int or str).
        self.optional_fields = tuple(optional_fields)
        self._lock = threading.Lock()
        self._cache: dict | None = None
        self._cache_path: str | None = None

    def path(self) -> str:
        override = os.environ.get(self.env)
        if override:
            return override
        from aiter.jit.core import AITER_ROOT_DIR

        return os.path.join(AITER_ROOT_DIR, "aiter", "configs", self.default_name)

    def _load(self, path: str) -> dict:
        table: dict = {}
        if not os.path.isfile(path):
            return table
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            fields = _SHAPE_FIELDS + self.value_fields
            missing = [f for f in fields if f not in (reader.fieldnames or ())]
            if missing:
                raise ValueError(f"{path}: tune table is missing columns {missing}")
            for lineno, raw in enumerate(reader, start=2):
                if not raw.get("gfx") or raw["gfx"].lstrip().startswith("#"):
                    continue
                key = tuple(
                    raw["gfx"].strip() if f == "gfx" else int(raw[f])
                    for f in _SHAPE_FIELDS
                ) + (
                    _quant_key(
                        (raw.get("quant") or "").strip() or "a4w4", f"{path}:{lineno}"
                    ),
                    int((raw.get("world_size") or "").strip() or 16),
                    int((raw.get("gpus_per_node") or "").strip() or 8),
                )
                if key[-2] <= 0 or key[-1] <= 0 or key[-2] % key[-1]:
                    raise ValueError(
                        f"{path}:{lineno}: world_size must divide into equal GPU nodes"
                    )
                row = {f: int(raw[f]) for f in self.value_fields}
                for f, default in self.optional_fields:
                    v = (raw.get(f) or "").strip()
                    row[f] = type(default)(v) if v else default
                self.validate(row, f"{path}:{lineno}")
                table[key] = row
        return table

    def lookup(
        self,
        *,
        gfx,
        cu_num,
        token,
        model_dim,
        inter_dim,
        expert,
        topk,
        quant="a4w4",
        world_size=16,
        gpus_per_node=8,
    ) -> dict | None:
        if (
            int(world_size) <= 0
            or int(gpus_per_node) <= 0
            or int(world_size) % int(gpus_per_node)
        ):
            raise ValueError("tune lookup: world_size must divide into equal GPU nodes")
        key = (
            str(gfx),
            int(cu_num),
            int(model_dim),
            int(inter_dim),
            int(expert),
            int(topk),
            int(token) // int(topk),
            _quant_key(quant, "tune lookup"),
            int(world_size),
            int(gpus_per_node),
        )
        path = self.path()
        with self._lock:
            if self._cache is None or self._cache_path != path:
                self._cache = self._load(path)
                self._cache_path = path
            return self._cache.get(key)


def _chip() -> tuple[str, int]:
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

    return get_gfx_runtime(), get_cu_num()


# Stage1
TILE_GROUPS = (1, 2, 3, 4)
_DEFAULT_TILE_GROUPS = (1, 2, 4)
_ARENA_BM = 32
# Rows per expert up to which G=1 is used.
_G1_MAX_ROWS = 48
# Split from this many tokens per rank up (token / topk; see ``split_local`` above).
_SPLIT_MIN_TPR = 1024
# The split path reserves 16 remote fanout shards plus the CCO ticket.
_FANOUT_CTAS_PER_DEST = {0: 16, 1: 17}
# T0 keeps its CCO role, so dest0 needs another shard; leave compute CTAs free.
FANOUT_CTAS_PER_DEST_MIN, FANOUT_CTAS_PER_DEST_MAX = 2, 31


def _validate_stage1(row: dict, where: str) -> None:
    # Reject out-of-domain values here rather than in the Stage1 factory, whose
    # error does not name the CSV row that produced them.
    if row["tile_group"] not in TILE_GROUPS:
        raise ValueError(
            f"{where}: tile_group must be one of {TILE_GROUPS} (got {row['tile_group']})"
        )
    if row["split_local"] not in (0, 1):
        raise ValueError(
            f"{where}: split_local must be 0 or 1 (got {row['split_local']})"
        )
    minimum = 17 if row["split_local"] else FANOUT_CTAS_PER_DEST_MIN
    fs = row.get("fanout_ctas_per_dest")
    if fs is None or not minimum <= fs <= FANOUT_CTAS_PER_DEST_MAX:
        fs = default_fanout_ctas_per_dest(row["split_local"])
    row["fanout_ctas_per_dest"] = fs


def default_split_local(token: int, topk: int) -> int:
    """Split from ``_SPLIT_MIN_TPR`` tokens per rank up (see docstring).

    ``token`` is the GEMM row count (tokens per rank x top-k)."""
    return 1 if int(token) // max(1, int(topk)) >= _SPLIT_MIN_TPR else 0


def default_tile_group(token: int, expert: int, split_local: int = 1) -> int:
    """Split path: G=1 for few rows per expert, G=4 otherwise (fitted).
    Unsplit path: the smallest G whose 32*G-row m_block holds the rows.

    ``token / expert`` is the balanced rows-per-expert estimate (``token`` is the
    GEMM row count, ``expert`` the per-rank count).
    """
    rows = int(token) / max(1, int(expert))
    if not split_local:
        return next(
            (g for g in _DEFAULT_TILE_GROUPS if _ARENA_BM * g >= rows),
            max(_DEFAULT_TILE_GROUPS),
        )
    return 1 if rows <= _G1_MAX_ROWS else 4


def default_fanout_ctas_per_dest(split_local: int) -> int:
    """Safe default shared by table validation and shape-rule fallback."""
    return _FANOUT_CTAS_PER_DEST[int(bool(split_local))]


def select_stage1_tile(
    *,
    token: int,
    model_dim: int,
    inter_dim: int,
    expert: int,
    topk: int,
    quant: str = "a4w4",
    world_size: int = 16,
    gpus_per_node: int = 8,
) -> dict:
    """Return ``{"tile_group", "split_local", "fanout_ctas_per_dest", "source"}`` for this
    shape; ``source`` is ``table`` or ``default`` (the built-in rule).  The kernel
    name (``_tg{G}``, ``_fos{N}``, ``_slg``) remains the ground truth for what ran.

    ``split_local`` comes from the table row when there is one; otherwise the tokens-per-rank rule
    (``default_split_local``) picks it and the built-in G and fanout follow.

    The split path needs 16 remote fanout shards plus the CCO ticket,
    so its minimum fanout count is 17; the unsplit minimum is 2.
    A missing or out-of-range table fanout uses the same path's default while
    preserving that row's tile and copy roles.
    """
    split = default_split_local(token, topk)
    tuned = None
    try:
        gfx, cu_num = _chip()
        tuned = lookup_tune(
            gfx=gfx,
            cu_num=cu_num,
            token=token,
            model_dim=model_dim,
            inter_dim=inter_dim,
            expert=expert,
            topk=topk,
            quant=quant,
            world_size=world_size,
            gpus_per_node=gpus_per_node,
        )
    except ValueError:
        # A malformed table is a configuration error, not a miss.
        raise
    except Exception:
        # Anything else (no device, no aiter root) falls back to the rule.
        logger.debug("Stage1 tune lookup failed; using built-in rules", exc_info=True)
        tuned = None
    if tuned is not None:
        out = {
            key: tuned[key]
            for key in (
                "tile_group",
                "split_local",
                "fanout_ctas_per_dest",
            )
        }
        source = "table"
    else:
        out = {
            "split_local": split,
            "tile_group": default_tile_group(token, expert, split),
            "fanout_ctas_per_dest": default_fanout_ctas_per_dest(split),
        }
        source = "default"
        _validate_stage1(out, "stage1 tile rule")
    out["source"] = source
    return out


# Fused stage2 CTA width.  8 waves (one CTA per CU, each CTA runs an n_block
# pair over a shared A tile), so model_dim / gemm2_bn must be even.
STAGE2_FUSED_WAVES = 8


def select_stage2_waves(n_blocks: int) -> int:
    """Waves per fused-stage2 CTA (see STAGE2_FUSED_WAVES)."""
    if int(n_blocks) % 2:
        raise ValueError(
            f"fused stage2 needs an even GEMM2 n_block count (model_dim / gemm2_bn = {n_blocks})"
        )
    return STAGE2_FUSED_WAVES


# Stage2
NUM_QP = 8
GEMM2_BNS = (128, 256)
GEMM2_BMS = (32, 64, 96, 128)
GEMM2_BKS = (256,)
DEFAULT_GEMM2_BK = 256
# Default GEMM2 M tile of the fused stage2 per intra-node path: xGMI 64
# (128 stalls at large TPR), SDMA 128.
GEMM2_BM_XGMI, GEMM2_BM_SDMA = 64, 128
STAGE2_INTRA_COMMS = ("xgmi", "sdma")
DEFAULT_GEMM2_BN = 128
DEFAULT_GEMM2_CU = 0
GEMM2_CU_MAX = 2048
# From this max_tok_per_rank the fused stage2 ships its intra-node rows with
# the copy engines (SDMA) instead of xGMI stores from the GEMM2 epilogue.
STAGE2_SDMA_MIN_TPR = 2048
# SDMA chunk = tickets (32-row tiles) per engine-side wait.  The waits are all
# posted up front and serialised on the queue, so the chunk grows with the
# rows to keep ~STAGE2_SDMA_CHUNKS of them.  See select_sdma_chunk_tiles.
STAGE2_SDMA_CHUNKS = 16
STAGE2_SDMA_CHUNK_TILES_MIN = 128
# Send-buffer size in rows per max_tok_per_rank * topk; rows past it go out as
# xGMI stores.
# Bytes per peer that build up before the posted SDMA put is cut (the copy
# engines need multi-MB puts to reach full link bandwidth).
STAGE2_SDMA_PUT_BYTES = 4 << 20
STAGE2_SDMA_SEND_ROWS_PER_ROUTE = 2
# Single kernel, remote-first stage2: the remote plane goes over the rail in
# token-completion order (tokens whose rows are all in are reduced between
# GEMM2 units and sent CHUNK at a time), not per token range at the end.
STAGE2_TOKEN_READY = True


def _validate_stage2(row: dict, where: str) -> None:
    # Reject out-of-domain values at load time rather than letting them reach
    # the Stage2 compile, whose error message says nothing about the CSV row.
    chunk = row["return_chunk_tokens"]
    if chunk < 4:
        raise ValueError(f"{where}: return_chunk_tokens must be >= 4 (got {chunk})")
    bn = row["gemm2_bn"]
    if bn not in GEMM2_BNS:
        raise ValueError(f"{where}: gemm2_bn must be one of {GEMM2_BNS} (got {bn})")
    cu = row["gemm2_cu"]
    if not 0 <= cu <= GEMM2_CU_MAX:
        raise ValueError(f"{where}: gemm2_cu must be in [0, {GEMM2_CU_MAX}] (got {cu})")
    bm = row["gemm2_bm"]
    if bm and bm not in GEMM2_BMS:
        raise ValueError(f"{where}: gemm2_bm must be one of {GEMM2_BMS} (got {bm})")
    bk = row["gemm2_bk"]
    if bk and bk not in GEMM2_BKS:
        raise ValueError(f"{where}: gemm2_bk must be one of {GEMM2_BKS} (got {bk})")
    comm = row["intra_comm"]
    if comm and comm not in STAGE2_INTRA_COMMS:
        raise ValueError(
            f"{where}: intra_comm must be one of {STAGE2_INTRA_COMMS} or empty (got {comm!r})"
        )


def _validate_tune(row: dict, where: str) -> None:
    """Validate the paired Stage1 and Stage2 parameters in one CSV row."""
    _validate_stage1(row, where)
    _validate_stage2(row, where)


TUNE_TABLE = _TuneTable(
    "AITER_CONFIG_MEGAMOE_INTERNODE",
    "mega_moe_internode_tuned.csv",
    ("tile_group", "split_local", "return_chunk_tokens", "gemm2_bn", "gemm2_cu"),
    _validate_tune,
    # 0 / "" = the built-in rule
    optional_fields=(
        ("fanout_ctas_per_dest", 0),
        ("gemm2_bm", 0),
        ("gemm2_bk", 0),
        ("intra_comm", ""),
    ),
)


def lookup_tune(
    *,
    gfx: str,
    cu_num: int,
    token: int,
    model_dim: int,
    inter_dim: int,
    expert: int,
    topk: int,
    quant: str = "a4w4",
    world_size: int = 16,
    gpus_per_node: int = 8,
) -> dict | None:
    """Return one paired configuration, or None when the shape/path has no row.

    ``token`` is the GEMM row count (tokens per rank times topk); the CSV uses
    per-rank ``tokens``. ``expert`` maps to the per-rank ``experts`` column.
    Stage1 and Stage2 share the row, including its split path.
    """
    row = TUNE_TABLE.lookup(
        gfx=gfx,
        cu_num=cu_num,
        token=token,
        model_dim=model_dim,
        inter_dim=inter_dim,
        expert=expert,
        topk=topk,
        quant=quant,
        world_size=world_size,
        gpus_per_node=gpus_per_node,
    )
    return row


def select_gemm2_bk(tuned: dict | None) -> tuple[int, str]:
    """GEMM2 K tile and its source: table > default."""
    bk = int((tuned or {}).get("gemm2_bk", 0))
    if not bk:
        return DEFAULT_GEMM2_BK, "default"
    if bk not in GEMM2_BKS:
        raise ValueError(f"gemm2_bk must be one of {GEMM2_BKS} (got {bk}, from table)")
    return bk, "table"


def select_gemm2_bn(tuned: dict | None) -> tuple[int, str]:
    """GEMM2 N tile and its source: table > default.

    ``tuned`` is the ``lookup_tune`` result (``None`` on a miss); the
    caller already has it, so this does not look the table up again.
    """
    if tuned:
        bn, source = int(tuned["gemm2_bn"]), "table"
    else:
        bn, source = DEFAULT_GEMM2_BN, "default"
    if bn not in GEMM2_BNS:
        raise ValueError(
            f"gemm2_bn must be one of {GEMM2_BNS} (got {bn}, from {source})"
        )
    return bn, source


def select_gemm2_bm(
    tuned: dict | None, *, tile_group: int, remote_first: bool, sdma: bool
) -> tuple[int, str]:
    """GEMM2 M tile and its source: table > rule (fused stage2: GEMM2_BM_SDMA
    or GEMM2_BM_XGMI by the intra-node path, while the tile group holds it) > 32.

    The default larger tile is selected for fused remote-first stage2. An
    explicit larger tile also works on the unsplit path when it covers whole
    32-row tiles of one Stage1 tile group (``32*tile_group % bm == 0``).
    """
    bm = int((tuned or {}).get("gemm2_bm", 0))
    if not bm:
        want = GEMM2_BM_SDMA if sdma else GEMM2_BM_XGMI
        if remote_first and (32 * int(tile_group)) % want == 0:
            return want, "default"
        return 32, "default"
    if bm == 32:
        return 32, "table"
    if bm not in GEMM2_BMS or (32 * int(tile_group)) % bm:
        raise ValueError(
            f"gemm2_bm {bm} must be one of {GEMM2_BMS} and divide 32*tile_group "
            f"(tile_group {tile_group})"
        )
    return bm, "table"


def select_gemm2_grid(tuned: dict | None) -> tuple[int, str]:
    """GEMM2 (kernel1) persistent grid and its source: table > 0.

    0 keeps the operator's behaviour of launching ``worker_blocks`` CTAs.
    """
    if tuned:
        cu, source = int(tuned["gemm2_cu"]), "table"
    else:
        cu, source = DEFAULT_GEMM2_CU, "default"
    if not 0 <= cu <= GEMM2_CU_MAX:
        raise ValueError(
            f"gemm2_cu must be in [0, {GEMM2_CU_MAX}] (got {cu}, from {source})"
        )
    return cu, source


def select_stage2_sdma(
    max_tok_per_rank: int, tuned: dict | None = None
) -> tuple[bool, str]:
    """Whether the fused stage2 moves intra-node rows by SDMA, and the source:
    table ``intra_comm`` > rule (TPR >= STAGE2_SDMA_MIN_TPR).  Only the fused
    stage2 has the SDMA path; the caller also needs Stage1's local/remote split.

    The best path depends on the model, not just the TPR, so the tune table
    row takes precedence.
    """
    comm = str((tuned or {}).get("intra_comm", "") or "")
    if comm:
        return comm == "sdma", "table"
    return int(max_tok_per_rank) >= STAGE2_SDMA_MIN_TPR, "default"


def select_sdma_chunk_tiles(rows: int, tile_group_gs: int = 1) -> int:
    """Tickets per SDMA chunk for ``rows`` GEMM rows (mtpr * topk): about
    STAGE2_SDMA_CHUNKS chunks, a power of two >= STAGE2_SDMA_CHUNK_TILES_MIN
    (a multiple of the GEMM2 tile's 32-row tiles, ``tile_group_gs``)."""
    tiles = -(-int(rows) // 32)
    want = max(STAGE2_SDMA_CHUNK_TILES_MIN, -(-tiles // STAGE2_SDMA_CHUNKS))
    ct = 1 << (want - 1).bit_length()
    if ct % int(tile_group_gs):
        raise ValueError(
            f"sdma chunk {ct} tiles is not a multiple of the GEMM2 tile ({tile_group_gs})"
        )
    return ct
