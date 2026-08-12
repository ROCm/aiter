# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Framework tuner for the opus fp8 e8m0 mxscale flatmm split-K BMM (DSV4 wo_a).

Wired into the canonical :class:`GemmCommonTuner`, so it runs like the other
aiter GEMM tuners: multi-GPU via ``mp_tuner``, standard ``-i/--untune_file`` /
``-o/--tune_file`` CLI, batching, and the shared post-process / CSV writer.

The candidate pool lives here (``_TUNE_POLICY``). Per kid it holds only the
split-K factors to sweep; tile geometry, kernelName and the M alignment come from
the codegen instance table, so a kid cannot be tuned on a shape its launcher
rejects. That alignment used to be a second hand-maintained column and was wrong
in both directions -- it hid kid326, which is really arbitrary-M, from every
unaligned shape while the runtime dispatched it there anyway.

Runtime schema (what the tuner emits, and what the runtime reads back):
    gfx,b,m,n,k,libtype,kernelId,splitK,us,kernelName,tflops,bw,errRatio
``aiter/ops/batched_gemm_op_a8w8.py:lookup_mxscale_bmm_config`` indexes on
``["gfx","b","m","n","k"]``, dispatches to a backend on the winning row's
``libtype``, and ``bmm_op.py`` reads ``kernelId`` / ``splitK`` off that row, so
those columns must match exactly.

Verification (the part that catches column-transpose / scale defects):
  * inputs are *signed* and have *per-128-K-block varied magnitude*
    (``randn * 2**randint(-4,4)`` per block) so the e8m0 128-block scales span
    many exponents. Uniform non-negative ``rand()/10`` data hides a pure output
    column permutation (kid312/313 measured ~0.007 there but ~0.7-1.0 on real
    signed data) -- see the opus_bmm.md root-cause note.
  * reference is a dequantized fp32 einsum.
  * gate: ``mp_tuner`` runs ``checkAllclose(rtol=1e-2, atol=1e-2)`` and
    ``post_process`` keeps the fastest candidate whose mismatch fraction is
    ``<= --errRatio`` (default 0.02). A still-broken tileN COM_REP_N>1 kernel
    measures ~0.5 here and is rejected; the fp8 e8m0 quant floor is ~1e-4.

Usage (gfx950 only; the repo root must be on PYTHONPATH so the edited/rebuilt
tree wins over any installed aiter):
    cd <repo> && PYTHONPATH=$PWD \\
        python3 csrc/opus_gemm/opus_bmm_mxscale_tune.py -g 16 -m 1,16,64 -n 1024 -k 4096

    # re-tune every shape already in the shipped CSV, write a diffable copy:
    ... opus_bmm_mxscale_tune.py

    # overwrite the shipped tuned CSV in place:
    ... opus_bmm_mxscale_tune.py --apply

    # from an untuned CSV (columns: b,m,n,k -- or g,m,n,k), 8-way parallel:
    ... opus_bmm_mxscale_tune.py -i my_untuned.csv -o /tmp/out.csv --mp 8

    # the shipped shapes retuned into the second, preshuffle-inclusive table
    # (BPRESHUFFLE_CSV below; leaves the shipped one alone):
    ... opus_bmm_mxscale_tune.py --bpreshuffle --all --mp 8

    # what preshuffling B is worth: the same shapes twice, pool split by B layout,
    # then compare the two tables cell by cell:
    ... opus_bmm_mxscale_tune.py -i shapes.csv -o /tmp/rowb.csv --pool rowb --mp 8
    ... opus_bmm_mxscale_tune.py -i shapes.csv -o /tmp/preb.csv --pool preb --mp 8
"""

import os
import sys
from typing import Any, ClassVar

import pandas as pd
import torch

from aiter import dtypes, logger
from aiter.ops.opus.bmm_op import _opus_bmm_a8w8_mxscale_raw
from aiter.ops.shuffle import shuffle_weight
from aiter.utility.base_tuner import GemmCommonTuner, TunerCommon
from aiter.utility.mp_tuner import mp_tuner

# Neither op_tests nor this directory is a package, so put both on sys.path. This
# also has to hold in the spawned mp_tuner subprocesses, which re-import this
# module top-to-bottom.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
_OPTESTS = os.path.join(_REPO, "op_tests")
for _p in (_HERE, _OPTESTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# opus_gemm_common is pure python (stdlib only), so importing the codegen kid
# table here does not pull in the build.
from opus_gemm_common import (
    _BMM_MXSCALE_BPRESHUFFLE_BLDS_TWIN_OF,
    a8w8_mxscale_bmm_kernel_lists,
)
from test_opus_a8w8_bmm import (
    GROUP,
    _quant_block_e8m0,
    _quant_per_token_e8m0,
    run_torch,
)

# kid -> OpusGemmInstance. Kids are disjoint across the BMM families today;
# assert so a future collision (which the codegen dedups by launcher name
# downstream) is caught here instead of silently tuning one of the two.
_CODEGEN_BMM = {}
for _fam in a8w8_mxscale_bmm_kernel_lists:
    for _kid, _inst in _fam.items():
        assert (
            _kid not in _CODEGEN_BMM
        ), f"bmm kid {_kid} collides across codegen families; disambiguate by name"
        _CODEGEN_BMM[_kid] = _inst

# Split-K sweep for the flatmm_splitk family. Small-M / few-tile shapes (the G16
# wo_a decode: 16 batch * n1024 * k4096) underfill the CUs at splitK=1, so split-K
# (fp32-workspace partials + fused reduce tail) can win by exposing parallelism
# along K. The correctness gate drops any combo a kernel mishandles, so an
# over-broad sweep is safe, just slower.
_SK = [1, 2, 4, 8]

# Tuning policy: kid -> splitK list. The ONLY hand-maintained per-kid metadata --
# it decides which kids to sweep and with which split-K factors, not their
# geometry and not their M alignment. Tile shape, kernelName and m_align all come
# from the codegen instance, so this cannot drift from what compiles. kid 0 (the
# heuristic default) is intentionally not tuned.
_TUNE_POLICY = {
    # flatmm_splitk family: the M=16/32 last-mile tiles, the mid-M SFA/SFB-preload
    # tiles and the 64x* tiles. All are split-K capable via the fused reduce tail,
    # except kid646 whose persistent DIRECT_ONLY schedule requires splitK == 1.
    32: _SK,
    64: _SK,
    138: _SK,
    139: _SK,
    256: _SK,
    311: _SK,
    312: _SK,
    313: _SK,
    314: _SK,
    316: _SK,
    317: _SK,
    318: _SK,
    319: _SK,
    320: _SK,
    321: _SK,
    322: _SK,
    323: _SK,
    324: _SK,
    326: _SK,
    327: _SK,
    640: _SK,
    642: _SK,
    646: [1],
    650: _SK,
    653: _SK,
    # fused single-tile launcher.
    100: [1],
    # pipeline family; kid158 preloads both the per-token SFA and the block SFB
    # panel into LDS.
    149: [1],
    150: [1],
    151: [1],
    152: [1],
    158: [1],
    # kid158's scale preload at the two half tiles. Narrow-N wo_a (n1024) gives
    # the 256x256 tile only 4 N-tiles, so mid-M shapes leave half the CUs idle;
    # these fill them from the M side (159) and the N side (164). kid164 also
    # covers n128, which the 256-wide tiles reject outright.
    159: [1],
    164: [1],
    # The preshuffled-B families, all of them, splitK=1 only: none carries the
    # flatmm_splitk launcher's fused reduce tail. Nothing here was ever actually
    # evaluated before -- B arrived row-major (see gen_bmm_mxscale_data) so every
    # one of them failed the correctness gate silently, which is why the shipped
    # table picks none of them despite three of them having been listed here.
    #
    # wave8n4, the 2x4 eight-wave grid: 256x256 for large M, 128x256 mid, and the
    # 128x64x256 tile that carries the small-M end.
    168: [1],
    175: [1],
    194: [1],
    # wavetm1, the 1x8 / 1x4 grids. kid205 is kid203 plus the banded tile map.
    202: [1],
    203: [1],
    205: [1],
    # bdirect, B straight to registers with no LDS hop: the 16x32 and 64x32
    # last-mile tiles, and the 128x128 tile that owns the mid band.
    171: [1],
    172: [1],
    173: [1],
    179: [1],
    184: [1],
    # kid158's pipeline with only B's layout flipped -- the direct A/B comparison
    # for what preshuffling B is worth at the tile the shipped table leans on.
    196: [1],
    # monolithic mouter / wave pipelines.
    131: [1],
    132: [1],
    134: [1],
    142: [1],
    144: [1],
    148: [1],
    160: [1],
    161: [1],
    # minterleave only exists in split-K form.
    162: [2, 4, 8],
    163: [2, 4, 8],
    # 128x128x128 tiles, splitK=1 only and deliberately so. They are the largest
    # BMM tile (COM_REP_M=4 x COM_REP_N=8 -> 32 C fragments, 128 fp32 C values per
    # lane) at 512 VGPRs / occupancy 1. At splitK=1 they run the Cbf16
    # direct-output kernel and are strong -- kid325 wins 5 shipped wo_a rows. At
    # splitK>1 they switch to the Cvoid fp32-workspace kernel, which spills and has
    # never won (g2/m256: best kid325 split-K is 23.6us against the 14.5us winner),
    # and which is also where the clang-22 gfx950 greedy-VGPR miscompile lives (one
    # C-fragment dword left unmaterialized under --amdgpu-mfma-vgpr-form).
    128: [1],
    137: [1],
    325: [1],
}

# The blds twins, derived rather than listed: a twin is its plain kid's tile with
# B preshuffled and B's LDS staging kept, so wherever the plain kid is worth trying
# the twin is the preshuffled pool's answer to that cell. Deriving it here is what
# keeps the two in step -- the catalog asserts every plain flatmm tile has a twin,
# and this makes every one of those twins a candidate, so neither half of the pair
# can be added without the other reaching the sweep.
#
# splitK stays at 1 even where the plain kid sweeps the full _SK. Split-K has never
# won a flatmm cell on this envelope (the only sk>1 winners are kid163's minterleave
# rows), and sweeping four factors over 27 more kids is the bulk of the tuning time.
# A plain kid that wins a cell at splitK>1 is therefore still a cell the preshuffled
# pool cannot answer; widen this if one ever appears.
_TUNE_POLICY.update(
    {
        twin: [1]
        for plain, twin in _BMM_MXSCALE_BPRESHUFFLE_BLDS_TWIN_OF.items()
        if plain in _TUNE_POLICY
    }
)

# Deliberately not candidates: kid208 (mpack_sfa) and kid210/213/214/215/216/217
# (shuffle_scale) need A's scale panel, and for the shuffle_scale kids B's too, relaid out by
# whatever produces them. That layout is fixed once per model, so a table meant
# for a model whose quantiser emits plain scales cannot dispatch to them at all --
# see the shuffle_scale notes in opus_gemm_common.py for what the switch would be worth.
_RELAYOUT_KIDS = sorted(
    kid
    for kid, inst in _CODEGEN_BMM.items()
    if inst.needs_mpacked_sfa is not None or inst.needs_shuffle_scale is not None
)
assert not (set(_TUNE_POLICY) & set(_RELAYOUT_KIDS)), (
    f"kids {sorted(set(_TUNE_POLICY) & set(_RELAYOUT_KIDS))} need a producer-side "
    "scale relayout and cannot be tuned against plain scales"
)

# A policy entry for a kid the codegen no longer emits used to KeyError inside
# _applicable on the first shape, i.e. after the data was built. kid165, kid174
# and kid192 sat here that way. Fail at import instead.
_dead = sorted(set(_TUNE_POLICY) - set(_CODEGEN_BMM))
assert not _dead, f"_TUNE_POLICY lists kids the codegen does not emit: {_dead}"

# Only the flatmm_splitk (non-direct) and minterleave launchers honor splitK>1.
# Any other family sweeping it is a policy bug, so fail loudly at import.
for _kid, _sks in _TUNE_POLICY.items():
    if any(s > 1 for s in _sks):
        _tag = _CODEGEN_BMM[_kid].kernel_tag
        assert (
            _tag == "a8w8_mxscale_bmm_flatmm_splitk"
            and not getattr(_CODEGEN_BMM[_kid], "direct_only", False)
        ) or _tag == "a8w8_mxscale_bmm_minterleave", (
            f"kid {_kid} ({_tag}) is not split-K capable but sweeps {_sks}"
        )


POOLS = ("all", "preb", "rowb")


def _applicable(kid, g, m, n, k, pool="all"):
    """Split-K factors worth trying for this kid on this shape ([] == skip it).

    ``pool`` restricts by B's layout: "preb" to the preshuffled-B kids, "rowb" to
    the row-major ones, "all" to both. A shipped table has to be all one layout to
    be usable -- the weight is shuffled offline, so a deployment holds one form of
    it, and a mixed table would need both resident. The first cut of the preshuffle
    table did mix them, row-major kids winning 15 of 77 shapes on merit, and those
    rows are exactly the ones a preshuffled deployment cannot dispatch. Tuning the
    two pools separately over one shape set is also how the layouts get compared
    at all: per shape, best row-major against best preshuffled.
    """
    k_inst = _CODEGEN_BMM[kid]
    if pool == "preb" and not k_inst.needs_preshuffled_b:
        return []
    if pool == "rowb" and k_inst.needs_preshuffled_b:
        return []
    if n % k_inst.B_N or k % k_inst.B_K or m % k_inst.m_align:
        return []
    return _TUNE_POLICY[kid]


SHIPPED_CSV = os.path.join(
    _REPO,
    "aiter",
    "configs",
    "model_configs",
    "dsv4_batched_gemm_a8w8_blockscale_mxscale_tuned.csv",
)
DEFAULT_OUT = os.path.join(_REPO, "dsv4_bmm_mxscale_retuned.csv")

# The same dsv4 shapes retuned with the preshuffled-B families in the pool, kept
# as a second table rather than applied over the first.
#
# The name is load-bearing in a way worth spelling out. get_config_file globs
# model_configs/ for "*batched_gemm_a8w8_blockscale_mxscale_tuned*.csv" and merges
# every hit; two tables covering the same shapes would collide on the (gfx,b,m,n,k)
# key, and update_config_files answers a collision by rewriting the source files
# down to the lowest-us row each and then raising. Putting "bpreshuffle" before
# "tuned" breaks the substring, so this file is invisible to that glob and cannot
# take the shipped table with it. Select it explicitly:
#
#   AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE=<this file> python3 ...
BPRESHUFFLE_CSV = os.path.join(
    _REPO,
    "aiter",
    "configs",
    "model_configs",
    "dsv4_batched_gemm_a8w8_blockscale_mxscale_bpreshuffle_tuned.csv",
)


# ---------------------------------------------------------------------------
# mp_tuner hooks (module-level so the spawn workers can import them by name).
# ---------------------------------------------------------------------------
def _gen_varied(shape, k, device):
    """Signed, per-128-K-block varied-magnitude bf16 (mirrors _block_varied)."""
    x = torch.randn(shape, dtype=dtypes.fp32, device=device)
    amp = torch.exp2(torch.randint(-4, 4, (k // GROUP,), device=device).float())
    return (x * amp.repeat_interleave(GROUP)).to(dtypes.bf16)


def gen_bmm_mxscale_data(batch, m, n, k, seed, out_dtype, device="cuda"):
    """Return the 7-tuple mp_tuner indexes into:

    0 O_in   [m,g,k]     fp8 (mmajor transposed view, K contiguous)
    1 W_mx   [g,n,k]     fp8 (batch-major)
    2 Y       [m,g,n]     out_dtype output buffer
    3 xs_in  [m,g,k/128] uint8 e8m0 per-token scale (mmajor view)
    4 ws_mx  [g,n/128,k/128] uint8 e8m0 128x128-block scale
    5 ref     [m,g,n]     out_dtype dequant fp32 einsum reference
    6 W_sh   [g,n,k]     the same B in the (16,16) preshuffled layout
    """
    torch.manual_seed(seed)
    O_bf16 = _gen_varied((batch, m, k), k, device)
    W_bf16 = _gen_varied((batch, n, k), k, device)
    O_mx, xs_mx, xs_fp32 = _quant_per_token_e8m0(O_bf16)
    W_mx, ws_mx, ws_fp32 = _quant_block_e8m0(W_bf16)
    O_in = O_mx.transpose(0, 1)  # [m,g,k]
    xs_in = xs_mx.transpose(0, 1)  # [m,g,k/128]
    Y = torch.empty((m, batch, n), dtype=out_dtype, device=device)
    ref = run_torch(O_mx, W_mx, xs_fp32, ws_fp32).transpose(0, 1).to(out_dtype)
    # The preshuffled-B kids read B through the (16,16) MFMA-fragment layout that
    # a serving stack bakes into the weight offline. It is a permutation, so the
    # reference above covers both forms. Building it here rather than in the bench
    # keeps it out of what is timed.
    W_sh = shuffle_weight(W_mx, layout=(16, 16))
    return (O_in, W_mx, Y, xs_in, ws_mx, ref, W_sh)


def run_bmm_mxscale_bench(O_in, W_mx, Y, xs_in, ws_mx, W_sh, kernelId, splitK):
    """Tuner bench func: run the kid in-place, return Y for checkAllclose.

    B's layout is per kid, so it is selected here rather than by the caller: a
    preshuffled-B kid handed row-major B reads the right bytes in the wrong order
    and fails the gate, which is how three of these kids sat in _TUNE_POLICY
    without ever being able to win a shape.
    """
    Wb = W_sh if _CODEGEN_BMM[kernelId].needs_preshuffled_b else W_mx
    _opus_bmm_a8w8_mxscale_raw(O_in, Wb, Y, xs_in, ws_mx, splitK, kernelId)
    return Y


def _bmm_ref_passthrough(ref):
    """ref_func: the fp32 reference is precomputed in gen_data (slot 5)."""
    return ref


# ---------------------------------------------------------------------------
# Tuner
# ---------------------------------------------------------------------------
class OpusBmmMxscaleTuner(GemmCommonTuner):
    ARG_DEFAULTS: ClassVar[dict[str, Any]] = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": DEFAULT_OUT,
        "untune_file": "",
        # Fraction-of-mismatch (rtol=atol=1e-2) accept threshold. Correct kids
        # sit at the ~1e-4 fp8 e8m0 quant floor; a column-transposed kid is ~0.5.
        "errRatio": 0.02,
        "batch": 100,
    }

    KEYS: ClassVar[list[str]] = ["gfx", "b", "m", "n", "k"]
    RESULTS: ClassVar[list[str]] = [
        "libtype",
        "kernelId",
        "splitK",
        "us",
        "kernelName",
        "tflops",
        "bw",
        "errRatio",
    ]

    def __init__(self):
        # Bypass GemmCommonTuner.__init__ (it force-swaps "M"/"N" in the key,
        # which assumes the uppercase gptoss schema). Go straight to the
        # grandparent with our lowercase batched schema.
        TunerCommon.__init__(
            self,
            "OpusBmmMxscaleTuner",
            self.KEYS,
            self.RESULTS,
            description="Tune opus fp8 e8m0 mxscale flatmm split-K BMM (DSV4 wo_a)",
        )
        # sort N before M like the GEMM tuners (cosmetic ordering of the CSV).
        self.sort_keys = ["gfx", "b", "n", "m", "k"]

    # --- schema helpers -----------------------------------------------------
    def getKernelName(self, kernelId):
        k_inst = _CODEGEN_BMM.get(int(kernelId))
        return k_inst.name if k_inst else None

    def calculate(self, results, bpes=None):
        info, time, _err = results
        if time == self.INVALID_TIME:
            return 0, 0
        _gfx, b, m, n, k = info[0]
        us_s = time * 1e-6
        tflops = round(2 * b * m * n * k / us_s / 1e12, 1)
        # fp8 A + fp8 W + bf16 out.
        bw = round((b * m * k + b * n * k + 2 * b * m * n) / us_s / 1e9, 2)
        return tflops, bw

    def result_to_df(self, results):
        rows = []
        for info, time, err in results:
            keys, kernelId, splitK, kernelName = info
            resolved = kernelName or self.getKernelName(kernelId)
            tflops, bw = self.calculate((info, time, err))
            row = dict(zip(self.keys, keys))
            row.update(
                {
                    "libtype": "opus",
                    "kernelId": int(kernelId),
                    "splitK": int(splitK),
                    "us": time,
                    "kernelName": "None" if resolved is None else str(resolved),
                    "tflops": tflops,
                    "bw": bw,
                    "errRatio": err,
                }
            )
            rows.append(row)
        return pd.DataFrame(rows, columns=self.columns)

    # --- CLI ----------------------------------------------------------------
    def _setup_specific_arguments(self):
        # Free the base "-k/--splitK" store_true so we can reuse -k for the K dim.
        for action in list(self.parser._actions):
            if "-k" in action.option_strings or "--splitK" in action.option_strings:
                self.parser._actions.remove(action)
                for s in action.option_strings:
                    self.parser._option_string_actions.pop(s, None)
                for grp in self.parser._action_groups:
                    if action in grp._group_actions:
                        grp._group_actions.remove(action)
                break

        def _intlist(s):
            return [int(x) for x in str(s).split(",") if x != ""]

        self.parser.add_argument(
            "-g",
            "--batch_g",
            type=_intlist,
            default=None,
            help="comma list of batch g (e.g. 2,8,16)",
        )
        self.parser.add_argument(
            "-m",
            "--M",
            type=_intlist,
            default=None,
            help="comma list of M (e.g. 1,16,64)",
        )
        self.parser.add_argument(
            "-n",
            "--N",
            type=_intlist,
            default=[1024],
            help="comma list of N (default 1024)",
        )
        self.parser.add_argument(
            "-k",
            "--K",
            type=_intlist,
            default=[4096],
            help="comma list of K (default 4096)",
        )
        self.parser.add_argument(
            "--apply",
            action="store_true",
            default=False,
            help="overwrite the shipped tuned CSV in place",
        )
        self.parser.add_argument(
            "--bpreshuffle",
            action="store_true",
            default=False,
            help="write to the preshuffle-inclusive table (BPRESHUFFLE_CSV)",
        )
        self.parser.add_argument(
            "--pool",
            choices=POOLS,
            default="all",
            help="restrict candidates by B layout (--bpreshuffle implies preb)",
        )

    # --- shape sourcing -----------------------------------------------------
    def _shapes_from_shipped(self):
        try:
            df = pd.read_csv(SHIPPED_CSV)
        except FileNotFoundError:
            return []
        return sorted(
            {(int(r.b), int(r.m), int(r.n), int(r.k)) for _, r in df.iterrows()}
        )

    def pre_process(self, args):
        if args.apply and args.bpreshuffle:
            raise SystemExit("--apply and --bpreshuffle write different tables")
        if args.apply:
            args.tune_file = SHIPPED_CSV
        elif args.bpreshuffle:
            args.tune_file = BPRESHUFFLE_CSV
            if args.pool == "all":
                args.pool = "preb"

        gfx = self.get_gfx()
        if args.batch_g and args.M:
            shapes = [
                (g, m, n, k)
                for g in args.batch_g
                for m in args.M
                for n in args.N
                for k in args.K
            ]
        elif args.untune_file and os.path.exists(args.untune_file):
            df = pd.read_csv(args.untune_file)
            df.columns = [c.strip().lower() for c in df.columns]
            bcol = "b" if "b" in df.columns else "g"
            shapes = [
                (int(r[bcol]), int(r["m"]), int(r["n"]), int(r["k"]))
                for _, r in df.iterrows()
            ]
        else:
            logger.info(
                "no -g/-m and no untune_file; re-tuning shapes from %s", SHIPPED_CSV
            )
            shapes = self._shapes_from_shipped()

        self.untunedf = pd.DataFrame(
            [{"gfx": gfx, "b": g, "m": m, "n": n, "k": k} for (g, m, n, k) in shapes],
            columns=self.keys,
        )
        self.tunedf = self.get_tuned_gemm_list(args.tune_file)

        # Skip shapes already present in the tuned CSV (unless --all forces retune).
        if not args.all and len(self.tunedf) and len(self.untunedf):
            td = self.tunedf
            if "gfx" not in td.columns:
                td = td.assign(gfx=gfx)
            have = set(td[self.keys].apply(lambda r: tuple(r), axis=1).tolist())
            mask = self.untunedf.apply(lambda r: tuple(r) in have, axis=1)
            if args.verbose and mask.any():
                logger.info("skipping %d already-tuned shapes", int(mask.sum()))
            self.untunedf = self.untunedf[~mask].reset_index(drop=True)

    # --- tuning -------------------------------------------------------------
    def tune(self, untunedf, tunedf, args):
        gfx = self.get_gfx()
        out_dtype = dtypes.bf16
        perf_kwargs = {"num_warmup": args.warmup, "num_iters": args.iters}

        task = []
        tasks_data = []
        for seed, i in enumerate(range(len(untunedf)), start=1):
            b = int(untunedf.loc[i, "b"])
            m = int(untunedf.loc[i, "m"])
            n = int(untunedf.loc[i, "n"])
            k = int(untunedf.loc[i, "k"])
            info_keys = (gfx, b, m, n, k)

            n_cand = 0
            for kid in _TUNE_POLICY:
                for sk in _applicable(kid, b, m, n, k, args.pool):
                    info = (info_keys, kid, sk, "")
                    task.append(
                        (
                            info,
                            gen_bmm_mxscale_data,
                            (b, m, n, k, seed, out_dtype),
                            run_bmm_mxscale_bench,
                            ([0, 1, 2, 3, 4, 6], kid, sk),
                            perf_kwargs,
                            _bmm_ref_passthrough,
                            ([5],),
                            {},
                            None,
                            1e-2,  # rtol
                            1e-2,  # atol
                            None,  # compare_fn
                            None,  # max_abs_delta
                            [2],  # output_keys: NaN-init Y to catch partial writes
                        )
                    )
                    n_cand += 1
            tasks_data.append((n_cand, ()))

        if not task:
            return []
        return mp_tuner(
            task,
            tasks_data,
            args.mp,
            False,
            args.shape_grouped,
            args.errRatio,
            timeout=args.timeout,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    tuner = OpusBmmMxscaleTuner()
    _args = tuner.parse_args()
    tuner.run(_args, False)
