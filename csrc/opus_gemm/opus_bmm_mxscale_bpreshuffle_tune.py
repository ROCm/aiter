# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Framework tuner for the fp8 e8m0 mxscale BMM with a PRESHUFFLED B (gfx1250).

Sweeps BOTH backends into one table: the opus tiles from
``a8w8_mxscale_bmm_bpreshuffle_kernels_list`` and the FlyDSL candidates from
``aiter.ops.flydsl.batched_gemm_a8w8_gfx1250.bmm_candidates``. One ``mp_tuner``
pass per shape, winner by measured us.

Sweeping both is the point, not a convenience. The shipped CSV
(``dsv4_batched_gemm_a8w8_blockscale_mxscale_bpreshuffle_tuned.csv``) holds 20
rows, every one ``libtype=flydsl, kernelId=-1``, hand-committed with the FlyDSL
PR; opus was never a candidate, and
``_batched_gemm_a8w8_mxscale_bpreshuffle_impl`` still raises on any other
libtype. Those rows are also stale -- b16/m16384 records 485 us where the same
kernel now measures ~401. And a ratio taken from two different sessions on this
box has repeatedly disagreed by more than the effect being measured, so only a
head-to-head inside one collection decides anything.

Candidate metadata comes from the codegen-adjacent kid table, never from a
second hand-kept copy: see that table's header for what the last hand-kept
column (m_align, wrong in BOTH directions) cost.

Runtime schema, matching what ``lookup_mxscale_bmm_config(..., bpreshuffle=True)``
reads back:
    gfx,b,m,n,k,libtype,kernelId,splitK,us,kernelName,tflops,bw,errRatio
``libtype`` picks the backend; ``kernelId`` is the opus tile id, or -1 for
FlyDSL whose variant is carried by ``kernelName`` (a string-encoded config the
launcher regex-decodes, not a lookup key).

Verification mirrors the raw-B tuner: inputs are SIGNED with per-128-K-block
varied magnitude, so the e8m0 scales span many exponents -- uniform
non-negative data hides a pure output-column permutation. Reference is a
dequantized fp32 einsum; ``mp_tuner`` gates on checkAllclose(rtol=atol=1e-2)
and ``post_process`` keeps the fastest candidate under ``--errRatio``.

Usage (gfx1250; repo root on PYTHONPATH so the rebuilt tree wins):
    cd <repo> && PYTHONPATH=$PWD \\
        python3 csrc/opus_gemm/opus_bmm_mxscale_bpreshuffle_tune.py \\
            -g 16 -m 32,1024,4096 -n 1024 -k 4096

    # re-tune every shape already in the shipped CSV, in place:
    PYTHONPATH=$PWD python3 csrc/opus_gemm/opus_bmm_mxscale_bpreshuffle_tune.py --all --apply
"""

from __future__ import annotations

import os
import sys
from typing import Any, ClassVar

import pandas as pd
import torch

from aiter import dtypes, logger
from aiter.ops.opus.bmm_op import _opus_bmm_a8w8_mxscale_bpreshuffle_raw
from aiter.ops.shuffle import shuffle_weight
from aiter.utility.base_tuner import GemmCommonTuner, TunerCommon
from aiter.utility.mp_tuner import mp_tuner

# Neither op_tests nor this directory is a package; both must be importable in
# the spawned mp_tuner children, which re-import this module top-to-bottom.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
_OPTESTS = os.path.join(_REPO, "op_tests")
for _p in (_HERE, _OPTESTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Pure python (stdlib only) -- importing the kid table does not pull in a build.
from opus_gemm_common import (  # noqa: E402
    A8W8_MXSCALE_BMM_BPRESHUFFLE_BAD_KIDS,
    a8w8_mxscale_bmm_bpreshuffle_kernels_list,
)

GROUP = 128
_CODEGEN = a8w8_mxscale_bmm_bpreshuffle_kernels_list

SHIPPED_CSV = os.path.join(
    _REPO,
    "aiter",
    "configs",
    "model_configs",
    "dsv4_batched_gemm_a8w8_blockscale_mxscale_bpreshuffle_tuned.csv",
)
DEFAULT_OUT = os.path.join(_REPO, "dsv4_bmm_mxscale_bpreshuffle_retuned.csv")


# ---------------------------------------------------------------------------
# Candidate applicability
# ---------------------------------------------------------------------------
def _opus_applicable(kid, g, m, n, k):
    """True when this opus tile can legally run this shape.

    Only the launcher's real constraints. There is deliberately no tile
    alignment check: the grid is ceil(m/B_M) x ceil(n/B_N) x batch and the
    pipelines mask partial M/N tails and ceil_div K, so unlike the gfx950
    family there is no m_align to respect.
    """
    if kid in A8W8_MXSCALE_BMM_BPRESHUFFLE_BAD_KIDS:
        return False
    inst = _CODEGEN[kid]
    if n % 16 or k % GROUP:
        return False
    if k % inst.GROUP_K:
        return False
    # LDS scale-panel budget. kid17/kid47 cap at K<=4096.
    if k > inst.k_cap:
        return False
    return True


def _flydsl_candidates(g, m, n, k):
    """FlyDSL variants for this shape, as (kernel_name, cfg) pairs.

    bmm_candidates() does NOT enumerate the preload flag, yet 13 of the 20
    shipped rows carry `_pre` -- i.e. the shipped rows cannot be reproduced from
    in-tree code alone. Cross it here, gated by the same LDS budget the launcher
    applies.
    """
    try:
        from aiter.ops.flydsl.batched_gemm_a8w8_gfx1250 import (
            _LDS_BYTES,
            bmm_candidates,
            bmm_kernel_name,
            preload_lds_bytes,
        )
    except ImportError as exc:  # flydsl not built for this arch
        logger.warning("flydsl bmm candidates unavailable: %s", exc)
        return []

    out = []
    for c in bmm_candidates(g, m, n, k):
        for preload in (False, True):
            if preload and preload_lds_bytes(c, k) > _LDS_BYTES:
                continue
            out.append((bmm_kernel_name(**c, preload=preload), c))
    return out


# ---------------------------------------------------------------------------
# mp_tuner hooks (module level so spawn children import them by name)
# ---------------------------------------------------------------------------
def _gen_varied(shape, k, device):
    """Signed, per-128-K-block varied-magnitude bf16."""
    x = torch.randn(shape, dtype=dtypes.fp32, device=device)
    amp = torch.exp2(torch.randint(-4, 4, (k // GROUP,), device=device).float())
    return (x * amp.repeat_interleave(GROUP)).to(dtypes.bf16)


def gen_bpreshuf_data(batch, m, n, k, seed, out_dtype, device="cuda"):
    """Return the 7-tuple mp_tuner indexes into.

    0 O_in      [m,g,k]           fp8, K innermost
    1 W_shuf    shuffle_weight(W, (16,16)) blob -- both backends take this one
    2 Y         [m,g,n]           out_dtype output buffer
    3 xs_in     [m,g,k/128]       e8m0 per-token A scale
    4 ws_block  [g,n/128,k/128]   e8m0 block B scale   (GROUP_N=128 tiles)
    5 ws_col    [g,n,k/128]       e8m0 per-column B scale (GROUP_N=1 tiles)
    6 ref       [m,g,n]           dequantized fp32 einsum reference

    Both scale families describe the SAME weights: ws_col is ws_block expanded
    along N, so one reference serves every candidate regardless of its GROUP_N.
    """
    from test_opus_a8w8_bmm import (
        _quant_block_e8m0,
        _quant_per_token_e8m0,
        run_torch,
    )

    torch.manual_seed(seed)
    O_bf16 = _gen_varied((batch, m, k), k, device)
    W_bf16 = _gen_varied((batch, n, k), k, device)
    O_mx, xs_mx, xs_fp32 = _quant_per_token_e8m0(O_bf16)
    W_mx, ws_block, ws_fp32 = _quant_block_e8m0(W_bf16)

    ws_col = ws_block.repeat_interleave(GROUP, dim=1).contiguous()

    W_shuf = shuffle_weight(W_mx, layout=(16, 16)).contiguous()
    O_in = O_mx.transpose(0, 1).contiguous()
    xs_in = xs_mx.transpose(0, 1).contiguous()
    Y = torch.empty((m, batch, n), dtype=out_dtype, device=device)
    ref = run_torch(O_mx, W_mx, xs_fp32, ws_fp32).transpose(0, 1).to(out_dtype)
    return (O_in, W_shuf, Y, xs_in, ws_block, ws_col, ref)


def run_opus_bpreshuf_bench(O_in, W_shuf, Y, xs_in, ws, kernelId, splitK):
    """Bench one opus tile in place; returns Y for checkAllclose."""
    _opus_bmm_a8w8_mxscale_bpreshuffle_raw(O_in, W_shuf, Y, xs_in, ws, splitK, kernelId)
    return Y


def run_flydsl_bpreshuf_bench(O_in, W_shuf, Y, xs_in, ws, kernel_name):
    """Bench one FlyDSL variant in place; returns Y for checkAllclose."""
    from aiter.ops.flydsl.batched_gemm_a8w8_gfx1250 import (
        run_bmm_a8w8_mxfp8_128_gfx1250,
    )

    return run_bmm_a8w8_mxfp8_128_gfx1250(
        O_in, W_shuf, xs_in, ws, Y, kernel_name=kernel_name
    )


def _ref_passthrough(ref):
    """ref_func: the fp32 reference is precomputed in gen_data (slot 6)."""
    return ref


# ---------------------------------------------------------------------------
# Tuner
# ---------------------------------------------------------------------------
class OpusBmmBpreshufTuner(GemmCommonTuner):
    ARG_DEFAULTS: ClassVar[dict[str, Any]] = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": DEFAULT_OUT,
        "untune_file": "",
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
        # GemmCommonTuner.__init__ force-swaps "M"/"N" in sort_keys, which
        # assumes the uppercase gptoss schema; go to the grandparent with the
        # lowercase batched one.
        TunerCommon.__init__(
            self,
            "OpusBmmBpreshufTuner",
            self.KEYS,
            self.RESULTS,
            description="Tune the preshuffled-B fp8 mxscale BMM (opus + flydsl)",
        )
        self.sort_keys = ["gfx", "b", "n", "m", "k"]

    # --- schema helpers -----------------------------------------------------
    def getKernelName(self, kernelId, libtype="opus"):
        if libtype == "flydsl":
            return None  # carried in the info tuple; never resolved by id
        inst = _CODEGEN.get(int(kernelId))
        if inst is None:
            return None
        return (
            f"opus_bmm_bpreshuf_{inst.B_M}x{inst.B_N}x{inst.B_K}"
            f"_bs{inst.BLOCK_SIZE}_nb{inst.num_slots}_gn{inst.GROUP_N}"
        )

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
        for el in results:
            info, time, err = el
            keys, kernelId, splitK, kernelName, libtype = info
            resolved = kernelName or self.getKernelName(kernelId, libtype)
            tflops, bw = self.calculate(el)
            row = dict(zip(self.keys, keys))
            row.update(
                {
                    "libtype": libtype,
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
        # Free the base "-k/--splitK" store_true so -k can mean the K dim.
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
            "-g", "--batch_g", type=_intlist, default=None,
            help="comma list of batch g (e.g. 2,8,16)",
        )
        self.parser.add_argument(
            "-m", "--M", type=_intlist, default=None,
            help="comma list of M (e.g. 32,1024,4096)",
        )
        self.parser.add_argument(
            "-n", "--N", type=_intlist, default=[1024],
            help="comma list of N (default 1024)",
        )
        self.parser.add_argument(
            "-k", "--K", type=_intlist, default=[4096],
            help="comma list of K (default 4096)",
        )
        self.parser.add_argument(
            "--libtype", type=str, default="both",
            choices=["both", "opus", "flydsl"],
            help="which backends to sweep (default both -- a cross-session "
                 "ratio on this box is not trustworthy)",
        )
        self.parser.add_argument(
            "--apply", action="store_true", default=False,
            help="overwrite the shipped tuned CSV in place",
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
        if args.apply:
            args.tune_file = SHIPPED_CSV

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
        want_opus = args.libtype in ("both", "opus")
        want_fly = args.libtype in ("both", "flydsl")

        task = []
        tasks_data = []
        for seed, i in enumerate(range(len(untunedf)), start=1):
            b = int(untunedf.loc[i, "b"])
            m = int(untunedf.loc[i, "m"])
            n = int(untunedf.loc[i, "n"])
            k = int(untunedf.loc[i, "k"])
            info_keys = (gfx, b, m, n, k)
            n_cand = 0

            if want_opus:
                for kid in sorted(_CODEGEN):
                    if not _opus_applicable(kid, b, m, n, k):
                        continue
                    # The w_scale slot IS the GROUP_N partition: slot 4 is the
                    # [g,n/128,k/128] block scale, slot 5 the [g,n,k/128]
                    # per-column one. Handing a tile the wrong one is not a
                    # silent mismatch -- the launcher throws on w_scale.size(1).
                    ws_slot = 4 if _CODEGEN[kid].GROUP_N == GROUP else 5
                    # nospec launcher requires splitK == 1 exactly; the
                    # specialized one accepts <= 1. So 1 for every tile.
                    task.append(
                        (
                            (info_keys, kid, 1, "", "opus"),
                            gen_bpreshuf_data,
                            (b, m, n, k, seed, out_dtype),
                            run_opus_bpreshuf_bench,
                            ([0, 1, 2, 3, ws_slot], kid, 1),
                            perf_kwargs,
                            _ref_passthrough,
                            ([6],),
                            {},
                            None,
                            1e-2,
                            1e-2,
                            None,
                            None,
                            [2],  # NaN-init Y so a partial write is caught
                        )
                    )
                    n_cand += 1

            if want_fly:
                for kname, _cfg in _flydsl_candidates(b, m, n, k):
                    task.append(
                        (
                            (info_keys, -1, 1, kname, "flydsl"),
                            gen_bpreshuf_data,
                            (b, m, n, k, seed, out_dtype),
                            run_flydsl_bpreshuf_bench,
                            ([0, 1, 2, 3, 4], kname),
                            perf_kwargs,
                            _ref_passthrough,
                            ([6],),
                            {},
                            None,
                            1e-2,
                            1e-2,
                            None,
                            None,
                            [2],
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
    tuner = OpusBmmBpreshufTuner()
    _args = tuner.parse_args()
    tuner.run(_args, False)
