# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Publish a tuned row only when it beats what serving runs and a fresh process proves it.

``GatedTuner`` is a ``TunerCommon`` for families whose tuned CSV pins one
candidate per exact key. The family supplies what only it knows: the problem
and its data, a reference, the candidate field, ``result_to_df``, the selection
probe and the hooks below. ``GatedTuner`` decides what reaches the CSV:

1. ``incumbents(row)`` names what serves the shape today: the published row if
   there is one, and what dispatch picks with no row, asked in a fresh process
   against an empty table. The family adds both to its candidate field, so the
   incumbent is timed by the same worker as its challengers.
2. ``post_process`` gates the fastest correct candidate against the incumbent
   with ``gate_against_incumbent``.
3. ``result_to_csv`` writes the promoted rows to a staging copy of the table,
   proves each in a fresh process against that copy, and replaces the table
   with only the rows that passed.

Hooks a family implements: ``RUNTIME_FIELDS``, ``CANDIDATE_FIELDS``,
``lookup_key``, ``candidate_from_selection``, ``expected_selection`` and the
``TunerCommon`` selection-probe attributes.
"""

import os
import tempfile
from pathlib import Path
from typing import Any, ClassVar, NamedTuple

import pandas as pd

from aiter import logger
from aiter.utility.base_tuner import TunerCommon
from aiter.utility.tuning_policy import (
    PROMOTE,
    PromotionPolicy,
    gate_against_incumbent,
)

HARDWARE_FIELDS = ("gfx", "gpu_model", "cu_num")


class Incumbents(NamedTuple):
    """What serves one shape today, as candidate fields."""

    # The published row's candidate, or None when the shape has no row.
    published: dict[str, str] | None
    # What dispatch picks with no row, or None when it could not be named.
    autoselect: dict[str, str] | None
    # Why autoselect is None.
    reason: str


class GatedTuner(TunerCommon):
    # Columns of the tuned CSV, key fields first.
    RUNTIME_FIELDS: ClassVar[tuple[str, ...]] = ()
    # Columns that name a candidate.
    CANDIDATE_FIELDS: ClassVar[tuple[str, ...]] = ()

    def __init__(self, name, key, resultList, description=None):
        super().__init__(name, key, resultList, description)
        self._args = None
        self._published: dict[tuple, dict[str, str]] = {}
        self._incumbents: dict[tuple, Incumbents] = {}
        self._dropped: set[tuple] = set()

    def lookup_key(self, row) -> tuple:
        """The key dispatch derives for ``row``. Override to canonicalize."""
        return tuple(_cell(row[field]) for field in self.keys)

    def candidate_identity(self, row) -> tuple:
        """What makes two candidates the same. Override to canonicalize."""
        return tuple(_cell(row.get(field)) for field in self.CANDIDATE_FIELDS)

    def candidate_from_selection(self, selection) -> dict[str, str] | None:
        """Candidate fields for a selection dispatch recorded, or None when
        the family's catalogue cannot name it."""
        raise NotImplementedError

    def stamp_hardware(self, frame: pd.DataFrame) -> pd.DataFrame:
        """``frame`` with the hardware columns set to this GPU's.

        A workload catalogue that names other hardware is an error: its rows
        would be measured here and published under a key they were not
        measured on.
        """
        live = {
            "gfx": self.get_gfx(),
            "gpu_model": self.get_gpu_model(),
            "cu_num": self.get_cu_num(),
        }
        frame = frame.copy()
        for field, value in live.items():
            if field in frame.columns:
                given = frame[field].fillna("").astype(str).str.strip().str.lower()
                if ((given != "") & (given != str(value).lower())).any():
                    raise ValueError(
                        f"workload catalogue {field} does not match this GPU ({value})"
                    )
            frame[field] = value
        return frame

    def load_published(self, args) -> None:
        """Read the tuned table and, unless ``--all``, skip shapes it covers."""
        self._args = args
        self.tunedf = _read_table(args.tune_file, self.RUNTIME_FIELDS)
        self._published = {}
        for _, row in self.tunedf.iterrows():
            key = self._key_or_none(row)
            if key is not None:
                self._published[key] = {
                    field: row[field] for field in self.CANDIDATE_FIELDS
                }
        if args.all or self.untunedf is None or self.untunedf.empty:
            return
        covered = self.untunedf.apply(
            lambda row: self.lookup_key(row) in self._published, axis=1
        ).astype(bool)
        if covered.any():
            logger.info(
                "skipping %d shapes that already have a row in %s; "
                "pass --all to re-tune them",
                int(covered.sum()),
                args.tune_file,
            )
        self.untunedf = self.untunedf[~covered].reset_index(drop=True)

    def incumbents(self, row) -> list[dict[str, str]]:
        """Candidates the family must add to this shape's field."""
        key = self.lookup_key(row)
        autoselect, reason = self._resolve_autoselect(row)
        found = Incumbents(self._published.get(key), autoselect, reason)
        self._incumbents[key] = found
        return [c for c in (found.published, found.autoselect) if c is not None]

    def _resolve_autoselect(self, row) -> tuple[dict[str, str] | None, str]:
        descriptor, empty_table = tempfile.mkstemp(prefix="autoselect-", suffix=".csv")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(",".join(self.RUNTIME_FIELDS) + "\n")
            proof = self.prove_selection(row, empty_table, self._args)
        finally:
            os.unlink(empty_table)
        if proof.status not in ("verified", "unchecked"):
            return None, f"auto-select probe {proof.status}: {proof.detail}"
        if len(proof.observed) != 1:
            return None, f"auto-select recorded {len(proof.observed)} selections"
        candidate = self.candidate_from_selection(proof.observed[0])
        if candidate is None:
            return None, f"auto-select runs {proof.observed[0]}, outside the catalogue"
        return candidate, ""

    def _policy(self, args) -> PromotionPolicy:
        return PromotionPolicy(min_improvement_pct=float(args.min_improvement_pct))

    def post_process(self, rets, args, topk=-1, fast_mode=False):
        self._args = args
        resultdf = self.result_to_df(rets)
        if args.profile_file:
            self._append_profile(resultdf, args.profile_file)
        promoted, settled, failed = [], [], []
        for _, group in resultdf.groupby(self.keys, dropna=False, sort=False):
            outcome, row = self._decide(group, args)
            if outcome == "promote":
                promoted.append(row)
            elif outcome in ("retain", "drop"):
                settled.append(row)
            else:
                failed.append(row)
        for row in (*promoted, *settled):
            print(
                f"{row['status']}: {self.lookup_key(row)}: {row['detail']}", flush=True
            )
        self._record(self.success, [*promoted, *settled], "success")
        self._record(self.failed, failed, "failed")
        return pd.DataFrame(promoted, columns=self.columns)

    def _decide(self, group, args) -> tuple[str, pd.Series]:
        key = self.lookup_key(group.iloc[0])
        valid = group[
            (group["status"] == "ok")
            & (group["us"] > 0)
            & (group["errRatio"] <= args.errRatio)
        ].sort_values("us")
        if valid.empty:
            return "fail", _settle(
                group.iloc[0], "failed", "no candidate passed the numerical check"
            )
        measured: dict[tuple, pd.Series] = {}
        for _, row in valid.iterrows():
            measured.setdefault(self.candidate_identity(row), row)
        fastest = valid.iloc[0]
        found = self._incumbents.get(key, Incumbents(None, None, "never resolved"))

        def timed(candidate):
            if candidate is None:
                return None
            return measured.get(self.candidate_identity(candidate))

        serving = timed(found.published)
        stale = found.published is not None and serving is None
        if serving is None:
            serving = timed(found.autoselect)
        if serving is None:
            reason = found.reason or "auto-select did not pass the numerical check"
            return "fail", _settle(
                fastest, "unverified", f"incumbent not measured ({reason})"
            )
        source = "published row" if not stale and found.published else "auto-select"
        settle = "drop" if stale else "retain"
        if stale:
            self._dropped.add(key)
        if self.candidate_identity(fastest) == self.candidate_identity(serving):
            return settle, _settle(serving, settle, f"{source} is fastest")
        policy = self._policy(args)
        decision = gate_against_incumbent(
            float(serving["us"]), float(fastest["us"]), policy
        )
        bar = policy.min_improvement_pct
        returns_to_autoselect = found.autoselect is not None and (
            self.candidate_identity(fastest)
            == self.candidate_identity(found.autoselect)
        )
        if decision.outcome == PROMOTE and returns_to_autoselect:
            self._dropped.add(key)
            return "drop", _settle(
                fastest,
                "drop",
                f"auto-select beat the {source} by {decision.margin_pct:.2f}%",
            )
        if decision.outcome == PROMOTE:
            self._dropped.discard(key)
            return "promote", _settle(
                fastest,
                "promoted",
                f"beat {source} by {decision.margin_pct:.2f}% (bar {bar:.2f}%)",
            )
        return settle, _settle(
            serving,
            settle,
            f"fastest beat {source} by {decision.margin_pct:.2f}%, under the {bar:.2f}% bar",
        )

    def result_to_csv(self, resultdf, file, concat=False):
        """Stage, prove, then commit only the rows that passed."""
        table = _read_table(file, self.RUNTIME_FIELDS)
        new = (
            pd.DataFrame(columns=self.RUNTIME_FIELDS)
            if resultdf is None or resultdf.empty
            else resultdf.loc[:, list(self.RUNTIME_FIELDS)].fillna("").astype(str)
        )
        new_keys = [self.lookup_key(row) for _, row in new.iterrows()]
        replaced = set(new_keys) | self._dropped
        kept = table[
            [self._key_or_none(row) not in replaced for _, row in table.iterrows()]
        ]
        staged = pd.concat([kept, new], ignore_index=True)

        failed_keys = set()
        if not new.empty:
            stage_path = _staging_path(file)
            try:
                _atomic_write_csv(staged, stage_path)
                for key, (_, row) in zip(new_keys, new.iterrows()):
                    proof = self.prove_selection(row, stage_path, self._args)
                    if proof.status != "verified":
                        logger.error(
                            "not publishing %s: selection proof %s: %s",
                            key,
                            proof.status,
                            proof.detail,
                        )
                        failed_keys.add(key)
            finally:
                if os.path.exists(stage_path):
                    os.unlink(stage_path)

        if failed_keys:
            restored = table[
                [self._key_or_none(row) in failed_keys for _, row in table.iterrows()]
            ]
            proven = new[[key not in failed_keys for key in new_keys]]
            staged = pd.concat([kept, restored, proven], ignore_index=True)
            self._mark_proof_failures(failed_keys)
        if not new.empty or self._dropped:
            _atomic_write_csv(staged, file)
        self._dropped = set()

    def _mark_proof_failures(self, failed_keys) -> None:
        in_success = self.success.apply(
            lambda row: self.lookup_key(row) in failed_keys, axis=1
        ).astype(bool)
        failures = self.success[in_success].copy()
        failures["status"] = "proof_failed"
        failures["detail"] = "fresh-process selection proof failed"
        self.success = self.success[~in_success]
        self._record(self.failed, [row for _, row in failures.iterrows()], "failed")

    def sortResults(self, tune_file, issorted, values):
        if not os.path.exists(tune_file):
            return
        table = _read_table(tune_file, self.RUNTIME_FIELDS)
        keys = [self._key_or_none(row) for _, row in table.iterrows()]
        table = table.assign(
            _key=[k if k is not None else i for i, k in enumerate(keys)]
        )
        table = table.drop_duplicates(subset="_key", keep="last").drop(columns="_key")
        if issorted:
            table = table.sort_values(list(self.keys))
        _atomic_write_csv(table, tune_file)

    def _append_profile(self, resultdf, path) -> None:
        frame = resultdf.astype(str)
        if os.path.exists(path):
            frame = pd.concat(
                [pd.read_csv(path, dtype=str, keep_default_na=False), frame],
                ignore_index=True,
            )
        frame = frame.drop_duplicates(
            subset=[*self.keys, *self.CANDIDATE_FIELDS], keep="last"
        )
        _atomic_write_csv(frame, path)

    def _key_or_none(self, row) -> tuple | None:
        try:
            return self.lookup_key(row)
        except (KeyError, TypeError, ValueError) as error:
            logger.warning(
                "tuned row does not parse as a problem (%s): %s", error, dict(row)
            )
            return None

    def _record(self, frame, rows, name) -> None:
        if not rows:
            return
        new = pd.DataFrame(rows, columns=self.columns)
        setattr(
            self,
            name,
            new if frame.empty else pd.concat([frame, new], ignore_index=True),
        )


def _cell(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _settle(row, status, detail) -> pd.Series:
    row = row.copy()
    row["status"] = status
    row["detail"] = detail
    return row


def _read_table(path, fields) -> pd.DataFrame:
    """The tuned table as text, so keys compare the way they were written."""
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=list(fields))
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [field for field in fields if field not in table.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return table[list(fields)]


def _staging_path(path) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staging = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".staged.csv", dir=destination.parent
    )
    os.close(descriptor)
    return staging


def _atomic_write_csv(frame: pd.DataFrame, path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
