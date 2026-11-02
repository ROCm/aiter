# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Search engine for tuning spaces: ask/tell strategies instead of sweeps.

Kernel-neutral: any explicit parameter list (:class:`SearchSpace`) or any
flat config list (:meth:`SearchSpace.from_configs`) can be searched. A kernel
plugs in through a :class:`Backend` (``compile`` / ``check`` / ``measure``),
for example a :class:`CallableBackend` over three plain functions.

Ported from rocKE's ``rocke.dispatch.tuning.search`` (ROCm/rocm-libraries,
branch ``users/vanantha/rocke-monokernels``, commit ``4478c174a1``) without
the rocKE dispatcher adapters (``KnobSearchSpace``, ``SpecBackend``,
``tune_candidate``, ``SearchSpace.from_axes``); the toolchain fingerprint
covers FlyDSL instead of rocKE/comgr.

- :mod:`.params` -- :class:`Param`, :class:`SearchSpace`: mixed
  ordinal/categorical/bool/gated parameters, validity, repair, keys;
- :mod:`.trial` -- :class:`Trial`, :class:`Observation`, :class:`Fidelity`,
  :class:`Budget`;
- :mod:`.strategy` -- the :class:`Strategy` protocol, random and exhaustive;
- :mod:`.de` -- mixed-variable differential evolution (rand/1, current-to-
  (p)best/1; jDE / SHADE adaptation);
- :mod:`.halving` -- successive halving and Hyperband over fidelities;
- :mod:`.stats` -- paired / Welch t, signed-rank, Holm, :class:`Race`, noise;
- :mod:`.surrogate` -- priors (selector, LightGBM), TPE, GP + EI;
- :mod:`.local` -- hill climbing with lookback and combination re-sweeps;
- :mod:`.portfolio` -- the bandit portfolio and warm starts;
- :mod:`.measure` -- the backend protocol and the measurement rules;
- :mod:`.db` -- the tuning DB a dispatcher serves and search traces;
- :mod:`.driver` -- the ask/tell loop with checkpoint/resume;
- :mod:`.api` -- :func:`tune`, :func:`record_result`, :func:`search_configs`;
- :mod:`.replay` -- offline scoring against recorded sweeps.

Usage: ``python -m aiter.utility.tuning_search --help``.
"""

from .api import (
    DEFAULT_PORTFOLIO,
    FAMILY_DEFAULTS,
    STRATEGY_NAMES,
    default_ladder,
    make_strategy,
    recommend,
    record_result,
    search_configs,
    tune,
)
from .db import (
    TraceWriter,
    TunedRecord,
    TuningDB,
    problem_key,
    read_trace,
    records_from_rows,
)
from .de import JDE, DERand, DifferentialEvolution
from .driver import SearchResult, run_search
from .halving import Hyperband, SuccessiveHalving
from .local import HillClimb
from .measure import (
    Backend,
    CallableBackend,
    CompileOutcome,
    Evaluator,
    SyntheticBackend,
    TableBackend,
    toolchain_environment,
)
from .params import BOOL, CATEGORICAL, ORDINAL, Param, SearchSpace, flatten_spec
from .portfolio import Portfolio, WarmStart
from .stats import NoiseModel, Race
from .strategy import Exhaustive, RandomSearch, Strategy
from .surrogate import (
    GPEI,
    TPE,
    CallablePrior,
    LightGBMPrior,
    Prior,
    PriorScale,
    TablePrior,
)
from .trial import Budget, Fidelity, Observation, Status, Trial

__all__ = [
    "BOOL",
    "CATEGORICAL",
    "DEFAULT_PORTFOLIO",
    "FAMILY_DEFAULTS",
    "GPEI",
    "JDE",
    "ORDINAL",
    "STRATEGY_NAMES",
    "TPE",
    "Backend",
    "Budget",
    "CallableBackend",
    "CallablePrior",
    "CompileOutcome",
    "DERand",
    "DifferentialEvolution",
    "Evaluator",
    "Exhaustive",
    "Fidelity",
    "HillClimb",
    "Hyperband",
    "LightGBMPrior",
    "NoiseModel",
    "Observation",
    "Param",
    "Portfolio",
    "Prior",
    "PriorScale",
    "Race",
    "RandomSearch",
    "SearchResult",
    "SearchSpace",
    "Status",
    "Strategy",
    "SuccessiveHalving",
    "SyntheticBackend",
    "TableBackend",
    "TablePrior",
    "TraceWriter",
    "Trial",
    "TunedRecord",
    "TuningDB",
    "WarmStart",
    "default_ladder",
    "flatten_spec",
    "make_strategy",
    "problem_key",
    "read_trace",
    "recommend",
    "record_result",
    "records_from_rows",
    "run_search",
    "search_configs",
    "toolchain_environment",
    "tune",
]
