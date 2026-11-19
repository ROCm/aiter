"""Run the existing source-rank shuffle checks at a full Flash shape."""

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_mega_moe_gfx1250 as suite
from mori.cco import Communicator

_ORIGINAL_INIT = Communicator.init


def init_test_comm(cls, *args, **kwargs):
    # The reference push and experimental pull arenas coexist in this test.
    kwargs["per_rank_vmm"] = max(kwargs.get("per_rank_vmm", 0), 16 * 1024**3)
    return _ORIGINAL_INIT(*args, **kwargs)


Communicator.init = classmethod(init_test_comm)
suite.HIDDEN, suite.INTER, suite.TOPK = 5120, 2304, 6
suite.TOKENS = int(os.environ.get("CHECK_TPR", "257"))
suite.EXPERTS_PER_RANK = 96
suite.CHECKS = (suite._check_pull_s1f_src_rank_shuffle,)

if __name__ == "__main__":
    raise SystemExit(suite.main())
