"""Make MegaMoEV2's cross-rank epoch buffering depth configurable.

Upstream double-buffers the P2P handshake: 2 parity slots, indexed as
parity*npes+rank / parity*epr+local_expert, with strict-equality waits on
monotonically increasing `expected` values. Two slots only tolerate one epoch
of drift between ranks. Issuing many mega calls back to back (ATOM issues 61,
one per MoE layer, per forward step) lets the ranks drift further and a peer's
flag is overwritten while someone is still waiting for its previous value.

Measured: depth 1 survives 610 calls; depth 61 wedges after 122.

AITER_MEGA_EPOCH_SLOTS (power of two, default 2 = upstream behaviour).
"""

import re
import sys

V2 = "/app/aiter-test/aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py"
S1 = "/app/aiter-test/aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py"

CONST = (
    "\n_EPOCH_SLOTS = int(os.environ.get('AITER_MEGA_EPOCH_SLOTS', '2'))\n"
    "assert _EPOCH_SLOTS >= 2 and (_EPOCH_SLOTS & (_EPOCH_SLOTS - 1)) == 0, (\n"
    "    'AITER_MEGA_EPOCH_SLOTS must be a power of two >= 2'\n"
    ")\n"
)

V2_SUBS = [
    (
        "self._s1_epoch_expected = torch.zeros(2, dtype=torch.int32, device=self.dev)",
        "self._s1_epoch_expected = torch.zeros(_EPOCH_SLOTS, dtype=torch.int32, device=self.dev)",
    ),
    (
        '"pair_ready": torch.zeros(2, dtype=torch.int32, device=self.dev),',
        '"pair_ready": torch.zeros(_EPOCH_SLOTS, dtype=torch.int32, device=self.dev),',
    ),
    (
        '"pair_order_ready": torch.zeros(2, dtype=torch.int32, device=self.dev),',
        '"pair_order_ready": torch.zeros(_EPOCH_SLOTS, dtype=torch.int32, device=self.dev),',
    ),
    (
        'workspace["count_done"] = op._sym((2 * self.world_size,), torch.int32)',
        'workspace["count_done"] = op._sym((_EPOCH_SLOTS * self.world_size,), torch.int32)',
    ),
    (
        'workspace["plan_ready"] = op._sym((2 * self.world_size,), torch.int32)',
        'workspace["plan_ready"] = op._sym((_EPOCH_SLOTS * self.world_size,), torch.int32)',
    ),
    (
        'workspace["payload_ready"] = op._sym((2 * self.epr,), torch.int32)',
        'workspace["payload_ready"] = op._sym((_EPOCH_SLOTS * self.epr,), torch.int32)',
    ),
]

S1_SUBS = [
    (
        "next_parity = old_parity ^ fx.Int32(1)",
        "next_parity = (old_parity + fx.Int32(1)) & fx.Int32(_EPOCH_SLOTS - 1)",
    ),
]


def add_const(text):
    if "_EPOCH_SLOTS" in text and "AITER_MEGA_EPOCH_SLOTS" in text:
        return text, False
    if not re.search(r"^import os$", text, re.M):
        text = "import os\n" + text
    lines = text.split("\n")
    last = max(
        i for i, l in enumerate(lines) if l.startswith("import ") or l.startswith("from ")
    )
    lines.insert(last + 1, CONST)
    return "\n".join(lines), True


def patch(path, subs):
    with open(path) as f:
        text = f.read()
    orig = text
    text, added = add_const(text)
    missing = []
    for old, new in subs:
        if new in text:
            continue
        if old not in text:
            missing.append(old)
            continue
        text = text.replace(old, new, 1)
    if missing:
        print(f"MISSING in {path}:")
        for m in missing:
            print(f"  {m}")
        return False
    if text != orig:
        with open(path + ".bak_epochslots", "w") as f:
            f.write(orig)
        with open(path, "w") as f:
            f.write(text)
    print(f"PATCHED {path} (const_added={added})")
    return True


ok = patch(V2, V2_SUBS) and patch(S1, S1_SUBS)
print("PATCH_OK" if ok else "PATCH_FAILED")
sys.exit(0 if ok else 1)
