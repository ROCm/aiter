#!/usr/bin/env python3
"""Ask amd-smi about the GPU this run claimed.

WHY THIS IS NOT INSIDE THE SHELL
--------------------------------
Two heredocs asked amd-smi the same two questions, and both opened with the same six lines of
``importlib`` boilerplate to borrow ``read_activity`` from the picker. Duplicated setup around a
device query is where an "unknown" quietly becomes a "0": the picker's own docstring records
that some driver and amd-smi combinations fail the activity query outright, and the distinction
between unknown and measured-idle only survives if every caller preserves it. There is one
caller now, and it is tested with a stub device rather than only on a host that happens to have
the hardware.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import socket
import sys

PICKER_PATH = pathlib.Path(__file__).resolve().parent / "pick-idle-gpu.py"

ACTIVITY_UNAVAILABLE = "unavailable"


def load_picker():
    """Import the shipped picker as a module, for its amd-smi handling.

    The picker is the thing that knows which amd-smi to import and how to read activity without
    reporting an unavailable value as an idle one. Borrowing it keeps one implementation of that
    judgement rather than a second copy here that could drift away from it.
    """
    spec = importlib.util.spec_from_file_location("validation_gpu_picker", PICKER_PATH)
    picker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(picker)
    return picker


def find_handle(amdsmi, hip_id: int):
    """The (amd-smi index, handle) whose HIP index is ``hip_id``.

    HIP and amd-smi order devices independently, so the two indices are not interchangeable and
    the report carries both.
    """
    for smi_index, handle in enumerate(amdsmi.amdsmi_get_processor_handles()):
        enumeration = amdsmi.amdsmi_get_gpu_enumeration_info(handle)
        if enumeration.get("hip_id") == hip_id:
            return smi_index, handle
    raise RuntimeError(f"HIP index {hip_id} has no amd-smi mapping")


def describe(amdsmi, picker, hip_id: int, hostname: str) -> dict:
    """The gpu_claim stage's record of which device this run holds."""
    smi_index, handle = find_handle(amdsmi, hip_id)
    asic = amdsmi.amdsmi_get_gpu_asic_info(handle)
    gfx_activity, _ = picker.read_activity(amdsmi, handle)
    return {
        "status": "pass",
        "hip_index": hip_id,
        "amd_smi_index": smi_index,
        "model": asic.get("market_name", "unknown"),
        "arch": asic.get("target_graphics_version", "unknown"),
        "bdf": amdsmi.amdsmi_get_gpu_device_bdf(handle),
        "gfx_activity_before_pct": gfx_activity,
        "host": hostname,
    }


def activity(amdsmi, picker, hip_id: int) -> str:
    """Current GFX busy percentage, or the word that means it was not readable.

    An unreadable activity query is not a quiet zero. Reporting unknown as 0 would publish an
    idleness nobody observed, which is the same substitution the verdict rule refuses.
    """
    _, handle = find_handle(amdsmi, hip_id)
    gfx, _ = picker.read_activity(amdsmi, handle)
    return ACTIVITY_UNAVAILABLE if gfx is None else str(gfx)


def _with_amdsmi(work):
    picker = load_picker()
    amdsmi = picker.import_amdsmi()
    amdsmi.amdsmi_init()
    try:
        return work(amdsmi, picker)
    finally:
        amdsmi.amdsmi_shut_down()


def cmd_identify(args: argparse.Namespace) -> int:
    info = _with_amdsmi(
        lambda amdsmi, picker: describe(
            amdsmi, picker, args.hip_id, socket.gethostname()
        )
    )
    print(json.dumps(info))
    return 0


def cmd_activity(args: argparse.Namespace) -> int:
    print(_with_amdsmi(lambda amdsmi, picker: activity(amdsmi, picker, args.hip_id)))
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    identify = sub.add_parser("identify", help="describe the claimed GPU as JSON")
    identify.add_argument("hip_id", type=int)
    identify.set_defaults(func=cmd_identify)

    busy = sub.add_parser("activity", help="print current GFX busy percent")
    busy.add_argument("hip_id", type=int)
    busy.set_defaults(func=cmd_activity)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
