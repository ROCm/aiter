"""Satisfy Proton's visible-device check on AMD GPUs.

Proton rejects HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES. It asks
for ROCR_VISIBLE_DEVICES instead.

This context manager removes both rejected variables for the block.
If ROCR_VISIBLE_DEVICES is unset, it copies the first non-empty value
from those two variables, HIP first. It restores the saved values
when the block ends.

This only satisfies Proton's environment check. Device visibility was
fixed when the HIP runtime initialized.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager

from aiter import logger

# Proton's _check_env rejects both names, including an empty string.
_BLOCKED_KEYS = ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
_ROCR_KEY = "ROCR_VISIBLE_DEVICES"

# Log the visible-device adjustment one time per process.
_WAS_LOGGED: bool = False


@contextmanager
def proton_visible_devices() -> Iterator[None]:
    saved = {name: os.environ.get(name) for name in (*_BLOCKED_KEYS, _ROCR_KEY)}
    # Proton rejects a set variable even when the value is "".
    present = [name for name in _BLOCKED_KEYS if saved[name] is not None]
    # An empty string is not a device mask. HIP wins when both are set.
    donor = next((name for name in _BLOCKED_KEYS if saved[name]), None)
    rocr_value = saved[_ROCR_KEY] or None

    log_lines: list[str] = []

    try:
        for name in present:
            os.environ.pop(name, None)

        unset = ", ".join(present)
        if donor is not None and rocr_value is None:
            os.environ[_ROCR_KEY] = saved[donor]
            log_lines.append(
                "ROCR_VISIBLE_DEVICES is unset. "
                f"Copying {donor}={saved[donor]} into ROCR_VISIBLE_DEVICES "
                f"for this block. Unsetting {unset}."
            )
        elif present:
            kept = (
                f" Keeping ROCR_VISIBLE_DEVICES={rocr_value}."
                if rocr_value is not None
                else ""
            )
            log_lines.append(f"Unsetting {unset} so Proton can start.{kept}")

        if log_lines:
            log_lines.append(
                "This only satisfies Proton's environment check. "
                "Device visibility was fixed at runtime init."
            )

        global _WAS_LOGGED
        if log_lines and not _WAS_LOGGED:
            _WAS_LOGGED = True
            log_lines.append("These variables are restored when this block returns.")
            logger.warning("%s", "\n".join(log_lines))

        yield

    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
