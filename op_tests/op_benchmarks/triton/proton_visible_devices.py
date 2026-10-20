"""If you set HIP_VISIBLE_DEVICES, Proton does not start on AMD GPUs.
Proton shows this error:

> ValueError: Proton does not work when the environment variable
> HIP_VISIBLE_DEVICES is set on AMD GPUs. Please unset it and use
> ROCR_VISIBLE_DEVICES instead

The context manager unsets HIP_VISIBLE_DEVICES for the block.
If ROCR_VISIBLE_DEVICES is unset, the context manager sets ROCR_VISIBLE_DEVICES
to the HIP_VISIBLE_DEVICES value.
If the two values are different, the context manager keeps ROCR_VISIBLE_DEVICES.
The context manager puts the two variables back when the block ends.
"""

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager

_told = False


@contextmanager
def proton_visible_devices() -> Iterator[None]:
    hip_key = "HIP_VISIBLE_DEVICES"
    rocr_key = "ROCR_VISIBLE_DEVICES"

    saved = {name: os.environ.get(name) for name in (hip_key, rocr_key)}

    # An empty string is the same as an unset variable.
    # Do not change an empty variable.
    hip_value = saved[hip_key] or None
    rocr_value = saved[rocr_key] or None

    log_lines: list[str] = []

    try:
        if hip_value:
            # Unset HIP_VISIBLE_DEVICES.
            # Proton can start.
            os.environ.pop(hip_key, None)

            if rocr_value is None:
                # ROCR_VISIBLE_DEVICES is unset.
                # Set ROCR_VISIBLE_DEVICES to the HIP_VISIBLE_DEVICES value.
                os.environ[rocr_key] = hip_value
                log_lines.append(
                    f"HIP_VISIBLE_DEVICES={hip_value} is set and "
                    "ROCR_VISIBLE_DEVICES is unset. "
                    "Unsetting HIP_VISIBLE_DEVICES and setting "
                    f"ROCR_VISIBLE_DEVICES={hip_value}."
                )
            elif hip_value == rocr_value:
                # The two values are the same.
                # Unset HIP_VISIBLE_DEVICES.
                log_lines.append(
                    "HIP_VISIBLE_DEVICES and ROCR_VISIBLE_DEVICES are both "
                    f"{hip_value}. Unsetting HIP_VISIBLE_DEVICES."
                )
            else:
                # The two values are different.
                # Keep ROCR_VISIBLE_DEVICES.
                log_lines.append(
                    f"HIP_VISIBLE_DEVICES={hip_value} and ROCR_VISIBLE_DEVICES={rocr_value} "
                    "differ. Unsetting HIP_VISIBLE_DEVICES and keeping "
                    f"ROCR_VISIBLE_DEVICES={rocr_value}. "
                    "Visible GPUs follow ROCR_VISIBLE_DEVICES."
                )

        # Print the message one time.
        global _told
        if log_lines and not _told:
            _told = True
            log_lines.append("These variables are restored when this block returns.")
            print("\n".join(log_lines), file=sys.__stderr__ or sys.stderr)

        yield

    finally:
        # Put each variable back to its saved value.
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
