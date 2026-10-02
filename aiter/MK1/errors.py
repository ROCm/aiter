# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Error taxonomy for the optional MK1 persistent decoder."""


class MK1Error(RuntimeError):
    """Base class for all MK1 integration failures."""


class UnsupportedConfigurationError(MK1Error):
    """The requested hardware, model, or runtime mode is unsupported."""


class NativeBinaryError(MK1Error):
    """The packaged native binary is absent, corrupt, or incompatible."""


class CheckpointError(MK1Error):
    """A persistent checkpoint failed validation or loading."""


class PrelaunchError(MK1Error):
    """A request was rejected before the native backend could modify state."""


class BackendLaunchError(MK1Error):
    """The native backend failed after a launch may have modified state."""
