# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import os
from torch.utils.file_baton import FileBaton


def mp_lock(
    lockPath: str,
    MainFunc: callable,
    FinalFunc: callable = None,
    WaitFunc: callable = None,
):
    """
    Using FileBaton for multiprocessing.
    """
    baton = FileBaton(lockPath)
    if baton.try_acquire():
        try:
            ret = MainFunc()
        finally:
            if FinalFunc is not None:
                FinalFunc()
            baton.release()
    else:
        baton.wait()
        if WaitFunc is not None:
            ret = WaitFunc()
        ret = None
    return ret
