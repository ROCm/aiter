# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

import os
from enum import IntEnum
from typing import Optional, Tuple
from chip_info import get_gfx


class BufferCoherenceType(IntEnum):
    DEFAULT = 0

    WAVE_NT = 1
    GROUP_NT = 2
    DEVICE_NT = 3
    SYSTEM_NT = 4

    GLC = 5
    SLC = 6
    GLC_SLC = 7

    HIGH_PRIORITY = 8
    LAST_USE = 9


class Gfx90aCoherence:
    DEFAULT = 0
    GLC = 1      # glc
    SLC = 2      # slc
    GLC_SLC = 3  # glc + slc

    DEVICE_NT = GLC
    SYSTEM_NT = SLC
    WAVE_NT = 0
    GROUP_NT = 0


class Gfx942Coherence:
    WAVE = 0
    GROUP = 1
    DEVICE = 16
    SYSTEM = 17

    # Temporal hints
    NT0 = 0  # temporal reuse expected
    NT1 = 2  # non-temporal

    WAVE_NT0 = NT0 | WAVE      # 0
    WAVE_NT1 = NT1 | WAVE      # 2
    GROUP_NT0 = NT0 | GROUP    # 1
    GROUP_NT1 = NT1 | GROUP    # 3
    DEVICE_NT0 = NT0 | DEVICE  # 16
    DEVICE_NT1 = NT1 | DEVICE  # 18
    SYSTEM_NT0 = NT0 | SYSTEM  # 17
    SYSTEM_NT1 = NT1 | SYSTEM  # 19

    DEFAULT = WAVE_NT0
    WAVE_NT = WAVE_NT1
    GROUP_NT = GROUP_NT1
    DEVICE_NT = DEVICE_NT1
    SYSTEM_NT = SYSTEM_NT1
    GLC = DEVICE_NT1
    SLC = SYSTEM_NT1
    GLC_SLC = DEVICE_NT1 | SYSTEM_NT1


class Gfx12Coherence:
    RT = 0     # regular temporal
    NT = 1     # non temporal
    HT = 2     # high priority temporal
    LU = 3     # last use (load op)
    WB = 3     # write-back (store op)
    NT_RT = 4
    RT_NT = 5
    NT_HT = 6
    NT_WB = 7

    # Scope
    CU = 0
    SE = 8
    DEVICE = 16
    SYSTEM = 24

    CU_NT = NT | CU          # 1
    SE_NT = NT | SE          # 9
    DEVICE_NT = NT | DEVICE  # 17
    SYSTEM_NT = NT | SYSTEM  # 25
    CU_HT = HT | CU          # 2
    DEVICE_HT = HT | DEVICE  # 18

    DEFAULT = RT
    WAVE_NT = CU_NT
    GROUP_NT = CU_NT
    GLC = DEVICE_NT
    SLC = SYSTEM_NT
    GLC_SLC = DEVICE_NT | SYSTEM_NT


class BufferCoherenceMapper:
    @staticmethod
    def map_to_gfx90a(coherence_type: BufferCoherenceType) -> int:
        mapping = {
            BufferCoherenceType.DEFAULT: Gfx90aCoherence.DEFAULT,
            BufferCoherenceType.WAVE_NT: Gfx90aCoherence.WAVE_NT,
            BufferCoherenceType.GROUP_NT: Gfx90aCoherence.GROUP_NT,
            BufferCoherenceType.DEVICE_NT: Gfx90aCoherence.DEVICE_NT,
            BufferCoherenceType.SYSTEM_NT: Gfx90aCoherence.SYSTEM_NT,
            BufferCoherenceType.GLC: Gfx90aCoherence.GLC,
            BufferCoherenceType.SLC: Gfx90aCoherence.SLC,
            BufferCoherenceType.GLC_SLC: Gfx90aCoherence.GLC_SLC,
            BufferCoherenceType.HIGH_PRIORITY: Gfx90aCoherence.DEFAULT,
            BufferCoherenceType.LAST_USE: Gfx90aCoherence.DEFAULT,
        }
        return mapping.get(coherence_type, Gfx90aCoherence.DEFAULT)

    @staticmethod
    def map_to_gfx942(coherence_type: BufferCoherenceType) -> int:
        mapping = {
            BufferCoherenceType.DEFAULT: Gfx942Coherence.DEFAULT,
            BufferCoherenceType.WAVE_NT: Gfx942Coherence.WAVE_NT,
            BufferCoherenceType.GROUP_NT: Gfx942Coherence.GROUP_NT,
            BufferCoherenceType.DEVICE_NT: Gfx942Coherence.DEVICE_NT,
            BufferCoherenceType.SYSTEM_NT: Gfx942Coherence.SYSTEM_NT,
            BufferCoherenceType.GLC: Gfx942Coherence.GLC,
            BufferCoherenceType.SLC: Gfx942Coherence.SLC,
            BufferCoherenceType.GLC_SLC: Gfx942Coherence.GLC_SLC,
            BufferCoherenceType.HIGH_PRIORITY: Gfx942Coherence.GROUP_NT0,  # temporal
            BufferCoherenceType.LAST_USE: Gfx942Coherence.GROUP_NT,
        }
        return mapping.get(coherence_type, Gfx942Coherence.DEFAULT)

    @staticmethod
    def map_to_gfx12(coherence_type: BufferCoherenceType) -> int:
        mapping = {
            BufferCoherenceType.DEFAULT: Gfx12Coherence.DEFAULT,
            BufferCoherenceType.WAVE_NT: Gfx12Coherence.WAVE_NT,
            BufferCoherenceType.GROUP_NT: Gfx12Coherence.GROUP_NT,
            BufferCoherenceType.DEVICE_NT: Gfx12Coherence.DEVICE_NT,
            BufferCoherenceType.SYSTEM_NT: Gfx12Coherence.SYSTEM_NT,
            BufferCoherenceType.GLC: Gfx12Coherence.GLC,
            BufferCoherenceType.SLC: Gfx12Coherence.SLC,
            BufferCoherenceType.GLC_SLC: Gfx12Coherence.GLC_SLC,
            BufferCoherenceType.HIGH_PRIORITY: Gfx12Coherence.CU_HT,
            BufferCoherenceType.LAST_USE: Gfx12Coherence.LU,
        }
        return mapping.get(coherence_type, Gfx12Coherence.DEFAULT)

    @staticmethod
    def map_to_arch_value(coherence_type: BufferCoherenceType) -> int:
        arch = get_gfx()

        if arch == "gfx90a":
            return BufferCoherenceMapper.map_to_gfx90a(coherence_type)
        elif arch in ["gfx950", "gfx942"]:
            return BufferCoherenceMapper.map_to_gfx942(coherence_type)
        elif arch in ["gfx1200", "gfx1201"]:
            return BufferCoherenceMapper.map_to_gfx12(coherence_type)
        return BufferCoherenceType.DEFAULT
