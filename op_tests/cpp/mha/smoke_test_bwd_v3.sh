#!/bin/sh
# SPDX-License-Identifier: MIT
# Copyright (C) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Backwards-compatible shim. The real driver lives in test_bwd_v3_coverage.sh.
# Runs the same three suites this file historically ran.
exec "$(dirname "$0")/test_bwd_v3_coverage.sh" batch group swa
