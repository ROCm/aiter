#!/bin/bash
# dmf.sh: number of GPU fault / driver-eviction lines in dmesg (compare before / after a run)
dmesg 2>/dev/null | grep -ciE "page fault|PROTECTION_FAULT|MES.*failed|Failed to evict|sq_intr: error|ring .* timeout|GPU reset"
