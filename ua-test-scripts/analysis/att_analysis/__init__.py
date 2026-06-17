"""Headless rocprofv3 ATT trace analysis for the FA4 unified-attention pipeline.

Modules:
    model      -- load ui_output_* dirs (filenames/code/per-wave timelines)
    phases     -- classify instructions into MATRIX/SOFTMAX/... (source or mnemonic)
    window     -- pick a few steady-state loop iterations to focus on
    timeline   -- render the two-warpgroup overlap PNG
"""
