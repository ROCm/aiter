# Numerical status

The full kernel is experimental and not accuracy-qualified. Across 51 cases (17 inputs, positions 1024/1536/2047), one launch and two fused regions are bitwise identical and health checks pass. Worst relative RMS error versus native full layer is 0.35747343. In that case native selects experts [6,61,62,91,99,161], while fused selects [6,44,61,62,99,161]. Native FFN fed the fused attention result selects the same experts as fused and differs from the fused final output by only 4.77e-8 relative RMS. This isolates that large discrepancy to inherited attention differences affecting routing.

The FFN-only kernel was also tested on 32 changing random residuals. All routing IDs match native; maximum final relative RMS error is 3.11396e-5. The serving adapter was checked at positions 1024,1536,2047 with shifted cache rows; both layer output and updated KV row match the standalone kernel bitwise.

The previous C1 table called routing HERD because the environment enabled it. AITER defaults HERD to M>=16; this C1 path actually uses stock top-6. The new in-kernel routing preserves this choice, native BF16 transformed-score rounding, tie preference, sorted IDs and BF16 gate weights.
