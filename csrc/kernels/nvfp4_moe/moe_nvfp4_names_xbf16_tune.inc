    // Tune-only (not in the default build): the kernels of the measured M3.1 TP4 table
    // for the other main-path mode (bf16 X into stage 1, fused sort without the cast)
    // that the production list lacks. Build:
    // -DNVFP4_MOE_TUNE_NAMES='"moe_nvfp4_names_xbf16_tune.inc"'.
    KERNEL("s1_16x32_xreg_ksplit4_wbuf4x128_kdim6144_gs6_upfront_tight_occ2_xbf16"),
    KERNEL("s1_16x32_xreg_ksplit8_wbuf2x128_kdim6144_gs6_upfront_xbf16"),
    KERNEL("s1_16x32_xreg_ksplit8_wbuf4x128_kdim6144_gs6_upfront_tight_xbf16"),
    KERNEL("s1_16x384_xstream_waves12_wbuf3x128_xbuf2_kdim6144_gs6_upfront_tight_occ2"
           "_xcdremap_xbf16"),
    KERNEL("s1_16x384_xstream_waves12_wbuf3x128_xbuf2_kdim6144_gs6_upfront_xcdremap_xbf16"),
    KERNEL("s1_16x64_xreg_ksplit4_wbuf3x128_kdim6144_gs6_upfront_occ2_xbf16"),
    KERNEL("s1_16x64_xreg_ksplit8_wbuf2x128_kdim6144_gs6_upfront_occ4_xbf16"),
    KERNEL("s1_16x64_xreg_ksplit8_wbuf2x128_kdim6144_gs6_upfront_tight_xbf16"),
    KERNEL("s1_16x64_xreg_ksplit8_wbuf2x128_kdim6144_gs6_upfront_xbf16"),
    KERNEL("s1_32x256_xdma_waves8_wbuf2x128_xbuf3_gs6_occ3_xcdremap_xbf16"),
    KERNEL("s1_32x384_xdma_waves12_wbuf2x128_xbuf3_gs6_xbf16"),
    KERNEL("s1_32x384_xdma_waves12_wbuf2x128_xbuf3_gs6_xcdremap_xbf16"),
    KERNEL("s1_48x384_xstream_waves12_wbuf2x256_xbuf4_kdim6144_gs6_xbf16"),
    KERNEL("s1_48x384_xstream_waves4_xbuf4_kdim6144_gs6_tight_occ2_xbf16"),
    KERNEL("s2_16x128_xstream_waves4_wbuf2x128_xbuf5_kdim768_gs6_upfront"),
    KERNEL("s2_16x128_xstream_waves4_wbuf2x128_xbuf5_kdim768_gs6_upfront_occ5"),
    KERNEL("s2_16x32_xreg_wbuf4x128_kdim768_gs6_upfront_tight_occ4"),
    KERNEL("s2_16x64_xstream_waves2_xbuf2_kdim768_gs6_upfront_occ5"),
    KERNEL("s2_16x64_xstream_waves4_wbuf3x128_xbuf4_kdim768_gs6_upfront_occ6"),
    KERNEL("s2_16x768_xstream_waves12_xbuf3_kdim768_gs6_upfront_tight"),
    KERNEL("s2_32x256_xstream_waves4_wbuf2x128_xbuf2_kdim768_gs6_tight_occ3"),
    KERNEL("s2_32x256_xstream_waves4_wbuf2x128_xbuf2_kdim768_gs6_upfront_staged"),
    KERNEL("s2_32x256_xstream_waves4_wbuf3x128_xbuf2_kdim768_gs6_tight"),
    KERNEL("s2_48x256_xstream_waves4_wbuf3x128_xbuf2_kdim768_gs6"),
