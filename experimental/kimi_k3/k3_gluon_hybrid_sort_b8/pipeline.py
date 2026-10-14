"""Four-launch B8 contract; this module has no GPU/runtime imports.

The caller owns buffers for the lifetime of graph replay and uses one stream.
The expert callback must enqueue exactly the selected AITER MoE1 and MoE2.
"""

METADATA_KEYS = (
    "SortIDs",
    "SortWeights",
    "SortExperts",
    "SortCounts",
    "SortOut",
    "SortM",
    "SortReverse",
)
MOE1 = "flydsl_mxmoe_g1_a4w4_16x256x256_f16in_nt_situv2_xcd4"
MOE2 = "flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x128_atomic_nt_sbm16"


def require_expert_config(config):
    """Reject dispatch changes that invalidate the fused BM16 metadata ABI."""
    first = getattr(config.stage1, "keywords", {}) or {}
    second = getattr(config.stage2, "keywords", {}) or {}
    if (
        first.get("kernelName1") != MOE1
        or second.get("kernelName2") != MOE2
        or config.block_m != 16
        or config.run_1stage
        or config.prequant
    ):
        raise ValueError(
            "B8 hybrid requires the recorded BM16 inline-quant MoE1 and atomic "
            "MoE2 configuration; revalidate metadata and launch count before "
            "using a different AITER tuning row."
        )
    return config


def stage_options(options, metadata, grid):
    """Build matching compile/launch specializations without touching buffers."""
    if grid != 512 or options.get("WORLD") != 8 or options.get("num_warps") != 8:
        raise ValueError("Fused B8 metadata requires TP8, 512 CTAs, and 8 warps")
    if options.get("START", 0) not in (0, 2):
        raise ValueError("Front must start at projection (0) or MoE (2)")
    missing = set(METADATA_KEYS).difference(metadata)
    if missing:
        raise ValueError(f"Missing fused metadata buffers: {sorted(missing)}")
    common = dict(options)
    common.update({key: metadata[key] for key in METADATA_KEYS})
    front = dict(common, END=2, BEGIN=True, ExternalRouted=None)
    tail = dict(common, START=4, END=4, BEGIN=False, ExternalRouted=metadata["SortOut"])
    return front, tail


def launch_hybrid(kernel, call, options, *, grid, state, weights, metadata, experts):
    """Front+metadata -> MoE1 -> MoE2 -> tail, with no output copy or sort."""
    if state["Latent"].shape != (8, 3584) or state["IDs"].shape != (8, 16):
        raise ValueError("Fused metadata supports B8, top-16, latent width 3584")
    if state["Weights"].shape != (8, 16):
        raise ValueError("Routing weights must have shape (8, 16)")
    front, tail = stage_options(options, metadata, grid)
    kernel[(grid,)](*call, **front)
    routed = experts(state["Latent"], weights, state["IDs"], state["Weights"], metadata)
    # Consume the actual returned allocation; never enqueue a copy into scratch.
    tail["ExternalRouted"] = routed
    kernel[(grid,)](*call, **tail)
    return routed
