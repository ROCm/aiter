def _ck_targets_flag_for_arches(gfxs: list[str]) -> str:
    gfxs = [gfx for gfx in gfxs if gfx != "cpu"]
    if not gfxs or all(gfx.startswith("gfx9") for gfx in gfxs):
        return ""
    return f" --targets {','.join(gfxs)}"


def _require_ck_batch_prefill_targets(gfxs: list[str]) -> None:
    unsupported = [gfx for gfx in gfxs if gfx != "cpu" and not gfx.startswith("gfx9")]
    if unsupported:
        raise RuntimeError(
            "CK batch-prefill code generation only supports gfx9 targets; "
            f"unsupported targets: {', '.join(unsupported)}"
        )


def _apply_ck_mha_prebuild_target_policy(build_args: list[dict], gfxs: list[str]):
    targets_flag = _ck_targets_flag_for_arches(gfxs)
    batch_prefill_supported = not any(
        gfx != "cpu" and not gfx.startswith("gfx9") for gfx in gfxs
    )
    target_modules = {"module_mha_fwd", "module_mha_varlen_fwd", "libmha_fwd"}
    adjusted = []

    for build in build_args:
        md_name = build["md_name"]
        if md_name == "module_mha_batch_prefill" and not batch_prefill_supported:
            continue
        if md_name not in target_modules:
            adjusted.append(build)
            continue

        commands = build.get("blob_gen_cmd", [])
        if isinstance(commands, str):
            commands = [commands]
        commands = list(commands)
        filtered = []
        for command in commands:
            parts = command.split()
            try:
                mode = parts[parts.index("-d") + 1]
            except (ValueError, IndexError):
                filtered.append(command)
                continue
            if mode == "batch_prefill" and not batch_prefill_supported:
                continue
            if mode in ("fwd", "fwd_splitkv") and targets_flag and "--targets" not in parts:
                command += targets_flag
            filtered.append(command)

        adjusted_build = dict(build)
        adjusted_build["blob_gen_cmd"] = filtered
        if md_name == "libmha_fwd" and not batch_prefill_supported:
            adjusted_build["srcs"] = [
                source
                for source in build.get("srcs", [])
                if not source.replace("\\", "/").rstrip("'\"").endswith(
                    "/cpp_itfs/mha_fwd_batch_prefill.cu"
                )
            ]
        adjusted.append(adjusted_build)

    return adjusted


def _ck_targets_flag() -> str:
    """Select every build architecture for CK FMHA code generation.

    CK's default covers gfx9 and gfx950. gfx10 targets have no FMHA factory in
    the pinned CK generator; other targets need explicit targets. GPU_ARCHS may
    name several architectures, which CK expects as a comma-separated list.
    """
    try:
        from chip_info import get_gfx_list

        gfxs = get_gfx_list()
    except Exception:  # noqa: BLE001
        return ""
    return _ck_targets_flag_for_arches(gfxs)


def compose_mha_fwd_variant_suffix_and_filter(
    dtype: str,
    logits_positive: bool,
    has_bias: bool,
    has_alibi: bool,
    use_mask: bool,
    return_lse: bool,
    dropout_zero: bool,
    skip_zero: bool,
    has_qscale: bool,
) -> tuple[str, str]:
    dtype_token = f"_{dtype}"
    logits_token = "_logits" if logits_positive else "_nlogits"
    if has_bias:
        bias_token = "_bias"
    elif has_alibi:
        bias_token = "_alibi"
    else:
        bias_token = "_nbias"
    mask_token = "_mask" if use_mask else "_nmask"
    lse_token = "_lse" if return_lse else "_nlse"
    dropout_token = "_ndropout" if dropout_zero else "_dropout"
    skip_token = "_nskip" if skip_zero else "_skip"
    qscale_token = "_nqscale" if has_qscale else "_pertensor"

    suffix = (
        dtype_token
        + logits_token
        + bias_token
        + mask_token
        + lse_token
        + dropout_token
        + skip_token
        + qscale_token
    )

    filt = (
        "*"
        + f"{dtype}*"
        + ("_logits*" if logits_positive else "_nlogits*")
        + ("_bias*" if has_bias else ("_alibi*" if has_alibi else "_nbias*"))
        + ("_mask*" if use_mask else "_nmask*")
        + ("_lse*" if return_lse else "_nlse*")
        + ("_ndropout*" if dropout_zero else "_dropout*")
        + ("_nskip*" if skip_zero else "_skip*")
        + ("_nqscale*" if has_qscale else "_pertensor*")
    )
    return suffix, filt


def _parse_mha_varlen_fwd_md_name(md_name: str):
    dtype = (
        "bf16" if "_bf16" in md_name else ("fp16" if "_fp16" in md_name else "fp8bf16")
    )
    logits_positive = "_logits" in md_name and "_nlogits" not in md_name
    has_bias = "_bias" in md_name
    has_alibi = "_alibi" in md_name
    use_mask = "_mask" in md_name and "_nmask" not in md_name
    return_lse = "_lse" in md_name and "_nlse" not in md_name
    dropout_zero = "_ndropout" in md_name
    skip_zero = "_nskip" in md_name
    has_qscale = "_nqscale" in md_name
    return (
        dtype,
        logits_positive,
        has_bias,
        has_alibi,
        use_mask,
        return_lse,
        dropout_zero,
        skip_zero,
        has_qscale,
    )


def get_mha_varlen_prebuild_variants_by_names(
    md_names: list[str], ck_dir: str, receipt: int = 200
) -> list[dict]:
    variants: list[dict] = []
    for md_name in md_names:
        (
            dtype,
            logits_positive,
            has_bias,
            has_alibi,
            use_mask,
            return_lse,
            dropout_zero,
            skip_zero,
            has_qscale,
        ) = _parse_mha_varlen_fwd_md_name(md_name)
        suffix, filter_pattern = compose_mha_fwd_variant_suffix_and_filter(
            dtype=dtype,
            logits_positive=logits_positive,
            has_bias=has_bias,
            has_alibi=has_alibi,
            use_mask=use_mask,
            return_lse=return_lse,
            dropout_zero=dropout_zero,
            skip_zero=skip_zero,
            has_qscale=has_qscale,
        )
        blob_gen_cmd = [
            f"{ck_dir}/example/ck_tile/01_fmha/generate.py -d fwd --receipt {receipt} --filter {filter_pattern} --output_dir {{}}{_ck_targets_flag()}",
            f'{ck_dir}/example/ck_tile/01_fmha/generate.py -d fwd_splitkv --receipt {receipt} --filter " @ " --output_dir {{}}{_ck_targets_flag()}',
        ]
        variants.append(
            {"md_name": f"mha_varlen_fwd{suffix}", "blob_gen_cmd": blob_gen_cmd}
        )
    return variants
