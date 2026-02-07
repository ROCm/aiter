from typing import List, Dict, Tuple


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
) -> Tuple[str, str]:
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
    md_names: List[str], ck_dir: str, receipt: int = 200
) -> List[Dict]:
    variants: List[Dict] = []
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
            f"{ck_dir}/example/ck_tile/01_fmha/generate.py -d fwd --receipt {receipt} --filter {filter_pattern} --output_dir {{}}",
            f'{ck_dir}/example/ck_tile/01_fmha/generate.py -d fwd_splitkv --receipt {receipt} --filter " @ " --output_dir {{}}',
        ]
        variants.append(
            {"md_name": f"mha_varlen_fwd{suffix}", "blob_gen_cmd": blob_gen_cmd}
        )
    return variants
