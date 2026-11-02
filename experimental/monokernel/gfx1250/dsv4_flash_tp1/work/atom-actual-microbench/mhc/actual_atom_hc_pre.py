def hc_pre(self, residual: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, norm_weight: torch.Tensor | None=None, norm_eps: float=1e-06, res_preshuffle: bool=False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduce mHC residual `[num_tokens, hc, dim]` to sub-layer input `[num_tokens, dim]`.

        Prefers the fused aiter `mhc_pre` kernel (single ROCm op for RMSNorm +
        hc-fn linear + Sinkhorn projection + weighted reduction). Falls back to
        the torch `hc_split_sinkhorn` reference when the aiter kernel is
        unavailable or `dim` doesn't satisfy the `% 256/512 == 0` constraint.

        Returns:
          y:    [num_tokens, dim]      sub-layer input
          post: [num_tokens, hc]       post-gate weights for hc_post
          comb: [num_tokens, hc, hc]   combination matrix for hc_post
        """
    if self._mhc_pre is not None:
        pack_kw = {'w_preshuffle_bf16': 1} if self.enable_hc_fn_pack_bf16 else {}
        if res_preshuffle:
            pack_kw['res_preshuffle'] = True
        post, comb, y = self._mhc_pre(residual, hc_fn, hc_scale, hc_base, float(self.norm_eps), float(self.hc_eps), float(self.hc_eps), self.HC_POST_MULT, int(self.hc_sinkhorn_iters), norm_weight, norm_eps, **pack_kw)
        return (y, post.squeeze(-1), comb)
    dtype = residual.dtype
    x_flat = residual.flatten(-2)
    x_normed = _rmsnorm_nw(x_flat, self.norm_eps, x_flat.shape[-1])
    mixes = F.linear(x_normed.float(), hc_fn)
    pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)
    y = torch.sum(pre.unsqueeze(-1) * residual, dim=-2)
    if norm_weight is not None:
        y = F.rms_norm(y.float(), (y.shape[-1],), norm_weight.float(), norm_eps).to(dtype)
    return (y.to(dtype), post, comb)
