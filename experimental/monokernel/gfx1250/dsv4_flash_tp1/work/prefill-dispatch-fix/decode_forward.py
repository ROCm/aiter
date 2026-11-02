"""Decode-only model entry; caller owns mode/shape eligibility outside compile.

Eager orchestration is recorded by ATOM's full GPU graph capture. Prefill never
calls this function, including single-token and chunked prefills.
"""
import aiter

def decode_forward(model,input_ids,positions,md):
    h=model.embed(input_ids)
    h=aiter.mhc_res_repeat(h,model.hc_mult,True)
    for layer in model.layers:
        adapter=layer._experimental_full_layer
        h=adapter.execute(h,positions,md)
    # Preserve the existing final residual contract and native head path.
    last=model.layers[-1];adapter=last._experimental_full_layer;m=input_ids.numel()
    return last.hc_post(adapter.zero[:m],h,adapter.post[:m],adapter.comb[:m],res_preshuffle=True)
