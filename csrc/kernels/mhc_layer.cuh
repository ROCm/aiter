#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "../include/mhc_types.h"
#include "fused_rmsnorm_matmul.cuh"
#include "rmsnorm.cuh"
#include "sinkhorn_knopp.cuh"
#include "stream_ops.cuh"
#include "../include/utils.cuh"

namespace mhc {

struct MHCLayerConfig {
    int batch_size;
    int hidden_dim;
    int expansion_rate;
    int sinkhorn_iters;
    float eps;
    float alpha_init;
    bool use_pdl;
    bool use_dynamic_h;

    MHCLayerConfig()
        : batch_size(0), hidden_dim(0), expansion_rate(4), sinkhorn_iters(20), eps(1e-5f),
          alpha_init(0.01f), use_pdl(true), use_dynamic_h(true) {}
};

struct MHCLayerWeights {
    __hip_bfloat16* rmsnorm_weight;

    __hip_bfloat16* phi_combined;
    __hip_bfloat16* phi_pre;
    __hip_bfloat16* phi_post;
    __hip_bfloat16* phi_res;

    float* b_pre;
    float* b_post;
    float* b_res;

    float alpha_pre;
    float alpha_post;
    float alpha_res;

    bool initialized;
    bool dynamic_h;
    int hidden_dim;
    int expansion_rate;

    MHCLayerWeights() : initialized(false), dynamic_h(true), phi_combined(nullptr) {}

    void init(int C, int n, bool use_dynamic = true, float alpha_init = 0.01f) {
        hidden_dim = C;
        expansion_rate = n;
        dynamic_h = use_dynamic;

        HIP_CALL(hipMalloc(&rmsnorm_weight, C * sizeof(__hip_bfloat16)));

        if (dynamic_h) {
            int nC = n * C;
            int total_H_dim = n + n + n * n;
            HIP_CALL(hipMalloc(&phi_combined, total_H_dim * nC * sizeof(__hip_bfloat16)));
            phi_pre = phi_combined;
            phi_post = phi_combined + n * nC;
            phi_res = phi_combined + 2 * n * nC;
        } else {
            phi_combined = nullptr;
            phi_pre = nullptr;
            phi_post = nullptr;
            phi_res = nullptr;
        }

        HIP_CALL(hipMalloc(&b_pre, n * sizeof(float)));
        HIP_CALL(hipMalloc(&b_post, n * sizeof(float)));
        HIP_CALL(hipMalloc(&b_res, n * n * sizeof(float)));

        alpha_pre = alpha_init;
        alpha_post = alpha_init;
        alpha_res = alpha_init;

        initialized = true;
    }

    void destroy() {
        if (!initialized)
            return;

        hipFree(rmsnorm_weight);
        if (dynamic_h) {
            hipFree(phi_combined);
        }
        hipFree(b_pre);
        hipFree(b_post);
        hipFree(b_res);

        initialized = false;
    }
};

struct MHCLayerBuffers {
    float* x_expanded;
    __hip_bfloat16* x_aggregated_bf16;
    float* x_aggregated_f32;
    float* rms_values;
    __hip_bfloat16* layer_out_bf16;
    float* layer_out_f32;
    float* y_distributed;
    float* sinkhorn_M;
    float* x_mixed;
    float* output;

    __hip_bfloat16* x_flat_bf16;
    float* rms_dynamic;
    float* H_proj_raw;

    float* H_pre_activated;
    float* H_post_activated;
    float* H_res_tilde;

    FusedRMSNormMatmul fused_rms_matmul;

    bool initialized;
    bool dynamic_h;
    int batch_size;
    int hidden_dim;
    int expansion_rate;

    MHCLayerBuffers() : initialized(false), x_mixed(nullptr), dynamic_h(true) {}

    void init(int B, int C, int n, bool needs_x_mixed = false, bool use_dynamic_h = true) {
        batch_size = B;
        hidden_dim = C;
        expansion_rate = n;
        dynamic_h = use_dynamic_h;

        HIP_CALL(hipMalloc(&x_expanded, B * n * C * sizeof(float)));
        HIP_CALL(hipMalloc(&x_aggregated_bf16, B * C * sizeof(__hip_bfloat16)));
        HIP_CALL(hipMalloc(&x_aggregated_f32, B * C * sizeof(float)));
        HIP_CALL(hipMalloc(&rms_values, B * sizeof(float)));
        HIP_CALL(hipMalloc(&layer_out_bf16, B * C * sizeof(__hip_bfloat16)));
        HIP_CALL(hipMalloc(&layer_out_f32, B * C * sizeof(float)));
        HIP_CALL(hipMalloc(&y_distributed, B * n * C * sizeof(float)));
        if (needs_x_mixed) {
            HIP_CALL(hipMalloc(&x_mixed, B * n * C * sizeof(float)));
        }
        HIP_CALL(hipMalloc(&output, B * n * C * sizeof(float)));

        if (dynamic_h) {
            int nC = n * C;
            int total_H_dim = n + n + n * n;
            HIP_CALL(hipMalloc(&x_flat_bf16, B * nC * sizeof(__hip_bfloat16)));
            HIP_CALL(hipMalloc(&rms_dynamic, B * sizeof(float)));
            HIP_CALL(hipMalloc(&H_proj_raw, B * total_H_dim * sizeof(float)));
            fused_rms_matmul.init(B, total_H_dim, nC);
            HIP_CALL(hipMalloc(&sinkhorn_M, B * n * n * sizeof(float)));
            HIP_CALL(hipMalloc(&H_pre_activated, B * n * sizeof(float)));
            HIP_CALL(hipMalloc(&H_post_activated, B * n * sizeof(float)));
            HIP_CALL(hipMalloc(&H_res_tilde, B * n * n * sizeof(float)));
        } else {
            x_flat_bf16 = nullptr;
            rms_dynamic = nullptr;
            H_proj_raw = nullptr;
            HIP_CALL(hipMalloc(&sinkhorn_M, n * n * sizeof(float)));
            HIP_CALL(hipMalloc(&H_pre_activated, n * sizeof(float)));
            HIP_CALL(hipMalloc(&H_post_activated, n * sizeof(float)));
            HIP_CALL(hipMalloc(&H_res_tilde, n * n * sizeof(float)));
        }

        initialized = true;
    }

    void destroy() {
        if (!initialized)
            return;

        hipFree(x_expanded);
        hipFree(x_aggregated_bf16);
        hipFree(x_aggregated_f32);
        hipFree(rms_values);
        hipFree(layer_out_bf16);
        hipFree(layer_out_f32);
        hipFree(y_distributed);
        hipFree(sinkhorn_M);
        if (x_mixed)
            hipFree(x_mixed);
        hipFree(output);

        if (dynamic_h) {
            hipFree(x_flat_bf16);
            hipFree(rms_dynamic);
            hipFree(H_proj_raw);
            fused_rms_matmul.destroy();
        }

        hipFree(H_pre_activated);
        hipFree(H_post_activated);
        hipFree(H_res_tilde);

        initialized = false;
    }
};

struct MHCLayerGradients {
    float* d_x_expanded;
    float* d_H_pre;
    float* d_rmsnorm_weight;
    float* d_H_post;
    float* d_H_res;
    float* d_x_aggregated;
    float* d_layer_out;
    float* d_y_distributed;
    float* d_x_mixed;
    float* d_M;

    float* d_H_pre_activated;
    float* d_H_post_activated;
    float* d_H_res_exp;

    float* workspace_dH;
    float* workspace_dM;
    int workspace_num_blocks;

    bool initialized;

    MHCLayerGradients() : initialized(false), workspace_dH(nullptr), workspace_dM(nullptr) {}

    void init(int B, int C, int n) {
        HIP_CALL(hipMalloc(&d_x_expanded, B * n * C * sizeof(float)));
        HIP_CALL(hipMalloc(&d_H_pre, n * sizeof(float)));
        HIP_CALL(hipMalloc(&d_rmsnorm_weight, C * sizeof(float)));
        HIP_CALL(hipMalloc(&d_H_post, n * sizeof(float)));
        HIP_CALL(hipMalloc(&d_H_res, n * n * sizeof(float)));
        HIP_CALL(hipMalloc(&d_x_aggregated, B * C * sizeof(float)));
        HIP_CALL(hipMalloc(&d_layer_out, B * C * sizeof(float)));
        HIP_CALL(hipMalloc(&d_y_distributed, B * n * C * sizeof(float)));
        HIP_CALL(hipMalloc(&d_x_mixed, B * n * C * sizeof(float)));
        HIP_CALL(hipMalloc(&d_M, n * n * sizeof(float)));

        HIP_CALL(hipMalloc(&d_H_pre_activated, n * sizeof(float)));
        HIP_CALL(hipMalloc(&d_H_post_activated, n * sizeof(float)));
        HIP_CALL(hipMalloc(&d_H_res_exp, n * n * sizeof(float)));

        constexpr int BLOCK_SIZE = 256;
        workspace_num_blocks = min(128, (B * C + BLOCK_SIZE - 1) / BLOCK_SIZE);
        HIP_CALL(hipMalloc(&workspace_dH, workspace_num_blocks * n * sizeof(float)));
        HIP_CALL(hipMalloc(&workspace_dM, workspace_num_blocks * n * n * sizeof(float)));

        initialized = true;
    }

    void destroy() {
        if (!initialized)
            return;

        hipFree(d_x_expanded);
        hipFree(d_H_pre);
        hipFree(d_rmsnorm_weight);
        hipFree(d_H_post);
        hipFree(d_H_res);
        hipFree(d_x_aggregated);
        hipFree(d_layer_out);
        hipFree(d_y_distributed);
        hipFree(d_x_mixed);
        hipFree(d_M);

        hipFree(d_H_pre_activated);
        hipFree(d_H_post_activated);
        hipFree(d_H_res_exp);

        hipFree(workspace_dH);
        hipFree(workspace_dM);

        initialized = false;
    }

    void zero_weight_grads(int C, int n, hipStream_t stream = nullptr) {
        HIP_CALL(hipMemsetAsync(d_H_pre, 0, n * sizeof(float), stream));
        HIP_CALL(hipMemsetAsync(d_rmsnorm_weight, 0, C * sizeof(float), stream));
        HIP_CALL(hipMemsetAsync(d_H_post, 0, n * sizeof(float), stream));
        HIP_CALL(hipMemsetAsync(d_H_res, 0, n * n * sizeof(float), stream));
    }
};

template<int BLOCK_SIZE>
__global__ void sigmoid_kernel(float* __restrict__ out, const float* __restrict__ inp, int size) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < size) {
        float x = inp[idx];
        out[idx] = fast_sigmoid(-x);
    }
}

template<int BLOCK_SIZE>
__global__ void sigmoid_scale_kernel(float* __restrict__ out, const float* __restrict__ inp,
                                     float scale, int size) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < size) {
        float x = inp[idx];
        out[idx] = scale / (1.0f + expf(-x));
    }
}

template<int BLOCK_SIZE>
__global__ void exp_kernel(float* __restrict__ out, const float* __restrict__ inp, int size) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < size) {
        out[idx] = fast_exp(inp[idx]);
    }
}

template<int BLOCK_SIZE>
__global__ void sigmoid_backward_kernel(float* __restrict__ d_inp, const float* __restrict__ d_out,
                                        const float* __restrict__ activated, int size) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < size) {
        float s = activated[idx];
        d_inp[idx] = d_out[idx] * s * (1.0f - s);
    }
}

template<int BLOCK_SIZE>
__global__ void
sigmoid_scale_backward_kernel(float* __restrict__ d_inp, const float* __restrict__ d_out,
                              const float* __restrict__ activated, float scale, int size) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < size) {
        float s = activated[idx] / scale;
        d_inp[idx] = d_out[idx] * scale * s * (1.0f - s);
    }
}

template<int BLOCK_SIZE>
__global__ void exp_backward_kernel(float* __restrict__ d_inp, const float* __restrict__ d_out,
                                    const float* __restrict__ exp_val, int size) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < size) {
        d_inp[idx] = d_out[idx] * exp_val[idx];
    }
}

inline void apply_exp(float* out, const float* inp, int size, hipStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;
    int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    exp_kernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE, 0, stream>>>(out, inp, size);
}

inline void sigmoid_backward(float* d_inp, const float* d_out, const float* activated, int size,
                             hipStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;
    int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sigmoid_backward_kernel<BLOCK_SIZE>
        <<<num_blocks, BLOCK_SIZE, 0, stream>>>(d_inp, d_out, activated, size);
}

inline void sigmoid_scale_backward(float* d_inp, const float* d_out, const float* activated,
                                   float scale, int size, hipStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;
    int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sigmoid_scale_backward_kernel<BLOCK_SIZE>
        <<<num_blocks, BLOCK_SIZE, 0, stream>>>(d_inp, d_out, activated, scale, size);
}

inline void exp_backward(float* d_inp, const float* d_out, const float* exp_val, int size,
                         hipStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;
    int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    exp_backward_kernel<BLOCK_SIZE>
        <<<num_blocks, BLOCK_SIZE, 0, stream>>>(d_inp, d_out, exp_val, size);
}

template<int BLOCK_SIZE>
__global__ void apply_dynamic_h_activations_kernel(
    float* __restrict__ H_pre_out, float* __restrict__ H_post_out, float* __restrict__ H_res_out,
    const float* __restrict__ H_proj_raw, const float* __restrict__ b_pre,
    const float* __restrict__ b_post, const float* __restrict__ b_res, float alpha_pre,
    float alpha_post, float alpha_res, int B, int n) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int n2 = n * n;
    int total_H_dim = n + n + n2;

    for (int b = idx; b < B; b += gridDim.x * BLOCK_SIZE) {
        const float* proj = H_proj_raw + b * total_H_dim;
        float* h_pre = H_pre_out + b * n;
        float* h_post = H_post_out + b * n;
        float* h_res = H_res_out + b * n2;

        for (int i = 0; i < n; i++) {
            float val = alpha_pre * proj[i] + b_pre[i];
            h_pre[i] = fast_sigmoid(val);
        }

        for (int i = 0; i < n; i++) {
            float val = alpha_post * proj[n + i] + b_post[i];
            h_post[i] = 2.0f * fast_sigmoid(val);
        }

        for (int i = 0; i < n2; i++) {
            float val = alpha_res * proj[2 * n + i] + b_res[i];
            h_res[i] = val;
        }
    }
}

inline void apply_dynamic_h_activations(float* H_pre_out, float* H_post_out, float* H_res_out,
                                        const float* H_proj_raw, const float* b_pre,
                                        const float* b_post, const float* b_res, float alpha_pre,
                                        float alpha_post, float alpha_res, int B, int n,
                                        hipStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;
    int num_blocks = (B + BLOCK_SIZE - 1) / BLOCK_SIZE;
    apply_dynamic_h_activations_kernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        H_pre_out, H_post_out, H_res_out, H_proj_raw, b_pre, b_post, b_res, alpha_pre, alpha_post,
        alpha_res, B, n);
}

struct MHCLayer {
    MHCLayerConfig config;
    MHCLayerWeights weights;
    MHCLayerBuffers buffers;
    MHCLayerGradients grads;

    StreamMixTC stream_mix_tc;
    bool use_tc_mix;
    bool backward_enabled;
    bool use_pipelining;

    hipStream_t stream;
    hipStream_t sinkhorn_stream;
    hipEvent_t sinkhorn_done;
    bool owns_stream;
    bool initialized;

    MHCLayer()
        : stream(nullptr), sinkhorn_stream(nullptr), sinkhorn_done(nullptr), owns_stream(false),
          initialized(false), use_tc_mix(false), backward_enabled(false), use_pipelining(true) {}

    void init(const MHCLayerConfig& cfg, hipStream_t s = nullptr, bool enable_backward = false,
              bool enable_pipelining = true) {
        config = cfg;
        int B = cfg.batch_size;
        int C = cfg.hidden_dim;
        int n = cfg.expansion_rate;

        use_tc_mix = (n >= STREAM_MIX_TC_THRESHOLD);
        backward_enabled = enable_backward;
        // Only use pipelining for large expansion rate (n >= 16) where Sinkhorn-Knopp iteration
        // takes long enough to benefit from overlap
        use_pipelining = enable_pipelining && (n >= 16);

        weights.init(C, n, cfg.use_dynamic_h, cfg.alpha_init);
        buffers.init(B, C, n, use_tc_mix || backward_enabled, cfg.use_dynamic_h);

        if (use_tc_mix) {
            stream_mix_tc.init(B, n, C);
        }

        if (backward_enabled) {
            grads.init(B, C, n);
        }

        if (s == nullptr) {
            HIP_CALL(hipStreamCreate(&stream));
            owns_stream = true;
        } else {
            stream = s;
            owns_stream = false;
        }

        if (use_pipelining) {
            HIP_CALL(hipStreamCreate(&sinkhorn_stream));
            HIP_CALL(hipEventCreate(&sinkhorn_done));
        }

        initialized = true;
    }

    void destroy() {
        if (!initialized)
            return;

        weights.destroy();
        buffers.destroy();
        if (backward_enabled) {
            grads.destroy();
        }

        if (use_tc_mix) {
            stream_mix_tc.destroy();
        }

        if (use_pipelining) {
            if (sinkhorn_stream) {
                hipStreamDestroy(sinkhorn_stream);
                sinkhorn_stream = nullptr;
            }
            if (sinkhorn_done) {
                hipEventDestroy(sinkhorn_done);
                sinkhorn_done = nullptr;
            }
        }

        if (owns_stream && stream) {
            hipStreamDestroy(stream);
            stream = nullptr;
        }

        initialized = false;
    }

    void set_weights_dynamic(const __hip_bfloat16* h_rmsnorm_weight, const __hip_bfloat16* h_phi_pre,
                             const __hip_bfloat16* h_phi_post, const __hip_bfloat16* h_phi_res,
                             const float* h_b_pre, const float* h_b_post, const float* h_b_res,
                             float alpha_pre, float alpha_post, float alpha_res) {
        int C = config.hidden_dim;
        int n = config.expansion_rate;
        int nC = n * C;

        HIP_CALL(hipMemcpyAsync(weights.rmsnorm_weight, h_rmsnorm_weight, C * sizeof(__hip_bfloat16),
                                   hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync(weights.phi_pre, h_phi_pre, nC * n * sizeof(__hip_bfloat16),
                                   hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync(weights.phi_post, h_phi_post, nC * n * sizeof(__hip_bfloat16),
                                   hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync(weights.phi_res, h_phi_res, nC * n * n * sizeof(__hip_bfloat16),
                                   hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync(weights.b_pre, h_b_pre, n * sizeof(float),
                                   hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync(weights.b_post, h_b_post, n * sizeof(float),
                                   hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync(weights.b_res, h_b_res, n * n * sizeof(float),
                                   hipMemcpyHostToDevice, stream));

        weights.alpha_pre = alpha_pre;
        weights.alpha_post = alpha_post;
        weights.alpha_res = alpha_res;
    }

    void set_weights_static(const __hip_bfloat16* h_rmsnorm_weight, const float* h_b_pre,
                            const float* h_b_post, const float* h_b_res) {
        int C = config.hidden_dim;
        int n = config.expansion_rate;

        HIP_CALL(hipMemcpyAsync(weights.rmsnorm_weight, h_rmsnorm_weight, C * sizeof(__hip_bfloat16),
                                   hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync(weights.b_pre, h_b_pre, n * sizeof(float),
                                   hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync(weights.b_post, h_b_post, n * sizeof(float),
                                   hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync(weights.b_res, h_b_res, n * n * sizeof(float),
                                   hipMemcpyHostToDevice, stream));
    }

    void set_weights(const __hip_bfloat16* h_rmsnorm_weight, const float* h_H_pre, const float* h_H_post,
                     const float* h_H_res) {
        set_weights_static(h_rmsnorm_weight, h_H_pre, h_H_post, h_H_res);
    }

    void compute_dynamic_h_internal(int B, int n, int C) {
        int nC = n * C;

        float_to_bf16(buffers.x_flat_bf16, buffers.x_expanded, B * nC, stream);

        buffers.fused_rms_matmul.forward(buffers.H_proj_raw, buffers.x_flat_bf16,
                                         weights.phi_combined, stream);

        apply_dynamic_h_activations(buffers.H_pre_activated, buffers.H_post_activated,
                                    buffers.H_res_tilde, buffers.H_proj_raw, weights.b_pre,
                                    weights.b_post, weights.b_res, weights.alpha_pre,
                                    weights.alpha_post, weights.alpha_res, B, n, stream);

        apply_exp(buffers.H_res_tilde, buffers.H_res_tilde, B * n * n, stream);
        sinkhorn_knopp_forward_batched(buffers.sinkhorn_M, buffers.H_res_tilde, B, n,
                                       config.sinkhorn_iters, config.eps, stream);
    }

    void forward(const float* x_expanded) {
        int B = config.batch_size;
        int C = config.hidden_dim;
        int n = config.expansion_rate;

        HIP_CALL(hipMemcpyAsync(buffers.x_expanded, x_expanded, B * n * C * sizeof(float),
                                   hipMemcpyHostToDevice, stream));

        if (config.use_dynamic_h) {
            compute_dynamic_h_internal(B, n, C);

            stream_aggregate_bf16_dynamic(buffers.x_aggregated_bf16, buffers.x_expanded,
                                          buffers.H_pre_activated, B, n, C, stream);
        } else {
            if (use_pipelining) {
                sinkhorn_knopp_forward_fused_exp(buffers.sinkhorn_M, buffers.H_res_tilde,
                                                 weights.b_res, n, n, config.sinkhorn_iters,
                                                 config.eps, sinkhorn_stream);
                HIP_CALL(hipEventRecord(sinkhorn_done, sinkhorn_stream));
            }

            stream_aggregate_bf16_fused_sigmoid(buffers.x_aggregated_bf16, buffers.H_pre_activated,
                                                buffers.x_expanded, weights.b_pre, B, n, C, stream);

            if (!use_pipelining) {
                sinkhorn_knopp_forward_fused_exp(buffers.sinkhorn_M, buffers.H_res_tilde,
                                                 weights.b_res, n, n, config.sinkhorn_iters,
                                                 config.eps, stream);
            } else {
                HIP_CALL(hipStreamWaitEvent(stream, sinkhorn_done, 0));
            }
        }

        rmsnorm_forward_with_rms(buffers.layer_out_bf16, buffers.rms_values,
                                 buffers.x_aggregated_bf16, weights.rmsnorm_weight, B, C,
                                 config.eps, stream);

        if (config.use_dynamic_h) {
            stream_distribute_mix_add_fused_dynamic(
                buffers.output, buffers.x_expanded, buffers.layer_out_bf16,
                buffers.H_post_activated, buffers.sinkhorn_M, B, n, C, stream);
        } else {
            if (use_tc_mix) {
                stream_mix_tc.forward_fused_distribute_add(
                    buffers.output, buffers.H_post_activated, buffers.x_expanded,
                    buffers.layer_out_bf16, buffers.sinkhorn_M, weights.b_post, buffers.x_mixed,
                    stream);
            } else {
                stream_distribute_mix_add_fused(
                    buffers.output, buffers.H_post_activated, buffers.x_expanded,
                    buffers.layer_out_bf16, weights.b_post, buffers.sinkhorn_M, B, n, C, stream);
            }
        }
    }

    void forward_device(const float* d_x_expanded) {
        int B = config.batch_size;
        int C = config.hidden_dim;
        int n = config.expansion_rate;

        HIP_CALL(hipMemcpyAsync(buffers.x_expanded, d_x_expanded, B * n * C * sizeof(float),
                                   hipMemcpyDeviceToDevice, stream));

        if (config.use_dynamic_h) {
            compute_dynamic_h_internal(B, n, C);

            stream_aggregate_bf16_dynamic(buffers.x_aggregated_bf16, buffers.x_expanded,
                                          buffers.H_pre_activated, B, n, C, stream);
        } else {
            if (use_pipelining) {
                sinkhorn_knopp_forward_fused_exp(buffers.sinkhorn_M, buffers.H_res_tilde,
                                                 weights.b_res, n, n, config.sinkhorn_iters,
                                                 config.eps, sinkhorn_stream);
                HIP_CALL(hipEventRecord(sinkhorn_done, sinkhorn_stream));
            }

            stream_aggregate_bf16_fused_sigmoid(buffers.x_aggregated_bf16, buffers.H_pre_activated,
                                                buffers.x_expanded, weights.b_pre, B, n, C, stream);

            if (!use_pipelining) {
                sinkhorn_knopp_forward_fused_exp(buffers.sinkhorn_M, buffers.H_res_tilde,
                                                 weights.b_res, n, n, config.sinkhorn_iters,
                                                 config.eps, stream);
            } else {
                HIP_CALL(hipStreamWaitEvent(stream, sinkhorn_done, 0));
            }
        }

        rmsnorm_forward_with_rms(buffers.layer_out_bf16, buffers.rms_values,
                                 buffers.x_aggregated_bf16, weights.rmsnorm_weight, B, C,
                                 config.eps, stream);

        if (config.use_dynamic_h) {
            stream_distribute_mix_add_fused_dynamic(
                buffers.output, buffers.x_expanded, buffers.layer_out_bf16,
                buffers.H_post_activated, buffers.sinkhorn_M, B, n, C, stream);
        } else {
            if (use_tc_mix) {
                stream_mix_tc.forward_fused_distribute_add(
                    buffers.output, buffers.H_post_activated, buffers.x_expanded,
                    buffers.layer_out_bf16, buffers.sinkhorn_M, weights.b_post, buffers.x_mixed,
                    stream);
            } else {
                stream_distribute_mix_add_fused(
                    buffers.output, buffers.H_post_activated, buffers.x_expanded,
                    buffers.layer_out_bf16, weights.b_post, buffers.sinkhorn_M, B, n, C, stream);
            }
        }
    }

    float* get_output() { return buffers.output; }

    float* get_rms_values() { return buffers.rms_values; }

    void backward(const float* d_output) {
        if (!backward_enabled) {
            fprintf(stderr, "MHCLayer::backward called but backward not enabled\n");
            return;
        }

        int B = config.batch_size;
        int C = config.hidden_dim;
        int n = config.expansion_rate;

        grads.zero_weight_grads(C, n, stream);

        bf16_to_float(buffers.layer_out_f32, buffers.layer_out_bf16, B * C, stream);

        stream_distribute_mix_backward_fused(
            grads.d_x_mixed, grads.d_layer_out, grads.d_M, grads.d_H_post_activated, d_output,
            buffers.x_expanded, buffers.layer_out_f32, buffers.sinkhorn_M, buffers.H_post_activated,
            B, n, C, grads.workspace_dM, grads.workspace_dH, grads.workspace_num_blocks, stream);

        sinkhorn_knopp_backward(grads.d_H_res_exp, grads.d_M, buffers.sinkhorn_M,
                                buffers.H_res_tilde, n, config.sinkhorn_iters, config.eps, stream);

        exp_backward(grads.d_H_res, grads.d_H_res_exp, buffers.H_res_tilde, n * n, stream);

        sigmoid_scale_backward(grads.d_H_post, grads.d_H_post_activated, buffers.H_post_activated,
                               2.0f, n, stream);

        bf16_to_float(buffers.x_aggregated_f32, buffers.x_aggregated_bf16, B * C, stream);

        rmsnorm_backward(grads.d_x_aggregated, grads.d_rmsnorm_weight, grads.d_layer_out,
                         buffers.x_aggregated_bf16, weights.rmsnorm_weight, buffers.rms_values, B,
                         C, stream);

        stream_aggregate_backward(grads.d_x_expanded, grads.d_H_pre_activated, grads.d_x_aggregated,
                                  buffers.x_expanded, buffers.H_pre_activated, B, n, C,
                                  grads.workspace_dH, grads.workspace_num_blocks, stream);

        sigmoid_backward(grads.d_H_pre, grads.d_H_pre_activated, buffers.H_pre_activated, n,
                         stream);
    }

    void sync() { HIP_CALL(hipStreamSynchronize(stream)); }
};
} // namespace mhc
