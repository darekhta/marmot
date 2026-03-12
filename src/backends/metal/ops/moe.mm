#include "marmot/ops/matmul.h"

#include "core/tensor/tensor_utils.h"
#include "internal/metal_matmul_qkv_shared.h"
#include "internal/stride_helpers.h"
#include "metal_backend_internal.h"

#ifdef __APPLE__

#include <stdlib.h>

#include <float.h>
#include <string.h>
#include <time.h>

#include "utils/dtype_ref.h"

#ifdef __cplusplus
extern "C" {
#endif

marmot_error_t cpu_moe_experts_impl(const void *device_ctx, const marmot_moe_experts_desc_t *desc);
marmot_error_t marmot_metal_gemm(
    metal_context_t *ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out, bool transpose_b
);

#ifdef __cplusplus
}
#endif

typedef struct {
    uint32_t rows;
    uint32_t cols;
    uint32_t input_stride;
    uint32_t values_stride;
    uint32_t indices_stride;
    uint32_t k;
} metal_topk_uniforms_t;

typedef struct {
    uint32_t rows;
    uint32_t cols;
    uint32_t row_stride;
} metal_moe_zero_uniforms_t;

typedef struct {
    uint32_t count;
    uint32_t hidden;
    uint32_t output_rows;
    uint32_t src_stride;
    uint32_t out_stride;
    uint32_t index_stride;
    uint32_t weight_stride;
} metal_moe_scatter_uniforms_t;

typedef struct {
    uint32_t tokens;
    uint32_t experts_per_token;
    uint32_t experts;
    uint32_t id_stride0;
    uint32_t id_stride1;
    uint32_t weight_stride0;
    uint32_t weight_stride1;
    uint32_t renormalize_selected;
    float weights_scale;
} metal_moe_route_uniforms_t;

typedef struct {
    uint32_t route_count;
    uint32_t max_batch;
    uint32_t active_experts;
    uint32_t reserved;
} metal_moe_route_summary_t;

typedef struct {
    uint32_t routes;
    uint32_t input_cols;
    uint32_t output_cols;
    uint32_t input_stride;
    uint32_t output_stride;
    uint32_t weight_blocks;
    uint32_t broadcast_input;
} metal_moe_decode_gate_up_uniforms_t;

typedef struct {
    uint32_t routes;
    uint32_t input_cols;
    uint32_t output_cols;
    uint32_t input_stride;
    uint32_t output_stride;
    uint32_t weight_blocks;
    uint32_t activation;
} metal_moe_decode_down_uniforms_t;

typedef struct {
    uint32_t routes;
    uint32_t input_cols;
    uint32_t output_cols;
    uint32_t input_stride;
    uint32_t output_stride;
    uint32_t weight_blocks;
    uint32_t activation;
    uint32_t output_rows;
} metal_moe_indexed_down_uniforms_t;

typedef struct {
    bool enabled;
    uint64_t total_start_ns;
    uint64_t route_ns;
    uint64_t gather_ns;
    uint64_t gate_up_ns;
    uint64_t glu_ns;
    uint64_t down_ns;
    uint64_t scatter_ns;
} metal_moe_stage_profile_t;

static const char *metal_moe_dtype_name(marmot_dtype_t dtype) {
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT32:
        return "f32";
    case MARMOT_DTYPE_FLOAT16:
        return "f16";
    case MARMOT_DTYPE_INT32:
        return "i32";
    default:
        return "other";
    }
}

static const char *metal_moe_quant_kind_name(marmot_quant_kind_t kind) {
    switch (kind) {
    case MARMOT_QUANT_KIND_Q4_K:
        return "Q4_K";
    case MARMOT_QUANT_KIND_Q5_K:
        return "Q5_K";
    case MARMOT_QUANT_KIND_Q6_K:
        return "Q6_K";
    default:
        return "other";
    }
}

static bool metal_moe_value_dtype_supported(marmot_dtype_t dtype) {
    return dtype == MARMOT_DTYPE_FLOAT32 || dtype == MARMOT_DTYPE_FLOAT16;
}

static bool metal_moe_stage_profiling_enabled(void) {
    static int cached = -1;
    if (cached == -1) {
        const char *env = getenv("MARMOT_PROFILE_MOE_STAGES");
        cached = (env != nullptr && env[0] != '\0' && !(env[0] == '0' && env[1] == '\0')) ? 1 : 0;
    }
    return cached == 1;
}

static uint64_t metal_moe_now_ns(void) {
    return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
}

typedef enum {
    METAL_MOE_ROUTE_MODE_AUTO = 0,
    METAL_MOE_ROUTE_MODE_HOST = 1,
    METAL_MOE_ROUTE_MODE_GPU = 2,
} metal_moe_route_mode_t;

typedef enum {
    METAL_MOE_EXPERT_BATCH_AUTO = 0,
    METAL_MOE_EXPERT_BATCH_DISABLE = 1,
    METAL_MOE_EXPERT_BATCH_FORCE = 2,
} metal_moe_expert_batch_mode_t;

typedef enum {
    METAL_MOE_GROUPED_DECODE_AUTO = 0,
    METAL_MOE_GROUPED_DECODE_DISABLE = 1,
    METAL_MOE_GROUPED_DECODE_FORCE = 2,
} metal_moe_grouped_decode_mode_t;

static metal_moe_route_mode_t metal_moe_route_mode_override(void) {
    const char *env = getenv("MARMOT_MOE_ROUTE_MODE");
    if (env == nullptr || env[0] == '\0' || strcmp(env, "auto") == 0) {
        return METAL_MOE_ROUTE_MODE_AUTO;
    } else if (strcmp(env, "host") == 0) {
        return METAL_MOE_ROUTE_MODE_HOST;
    } else if (strcmp(env, "gpu") == 0) {
        return METAL_MOE_ROUTE_MODE_GPU;
    } else {
        return METAL_MOE_ROUTE_MODE_AUTO;
    }
}

static metal_moe_expert_batch_mode_t metal_moe_expert_batch_mode_override(void) {
    const char *env = getenv("MARMOT_MOE_EXPERT_BATCH");
    if (env == nullptr || env[0] == '\0') {
        env = getenv("MARMOT_MOE_GROUPED_PREFILL");
    }
    if (env == nullptr || env[0] == '\0' || strcmp(env, "auto") == 0) {
        return METAL_MOE_EXPERT_BATCH_AUTO;
    } else if (strcmp(env, "0") == 0 || strcmp(env, "off") == 0 || strcmp(env, "disable") == 0 ||
               strcmp(env, "disabled") == 0 || strcmp(env, "false") == 0 || strcmp(env, "loop") == 0) {
        return METAL_MOE_EXPERT_BATCH_DISABLE;
    } else if (strcmp(env, "1") == 0 || strcmp(env, "on") == 0 || strcmp(env, "force") == 0 ||
               strcmp(env, "enabled") == 0 || strcmp(env, "true") == 0 || strcmp(env, "grouped") == 0) {
        return METAL_MOE_EXPERT_BATCH_FORCE;
    } else {
        return METAL_MOE_EXPERT_BATCH_AUTO;
    }
}

static metal_moe_grouped_decode_mode_t metal_moe_grouped_decode_mode_override(void) {
    const char *env = getenv("MARMOT_MOE_GROUPED_DECODE");
    if (env == nullptr || env[0] == '\0' || strcmp(env, "auto") == 0) {
        return METAL_MOE_GROUPED_DECODE_AUTO;
    } else if (strcmp(env, "0") == 0 || strcmp(env, "off") == 0 || strcmp(env, "disable") == 0 ||
               strcmp(env, "disabled") == 0 || strcmp(env, "false") == 0 || strcmp(env, "loop") == 0) {
        return METAL_MOE_GROUPED_DECODE_DISABLE;
    } else if (strcmp(env, "1") == 0 || strcmp(env, "on") == 0 || strcmp(env, "force") == 0 ||
               strcmp(env, "enabled") == 0 || strcmp(env, "true") == 0 || strcmp(env, "grouped") == 0) {
        return METAL_MOE_GROUPED_DECODE_FORCE;
    } else {
        return METAL_MOE_GROUPED_DECODE_AUTO;
    }
}

static uint64_t metal_moe_stage_profile_begin(const metal_moe_stage_profile_t *profile) {
    return profile != nullptr && profile->enabled ? metal_moe_now_ns() : 0;
}

static void metal_moe_stage_profile_end(
    metal_moe_stage_profile_t *profile, metal_context_t *ctx, uint64_t start_ns, uint64_t *accumulator
) {
    if (profile == nullptr || !profile->enabled || start_ns == 0 || accumulator == nullptr) {
        return;
    }
    metal_command_stream_flush(ctx, true);
    *accumulator += metal_moe_now_ns() - start_ns;
}

static void metal_moe_stage_profile_emit(
    const metal_moe_stage_profile_t *profile, size_t tokens, size_t route_count, size_t experts, size_t max_batch,
    bool use_host_routes, bool use_grouped_decode, bool use_expert_batch_gate_up, bool use_expert_batch_down
) {
    if (profile == nullptr || !profile->enabled) {
        return;
    }
    const uint64_t total_ns = metal_moe_now_ns() - profile->total_start_ns;
    const char *route_mode = use_host_routes ? "host" : "gpu";
    const char *decode_mode = use_grouped_decode ? "decode-grouped" : "standard";
    const char *gate_up_mode = use_expert_batch_gate_up ? "expert-batched" : "gate-up-loop";
    const char *down_mode = use_expert_batch_down ? "expert-batched" : "down-loop";
    fprintf(
        stderr,
        "[moe profile] tokens=%zu routes=%zu experts=%zu max_batch=%zu route_mode=%s exec=%s gate_up=%s down=%s "
        "route=%.3fms gather=%.3fms gate_up=%.3fms glu=%.3fms down=%.3fms scatter=%.3fms total=%.3fms\n",
        tokens, route_count, experts, max_batch, route_mode, decode_mode, gate_up_mode, down_mode,
        (double)profile->route_ns / 1000000.0, (double)profile->gather_ns / 1000000.0,
        (double)profile->gate_up_ns / 1000000.0, (double)profile->glu_ns / 1000000.0,
        (double)profile->down_ns / 1000000.0, (double)profile->scatter_ns / 1000000.0, (double)total_ns / 1000000.0
    );
}

static void metal_moe_stage_profile_emit_details(
    const metal_moe_stage_profile_t *profile, const marmot_moe_experts_desc_t *desc, bool quantized_supported,
    size_t ordered_active_experts, bool use_grouped_decode, bool use_grouped_decode_gate_up,
    bool use_grouped_decode_down, bool use_expert_batch_prefill
) {
    if (profile == nullptr || !profile->enabled || desc == nullptr) {
        return;
    }
    fprintf(
        stderr,
        "[moe profile detail] quantized=%d gate_kind=%s up_kind=%s down_kind=%s hidden_dtype=%s topk_dtype=%s "
        "out_dtype=%s hidden_contig=%d active_experts=%zu grouped_decode=%d grouped_decode_gate_up=%d "
        "grouped_decode_down=%d expert_batch_prefill=%d\n",
        quantized_supported ? 1 : 0, metal_moe_quant_kind_name(desc->gate_exps->quant_kind),
        metal_moe_quant_kind_name(desc->up_exps->quant_kind), metal_moe_quant_kind_name(desc->down_exps->quant_kind),
        metal_moe_dtype_name(desc->hidden_states->dtype), metal_moe_dtype_name(desc->topk_weights->dtype),
        metal_moe_dtype_name(desc->out->dtype), marmot_tensor_is_contiguous(desc->hidden_states) ? 1 : 0,
        ordered_active_experts, use_grouped_decode ? 1 : 0, use_grouped_decode_gate_up ? 1 : 0,
        use_grouped_decode_down ? 1 : 0, use_expert_batch_prefill ? 1 : 0
    );
}

static float metal_moe_load_value(const void *data, marmot_dtype_t dtype, size_t index) {
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT32:
        return ((const float *)data)[index];
    case MARMOT_DTYPE_FLOAT16:
        return marmot_f16_to_f32_ref(((const marmot_float16_t *)data)[index]);
    default:
        return 0.0f;
    }
}

static void metal_moe_store_value(void *data, marmot_dtype_t dtype, size_t index, float value) {
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT32:
        ((float *)data)[index] = value;
        break;
    case MARMOT_DTYPE_FLOAT16:
        ((marmot_float16_t *)data)[index] = marmot_f32_to_f16_ref(value);
        break;
    default:
        break;
    }
}

constexpr size_t kMetalTopkSmallCols = 256;
constexpr size_t kMetalTopkSmallK = 16;

static inline void metal_moe_mark_device_write(marmot_tensor_t *tensor) {
    if (tensor == nullptr) {
        return;
    }
    tensor->memory_location = MARMOT_MEMORY_DEVICE;
    tensor->needs_sync = true;
}

static const char *metal_topk_pipeline_name(marmot_dtype_t dtype, bool use_small) {
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT32:
        return use_small ? "topk_f32_last_axis_small" : "topk_f32_last_axis";
    case MARMOT_DTYPE_FLOAT16:
        return use_small ? "topk_f16_last_axis_small" : "topk_f16_last_axis";
    default:
        return nullptr;
    }
}

static const char *metal_moe_zero_pipeline_name(marmot_dtype_t dtype) {
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT32:
        return "moe_zero_f32_2d";
    case MARMOT_DTYPE_FLOAT16:
        return "moe_zero_f16_2d";
    default:
        return nullptr;
    }
}

static const char *metal_moe_scatter_pipeline_name(marmot_dtype_t dtype) {
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT32:
        return "moe_scatter_add_f32";
    case MARMOT_DTYPE_FLOAT16:
        return "moe_scatter_add_f16";
    default:
        return nullptr;
    }
}

static const char *metal_moe_route_pack_pipeline_name(marmot_dtype_t dtype) {
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT32:
        return "moe_route_pack_stable_f32";
    case MARMOT_DTYPE_FLOAT16:
        return "moe_route_pack_stable_f16";
    default:
        return nullptr;
    }
}

static const char *metal_moe_decode_down_pipeline_name(
    marmot_quant_kind_t quant_kind, marmot_dtype_t input_dtype, marmot_dtype_t output_dtype
) {
    if (quant_kind != MARMOT_QUANT_KIND_Q6_K) {
        return nullptr;
    }
    switch (input_dtype) {
    case MARMOT_DTYPE_FLOAT32:
        return output_dtype == MARMOT_DTYPE_FLOAT32 ? "moe_decode_down_q6_k_f32_f32" : nullptr;
    case MARMOT_DTYPE_FLOAT16:
        if (output_dtype == MARMOT_DTYPE_FLOAT32) {
            return "moe_decode_down_q6_k_f16_f32";
        }
        return output_dtype == MARMOT_DTYPE_FLOAT16 ? "moe_decode_down_q6_k_f16_f16" : nullptr;
    default:
        return nullptr;
    }
}

static const char *metal_moe_decode_glu_down_pipeline_name(
    marmot_quant_kind_t quant_kind, marmot_dtype_t input_dtype, marmot_dtype_t output_dtype
) {
    if (quant_kind != MARMOT_QUANT_KIND_Q6_K) {
        return nullptr;
    }
    switch (input_dtype) {
    case MARMOT_DTYPE_FLOAT32:
        return output_dtype == MARMOT_DTYPE_FLOAT32 ? "moe_decode_glu_down_q6_k_f32_f32" : nullptr;
    case MARMOT_DTYPE_FLOAT16:
        if (output_dtype == MARMOT_DTYPE_FLOAT32) {
            return "moe_decode_glu_down_q6_k_f16_f32";
        }
        return output_dtype == MARMOT_DTYPE_FLOAT16 ? "moe_decode_glu_down_q6_k_f16_f16" : nullptr;
    default:
        return nullptr;
    }
}

static const char *metal_moe_expert_batch_q4_k_pipeline_name(marmot_dtype_t input_dtype, marmot_dtype_t output_dtype) {
    switch (input_dtype) {
    case MARMOT_DTYPE_FLOAT32:
        return output_dtype == MARMOT_DTYPE_FLOAT32 ? "moe_expert_matmul_q4_k_f32_f32" : nullptr;
    case MARMOT_DTYPE_FLOAT16:
        if (output_dtype == MARMOT_DTYPE_FLOAT32) {
            return "moe_expert_matmul_q4_k_f16_f32";
        }
        return output_dtype == MARMOT_DTYPE_FLOAT16 ? "moe_expert_matmul_q4_k_f16_f16" : nullptr;
    default:
        return nullptr;
    }
}

static const char *metal_moe_expert_batch_q6_k_pipeline_name(marmot_dtype_t input_dtype, marmot_dtype_t output_dtype) {
    switch (input_dtype) {
    case MARMOT_DTYPE_FLOAT32:
        return output_dtype == MARMOT_DTYPE_FLOAT32 ? "moe_expert_matmul_q6_k_f32_f32" : nullptr;
    case MARMOT_DTYPE_FLOAT16:
        if (output_dtype == MARMOT_DTYPE_FLOAT32) {
            return "moe_expert_matmul_q6_k_f16_f32";
        }
        return output_dtype == MARMOT_DTYPE_FLOAT16 ? "moe_expert_matmul_q6_k_f16_f16" : nullptr;
    default:
        return nullptr;
    }
}

static const char *
metal_moe_expert_batch_glu_q6_k_pipeline_name(marmot_dtype_t input_dtype, marmot_dtype_t output_dtype) {
    switch (input_dtype) {
    case MARMOT_DTYPE_FLOAT32:
        return output_dtype == MARMOT_DTYPE_FLOAT32 ? "moe_expert_glu_down_q6_k_f32_f32" : nullptr;
    case MARMOT_DTYPE_FLOAT16:
        if (output_dtype == MARMOT_DTYPE_FLOAT32) {
            return "moe_expert_glu_down_q6_k_f16_f32";
        }
        return output_dtype == MARMOT_DTYPE_FLOAT16 ? "moe_expert_glu_down_q6_k_f16_f16" : nullptr;
    default:
        return nullptr;
    }
}

static const char *metal_moe_fused_gate_up_q4_k_pipeline_name(marmot_dtype_t input_dtype, marmot_dtype_t output_dtype) {
    switch (input_dtype) {
    case MARMOT_DTYPE_FLOAT32:
        return output_dtype == MARMOT_DTYPE_FLOAT32 ? "moe_expert_fused_gate_up_q4_k_f32_f32" : nullptr;
    case MARMOT_DTYPE_FLOAT16:
        if (output_dtype == MARMOT_DTYPE_FLOAT32) {
            return "moe_expert_fused_gate_up_q4_k_f16_f32";
        }
        return output_dtype == MARMOT_DTYPE_FLOAT16 ? "moe_expert_fused_gate_up_q4_k_f16_f16" : nullptr;
    default:
        return nullptr;
    }
}

static const char *metal_moe_indexed_down_pipeline_name(
    marmot_quant_kind_t quant_kind, marmot_dtype_t input_dtype, marmot_dtype_t output_dtype
) {
    if (quant_kind != MARMOT_QUANT_KIND_Q6_K) {
        return nullptr;
    }
    switch (input_dtype) {
    case MARMOT_DTYPE_FLOAT32:
        return output_dtype == MARMOT_DTYPE_FLOAT32 ? "moe_indexed_down_q6_k_f32_f32" : nullptr;
    case MARMOT_DTYPE_FLOAT16:
        if (output_dtype == MARMOT_DTYPE_FLOAT32) {
            return "moe_indexed_down_q6_k_f16_f32";
        }
        return output_dtype == MARMOT_DTYPE_FLOAT16 ? "moe_indexed_down_q6_k_f16_f16" : nullptr;
    default:
        return nullptr;
    }
}

static const char *metal_moe_indexed_glu_down_pipeline_name(
    marmot_quant_kind_t quant_kind, marmot_dtype_t input_dtype, marmot_dtype_t output_dtype
) {
    if (quant_kind != MARMOT_QUANT_KIND_Q6_K) {
        return nullptr;
    }
    switch (input_dtype) {
    case MARMOT_DTYPE_FLOAT32:
        return output_dtype == MARMOT_DTYPE_FLOAT32 ? "moe_indexed_glu_down_q6_k_f32_f32" : nullptr;
    case MARMOT_DTYPE_FLOAT16:
        if (output_dtype == MARMOT_DTYPE_FLOAT32) {
            return "moe_indexed_glu_down_q6_k_f16_f32";
        }
        return output_dtype == MARMOT_DTYPE_FLOAT16 ? "moe_indexed_glu_down_q6_k_f16_f16" : nullptr;
    default:
        return nullptr;
    }
}

static const char *metal_moe_decode_gate_up_pipeline_name(
    marmot_quant_kind_t quant_kind, marmot_dtype_t input_dtype, marmot_dtype_t output_dtype
) {
    if (quant_kind != MARMOT_QUANT_KIND_Q4_K) {
        return nullptr;
    }
    switch (input_dtype) {
    case MARMOT_DTYPE_FLOAT32:
        return output_dtype == MARMOT_DTYPE_FLOAT32 ? "moe_decode_gate_up_q4_k_f32_f32" : nullptr;
    case MARMOT_DTYPE_FLOAT16:
        if (output_dtype == MARMOT_DTYPE_FLOAT32) {
            return "moe_decode_gate_up_q4_k_f16_f32";
        }
        return output_dtype == MARMOT_DTYPE_FLOAT16 ? "moe_decode_gate_up_q4_k_f16_f16" : nullptr;
    default:
        return nullptr;
    }
}

static void metal_moe_reset_alias_runtime(marmot_tensor_t *tensor) {
    if (tensor == nullptr) {
        return;
    }
    tensor->owns_data = false;
    tensor->packed_data = nullptr;
    tensor->packed_src_data = nullptr;
    tensor->packed_bytes = 0;
    tensor->packed_row_bytes = 0;
    tensor->packed_rows = 0;
}

static void
metal_moe_init_cpu_alias(marmot_tensor_t *dst, const marmot_tensor_t *src, const marmot_context_t *cpu_ctx) {
    *dst = *src;
    dst->backend = MARMOT_BACKEND_CPU;
    dst->ctx = (marmot_context_t *)cpu_ctx;
    dst->memory_location = MARMOT_MEMORY_HOST;
    dst->needs_sync = false;
    dst->packed_data = nullptr;
    dst->packed_src_data = nullptr;
    dst->packed_bytes = 0;
    dst->packed_row_bytes = 0;
    dst->packed_rows = 0;
}

static void metal_moe_init_dense_view(
    marmot_tensor_t *dst, const marmot_tensor_t *src, size_t rows, size_t cols, size_t byte_offset
) {
    *dst = *src;
    dst->shape.ndim = 2;
    dst->shape.shape[0] = rows;
    dst->shape.shape[1] = cols;
    dst->shape.strides[0] = cols;
    dst->shape.strides[1] = 1;
    dst->data = (uint8_t *)src->data + byte_offset;
    dst->capacity_bytes = rows * cols * marmot_dtype_size(src->dtype);
    metal_moe_reset_alias_runtime(dst);
}

static size_t metal_moe_quant_row_bytes(marmot_quant_kind_t kind, size_t cols) {
    const marmot_quant_kind_traits_t *traits = marmot_get_quant_kind_traits(kind);
    if (traits == nullptr || traits->block_values == 0) {
        return 0;
    }
    const size_t block_bytes = traits->header_bytes + traits->payload_bytes;
    const size_t blocks_per_row = (cols + traits->block_values - 1) / traits->block_values;
    return blocks_per_row * block_bytes;
}

static size_t metal_moe_expert_slice_bytes(const marmot_tensor_t *tensor, size_t rows, size_t cols) {
    if (tensor == nullptr) {
        return 0;
    }
    if (marmot_tensor_is_block_quantized_weight(tensor)) {
        return rows * metal_moe_quant_row_bytes(tensor->quant_kind, cols);
    }
    return rows * cols * marmot_dtype_size(tensor->dtype);
}

static void metal_moe_init_quant_view(
    marmot_tensor_t *dst, const marmot_tensor_t *src, size_t rows, size_t cols, size_t byte_offset
) {
    *dst = *src;
    dst->shape.ndim = 2;
    dst->shape.shape[0] = rows;
    dst->shape.shape[1] = cols;
    dst->shape.strides[0] = cols;
    dst->shape.strides[1] = 1;
    dst->data = (uint8_t *)src->data + byte_offset;
    dst->capacity_bytes = metal_moe_expert_slice_bytes(src, rows, cols);
    metal_moe_reset_alias_runtime(dst);
}

static void metal_moe_init_expert_view(
    marmot_tensor_t *dst, const marmot_tensor_t *src, size_t rows, size_t cols, size_t expert_idx
) {
    const size_t slice_bytes = metal_moe_expert_slice_bytes(src, rows, cols);
    const size_t byte_offset = expert_idx * slice_bytes;
    if (marmot_tensor_is_block_quantized_weight(src)) {
        metal_moe_init_quant_view(dst, src, rows, cols, byte_offset);
        return;
    }
    metal_moe_init_dense_view(dst, src, rows, cols, byte_offset);
}

static void metal_moe_init_2d_alias(marmot_tensor_t *dst, const marmot_tensor_t *src, size_t rows, size_t cols) {
    metal_moe_init_dense_view(dst, src, rows, cols, 0);
}

static void metal_moe_init_2d_slice_alias(
    marmot_tensor_t *dst, const marmot_tensor_t *src, size_t rows, size_t cols, size_t row_offset
) {
    const size_t byte_offset = row_offset * src->shape.strides[0] * marmot_dtype_size(src->dtype);
    metal_moe_init_dense_view(dst, src, rows, cols, byte_offset);
}

static void
metal_moe_init_1d_slice_alias(marmot_tensor_t *dst, const marmot_tensor_t *src, size_t length, size_t offset) {
    *dst = *src;
    dst->shape.ndim = 1;
    dst->shape.shape[0] = length;
    dst->shape.strides[0] = 1;
    dst->data = (uint8_t *)src->data + offset * marmot_dtype_size(src->dtype);
    dst->capacity_bytes = length * marmot_dtype_size(src->dtype);
    metal_moe_reset_alias_runtime(dst);
}

static void metal_moe_init_temp_tensor(
    marmot_tensor_t *dst, const marmot_context_t *src_ctx, const size_t *shape, size_t ndim, marmot_dtype_t dtype,
    void *data, size_t capacity_bytes
) {
    memset(dst, 0, sizeof(*dst));
    dst->shape.ndim = ndim;
    memcpy(dst->shape.shape, shape, ndim * sizeof(size_t));
    dst->shape.strides[ndim - 1] = 1;
    for (size_t i = ndim; i-- > 1;) {
        dst->shape.strides[i - 1] = dst->shape.strides[i] * shape[i];
    }
    dst->dtype = dtype;
    dst->backend = src_ctx != nullptr ? src_ctx->backend_type : MARMOT_BACKEND_METAL;
    dst->ctx = (marmot_context_t *)src_ctx;
    dst->owns_data = false;
    dst->quant_kind = MARMOT_QUANT_KIND_GENERIC;
    dst->quant_layout = MARMOT_QUANT_LAYOUT_GENERIC;
    dst->memory_location = MARMOT_MEMORY_HOST;
    dst->needs_sync = false;
    dst->data = data;
    dst->capacity_bytes = capacity_bytes;
}

static marmot_error_t metal_moe_project(
    const void *device_ctx, metal_context_t *ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    marmot_tensor_t *out
) {
    if (marmot_tensor_is_block_quantized_weight(weight)) {
        return metal_matmul_quantized(device_ctx, input, weight, nullptr, out);
    }
    return marmot_metal_gemm(ctx, input, weight, nullptr, out, true);
}

static bool metal_moe_can_fused_gate_up(
    const marmot_tensor_t *input, const marmot_tensor_t *gate_weight, const marmot_tensor_t *up_weight,
    const marmot_tensor_t *gate_out, const marmot_tensor_t *up_out, const marmot_tensor_t *scratch_out
) {
    if (input == nullptr || gate_weight == nullptr || up_weight == nullptr || gate_out == nullptr ||
        up_out == nullptr || scratch_out == nullptr) {
        return false;
    }
    if (input->shape.ndim != 2 || gate_weight->shape.ndim != 2 || up_weight->shape.ndim != 2 ||
        gate_out->shape.ndim != 2 || up_out->shape.ndim != 2 || scratch_out->shape.ndim != 2) {
        return false;
    }
    if (!marmot_tensor_is_block_quantized_weight(gate_weight) || !marmot_tensor_is_block_quantized_weight(up_weight)) {
        return false;
    }
    if (gate_weight->dtype != up_weight->dtype || gate_weight->quant_kind != up_weight->quant_kind ||
        gate_weight->quant_layout != up_weight->quant_layout) {
        return false;
    }
    if (gate_weight->shape.shape[0] != up_weight->shape.shape[0] ||
        gate_weight->shape.shape[1] != up_weight->shape.shape[1]) {
        return false;
    }
    if (input->shape.shape[1] != gate_weight->shape.shape[1]) {
        return false;
    }
    if (gate_out->shape.shape[0] != input->shape.shape[0] || up_out->shape.shape[0] != input->shape.shape[0] ||
        scratch_out->shape.shape[0] != input->shape.shape[0]) {
        return false;
    }
    if (gate_out->shape.shape[1] != gate_weight->shape.shape[0] ||
        up_out->shape.shape[1] != up_weight->shape.shape[0] ||
        scratch_out->shape.shape[1] != gate_weight->shape.shape[0]) {
        return false;
    }
    return gate_out->dtype == up_out->dtype && gate_out->dtype == scratch_out->dtype;
}

static bool metal_moe_should_fused_gate_up(size_t batch_rows, size_t hidden, size_t ff_length) {
    (void)hidden;
    return batch_rows > 0 && batch_rows <= 64 && ff_length > 0;
}

static marmot_error_t metal_moe_project_gate_up(
    const marmot_context_t *src_ctx, const marmot_tensor_t *input, const marmot_tensor_t *gate_weight,
    const marmot_tensor_t *up_weight, marmot_tensor_t *gate_out, marmot_tensor_t *up_out, marmot_tensor_t *scratch_out
) {
    metal_context_t *ctx = static_cast<metal_context_t *>(src_ctx->device_ctx);
    marmot_error_t dual_status =
        metal_matmul_qkv_run_quantized_dual_output(ctx, input, gate_weight, up_weight, gate_out, up_out);
    if (dual_status == MARMOT_SUCCESS) {
        return dual_status;
    }
    if (dual_status != MARMOT_ERROR_NOT_IMPLEMENTED && dual_status != MARMOT_ERROR_UNSUPPORTED_DTYPE) {
        return dual_status;
    }

    marmot_matmul_qkv_desc_t desc = marmot_matmul_qkv_desc_default();
    desc.input = input;
    desc.layout = MARMOT_QKV_LAYOUT_SEPARATE;
    desc.separate.wq = gate_weight;
    desc.separate.wk = up_weight;
    desc.separate.wv = gate_weight;
    desc.out_q = gate_out;
    desc.out_k = up_out;
    desc.out_v = scratch_out;
    return marmot_matmul_qkv_shared_input(src_ctx, &desc);
}

static uint32_t metal_moe_quant_matmul_hints(const marmot_tensor_t *weight, size_t batch_rows) {
    if (weight == nullptr || !marmot_tensor_is_block_quantized_weight(weight)) {
        return 0;
    }
    if (weight->quant_kind != MARMOT_QUANT_KIND_Q4_K) {
        return 0;
    }
    if (batch_rows <= 8 || batch_rows > 16) {
        return 0;
    }
    return METAL_MATMUL_QUANT_HINT_PREFER_MV;
}

static void metal_moe_release_tracked_allocation(metal_context_t *ctx, marmot_allocation_t *alloc) {
    if (ctx == nullptr || alloc == nullptr || alloc->ptr == nullptr) {
        return;
    }
    metal_allocator_ops.free(ctx, alloc);
}

static void metal_moe_workspace_reclaim_locked(metal_context_t *ctx) {
    if (ctx == nullptr) {
        return;
    }

    for (metal_moe_workspace_t *workspace = ctx->moe_workspaces; workspace != nullptr; workspace = workspace->next) {
        if (workspace->in_use) {
            continue;
        }
        if (workspace->release_serial == UINT64_MAX && ctx->active_command_buffer == nil) {
            workspace->release_serial =
                ctx->active_command_buffer == nil && ctx->has_in_flight_work ? ctx->last_submitted_command_serial : 0;
        }
        if (workspace->release_serial != 0 && workspace->release_serial != UINT64_MAX &&
            ctx->completed_command_serial >= workspace->release_serial) {
            workspace->release_serial = 0;
        }
        if (ctx->active_command_buffer == nil && !ctx->has_in_flight_work) {
            workspace->release_serial = 0;
        }
    }
}

static marmot_error_t metal_moe_workspace_ensure_host_array(size_t **buffer, size_t count) {
    if (buffer == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (count == 0) {
        return MARMOT_SUCCESS;
    }
    if (count > SIZE_MAX / sizeof(**buffer)) {
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Metal MoE host workspace size overflow");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }
    size_t *resized = (size_t *)realloc(*buffer, count * sizeof(*resized));
    if (resized == nullptr) {
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to grow Metal MoE host workspace");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }
    *buffer = resized;
    return MARMOT_SUCCESS;
}

static marmot_error_t
metal_moe_workspace_ensure_allocation(metal_context_t *ctx, marmot_allocation_t *alloc, size_t bytes) {
    if (ctx == nullptr || alloc == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (alloc->ptr != nullptr && alloc->size >= bytes) {
        return MARMOT_SUCCESS;
    }

    metal_moe_release_tracked_allocation(ctx, alloc);
    return metal_allocate_tracked(ctx, bytes, MARMOT_ALLOC_GPU_SHARED, alloc);
}

static marmot_error_t metal_moe_workspace_acquire(
    metal_context_t *ctx, size_t experts, size_t route_indices_bytes, size_t route_weights_bytes, size_t hidden_bytes,
    size_t ff_bytes, metal_moe_workspace_t **out_workspace
) {
    if (ctx == nullptr || out_workspace == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Metal MoE workspace acquisition received invalid arguments");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    metal_moe_workspace_t *workspace = nullptr;
    pthread_mutex_lock(&ctx->command_serial_mutex);
    metal_moe_workspace_reclaim_locked(ctx);
    for (metal_moe_workspace_t *candidate = ctx->moe_workspaces; candidate != nullptr; candidate = candidate->next) {
        if (candidate->in_use || candidate->release_serial != 0) {
            continue;
        }
        workspace = candidate;
        workspace->in_use = true;
        break;
    }
    pthread_mutex_unlock(&ctx->command_serial_mutex);

    if (workspace == nullptr) {
        workspace = (metal_moe_workspace_t *)calloc(1, sizeof(*workspace));
        if (workspace == nullptr) {
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to allocate Metal MoE workspace record");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        workspace->in_use = true;
        pthread_mutex_lock(&ctx->command_serial_mutex);
        workspace->next = ctx->moe_workspaces;
        ctx->moe_workspaces = workspace;
        pthread_mutex_unlock(&ctx->command_serial_mutex);
    }

    marmot_error_t status = metal_moe_workspace_ensure_host_array(&workspace->expert_counts, experts);
    if (status == MARMOT_SUCCESS) {
        status = metal_moe_workspace_ensure_host_array(&workspace->expert_offsets, experts);
    }
    if (status == MARMOT_SUCCESS) {
        status = metal_moe_workspace_ensure_host_array(&workspace->expert_cursor, experts);
    }
    if (status == MARMOT_SUCCESS) {
        status = metal_moe_workspace_ensure_host_array(&workspace->expert_order, experts);
    }
    const size_t route_meta_bytes = experts > 0 ? experts * sizeof(uint32_t) : sizeof(uint32_t);
    if (status == MARMOT_SUCCESS) {
        status = metal_moe_workspace_ensure_allocation(ctx, &workspace->route_counts_alloc, route_meta_bytes);
    }
    if (status == MARMOT_SUCCESS) {
        status = metal_moe_workspace_ensure_allocation(ctx, &workspace->route_offsets_alloc, route_meta_bytes);
    }
    if (status == MARMOT_SUCCESS) {
        status = metal_moe_workspace_ensure_allocation(ctx, &workspace->route_status_alloc, sizeof(uint32_t));
    }
    if (status == MARMOT_SUCCESS) {
        status = metal_moe_workspace_ensure_allocation(
            ctx, &workspace->route_summary_alloc, sizeof(metal_moe_route_summary_t)
        );
    }
    if (status == MARMOT_SUCCESS) {
        status = metal_moe_workspace_ensure_allocation(ctx, &workspace->route_indices_alloc, route_indices_bytes);
    }
    if (status == MARMOT_SUCCESS) {
        status = metal_moe_workspace_ensure_allocation(ctx, &workspace->route_experts_alloc, route_indices_bytes);
    }
    if (status == MARMOT_SUCCESS) {
        status = metal_moe_workspace_ensure_allocation(ctx, &workspace->route_weights_alloc, route_weights_bytes);
    }
    if (status == MARMOT_SUCCESS) {
        status = metal_moe_workspace_ensure_allocation(ctx, &workspace->hidden_batch_alloc, hidden_bytes);
    }
    if (status == MARMOT_SUCCESS) {
        status = metal_moe_workspace_ensure_allocation(ctx, &workspace->gate_batch_alloc, ff_bytes);
    }
    if (status == MARMOT_SUCCESS) {
        status = metal_moe_workspace_ensure_allocation(ctx, &workspace->up_batch_alloc, ff_bytes);
    }
    if (status == MARMOT_SUCCESS) {
        status = metal_moe_workspace_ensure_allocation(ctx, &workspace->fused_batch_alloc, ff_bytes);
    }
    if (status == MARMOT_SUCCESS) {
        status = metal_moe_workspace_ensure_allocation(ctx, &workspace->down_batch_alloc, hidden_bytes);
    }
    if (status != MARMOT_SUCCESS) {
        pthread_mutex_lock(&ctx->command_serial_mutex);
        workspace->in_use = false;
        workspace->release_serial = 0;
        pthread_mutex_unlock(&ctx->command_serial_mutex);
        return status;
    }

    workspace->experts_capacity = experts;
    *out_workspace = workspace;
    return MARMOT_SUCCESS;
}

static void metal_moe_workspace_release(metal_context_t *ctx, metal_moe_workspace_t *workspace) {
    if (ctx == nullptr || workspace == nullptr) {
        return;
    }

    pthread_mutex_lock(&ctx->command_serial_mutex);
    workspace->in_use = false;
    if (ctx->active_command_buffer != nil) {
        workspace->release_serial = UINT64_MAX;
    } else if (ctx->has_in_flight_work) {
        workspace->release_serial = ctx->last_submitted_command_serial;
    } else {
        workspace->release_serial = 0;
    }
    pthread_mutex_unlock(&ctx->command_serial_mutex);
}

void metal_moe_workspace_pool_destroy(metal_context_t *ctx) {
    if (ctx == nullptr) {
        return;
    }

    metal_moe_workspace_t *workspace = ctx->moe_workspaces;
    ctx->moe_workspaces = nullptr;
    while (workspace != nullptr) {
        metal_moe_workspace_t *next = workspace->next;
        free(workspace->expert_counts);
        free(workspace->expert_offsets);
        free(workspace->expert_cursor);
        free(workspace->expert_order);
        free(workspace);
        workspace = next;
    }
}

static size_t metal_moe_build_expert_order(size_t *expert_order, const size_t *expert_counts, size_t experts) {
    if (expert_order == nullptr || expert_counts == nullptr) {
        return 0;
    }

    size_t active = 0;
    for (size_t expert = 0; expert < experts; ++expert) {
        if (expert_counts[expert] == 0) {
            continue;
        }
        expert_order[active++] = expert;
    }

    for (size_t i = 1; i < active; ++i) {
        const size_t expert = expert_order[i];
        const size_t count = expert_counts[expert];
        size_t j = i;
        while (j > 0) {
            const size_t prev_expert = expert_order[j - 1];
            const size_t prev_count = expert_counts[prev_expert];
            if (prev_count > count || (prev_count == count && prev_expert < expert)) {
                break;
            }
            expert_order[j] = prev_expert;
            --j;
        }
        expert_order[j] = expert;
    }
    return active;
}

static size_t metal_moe_build_route_metadata_from_counts(
    const uint32_t *route_counts, size_t experts, size_t *expert_counts, size_t *expert_offsets, size_t *expert_order,
    size_t *out_route_count, size_t *out_max_batch
) {
    size_t prefix = 0;
    size_t max_batch = 0;
    size_t active_experts = 0;
    if (route_counts == nullptr) {
        if (out_route_count != nullptr) {
            *out_route_count = 0;
        }
        if (out_max_batch != nullptr) {
            *out_max_batch = 0;
        }
        return 0;
    }

    for (size_t expert = 0; expert < experts; ++expert) {
        const size_t count = route_counts[expert];
        if (expert_counts != nullptr) {
            expert_counts[expert] = count;
        }
        if (expert_offsets != nullptr) {
            expert_offsets[expert] = prefix;
        }
        prefix += count;
        if (count != 0) {
            active_experts++;
            if (count > max_batch) {
                max_batch = count;
            }
        }
    }
    if (out_route_count != nullptr) {
        *out_route_count = prefix;
    }
    if (out_max_batch != nullptr) {
        *out_max_batch = max_batch;
    }
    if (expert_order != nullptr && expert_counts != nullptr) {
        return metal_moe_build_expert_order(expert_order, expert_counts, experts);
    }
    return active_experts;
}

static marmot_error_t metal_moe_fill_route_experts(
    marmot_tensor_t *route_experts, const size_t *expert_counts, const size_t *expert_offsets,
    const size_t *expert_order, size_t ordered_active_experts
) {
    if (route_experts == nullptr || expert_counts == nullptr || expert_offsets == nullptr || expert_order == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Metal MoE route experts fill received invalid arguments");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    int32_t *route_expert_data = (int32_t *)route_experts->data;
    if (route_expert_data == nullptr) {
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Metal MoE route experts buffer is missing");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    const size_t routes = route_experts->shape.shape[0];
    for (size_t ordered_idx = 0; ordered_idx < ordered_active_experts; ++ordered_idx) {
        const size_t expert = expert_order[ordered_idx];
        const size_t count = expert_counts[expert];
        const size_t offset = expert_offsets[expert];
        if (expert > INT32_MAX || offset > routes || count > routes - offset) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Metal MoE route experts metadata is out of range");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        for (size_t route = 0; route < count; ++route) {
            route_expert_data[offset + route] = (int32_t)expert;
        }
    }

    return MARMOT_SUCCESS;
}

static bool metal_moe_should_use_host_routes(const marmot_moe_experts_desc_t *desc) {
    if (desc == nullptr) {
        return false;
    }
    switch (metal_moe_route_mode_override()) {
    case METAL_MOE_ROUTE_MODE_HOST:
        return true;
    case METAL_MOE_ROUTE_MODE_GPU:
        return false;
    case METAL_MOE_ROUTE_MODE_AUTO:
    default:
        if (desc->hidden_states != nullptr && desc->hidden_states->shape.ndim == 2 &&
            desc->hidden_states->shape.shape[0] <= 64) {
            return true;
        }
        break;
    }
    if (desc->hidden_states != nullptr && desc->topk_ids != nullptr && desc->topk_weights != nullptr &&
        desc->out != nullptr && desc->hidden_states->shape.ndim == 2 && desc->hidden_states->shape.shape[0] > 1 &&
        marmot_tensor_is_contiguous(desc->hidden_states) && marmot_tensor_is_block_quantized_weight(desc->gate_exps) &&
        marmot_tensor_is_block_quantized_weight(desc->up_exps) &&
        marmot_tensor_is_block_quantized_weight(desc->down_exps) &&
        desc->gate_exps->quant_kind == MARMOT_QUANT_KIND_Q4_K && desc->up_exps->quant_kind == MARMOT_QUANT_KIND_Q4_K &&
        desc->down_exps->quant_kind == MARMOT_QUANT_KIND_Q6_K &&
        desc->gate_exps->quant_layout == desc->up_exps->quant_layout &&
        desc->gate_exps->dtype == desc->up_exps->dtype && metal_moe_value_dtype_supported(desc->hidden_states->dtype) &&
        desc->topk_ids->dtype == MARMOT_DTYPE_INT32 && desc->topk_weights->dtype == desc->hidden_states->dtype &&
        desc->out->dtype == desc->hidden_states->dtype &&
        metal_moe_decode_gate_up_pipeline_name(
            desc->gate_exps->quant_kind, desc->hidden_states->dtype, desc->hidden_states->dtype
        ) != nullptr &&
        metal_moe_decode_down_pipeline_name(
            desc->down_exps->quant_kind, desc->hidden_states->dtype, desc->hidden_states->dtype
        ) != nullptr) {
        return false;
    }
    return true;
}

static bool metal_moe_grouped_decode_base_ready(
    const marmot_moe_experts_desc_t *desc, size_t tokens, size_t route_count, size_t max_batch
) {
    return desc != nullptr && tokens == 1 && max_batch == 1 && route_count > 1 && desc->hidden_states != nullptr &&
        marmot_tensor_is_contiguous(desc->hidden_states);
}

static bool metal_moe_should_use_grouped_decode(
    const marmot_moe_experts_desc_t *desc, size_t tokens, size_t route_count, size_t max_batch
) {
    const bool base_ready = metal_moe_grouped_decode_base_ready(desc, tokens, route_count, max_batch);
    switch (metal_moe_grouped_decode_mode_override()) {
    case METAL_MOE_GROUPED_DECODE_DISABLE:
        return false;
    case METAL_MOE_GROUPED_DECODE_FORCE:
        return base_ready;
    case METAL_MOE_GROUPED_DECODE_AUTO:
    default:
        return base_ready;
    }
}

static marmot_error_t metal_moe_build_routes_cpu(
    metal_context_t *ctx, const marmot_moe_experts_desc_t *desc, size_t experts, size_t *expert_counts,
    size_t *expert_offsets, size_t *expert_cursor, size_t *expert_order, marmot_tensor_t *route_indices,
    marmot_tensor_t *route_weights, size_t *out_route_count, size_t *out_max_batch, size_t *out_ordered_active_experts
) {
    if (ctx == nullptr || desc == nullptr || expert_counts == nullptr || expert_offsets == nullptr ||
        expert_cursor == nullptr || expert_order == nullptr || route_indices == nullptr || route_weights == nullptr ||
        out_route_count == nullptr || out_max_batch == nullptr || out_ordered_active_experts == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Metal MoE CPU route build requires valid arguments");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const size_t ids_bytes = marmot_tensor_size_bytes(desc->topk_ids);
    const size_t weights_bytes = marmot_tensor_size_bytes(desc->topk_weights);
    if (desc->topk_ids->needs_sync && desc->topk_ids->memory_location == MARMOT_MEMORY_DEVICE) {
        marmot_error_t sync_status =
            metal_memcpy_from_device(ctx, desc->topk_ids->data, desc->topk_ids->data, ids_bytes);
        if (sync_status != MARMOT_SUCCESS) {
            return sync_status;
        }
    }
    if (desc->topk_weights->needs_sync && desc->topk_weights->memory_location == MARMOT_MEMORY_DEVICE) {
        marmot_error_t sync_status =
            metal_memcpy_from_device(ctx, desc->topk_weights->data, desc->topk_weights->data, weights_bytes);
        if (sync_status != MARMOT_SUCCESS) {
            return sync_status;
        }
    }
    if (!metal_command_stream_wait_for_shared_read(ctx, desc->topk_ids->data, ids_bytes)) {
        metal_command_stream_flush(ctx, true);
    }
    if (!metal_command_stream_wait_for_shared_read(ctx, desc->topk_weights->data, weights_bytes)) {
        metal_command_stream_flush(ctx, true);
    }

    const size_t tokens = desc->topk_ids->shape.shape[0];
    const size_t experts_per_token = desc->topk_ids->shape.shape[1];
    const size_t topk_id_stride = desc->topk_ids->shape.strides[0];
    const size_t topk_weight_stride = desc->topk_weights->shape.strides[0];
    const marmot_int32_t *topk_ids = (const marmot_int32_t *)desc->topk_ids->data;
    const void *topk_weights = desc->topk_weights->data;

    memset(expert_counts, 0, experts * sizeof(*expert_counts));
    size_t active_route_count = 0;
    for (size_t token = 0; token < tokens; ++token) {
        for (size_t slot = 0; slot < experts_per_token; ++slot) {
            const int32_t expert_idx = topk_ids[token * topk_id_stride + slot].value;
            if (expert_idx < 0 || (size_t)expert_idx >= experts) {
                marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "MoE expert id is out of range");
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            bool duplicate = false;
            for (size_t prev = 0; prev < slot; ++prev) {
                if (topk_ids[token * topk_id_stride + prev].value == expert_idx) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate) {
                continue;
            }
            expert_counts[(size_t)expert_idx]++;
            active_route_count++;
        }
    }

    size_t prefix = 0;
    size_t max_batch = 0;
    for (size_t expert = 0; expert < experts; ++expert) {
        const size_t count = expert_counts[expert];
        expert_offsets[expert] = prefix;
        expert_cursor[expert] = prefix;
        prefix += count;
        if (count > max_batch) {
            max_batch = count;
        }
    }
    if (prefix != active_route_count) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "MoE routing workspace prefix sum mismatch");
        return MARMOT_ERROR_INVALID_OPERATION;
    }

    int32_t *route_ids = (int32_t *)route_indices->data;
    void *route_weight_data = route_weights->data;
    for (size_t token = 0; token < tokens; ++token) {
        float token_weight_norm = desc->weights_scale;
        if (desc->router_weight_policy == MARMOT_ROUTER_WEIGHT_POLICY_RENORMALIZE_SELECTED) {
            float weight_sum = 0.0f;
            for (size_t slot = 0; slot < experts_per_token; ++slot) {
                weight_sum +=
                    metal_moe_load_value(topk_weights, desc->topk_weights->dtype, token * topk_weight_stride + slot);
            }
            token_weight_norm = weight_sum > FLT_MIN ? desc->weights_scale / weight_sum : 0.0f;
        }
        for (size_t slot = 0; slot < experts_per_token; ++slot) {
            const size_t expert_idx = (size_t)topk_ids[token * topk_id_stride + slot].value;
            bool duplicate = false;
            for (size_t prev = 0; prev < slot; ++prev) {
                if ((size_t)topk_ids[token * topk_id_stride + prev].value == expert_idx) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate) {
                continue;
            }

            float combined_weight = 0.0f;
            for (size_t scan = slot; scan < experts_per_token; ++scan) {
                if ((size_t)topk_ids[token * topk_id_stride + scan].value != expert_idx) {
                    continue;
                }
                combined_weight +=
                    metal_moe_load_value(topk_weights, desc->topk_weights->dtype, token * topk_weight_stride + scan);
            }

            const size_t pos = expert_cursor[expert_idx]++;
            route_ids[pos] = (int32_t)token;
            metal_moe_store_value(route_weight_data, route_weights->dtype, pos, combined_weight * token_weight_norm);
        }
    }

    metal_residency_mark_shared_write(ctx, route_indices->data);
    metal_residency_mark_shared_write(ctx, route_weights->data);

    *out_route_count = active_route_count;
    *out_max_batch = max_batch;
    *out_ordered_active_experts = metal_moe_build_expert_order(expert_order, expert_counts, experts);
    return MARMOT_SUCCESS;
}

static marmot_error_t metal_moe_cpu_fallback(const marmot_moe_experts_desc_t *desc) {
    if (desc == nullptr || desc->hidden_states == nullptr || desc->out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "MoE descriptor is incomplete");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const marmot_context_t *src_ctx = desc->hidden_states->ctx != nullptr ? desc->hidden_states->ctx : desc->out->ctx;
    if (src_ctx == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Metal MoE execution requires tensors bound to a context");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_context_t *cpu_ctx = marmot_init(MARMOT_BACKEND_CPU);
    if (cpu_ctx == nullptr) {
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to create CPU context for Metal MoE fallback");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    size_t out_shape[MARMOT_MAX_DIMS] = {0};
    for (size_t i = 0; i < desc->out->shape.ndim; ++i) {
        out_shape[i] = desc->out->shape.shape[i];
    }

    marmot_tensor_t *cpu_out = marmot_tensor_create(cpu_ctx, out_shape, desc->out->shape.ndim, desc->out->dtype);
    if (cpu_out == nullptr) {
        marmot_destroy(cpu_ctx);
        return marmot_get_last_error();
    }

    marmot_tensor_t hidden_cpu;
    marmot_tensor_t gate_cpu;
    marmot_tensor_t up_cpu;
    marmot_tensor_t down_cpu;
    marmot_tensor_t topk_ids_cpu;
    marmot_tensor_t topk_weights_cpu;
    metal_moe_init_cpu_alias(&hidden_cpu, desc->hidden_states, cpu_ctx);
    metal_moe_init_cpu_alias(&gate_cpu, desc->gate_exps, cpu_ctx);
    metal_moe_init_cpu_alias(&up_cpu, desc->up_exps, cpu_ctx);
    metal_moe_init_cpu_alias(&down_cpu, desc->down_exps, cpu_ctx);
    metal_moe_init_cpu_alias(&topk_ids_cpu, desc->topk_ids, cpu_ctx);
    metal_moe_init_cpu_alias(&topk_weights_cpu, desc->topk_weights, cpu_ctx);

    marmot_moe_experts_desc_t cpu_desc = *desc;
    cpu_desc.hidden_states = &hidden_cpu;
    cpu_desc.gate_exps = &gate_cpu;
    cpu_desc.up_exps = &up_cpu;
    cpu_desc.down_exps = &down_cpu;
    cpu_desc.topk_ids = &topk_ids_cpu;
    cpu_desc.topk_weights = &topk_weights_cpu;
    cpu_desc.out = cpu_out;

    marmot_error_t status = cpu_moe_experts_impl(nullptr, &cpu_desc);
    if (status == MARMOT_SUCCESS) {
        const size_t out_bytes = marmot_tensor_size_bytes(cpu_out);
        status = marmot_tensor_copy_from_host_buffer(src_ctx, desc->out, cpu_out->data, out_bytes);
    }

    marmot_tensor_destroy(cpu_out);
    marmot_destroy(cpu_ctx);
    return status;
}

static marmot_error_t metal_moe_zero_output(metal_context_t *ctx, marmot_tensor_t *out) {
    if (ctx == nullptr || out == nullptr || !metal_moe_value_dtype_supported(out->dtype) || out->shape.ndim != 2) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Metal MoE zero requires a 2D FLOAT16/FLOAT32 output tensor");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const size_t rows = out->shape.shape[0];
    const size_t cols = out->shape.shape[1];
    if (rows == 0 || cols == 0) {
        return MARMOT_SUCCESS;
    }
    if (rows > UINT32_MAX || cols > UINT32_MAX || out->shape.strides[0] > UINT32_MAX) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Metal MoE output shape exceeds current kernel limits");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    const size_t bytes = marmot::metal::tensor_span_bytes(out);
    metal_tensor_buffer_t out_view = metal_buffer_acquire_view(ctx, out, out->dtype, bytes);
    if (out_view.buffer == nil) {
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Metal MoE failed to acquire output buffer");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    const char *pipeline_name = metal_moe_zero_pipeline_name(out->dtype);
    id<MTLComputePipelineState> pipeline = pipeline_name != nullptr ? metal_pipeline_get(ctx, pipeline_name) : nil;
    if (pipeline == nil) {
        [out_view.buffer release];
        marmot_set_error(MARMOT_ERROR_BACKEND_INIT_FAILED, "Metal MoE zero pipeline initialization failed");
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
    if (encoder == nil) {
        [pipeline release];
        [out_view.buffer release];
        marmot_set_error(MARMOT_ERROR_BACKEND_INIT_FAILED, "Metal MoE zero encoder acquisition failed");
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    metal_moe_zero_uniforms_t params = {
        .rows = (uint32_t)rows,
        .cols = (uint32_t)cols,
        .row_stride = (uint32_t)out->shape.strides[0],
    };
    const NSUInteger total = (NSUInteger)(rows * cols);
    [encoder setBuffer:out_view.buffer offset:out_view.offset atIndex:0];
    [encoder setBytes:&params length:sizeof(params) atIndex:1];
    MTLSize threads = metal_threads_for_elements(pipeline, total, 512);
    [encoder dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:threads];
    if (!out_view.is_private) {
        metal_command_stream_track_shared_write(ctx, out->data);
    }
    metal_command_stream_flush(ctx, false);

    [pipeline release];
    [out_view.buffer release];
    if (out_view.is_private) {
        metal_residency_mark_dirty(ctx, out, out->dtype);
    }
    return MARMOT_SUCCESS;
}

static marmot_error_t metal_moe_route_count(
    metal_context_t *ctx, const marmot_tensor_t *topk_ids, size_t experts, uint32_t *expert_counts,
    uint32_t *status_flag
) {
    if (ctx == nullptr || topk_ids == nullptr || expert_counts == nullptr || status_flag == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Metal MoE route_count requires non-null arguments");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (topk_ids->shape.ndim != 2 || topk_ids->dtype != MARMOT_DTYPE_INT32) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Metal MoE route_count requires 2D INT32 top-k ids");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const size_t tokens = topk_ids->shape.shape[0];
    const size_t experts_per_token = topk_ids->shape.shape[1];
    const size_t route_capacity = tokens * experts_per_token;
    if (route_capacity == 0) {
        return MARMOT_SUCCESS;
    }
    if (tokens > UINT32_MAX || experts_per_token > UINT32_MAX || experts > UINT32_MAX ||
        topk_ids->shape.strides[0] > UINT32_MAX || topk_ids->shape.strides[1] > UINT32_MAX) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Metal MoE route_count shape exceeds current kernel limits");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    const size_t ids_bytes = marmot::metal::tensor_span_bytes(topk_ids);
    metal_tensor_buffer_t ids_view = metal_buffer_acquire_view(ctx, topk_ids, topk_ids->dtype, ids_bytes);
    id<MTLBuffer> counts_buffer =
        metal_buffer_acquire(ctx, expert_counts, (experts > 0 ? experts : 1) * sizeof(uint32_t));
    id<MTLBuffer> status_buffer = metal_buffer_acquire(ctx, status_flag, sizeof(uint32_t));
    if (ids_view.buffer == nil || counts_buffer == nil || status_buffer == nil) {
        if (ids_view.buffer != nil) {
            [ids_view.buffer release];
        }
        if (counts_buffer != nil) {
            [counts_buffer release];
        }
        if (status_buffer != nil) {
            [status_buffer release];
        }
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Metal MoE route_count buffer acquisition failed");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    id<MTLComputePipelineState> pipeline = metal_pipeline_get(ctx, "moe_route_count_i32");
    if (pipeline == nil) {
        [ids_view.buffer release];
        [counts_buffer release];
        [status_buffer release];
        marmot_set_error(MARMOT_ERROR_BACKEND_INIT_FAILED, "Metal MoE route_count pipeline initialization failed");
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
    if (encoder == nil) {
        [pipeline release];
        [ids_view.buffer release];
        [counts_buffer release];
        [status_buffer release];
        marmot_set_error(MARMOT_ERROR_BACKEND_INIT_FAILED, "Metal MoE route_count encoder acquisition failed");
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    metal_moe_route_uniforms_t params = {
        .tokens = (uint32_t)tokens,
        .experts_per_token = (uint32_t)experts_per_token,
        .experts = (uint32_t)experts,
        .id_stride0 = (uint32_t)topk_ids->shape.strides[0],
        .id_stride1 = (uint32_t)topk_ids->shape.strides[1],
        .weight_stride0 = 0,
        .weight_stride1 = 0,
        .renormalize_selected = 0,
        .weights_scale = 0.0f,
    };

    [encoder setBuffer:ids_view.buffer offset:ids_view.offset atIndex:0];
    [encoder setBuffer:counts_buffer offset:0 atIndex:1];
    [encoder setBuffer:status_buffer offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];
    MTLSize threads = metal_threads_for_elements(pipeline, (NSUInteger)route_capacity, 512);
    [encoder dispatchThreads:MTLSizeMake((NSUInteger)route_capacity, 1, 1) threadsPerThreadgroup:threads];
    metal_command_stream_track_shared_write(ctx, expert_counts);
    metal_command_stream_track_shared_write(ctx, status_flag);
    metal_command_stream_flush(ctx, false);

    [pipeline release];
    [ids_view.buffer release];
    [counts_buffer release];
    [status_buffer release];
    return MARMOT_SUCCESS;
}

static marmot_error_t metal_moe_route_prepare(
    metal_context_t *ctx, size_t experts, const uint32_t *expert_counts, uint32_t *expert_offsets,
    metal_moe_route_summary_t *route_summary
) {
    if (ctx == nullptr || expert_counts == nullptr || expert_offsets == nullptr || route_summary == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Metal MoE route_prepare requires non-null arguments");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (experts == 0) {
        memset(route_summary, 0, sizeof(*route_summary));
        return MARMOT_SUCCESS;
    }
    if (experts > UINT32_MAX) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Metal MoE route_prepare shape exceeds current kernel limits");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    id<MTLBuffer> counts_buffer =
        metal_buffer_acquire(ctx, expert_counts, (experts > 0 ? experts : 1) * sizeof(uint32_t));
    id<MTLBuffer> offsets_buffer =
        metal_buffer_acquire(ctx, expert_offsets, (experts > 0 ? experts : 1) * sizeof(uint32_t));
    id<MTLBuffer> summary_buffer = metal_buffer_acquire(ctx, route_summary, sizeof(*route_summary));
    if (counts_buffer == nil || offsets_buffer == nil || summary_buffer == nil) {
        if (counts_buffer != nil) {
            [counts_buffer release];
        }
        if (offsets_buffer != nil) {
            [offsets_buffer release];
        }
        if (summary_buffer != nil) {
            [summary_buffer release];
        }
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Metal MoE route_prepare buffer acquisition failed");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    id<MTLComputePipelineState> pipeline = metal_pipeline_get(ctx, "moe_route_prepare_i32");
    if (pipeline == nil) {
        [counts_buffer release];
        [offsets_buffer release];
        [summary_buffer release];
        marmot_set_error(MARMOT_ERROR_BACKEND_INIT_FAILED, "Metal MoE route_prepare pipeline initialization failed");
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
    if (encoder == nil) {
        [pipeline release];
        [counts_buffer release];
        [offsets_buffer release];
        [summary_buffer release];
        marmot_set_error(MARMOT_ERROR_BACKEND_INIT_FAILED, "Metal MoE route_prepare encoder acquisition failed");
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    const uint32_t expert_count_u32 = (uint32_t)experts;
    [encoder setBuffer:counts_buffer offset:0 atIndex:0];
    [encoder setBuffer:offsets_buffer offset:0 atIndex:1];
    [encoder setBuffer:summary_buffer offset:0 atIndex:2];
    [encoder setBytes:&expert_count_u32 length:sizeof(expert_count_u32) atIndex:3];
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    metal_command_stream_track_shared_write(ctx, expert_offsets);
    metal_command_stream_track_shared_write(ctx, route_summary);

    [pipeline release];
    [counts_buffer release];
    [offsets_buffer release];
    [summary_buffer release];
    return MARMOT_SUCCESS;
}

static marmot_error_t metal_moe_route_pack(
    metal_context_t *ctx, const marmot_moe_experts_desc_t *desc, const uint32_t *expert_offsets, size_t experts,
    const marmot_tensor_t *route_indices, const marmot_tensor_t *route_weights, const marmot_tensor_t *route_experts
) {
    if (ctx == nullptr || desc == nullptr || expert_offsets == nullptr || route_indices == nullptr ||
        route_weights == nullptr || route_experts == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Metal MoE route_pack requires non-null arguments");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const size_t tokens = desc->topk_ids->shape.shape[0];
    const size_t experts_per_token = desc->topk_ids->shape.shape[1];
    const size_t route_count = tokens * experts_per_token;
    if (route_count == 0) {
        return MARMOT_SUCCESS;
    }
    if (tokens > UINT32_MAX || experts_per_token > UINT32_MAX || experts > UINT32_MAX ||
        desc->topk_ids->shape.strides[0] > UINT32_MAX || desc->topk_ids->shape.strides[1] > UINT32_MAX ||
        desc->topk_weights->shape.strides[0] > UINT32_MAX || desc->topk_weights->shape.strides[1] > UINT32_MAX) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Metal MoE route_pack shape exceeds current kernel limits");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    const size_t ids_bytes = marmot::metal::tensor_span_bytes(desc->topk_ids);
    const size_t weights_bytes = marmot::metal::tensor_span_bytes(desc->topk_weights);
    const size_t route_indices_bytes = marmot::metal::tensor_span_bytes(route_indices);
    const size_t route_weights_bytes = marmot::metal::tensor_span_bytes(route_weights);
    const size_t route_experts_bytes = marmot::metal::tensor_span_bytes(route_experts);
    metal_tensor_buffer_t ids_view = metal_buffer_acquire_view(ctx, desc->topk_ids, desc->topk_ids->dtype, ids_bytes);
    metal_tensor_buffer_t topk_weights_view =
        metal_buffer_acquire_view(ctx, desc->topk_weights, desc->topk_weights->dtype, weights_bytes);
    id<MTLBuffer> offsets_buffer =
        metal_buffer_acquire(ctx, expert_offsets, (experts > 0 ? experts : 1) * sizeof(uint32_t));
    metal_tensor_buffer_t route_indices_view =
        metal_buffer_acquire_view(ctx, route_indices, route_indices->dtype, route_indices_bytes);
    metal_tensor_buffer_t route_weights_view =
        metal_buffer_acquire_view(ctx, route_weights, route_weights->dtype, route_weights_bytes);
    metal_tensor_buffer_t route_experts_view =
        metal_buffer_acquire_view(ctx, route_experts, route_experts->dtype, route_experts_bytes);
    if (ids_view.buffer == nil || topk_weights_view.buffer == nil || offsets_buffer == nil ||
        route_indices_view.buffer == nil || route_weights_view.buffer == nil || route_experts_view.buffer == nil) {
        if (ids_view.buffer != nil) {
            [ids_view.buffer release];
        }
        if (topk_weights_view.buffer != nil) {
            [topk_weights_view.buffer release];
        }
        if (offsets_buffer != nil) {
            [offsets_buffer release];
        }
        if (route_indices_view.buffer != nil) {
            [route_indices_view.buffer release];
        }
        if (route_weights_view.buffer != nil) {
            [route_weights_view.buffer release];
        }
        if (route_experts_view.buffer != nil) {
            [route_experts_view.buffer release];
        }
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Metal MoE route_pack buffer acquisition failed");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    const char *pipeline_name = metal_moe_route_pack_pipeline_name(desc->topk_weights->dtype);
    id<MTLComputePipelineState> pipeline = pipeline_name != nullptr ? metal_pipeline_get(ctx, pipeline_name) : nil;
    if (pipeline == nil) {
        [ids_view.buffer release];
        [topk_weights_view.buffer release];
        [offsets_buffer release];
        [route_indices_view.buffer release];
        [route_weights_view.buffer release];
        [route_experts_view.buffer release];
        marmot_set_error(MARMOT_ERROR_BACKEND_INIT_FAILED, "Metal MoE route_pack pipeline initialization failed");
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
    if (encoder == nil) {
        [pipeline release];
        [ids_view.buffer release];
        [topk_weights_view.buffer release];
        [offsets_buffer release];
        [route_indices_view.buffer release];
        [route_weights_view.buffer release];
        [route_experts_view.buffer release];
        marmot_set_error(MARMOT_ERROR_BACKEND_INIT_FAILED, "Metal MoE route_pack encoder acquisition failed");
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    metal_moe_route_uniforms_t params = {
        .tokens = (uint32_t)tokens,
        .experts_per_token = (uint32_t)experts_per_token,
        .experts = (uint32_t)experts,
        .id_stride0 = (uint32_t)desc->topk_ids->shape.strides[0],
        .id_stride1 = (uint32_t)desc->topk_ids->shape.strides[1],
        .weight_stride0 = (uint32_t)desc->topk_weights->shape.strides[0],
        .weight_stride1 = (uint32_t)desc->topk_weights->shape.strides[1],
        .renormalize_selected =
            desc->router_weight_policy == MARMOT_ROUTER_WEIGHT_POLICY_RENORMALIZE_SELECTED ? 1u : 0u,
        .weights_scale = desc->weights_scale,
    };

    [encoder setBuffer:ids_view.buffer offset:ids_view.offset atIndex:0];
    [encoder setBuffer:topk_weights_view.buffer offset:topk_weights_view.offset atIndex:1];
    [encoder setBuffer:offsets_buffer offset:0 atIndex:2];
    [encoder setBuffer:route_indices_view.buffer offset:route_indices_view.offset atIndex:3];
    [encoder setBuffer:route_weights_view.buffer offset:route_weights_view.offset atIndex:4];
    [encoder setBuffer:route_experts_view.buffer offset:route_experts_view.offset atIndex:5];
    [encoder setBytes:&params length:sizeof(params) atIndex:6];
    MTLSize threads = metal_threads_for_elements(pipeline, (NSUInteger)experts, 256);
    [encoder dispatchThreads:MTLSizeMake((NSUInteger)experts, 1, 1) threadsPerThreadgroup:threads];
    if (!route_indices_view.is_private) {
        metal_command_stream_track_shared_write(ctx, route_indices->data);
        metal_moe_mark_device_write((marmot_tensor_t *)route_indices);
    }
    if (!route_weights_view.is_private) {
        metal_command_stream_track_shared_write(ctx, route_weights->data);
        metal_moe_mark_device_write((marmot_tensor_t *)route_weights);
    }
    if (!route_experts_view.is_private) {
        metal_command_stream_track_shared_write(ctx, route_experts->data);
        metal_moe_mark_device_write((marmot_tensor_t *)route_experts);
    }

    [pipeline release];
    [ids_view.buffer release];
    [topk_weights_view.buffer release];
    [offsets_buffer release];
    [route_indices_view.buffer release];
    [route_weights_view.buffer release];
    [route_experts_view.buffer release];
    if (route_indices_view.is_private) {
        metal_residency_mark_dirty(ctx, route_indices, route_indices->dtype);
    }
    if (route_weights_view.is_private) {
        metal_residency_mark_dirty(ctx, route_weights, route_weights->dtype);
    }
    if (route_experts_view.is_private) {
        metal_residency_mark_dirty(ctx, route_experts, route_experts->dtype);
    }
    return MARMOT_SUCCESS;
}

static marmot_error_t metal_moe_scatter_add(
    metal_context_t *ctx, const marmot_tensor_t *src, const marmot_tensor_t *indices, const marmot_tensor_t *weights,
    marmot_tensor_t *out, bool use_atomic = false
) {
    if (ctx == nullptr || src == nullptr || indices == nullptr || weights == nullptr || out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Metal MoE scatter_add requires non-null tensors");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (!metal_moe_value_dtype_supported(src->dtype) || weights->dtype != src->dtype || out->dtype != src->dtype ||
        indices->dtype != MARMOT_DTYPE_INT32) {
        marmot_set_error(
            MARMOT_ERROR_UNSUPPORTED_DTYPE,
            "Metal MoE scatter_add requires FLOAT16/FLOAT32 data with matching weights/output and INT32 indices"
        );
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (src->shape.ndim != 2 || out->shape.ndim != 2 || indices->shape.ndim != 1 || weights->shape.ndim != 1) {
        marmot_set_error(
            MARMOT_ERROR_DIMENSION_MISMATCH, "Metal MoE scatter_add expects 2D data tensors and 1D routes"
        );
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    const size_t count = src->shape.shape[0];
    const size_t hidden = src->shape.shape[1];
    const size_t rows = out->shape.shape[0];
    if (indices->shape.shape[0] != count || weights->shape.shape[0] != count || out->shape.shape[1] != hidden) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Metal MoE scatter_add shapes do not match");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (count == 0 || hidden == 0) {
        return MARMOT_SUCCESS;
    }
    if (count > UINT32_MAX || hidden > UINT32_MAX || rows > UINT32_MAX || src->shape.strides[0] > UINT32_MAX ||
        out->shape.strides[0] > UINT32_MAX || indices->shape.strides[0] > UINT32_MAX ||
        weights->shape.strides[0] > UINT32_MAX) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Metal MoE scatter_add shape exceeds current kernel limits");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    const size_t src_bytes = marmot_tensor_size_bytes(src);
    const size_t indices_bytes = marmot_tensor_size_bytes(indices);
    const size_t weights_bytes = marmot_tensor_size_bytes(weights);
    const size_t out_bytes = marmot::metal::tensor_span_bytes(out);
    metal_tensor_buffer_t src_view = metal_buffer_acquire_view(ctx, src, src->dtype, src_bytes);
    id<MTLBuffer> indices_buffer = metal_residency_acquire_existing(ctx, indices, indices->dtype);
    if (indices_buffer == nil) {
        indices_buffer = metal_buffer_acquire(ctx, indices->data, indices_bytes);
    }
    id<MTLBuffer> weights_buffer = metal_residency_acquire_existing(ctx, weights, weights->dtype);
    if (weights_buffer == nil) {
        weights_buffer = metal_buffer_acquire(ctx, weights->data, weights_bytes);
    }
    metal_tensor_buffer_t out_view = metal_buffer_acquire_view(ctx, out, out->dtype, out_bytes);
    if (src_view.buffer == nil || indices_buffer == nil || weights_buffer == nil || out_view.buffer == nil) {
        if (src_view.buffer != nil) {
            [src_view.buffer release];
        }
        if (indices_buffer != nil) {
            [indices_buffer release];
        }
        if (weights_buffer != nil) {
            [weights_buffer release];
        }
        if (out_view.buffer != nil) {
            [out_view.buffer release];
        }
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Metal MoE scatter_add buffer acquisition failed");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    const char *pipeline_name = nullptr;
    if (use_atomic && src->dtype == MARMOT_DTYPE_FLOAT32) {
        pipeline_name = "moe_scatter_add_atomic_f32";
    } else {
        pipeline_name = metal_moe_scatter_pipeline_name(src->dtype);
    }
    id<MTLComputePipelineState> pipeline = pipeline_name != nullptr ? metal_pipeline_get(ctx, pipeline_name) : nil;
    if (pipeline == nil) {
        [src_view.buffer release];
        [indices_buffer release];
        [weights_buffer release];
        [out_view.buffer release];
        marmot_set_error(MARMOT_ERROR_BACKEND_INIT_FAILED, "Metal MoE scatter_add pipeline initialization failed");
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
    if (encoder == nil) {
        [pipeline release];
        [src_view.buffer release];
        [indices_buffer release];
        [weights_buffer release];
        [out_view.buffer release];
        marmot_set_error(MARMOT_ERROR_BACKEND_INIT_FAILED, "Metal MoE scatter_add encoder acquisition failed");
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    metal_moe_scatter_uniforms_t params = {
        .count = (uint32_t)count,
        .hidden = (uint32_t)hidden,
        .output_rows = (uint32_t)rows,
        .src_stride = (uint32_t)src->shape.strides[0],
        .out_stride = (uint32_t)out->shape.strides[0],
        .index_stride = (uint32_t)indices->shape.strides[0],
        .weight_stride = (uint32_t)weights->shape.strides[0],
    };
    const NSUInteger total = (NSUInteger)(count * hidden);
    [encoder setBuffer:src_view.buffer offset:src_view.offset atIndex:0];
    [encoder setBuffer:indices_buffer offset:0 atIndex:1];
    [encoder setBuffer:weights_buffer offset:0 atIndex:2];
    [encoder setBuffer:out_view.buffer offset:out_view.offset atIndex:3];
    [encoder setBytes:&params length:sizeof(params) atIndex:4];
    MTLSize threads = metal_threads_for_elements(pipeline, total, 512);
    [encoder dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:threads];
    if (!out_view.is_private) {
        metal_command_stream_track_shared_write(ctx, out->data);
    }
    metal_command_stream_flush(ctx, false);

    [pipeline release];
    [src_view.buffer release];
    [indices_buffer release];
    [weights_buffer release];
    [out_view.buffer release];
    if (out_view.is_private) {
        metal_residency_mark_dirty(ctx, out, out->dtype);
    }
    return MARMOT_SUCCESS;
}

static bool metal_moe_can_grouped_decode_gate_up(
    const marmot_moe_experts_desc_t *desc, const marmot_tensor_t *input, const marmot_tensor_t *route_experts,
    const marmot_tensor_t *gate_out, const marmot_tensor_t *up_out
) {
    if (desc == nullptr || route_experts == nullptr || gate_out == nullptr || up_out == nullptr || input == nullptr ||
        desc->gate_exps == nullptr || desc->up_exps == nullptr) {
        return false;
    }
    if (!marmot_tensor_is_block_quantized_weight(desc->gate_exps) ||
        !marmot_tensor_is_block_quantized_weight(desc->up_exps)) {
        return false;
    }
    if (desc->gate_exps->quant_kind != desc->up_exps->quant_kind ||
        desc->gate_exps->quant_layout != desc->up_exps->quant_layout ||
        desc->gate_exps->dtype != desc->up_exps->dtype) {
        return false;
    }
    if (!metal_moe_value_dtype_supported(input->dtype) || gate_out->dtype != input->dtype ||
        up_out->dtype != input->dtype || route_experts->dtype != MARMOT_DTYPE_INT32) {
        return false;
    }
    if (input->shape.ndim != 2 || route_experts->shape.ndim != 1 || gate_out->shape.ndim != 2 ||
        up_out->shape.ndim != 2 || desc->gate_exps->shape.ndim != 3 || desc->up_exps->shape.ndim != 3) {
        return false;
    }
    if ((input->shape.shape[0] != 1 && input->shape.shape[0] != route_experts->shape.shape[0]) ||
        !marmot_tensor_is_contiguous(input) || !marmot_tensor_is_contiguous(route_experts) ||
        !marmot_tensor_is_contiguous(gate_out) || !marmot_tensor_is_contiguous(up_out)) {
        return false;
    }
    if (gate_out->shape.shape[0] != route_experts->shape.shape[0] ||
        up_out->shape.shape[0] != route_experts->shape.shape[0] ||
        gate_out->shape.shape[1] != desc->gate_exps->shape.shape[1] ||
        up_out->shape.shape[1] != desc->up_exps->shape.shape[1] ||
        desc->gate_exps->shape.shape[0] != input->shape.shape[1] ||
        desc->up_exps->shape.shape[0] != input->shape.shape[1] ||
        desc->gate_exps->shape.shape[2] != desc->up_exps->shape.shape[2]) {
        return false;
    }
    return metal_moe_decode_gate_up_pipeline_name(desc->gate_exps->quant_kind, input->dtype, gate_out->dtype) !=
        nullptr;
}

static marmot_error_t metal_moe_decode_gate_up_grouped(
    metal_context_t *ctx, const marmot_tensor_t *input, const marmot_tensor_t *gate_exps,
    const marmot_tensor_t *up_exps, const marmot_tensor_t *route_experts, marmot_tensor_t *gate_out,
    marmot_tensor_t *up_out
) {
    if (ctx == nullptr || input == nullptr || gate_exps == nullptr || up_exps == nullptr || route_experts == nullptr ||
        gate_out == nullptr || up_out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Metal grouped MoE decode gate/up requires valid tensors");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const size_t routes = route_experts->shape.shape[0];
    const size_t input_cols = input->shape.shape[1];
    const size_t output_cols = gate_out->shape.shape[1];
    if (routes == 0 || input_cols == 0 || output_cols == 0) {
        return MARMOT_SUCCESS;
    }
    if (routes > UINT32_MAX || input_cols > UINT32_MAX || output_cols > UINT32_MAX ||
        input->shape.strides[0] > UINT32_MAX || gate_out->shape.strides[0] > UINT32_MAX ||
        up_out->shape.strides[0] > UINT32_MAX) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Metal grouped MoE decode gate/up shape exceeds kernel limits");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    const marmot_quant_kind_traits_t *traits = marmot_get_quant_kind_traits(gate_exps->quant_kind);
    if (traits == nullptr || traits->block_values == 0) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Metal grouped MoE decode gate/up requires block quant weights");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    const size_t weight_blocks = (input_cols + traits->block_values - 1u) / traits->block_values;
    if (weight_blocks == 0 || weight_blocks > UINT32_MAX) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Metal grouped MoE decode gate/up weight blocks overflow");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    const char *pipeline_name =
        metal_moe_decode_gate_up_pipeline_name(gate_exps->quant_kind, input->dtype, gate_out->dtype);
    id<MTLComputePipelineState> pipeline = pipeline_name != nullptr ? metal_pipeline_get(ctx, pipeline_name) : nil;
    if (pipeline == nil) {
        marmot_set_error(
            MARMOT_ERROR_BACKEND_INIT_FAILED, "Metal grouped MoE decode gate/up pipeline initialization failed"
        );
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    const size_t gate_weight_bytes = marmot_tensor_quant_storage_bytes(gate_exps);
    const size_t up_weight_bytes = marmot_tensor_quant_storage_bytes(up_exps);
    const size_t input_bytes = marmot_tensor_size_bytes(input);
    const size_t route_experts_bytes = marmot_tensor_size_bytes(route_experts);
    const size_t gate_out_bytes = marmot::metal::tensor_span_bytes(gate_out);
    const size_t up_out_bytes = marmot::metal::tensor_span_bytes(up_out);

    const size_t gate_weight_span = gate_weight_bytes != 0 ? gate_weight_bytes : marmot_tensor_size_bytes(gate_exps);
    const size_t up_weight_span = up_weight_bytes != 0 ? up_weight_bytes : marmot_tensor_size_bytes(up_exps);
    metal_tensor_buffer_t gate_weight_view =
        metal_buffer_acquire_view(ctx, gate_exps, gate_exps->dtype, gate_weight_span);
    metal_tensor_buffer_t up_weight_view = metal_buffer_acquire_view(ctx, up_exps, up_exps->dtype, up_weight_span);
    metal_tensor_buffer_t input_view = metal_buffer_acquire_view(ctx, input, input->dtype, input_bytes);
    metal_tensor_buffer_t route_experts_view =
        metal_buffer_acquire_view(ctx, route_experts, route_experts->dtype, route_experts_bytes);
    metal_tensor_buffer_t gate_out_view = metal_buffer_acquire_view(ctx, gate_out, gate_out->dtype, gate_out_bytes);
    metal_tensor_buffer_t up_out_view = metal_buffer_acquire_view(ctx, up_out, up_out->dtype, up_out_bytes);

    if (gate_weight_view.buffer == nil || up_weight_view.buffer == nil || input_view.buffer == nil ||
        route_experts_view.buffer == nil || gate_out_view.buffer == nil || up_out_view.buffer == nil) {
        if (gate_weight_view.buffer != nil) {
            [gate_weight_view.buffer release];
        }
        if (up_weight_view.buffer != nil) {
            [up_weight_view.buffer release];
        }
        if (input_view.buffer != nil) {
            [input_view.buffer release];
        }
        if (route_experts_view.buffer != nil) {
            [route_experts_view.buffer release];
        }
        if (gate_out_view.buffer != nil) {
            [gate_out_view.buffer release];
        }
        if (up_out_view.buffer != nil) {
            [up_out_view.buffer release];
        }
        [pipeline release];
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Metal grouped MoE decode gate/up buffer acquisition failed");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
    if (encoder == nil) {
        [gate_weight_view.buffer release];
        [up_weight_view.buffer release];
        [input_view.buffer release];
        [route_experts_view.buffer release];
        [gate_out_view.buffer release];
        [up_out_view.buffer release];
        [pipeline release];
        marmot_set_error(
            MARMOT_ERROR_BACKEND_INIT_FAILED, "Metal grouped MoE decode gate/up encoder acquisition failed"
        );
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    metal_moe_decode_gate_up_uniforms_t params = {
        .routes = (uint32_t)routes,
        .input_cols = (uint32_t)input_cols,
        .output_cols = (uint32_t)output_cols,
        .input_stride = (uint32_t)input->shape.strides[0],
        .output_stride = (uint32_t)gate_out->shape.strides[0],
        .weight_blocks = (uint32_t)weight_blocks,
        .broadcast_input = input->shape.shape[0] == 1 ? 1u : 0u,
    };

    [encoder setBuffer:gate_weight_view.buffer offset:gate_weight_view.offset atIndex:0];
    [encoder setBuffer:up_weight_view.buffer offset:up_weight_view.offset atIndex:1];
    [encoder setBuffer:input_view.buffer offset:input_view.offset atIndex:2];
    [encoder setBuffer:route_experts_view.buffer offset:route_experts_view.offset atIndex:3];
    [encoder setBuffer:gate_out_view.buffer offset:gate_out_view.offset atIndex:4];
    [encoder setBuffer:up_out_view.buffer offset:up_out_view.offset atIndex:5];
    [encoder setBytes:&params length:sizeof(params) atIndex:6];
    [encoder dispatchThreadgroups:MTLSizeMake((output_cols + 3u) / 4u, routes, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    if (!gate_out_view.is_private) {
        metal_command_stream_track_shared_write(ctx, gate_out->data);
    }
    if (!up_out_view.is_private) {
        metal_command_stream_track_shared_write(ctx, up_out->data);
    }
    metal_command_stream_flush(ctx, false);

    [gate_weight_view.buffer release];
    [up_weight_view.buffer release];
    [input_view.buffer release];
    [route_experts_view.buffer release];
    [gate_out_view.buffer release];
    [up_out_view.buffer release];
    [pipeline release];
    if (gate_out_view.is_private) {
        metal_residency_mark_dirty(ctx, gate_out, gate_out->dtype);
    }
    if (up_out_view.is_private) {
        metal_residency_mark_dirty(ctx, up_out, up_out->dtype);
    }
    return MARMOT_SUCCESS;
}

static bool metal_moe_can_grouped_decode_down(
    const marmot_moe_experts_desc_t *desc, const marmot_tensor_t *fused_batch, const marmot_tensor_t *route_experts,
    const marmot_tensor_t *route_weights
) {
    if (desc == nullptr || fused_batch == nullptr || route_experts == nullptr || route_weights == nullptr) {
        return false;
    }
    if (!marmot_tensor_is_block_quantized_weight(desc->down_exps) ||
        desc->down_exps->quant_kind != MARMOT_QUANT_KIND_Q6_K) {
        return false;
    }
    if (!metal_moe_value_dtype_supported(fused_batch->dtype) || desc->out->dtype != fused_batch->dtype ||
        route_weights->dtype != fused_batch->dtype || route_experts->dtype != MARMOT_DTYPE_INT32) {
        return false;
    }
    if (route_experts->shape.shape[0] < 4) {
        return false;
    }
    if (fused_batch->shape.ndim != 2 || route_experts->shape.ndim != 1 || route_weights->shape.ndim != 1 ||
        desc->out->shape.ndim != 2 || desc->down_exps->shape.ndim != 3) {
        return false;
    }
    if (!marmot_tensor_is_contiguous(fused_batch) || !marmot_tensor_is_contiguous(route_experts) ||
        !marmot_tensor_is_contiguous(route_weights) || !marmot_tensor_is_contiguous(desc->out)) {
        return false;
    }
    if (desc->out->shape.shape[0] != 1 || desc->down_exps->shape.shape[1] != desc->out->shape.shape[1] ||
        desc->down_exps->shape.shape[0] != fused_batch->shape.shape[1]) {
        return false;
    }
    if (route_experts->shape.shape[0] != fused_batch->shape.shape[0] ||
        route_weights->shape.shape[0] != fused_batch->shape.shape[0]) {
        return false;
    }
    return metal_moe_decode_down_pipeline_name(desc->down_exps->quant_kind, fused_batch->dtype, desc->out->dtype) !=
        nullptr;
}

static marmot_error_t metal_moe_decode_down_grouped(
    metal_context_t *ctx, const marmot_tensor_t *fused_batch, const marmot_tensor_t *down_exps,
    const marmot_tensor_t *route_experts, const marmot_tensor_t *route_weights, marmot_tensor_t *out
) {
    if (ctx == nullptr || fused_batch == nullptr || down_exps == nullptr || route_experts == nullptr ||
        route_weights == nullptr || out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Metal grouped MoE decode down requires valid tensors");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const size_t routes = fused_batch->shape.shape[0];
    const size_t input_cols = fused_batch->shape.shape[1];
    const size_t output_cols = out->shape.shape[1];
    if (routes == 0 || input_cols == 0 || output_cols == 0) {
        return MARMOT_SUCCESS;
    }
    if (routes > UINT32_MAX || input_cols > UINT32_MAX || output_cols > UINT32_MAX ||
        fused_batch->shape.strides[0] > UINT32_MAX) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Metal grouped MoE decode down shape exceeds kernel limits");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    const marmot_quant_kind_traits_t *traits = marmot_get_quant_kind_traits(down_exps->quant_kind);
    if (traits == nullptr || traits->block_values == 0) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Metal grouped MoE decode down requires block quant weights");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    const size_t weight_blocks = (input_cols + traits->block_values - 1u) / traits->block_values;
    if (weight_blocks == 0 || weight_blocks > UINT32_MAX) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Metal grouped MoE decode down weight blocks overflow");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    const char *pipeline_name =
        metal_moe_decode_down_pipeline_name(down_exps->quant_kind, fused_batch->dtype, out->dtype);
    id<MTLComputePipelineState> pipeline = pipeline_name != nullptr ? metal_pipeline_get(ctx, pipeline_name) : nil;
    if (pipeline == nil) {
        marmot_set_error(
            MARMOT_ERROR_BACKEND_INIT_FAILED, "Metal grouped MoE decode down pipeline initialization failed"
        );
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    const size_t weight_bytes = marmot_tensor_quant_storage_bytes(down_exps);
    const size_t weight_span = weight_bytes != 0 ? weight_bytes : marmot_tensor_size_bytes(down_exps);
    const size_t input_bytes = marmot_tensor_size_bytes(fused_batch);
    const size_t route_experts_bytes = marmot_tensor_size_bytes(route_experts);
    const size_t route_weights_bytes = marmot_tensor_size_bytes(route_weights);
    const size_t out_bytes = marmot::metal::tensor_span_bytes(out);
    metal_tensor_buffer_t weight_view = metal_buffer_acquire_view(ctx, down_exps, down_exps->dtype, weight_span);
    metal_tensor_buffer_t input_view = metal_buffer_acquire_view(ctx, fused_batch, fused_batch->dtype, input_bytes);
    metal_tensor_buffer_t route_experts_view =
        metal_buffer_acquire_view(ctx, route_experts, route_experts->dtype, route_experts_bytes);
    metal_tensor_buffer_t route_weights_view =
        metal_buffer_acquire_view(ctx, route_weights, route_weights->dtype, route_weights_bytes);
    metal_tensor_buffer_t out_view = metal_buffer_acquire_view(ctx, out, out->dtype, out_bytes);
    if (weight_view.buffer == nil || input_view.buffer == nil || route_experts_view.buffer == nil ||
        route_weights_view.buffer == nil || out_view.buffer == nil) {
        if (weight_view.buffer != nil) {
            [weight_view.buffer release];
        }
        if (input_view.buffer != nil) {
            [input_view.buffer release];
        }
        if (route_experts_view.buffer != nil) {
            [route_experts_view.buffer release];
        }
        if (route_weights_view.buffer != nil) {
            [route_weights_view.buffer release];
        }
        if (out_view.buffer != nil) {
            [out_view.buffer release];
        }
        [pipeline release];
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Metal grouped MoE decode down buffer acquisition failed");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
    if (encoder == nil) {
        [weight_view.buffer release];
        [input_view.buffer release];
        [route_experts_view.buffer release];
        [route_weights_view.buffer release];
        [out_view.buffer release];
        [pipeline release];
        marmot_set_error(MARMOT_ERROR_BACKEND_INIT_FAILED, "Metal grouped MoE decode down encoder acquisition failed");
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    metal_moe_decode_down_uniforms_t params = {
        .routes = (uint32_t)routes,
        .input_cols = (uint32_t)input_cols,
        .output_cols = (uint32_t)output_cols,
        .input_stride = (uint32_t)fused_batch->shape.strides[0],
        .output_stride = (uint32_t)out->shape.strides[0],
        .weight_blocks = (uint32_t)weight_blocks,
        .activation = 0,
    };

    [encoder setBuffer:weight_view.buffer offset:weight_view.offset atIndex:0];
    [encoder setBuffer:input_view.buffer offset:input_view.offset atIndex:1];
    [encoder setBuffer:route_experts_view.buffer offset:route_experts_view.offset atIndex:2];
    [encoder setBuffer:route_weights_view.buffer offset:route_weights_view.offset atIndex:3];
    [encoder setBuffer:out_view.buffer offset:out_view.offset atIndex:4];
    [encoder setBytes:&params length:sizeof(params) atIndex:5];
    [encoder dispatchThreadgroups:MTLSizeMake((output_cols + 3u) / 4u, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    if (!out_view.is_private) {
        metal_command_stream_track_shared_write(ctx, out->data);
    }
    metal_command_stream_flush(ctx, false);

    [weight_view.buffer release];
    [input_view.buffer release];
    [route_experts_view.buffer release];
    [route_weights_view.buffer release];
    [out_view.buffer release];
    [pipeline release];
    if (out_view.is_private) {
        metal_residency_mark_dirty(ctx, out, out->dtype);
    }
    return MARMOT_SUCCESS;
}

static marmot_error_t metal_moe_decode_down_glu_grouped(
    metal_context_t *ctx, const marmot_tensor_t *gate_batch, const marmot_tensor_t *up_batch,
    const marmot_tensor_t *down_exps, const marmot_tensor_t *route_experts, const marmot_tensor_t *route_weights,
    marmot_ffn_type_t ffn_type, marmot_tensor_t *out
) {
    if (ctx == nullptr || gate_batch == nullptr || up_batch == nullptr || down_exps == nullptr ||
        route_experts == nullptr || route_weights == nullptr || out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Metal grouped MoE decode GLU down requires valid tensors");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (gate_batch->shape.ndim != 2 || up_batch->shape.ndim != 2 ||
        gate_batch->shape.shape[0] != up_batch->shape.shape[0] ||
        gate_batch->shape.shape[1] != up_batch->shape.shape[1] || gate_batch->dtype != up_batch->dtype ||
        gate_batch->shape.strides[0] != up_batch->shape.strides[0]) {
        marmot_set_error(
            MARMOT_ERROR_DIMENSION_MISMATCH, "Metal grouped MoE decode GLU down requires matching gate/up tensors"
        );
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (ffn_type != MARMOT_FFN_SWIGLU && ffn_type != MARMOT_FFN_GEGLU) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Metal grouped MoE decode GLU down requires SwiGLU or GeGLU");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const size_t routes = gate_batch->shape.shape[0];
    const size_t input_cols = gate_batch->shape.shape[1];
    const size_t output_cols = out->shape.shape[1];
    if (routes == 0 || input_cols == 0 || output_cols == 0) {
        return MARMOT_SUCCESS;
    }
    if (routes > UINT32_MAX || input_cols > UINT32_MAX || output_cols > UINT32_MAX ||
        gate_batch->shape.strides[0] > UINT32_MAX) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Metal grouped MoE decode GLU down shape exceeds kernel limits");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    const marmot_quant_kind_traits_t *traits = marmot_get_quant_kind_traits(down_exps->quant_kind);
    if (traits == nullptr || traits->block_values == 0) {
        marmot_set_error(
            MARMOT_ERROR_NOT_IMPLEMENTED, "Metal grouped MoE decode GLU down requires block quant weights"
        );
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    const size_t weight_blocks = (input_cols + traits->block_values - 1u) / traits->block_values;
    if (weight_blocks == 0 || weight_blocks > UINT32_MAX) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Metal grouped MoE decode GLU down weight blocks overflow");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    const char *pipeline_name =
        metal_moe_decode_glu_down_pipeline_name(down_exps->quant_kind, gate_batch->dtype, out->dtype);
    id<MTLComputePipelineState> pipeline = pipeline_name != nullptr ? metal_pipeline_get(ctx, pipeline_name) : nil;
    if (pipeline == nil) {
        marmot_set_error(
            MARMOT_ERROR_BACKEND_INIT_FAILED, "Metal grouped MoE decode GLU down pipeline initialization failed"
        );
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    const size_t weight_bytes = marmot_tensor_quant_storage_bytes(down_exps);
    const size_t weight_span = weight_bytes != 0 ? weight_bytes : marmot_tensor_size_bytes(down_exps);
    const size_t gate_bytes = marmot_tensor_size_bytes(gate_batch);
    const size_t up_bytes = marmot_tensor_size_bytes(up_batch);
    const size_t route_experts_bytes = marmot_tensor_size_bytes(route_experts);
    const size_t route_weights_bytes = marmot_tensor_size_bytes(route_weights);
    const size_t out_bytes = marmot::metal::tensor_span_bytes(out);
    metal_tensor_buffer_t weight_view = metal_buffer_acquire_view(ctx, down_exps, down_exps->dtype, weight_span);
    metal_tensor_buffer_t gate_view = metal_buffer_acquire_view(ctx, gate_batch, gate_batch->dtype, gate_bytes);
    metal_tensor_buffer_t up_view = metal_buffer_acquire_view(ctx, up_batch, up_batch->dtype, up_bytes);
    metal_tensor_buffer_t route_experts_view =
        metal_buffer_acquire_view(ctx, route_experts, route_experts->dtype, route_experts_bytes);
    metal_tensor_buffer_t route_weights_view =
        metal_buffer_acquire_view(ctx, route_weights, route_weights->dtype, route_weights_bytes);
    metal_tensor_buffer_t out_view = metal_buffer_acquire_view(ctx, out, out->dtype, out_bytes);
    if (weight_view.buffer == nil || gate_view.buffer == nil || up_view.buffer == nil ||
        route_experts_view.buffer == nil || route_weights_view.buffer == nil || out_view.buffer == nil) {
        if (weight_view.buffer != nil) {
            [weight_view.buffer release];
        }
        if (gate_view.buffer != nil) {
            [gate_view.buffer release];
        }
        if (up_view.buffer != nil) {
            [up_view.buffer release];
        }
        if (route_experts_view.buffer != nil) {
            [route_experts_view.buffer release];
        }
        if (route_weights_view.buffer != nil) {
            [route_weights_view.buffer release];
        }
        if (out_view.buffer != nil) {
            [out_view.buffer release];
        }
        [pipeline release];
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Metal grouped MoE decode GLU down buffer acquisition failed");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
    if (encoder == nil) {
        [weight_view.buffer release];
        [gate_view.buffer release];
        [up_view.buffer release];
        [route_experts_view.buffer release];
        [route_weights_view.buffer release];
        [out_view.buffer release];
        [pipeline release];
        marmot_set_error(
            MARMOT_ERROR_BACKEND_INIT_FAILED, "Metal grouped MoE decode GLU down encoder acquisition failed"
        );
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    metal_moe_decode_down_uniforms_t params = {
        .routes = (uint32_t)routes,
        .input_cols = (uint32_t)input_cols,
        .output_cols = (uint32_t)output_cols,
        .input_stride = (uint32_t)gate_batch->shape.strides[0],
        .output_stride = (uint32_t)out->shape.strides[0],
        .weight_blocks = (uint32_t)weight_blocks,
        .activation =
            (uint32_t)(ffn_type == MARMOT_FFN_GEGLU ? MARMOT_DEVICE_BINARY_GEGLU : MARMOT_DEVICE_BINARY_SWIGLU),
    };

    [encoder setBuffer:weight_view.buffer offset:weight_view.offset atIndex:0];
    [encoder setBuffer:gate_view.buffer offset:gate_view.offset atIndex:1];
    [encoder setBuffer:up_view.buffer offset:up_view.offset atIndex:2];
    [encoder setBuffer:route_experts_view.buffer offset:route_experts_view.offset atIndex:3];
    [encoder setBuffer:route_weights_view.buffer offset:route_weights_view.offset atIndex:4];
    [encoder setBuffer:out_view.buffer offset:out_view.offset atIndex:5];
    [encoder setBytes:&params length:sizeof(params) atIndex:6];
    [encoder dispatchThreadgroups:MTLSizeMake((output_cols + 3u) / 4u, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    if (!out_view.is_private) {
        metal_command_stream_track_shared_write(ctx, out->data);
    }
    [weight_view.buffer release];
    [gate_view.buffer release];
    [up_view.buffer release];
    [route_experts_view.buffer release];
    [route_weights_view.buffer release];
    [out_view.buffer release];
    [pipeline release];
    if (out_view.is_private) {
        metal_residency_mark_dirty(ctx, out, out->dtype);
    }
    return MARMOT_SUCCESS;
}

typedef struct {
    uint32_t output_cols;
    uint32_t input_cols;
    uint32_t input_stride;
    uint32_t output_stride;
    uint32_t weight_blocks;
    uint32_t total_experts;
    uint32_t activation;
} metal_moe_expert_batch_uniforms_t;

static bool metal_moe_can_expert_batch_prefill(const marmot_moe_experts_desc_t *desc, size_t tokens) {
    if (desc == nullptr || tokens <= 1) {
        return false;
    }
    if (!marmot_tensor_is_block_quantized_weight(desc->gate_exps) ||
        !marmot_tensor_is_block_quantized_weight(desc->up_exps) ||
        !marmot_tensor_is_block_quantized_weight(desc->down_exps)) {
        return false;
    }
    if (desc->gate_exps->quant_kind != MARMOT_QUANT_KIND_Q4_K || desc->up_exps->quant_kind != MARMOT_QUANT_KIND_Q4_K ||
        desc->down_exps->quant_kind != MARMOT_QUANT_KIND_Q6_K) {
        return false;
    }
    if (!metal_moe_value_dtype_supported(desc->hidden_states->dtype) ||
        desc->out->dtype != desc->hidden_states->dtype) {
        return false;
    }
    switch (metal_moe_expert_batch_mode_override()) {
    case METAL_MOE_EXPERT_BATCH_DISABLE:
        return false;
    case METAL_MOE_EXPERT_BATCH_FORCE:
    case METAL_MOE_EXPERT_BATCH_AUTO:
    default:
        break;
    }
    if (metal_moe_expert_batch_q4_k_pipeline_name(desc->hidden_states->dtype, desc->hidden_states->dtype) == nullptr) {
        return false;
    }
    if (metal_moe_expert_batch_glu_q6_k_pipeline_name(desc->hidden_states->dtype, desc->hidden_states->dtype) ==
        nullptr) {
        return false;
    }
    return true;
}

static marmot_error_t metal_moe_expert_batch_matmul(
    metal_context_t *ctx, const marmot_tensor_t *weight_exps, const marmot_tensor_t *input_batch,
    marmot_tensor_t *output_batch, const uint32_t *route_counts_gpu, const uint32_t *route_offsets_gpu,
    size_t total_experts, size_t max_batch, marmot_quant_kind_t quant_kind, const uint32_t *active_expert_ids,
    size_t active_count
) {
    if (ctx == nullptr || weight_exps == nullptr || input_batch == nullptr || output_batch == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Metal expert-batched MoE matmul requires valid tensors");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const size_t output_cols = output_batch->shape.shape[1];
    const size_t input_cols = input_batch->shape.shape[1];
    if (output_cols == 0 || input_cols == 0 || max_batch == 0) {
        return MARMOT_SUCCESS;
    }

    const marmot_quant_kind_traits_t *traits = marmot_get_quant_kind_traits(quant_kind);
    if (traits == nullptr || traits->block_values == 0) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Metal expert-batched MoE matmul requires block quant weights");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    const size_t weight_blocks = (input_cols + traits->block_values - 1u) / traits->block_values;

    const char *pipeline_name = quant_kind == MARMOT_QUANT_KIND_Q4_K
        ? metal_moe_expert_batch_q4_k_pipeline_name(input_batch->dtype, output_batch->dtype)
        : metal_moe_expert_batch_q6_k_pipeline_name(input_batch->dtype, output_batch->dtype);
    id<MTLComputePipelineState> pipeline = pipeline_name != nullptr ? metal_pipeline_get(ctx, pipeline_name) : nil;
    if (pipeline == nil) {
        marmot_set_error(
            MARMOT_ERROR_BACKEND_INIT_FAILED, "Metal expert-batched MoE matmul pipeline initialization failed"
        );
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    const size_t weight_bytes = marmot_tensor_quant_storage_bytes(weight_exps);
    const size_t weight_span = weight_bytes != 0 ? weight_bytes : marmot_tensor_size_bytes(weight_exps);
    const size_t input_bytes = marmot_tensor_size_bytes(input_batch);
    const size_t out_bytes = marmot::metal::tensor_span_bytes(output_batch);
    metal_tensor_buffer_t weight_view = metal_buffer_acquire_view(ctx, weight_exps, weight_exps->dtype, weight_span);
    metal_tensor_buffer_t input_view = metal_buffer_acquire_view(ctx, input_batch, input_batch->dtype, input_bytes);
    metal_tensor_buffer_t out_view = metal_buffer_acquire_view(ctx, output_batch, output_batch->dtype, out_bytes);
    if (weight_view.buffer == nil || input_view.buffer == nil || out_view.buffer == nil) {
        if (weight_view.buffer != nil) {
            [weight_view.buffer release];
        }
        if (input_view.buffer != nil) {
            [input_view.buffer release];
        }
        if (out_view.buffer != nil) {
            [out_view.buffer release];
        }
        [pipeline release];
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Metal expert-batched MoE matmul buffer acquisition failed");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
    if (encoder == nil) {
        [weight_view.buffer release];
        [input_view.buffer release];
        [out_view.buffer release];
        [pipeline release];
        marmot_set_error(
            MARMOT_ERROR_BACKEND_INIT_FAILED, "Metal expert-batched MoE matmul encoder acquisition failed"
        );
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    metal_moe_expert_batch_uniforms_t params = {
        .output_cols = (uint32_t)output_cols,
        .input_cols = (uint32_t)input_cols,
        .input_stride = (uint32_t)input_batch->shape.strides[0],
        .output_stride = (uint32_t)output_batch->shape.strides[0],
        .weight_blocks = (uint32_t)weight_blocks,
        .total_experts = (uint32_t)total_experts,
        .activation = 0,
    };
    const size_t counts_bytes = total_experts * sizeof(uint32_t);
    [encoder setBuffer:weight_view.buffer offset:weight_view.offset atIndex:0];
    [encoder setBuffer:input_view.buffer offset:input_view.offset atIndex:1];
    [encoder setBuffer:out_view.buffer offset:out_view.offset atIndex:2];
    [encoder setBytes:route_counts_gpu length:counts_bytes atIndex:3];
    [encoder setBytes:route_offsets_gpu length:counts_bytes atIndex:4];
    [encoder setBytes:&params length:sizeof(params) atIndex:5];
    [encoder setBytes:active_expert_ids length:active_count * sizeof(uint32_t) atIndex:6];
    [encoder dispatchThreadgroups:MTLSizeMake((output_cols + 63u) / 64u, (max_batch + 31u) / 32u, active_count)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    if (!out_view.is_private) {
        metal_command_stream_track_shared_write(ctx, output_batch->data);
    }

    [weight_view.buffer release];
    [input_view.buffer release];
    [out_view.buffer release];
    [pipeline release];
    if (out_view.is_private) {
        metal_residency_mark_dirty(ctx, output_batch, output_batch->dtype);
    }
    return MARMOT_SUCCESS;
}

static marmot_error_t metal_moe_expert_batch_glu_down(
    metal_context_t *ctx, const marmot_tensor_t *down_exps, const marmot_tensor_t *gate_batch,
    const marmot_tensor_t *up_batch, marmot_tensor_t *output_batch, const uint32_t *route_counts_gpu,
    const uint32_t *route_offsets_gpu, size_t total_experts, size_t max_batch, marmot_ffn_type_t ffn_type,
    const uint32_t *active_expert_ids, size_t active_count
) {
    if (ctx == nullptr || down_exps == nullptr || gate_batch == nullptr || up_batch == nullptr ||
        output_batch == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Metal expert-batched MoE GLU down requires valid tensors");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (ffn_type != MARMOT_FFN_SWIGLU && ffn_type != MARMOT_FFN_GEGLU) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Metal expert-batched MoE GLU down requires SwiGLU or GeGLU");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const size_t output_cols = output_batch->shape.shape[1];
    const size_t input_cols = gate_batch->shape.shape[1];
    if (output_cols == 0 || input_cols == 0 || max_batch == 0) {
        return MARMOT_SUCCESS;
    }

    const marmot_quant_kind_traits_t *traits = marmot_get_quant_kind_traits(down_exps->quant_kind);
    if (traits == nullptr || traits->block_values == 0) {
        marmot_set_error(
            MARMOT_ERROR_NOT_IMPLEMENTED, "Metal expert-batched MoE GLU down requires block quant weights"
        );
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    const size_t weight_blocks = (input_cols + traits->block_values - 1u) / traits->block_values;

    const char *pipeline_name = metal_moe_expert_batch_glu_q6_k_pipeline_name(gate_batch->dtype, output_batch->dtype);
    id<MTLComputePipelineState> pipeline = pipeline_name != nullptr ? metal_pipeline_get(ctx, pipeline_name) : nil;
    if (pipeline == nil) {
        marmot_set_error(
            MARMOT_ERROR_BACKEND_INIT_FAILED, "Metal expert-batched MoE GLU down pipeline initialization failed"
        );
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    const size_t weight_bytes = marmot_tensor_quant_storage_bytes(down_exps);
    const size_t weight_span = weight_bytes != 0 ? weight_bytes : marmot_tensor_size_bytes(down_exps);
    const size_t gate_bytes = marmot_tensor_size_bytes(gate_batch);
    const size_t up_bytes = marmot_tensor_size_bytes(up_batch);
    const size_t out_bytes = marmot::metal::tensor_span_bytes(output_batch);
    metal_tensor_buffer_t weight_view = metal_buffer_acquire_view(ctx, down_exps, down_exps->dtype, weight_span);
    metal_tensor_buffer_t gate_view = metal_buffer_acquire_view(ctx, gate_batch, gate_batch->dtype, gate_bytes);
    metal_tensor_buffer_t up_view = metal_buffer_acquire_view(ctx, up_batch, up_batch->dtype, up_bytes);
    metal_tensor_buffer_t out_view = metal_buffer_acquire_view(ctx, output_batch, output_batch->dtype, out_bytes);
    if (weight_view.buffer == nil || gate_view.buffer == nil || up_view.buffer == nil || out_view.buffer == nil) {
        if (weight_view.buffer != nil) {
            [weight_view.buffer release];
        }
        if (gate_view.buffer != nil) {
            [gate_view.buffer release];
        }
        if (up_view.buffer != nil) {
            [up_view.buffer release];
        }
        if (out_view.buffer != nil) {
            [out_view.buffer release];
        }
        [pipeline release];
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Metal expert-batched MoE GLU down buffer acquisition failed");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
    if (encoder == nil) {
        [weight_view.buffer release];
        [gate_view.buffer release];
        [up_view.buffer release];
        [out_view.buffer release];
        [pipeline release];
        marmot_set_error(
            MARMOT_ERROR_BACKEND_INIT_FAILED, "Metal expert-batched MoE GLU down encoder acquisition failed"
        );
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    metal_moe_expert_batch_uniforms_t params = {
        .output_cols = (uint32_t)output_cols,
        .input_cols = (uint32_t)input_cols,
        .input_stride = (uint32_t)gate_batch->shape.strides[0],
        .output_stride = (uint32_t)output_batch->shape.strides[0],
        .weight_blocks = (uint32_t)weight_blocks,
        .total_experts = (uint32_t)total_experts,
        .activation =
            (uint32_t)(ffn_type == MARMOT_FFN_GEGLU ? MARMOT_DEVICE_BINARY_GEGLU : MARMOT_DEVICE_BINARY_SWIGLU),
    };

    const size_t counts_bytes = total_experts * sizeof(uint32_t);
    [encoder setBuffer:weight_view.buffer offset:weight_view.offset atIndex:0];
    [encoder setBuffer:gate_view.buffer offset:gate_view.offset atIndex:1];
    [encoder setBuffer:up_view.buffer offset:up_view.offset atIndex:2];
    [encoder setBuffer:out_view.buffer offset:out_view.offset atIndex:3];
    [encoder setBytes:route_counts_gpu length:counts_bytes atIndex:4];
    [encoder setBytes:route_offsets_gpu length:counts_bytes atIndex:5];
    [encoder setBytes:&params length:sizeof(params) atIndex:6];
    [encoder setBytes:active_expert_ids length:active_count * sizeof(uint32_t) atIndex:7];
    [encoder dispatchThreadgroups:MTLSizeMake((output_cols + 63u) / 64u, (max_batch + 31u) / 32u, active_count)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    if (!out_view.is_private) {
        metal_command_stream_track_shared_write(ctx, output_batch->data);
    }

    [weight_view.buffer release];
    [gate_view.buffer release];
    [up_view.buffer release];
    [out_view.buffer release];
    [pipeline release];
    if (out_view.is_private) {
        metal_residency_mark_dirty(ctx, output_batch, output_batch->dtype);
    }
    return MARMOT_SUCCESS;
}

static marmot_error_t metal_moe_expert_batch_fused_gate_up(
    metal_context_t *ctx, const marmot_tensor_t *gate_exps, const marmot_tensor_t *up_exps,
    const marmot_tensor_t *input_batch, marmot_tensor_t *gate_batch, marmot_tensor_t *up_batch,
    const uint32_t *route_counts_gpu, const uint32_t *route_offsets_gpu, size_t total_experts, size_t max_batch,
    const uint32_t *active_expert_ids, size_t active_count
) {
    if (ctx == nullptr || gate_exps == nullptr || up_exps == nullptr || input_batch == nullptr ||
        gate_batch == nullptr || up_batch == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Metal fused gate+up MoE matmul requires valid tensors");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const size_t output_cols = gate_batch->shape.shape[1];
    const size_t input_cols = input_batch->shape.shape[1];
    if (output_cols == 0 || input_cols == 0 || max_batch == 0 || active_count == 0) {
        return MARMOT_SUCCESS;
    }

    const marmot_quant_kind_traits_t *traits = marmot_get_quant_kind_traits(MARMOT_QUANT_KIND_Q4_K);
    if (traits == nullptr || traits->block_values == 0) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Metal fused gate+up requires Q4_K weights");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    const size_t weight_blocks = (input_cols + traits->block_values - 1u) / traits->block_values;

    const char *pipeline_name = metal_moe_fused_gate_up_q4_k_pipeline_name(input_batch->dtype, gate_batch->dtype);
    id<MTLComputePipelineState> pipeline = pipeline_name != nullptr ? metal_pipeline_get(ctx, pipeline_name) : nil;
    if (pipeline == nil) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    const size_t gate_weight_bytes = marmot_tensor_quant_storage_bytes(gate_exps);
    const size_t gate_weight_span = gate_weight_bytes != 0 ? gate_weight_bytes : marmot_tensor_size_bytes(gate_exps);
    const size_t up_weight_bytes = marmot_tensor_quant_storage_bytes(up_exps);
    const size_t up_weight_span = up_weight_bytes != 0 ? up_weight_bytes : marmot_tensor_size_bytes(up_exps);
    const size_t input_bytes = marmot_tensor_size_bytes(input_batch);
    const size_t gate_out_bytes = marmot::metal::tensor_span_bytes(gate_batch);
    const size_t up_out_bytes = marmot::metal::tensor_span_bytes(up_batch);
    metal_tensor_buffer_t gate_w_view = metal_buffer_acquire_view(ctx, gate_exps, gate_exps->dtype, gate_weight_span);
    metal_tensor_buffer_t up_w_view = metal_buffer_acquire_view(ctx, up_exps, up_exps->dtype, up_weight_span);
    metal_tensor_buffer_t input_view = metal_buffer_acquire_view(ctx, input_batch, input_batch->dtype, input_bytes);
    metal_tensor_buffer_t gate_out_view = metal_buffer_acquire_view(ctx, gate_batch, gate_batch->dtype, gate_out_bytes);
    metal_tensor_buffer_t up_out_view = metal_buffer_acquire_view(ctx, up_batch, up_batch->dtype, up_out_bytes);
    if (gate_w_view.buffer == nil || up_w_view.buffer == nil || input_view.buffer == nil ||
        gate_out_view.buffer == nil || up_out_view.buffer == nil) {
        if (gate_w_view.buffer != nil) {
            [gate_w_view.buffer release];
        }
        if (up_w_view.buffer != nil) {
            [up_w_view.buffer release];
        }
        if (input_view.buffer != nil) {
            [input_view.buffer release];
        }
        if (gate_out_view.buffer != nil) {
            [gate_out_view.buffer release];
        }
        if (up_out_view.buffer != nil) {
            [up_out_view.buffer release];
        }
        [pipeline release];
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Metal fused gate+up buffer acquisition failed");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
    if (encoder == nil) {
        [gate_w_view.buffer release];
        [up_w_view.buffer release];
        [input_view.buffer release];
        [gate_out_view.buffer release];
        [up_out_view.buffer release];
        [pipeline release];
        marmot_set_error(MARMOT_ERROR_BACKEND_INIT_FAILED, "Metal fused gate+up encoder acquisition failed");
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    metal_moe_expert_batch_uniforms_t params = {
        .output_cols = (uint32_t)output_cols,
        .input_cols = (uint32_t)input_cols,
        .input_stride = (uint32_t)input_batch->shape.strides[0],
        .output_stride = (uint32_t)gate_batch->shape.strides[0],
        .weight_blocks = (uint32_t)weight_blocks,
        .total_experts = (uint32_t)total_experts,
        .activation = 0,
    };
    const size_t counts_bytes = total_experts * sizeof(uint32_t);
    [encoder setBuffer:gate_w_view.buffer offset:gate_w_view.offset atIndex:0];
    [encoder setBuffer:up_w_view.buffer offset:up_w_view.offset atIndex:1];
    [encoder setBuffer:input_view.buffer offset:input_view.offset atIndex:2];
    [encoder setBuffer:gate_out_view.buffer offset:gate_out_view.offset atIndex:3];
    [encoder setBuffer:up_out_view.buffer offset:up_out_view.offset atIndex:4];
    [encoder setBytes:route_counts_gpu length:counts_bytes atIndex:5];
    [encoder setBytes:route_offsets_gpu length:counts_bytes atIndex:6];
    [encoder setBytes:active_expert_ids length:active_count * sizeof(uint32_t) atIndex:7];
    [encoder setBytes:&params length:sizeof(params) atIndex:8];
    [encoder dispatchThreadgroups:MTLSizeMake((output_cols + 63u) / 64u, (max_batch + 31u) / 32u, active_count)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    if (!gate_out_view.is_private) {
        metal_command_stream_track_shared_write(ctx, gate_batch->data);
    }
    if (!up_out_view.is_private) {
        metal_command_stream_track_shared_write(ctx, up_batch->data);
    }

    [gate_w_view.buffer release];
    [up_w_view.buffer release];
    [input_view.buffer release];
    [gate_out_view.buffer release];
    [up_out_view.buffer release];
    [pipeline release];
    if (gate_out_view.is_private) {
        metal_residency_mark_dirty(ctx, gate_batch, gate_batch->dtype);
    }
    if (up_out_view.is_private) {
        metal_residency_mark_dirty(ctx, up_batch, up_batch->dtype);
    }
    return MARMOT_SUCCESS;
}

static bool metal_moe_can_indexed_down(
    const marmot_tensor_t *down_weight, const marmot_tensor_t *input, const marmot_tensor_t *route_indices,
    const marmot_tensor_t *route_weights, const marmot_tensor_t *out
) {
    if (down_weight == nullptr || input == nullptr || route_indices == nullptr || route_weights == nullptr ||
        out == nullptr) {
        return false;
    }
    if (!marmot_tensor_is_block_quantized_weight(down_weight) || down_weight->quant_kind != MARMOT_QUANT_KIND_Q6_K) {
        return false;
    }
    if (!metal_moe_value_dtype_supported(input->dtype) || route_weights->dtype != input->dtype ||
        out->dtype != input->dtype || route_indices->dtype != MARMOT_DTYPE_INT32) {
        return false;
    }
    if (input->shape.ndim != 2 || down_weight->shape.ndim != 2 || route_indices->shape.ndim != 1 ||
        route_weights->shape.ndim != 1 || out->shape.ndim != 2) {
        return false;
    }
    if (!marmot_tensor_is_contiguous(input) || !marmot_tensor_is_contiguous(route_indices) ||
        !marmot_tensor_is_contiguous(route_weights) || !marmot_tensor_is_contiguous(out)) {
        return false;
    }
    if (route_indices->shape.shape[0] != input->shape.shape[0] ||
        route_weights->shape.shape[0] != input->shape.shape[0] || down_weight->shape.shape[0] != out->shape.shape[1] ||
        down_weight->shape.shape[1] != input->shape.shape[1]) {
        return false;
    }
    return metal_moe_indexed_down_pipeline_name(down_weight->quant_kind, input->dtype, out->dtype) != nullptr;
}

static bool metal_moe_can_indexed_glu_down(
    const marmot_tensor_t *down_weight, const marmot_tensor_t *gate_batch, const marmot_tensor_t *up_batch,
    const marmot_tensor_t *route_indices, const marmot_tensor_t *route_weights, marmot_ffn_type_t ffn_type,
    const marmot_tensor_t *out
) {
    if (ffn_type != MARMOT_FFN_SWIGLU && ffn_type != MARMOT_FFN_GEGLU) {
        return false;
    }
    if (gate_batch == nullptr || up_batch == nullptr || gate_batch->shape.ndim != 2 || up_batch->shape.ndim != 2 ||
        gate_batch->shape.shape[0] != up_batch->shape.shape[0] ||
        gate_batch->shape.shape[1] != up_batch->shape.shape[1] || gate_batch->dtype != up_batch->dtype ||
        gate_batch->shape.strides[0] != up_batch->shape.strides[0]) {
        return false;
    }
    if (!metal_moe_can_indexed_down(down_weight, gate_batch, route_indices, route_weights, out)) {
        return false;
    }
    return metal_moe_indexed_glu_down_pipeline_name(down_weight->quant_kind, gate_batch->dtype, out->dtype) != nullptr;
}

static marmot_error_t metal_moe_expert_down_indexed(
    metal_context_t *ctx, const marmot_tensor_t *input, const marmot_tensor_t *down_weight,
    const marmot_tensor_t *route_indices, const marmot_tensor_t *route_weights, marmot_tensor_t *out
) {
    if (ctx == nullptr || input == nullptr || down_weight == nullptr || route_indices == nullptr ||
        route_weights == nullptr || out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Metal indexed MoE down requires valid tensors");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const size_t routes = input->shape.shape[0];
    const size_t input_cols = input->shape.shape[1];
    const size_t output_cols = out->shape.shape[1];
    const size_t output_rows = out->shape.shape[0];
    if (routes == 0 || input_cols == 0 || output_cols == 0 || output_rows == 0) {
        return MARMOT_SUCCESS;
    }
    if (routes > UINT32_MAX || input_cols > UINT32_MAX || output_cols > UINT32_MAX || output_rows > UINT32_MAX ||
        input->shape.strides[0] > UINT32_MAX || out->shape.strides[0] > UINT32_MAX) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Metal indexed MoE down shape exceeds kernel limits");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    const marmot_quant_kind_traits_t *traits = marmot_get_quant_kind_traits(down_weight->quant_kind);
    if (traits == nullptr || traits->block_values == 0) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Metal indexed MoE down requires block quant weights");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    const size_t weight_blocks = (input_cols + traits->block_values - 1u) / traits->block_values;
    if (weight_blocks == 0 || weight_blocks > UINT32_MAX) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Metal indexed MoE down weight blocks overflow");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    const char *pipeline_name = metal_moe_indexed_down_pipeline_name(down_weight->quant_kind, input->dtype, out->dtype);
    id<MTLComputePipelineState> pipeline = pipeline_name != nullptr ? metal_pipeline_get(ctx, pipeline_name) : nil;
    if (pipeline == nil) {
        marmot_set_error(MARMOT_ERROR_BACKEND_INIT_FAILED, "Metal indexed MoE down pipeline initialization failed");
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    const size_t weight_bytes = marmot_tensor_quant_storage_bytes(down_weight);
    const size_t weight_span = weight_bytes != 0 ? weight_bytes : marmot_tensor_size_bytes(down_weight);
    const size_t input_bytes = marmot_tensor_size_bytes(input);
    const size_t route_indices_bytes = marmot_tensor_size_bytes(route_indices);
    const size_t route_weights_bytes = marmot_tensor_size_bytes(route_weights);
    const size_t out_bytes = marmot::metal::tensor_span_bytes(out);
    metal_tensor_buffer_t weight_view = metal_buffer_acquire_view(ctx, down_weight, down_weight->dtype, weight_span);
    metal_tensor_buffer_t input_view = metal_buffer_acquire_view(ctx, input, input->dtype, input_bytes);
    metal_tensor_buffer_t route_indices_view =
        metal_buffer_acquire_view(ctx, route_indices, route_indices->dtype, route_indices_bytes);
    metal_tensor_buffer_t route_weights_view =
        metal_buffer_acquire_view(ctx, route_weights, route_weights->dtype, route_weights_bytes);
    metal_tensor_buffer_t out_view = metal_buffer_acquire_view(ctx, out, out->dtype, out_bytes);
    if (weight_view.buffer == nil || input_view.buffer == nil || route_indices_view.buffer == nil ||
        route_weights_view.buffer == nil || out_view.buffer == nil) {
        if (weight_view.buffer != nil) {
            [weight_view.buffer release];
        }
        if (input_view.buffer != nil) {
            [input_view.buffer release];
        }
        if (route_indices_view.buffer != nil) {
            [route_indices_view.buffer release];
        }
        if (route_weights_view.buffer != nil) {
            [route_weights_view.buffer release];
        }
        if (out_view.buffer != nil) {
            [out_view.buffer release];
        }
        [pipeline release];
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Metal indexed MoE down buffer acquisition failed");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
    if (encoder == nil) {
        [weight_view.buffer release];
        [input_view.buffer release];
        [route_indices_view.buffer release];
        [route_weights_view.buffer release];
        [out_view.buffer release];
        [pipeline release];
        marmot_set_error(MARMOT_ERROR_BACKEND_INIT_FAILED, "Metal indexed MoE down encoder acquisition failed");
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    metal_moe_indexed_down_uniforms_t params = {
        .routes = (uint32_t)routes,
        .input_cols = (uint32_t)input_cols,
        .output_cols = (uint32_t)output_cols,
        .input_stride = (uint32_t)input->shape.strides[0],
        .output_stride = (uint32_t)out->shape.strides[0],
        .weight_blocks = (uint32_t)weight_blocks,
        .activation = 0,
        .output_rows = (uint32_t)output_rows,
    };

    [encoder setBuffer:weight_view.buffer offset:weight_view.offset atIndex:0];
    [encoder setBuffer:input_view.buffer offset:input_view.offset atIndex:1];
    [encoder setBuffer:route_indices_view.buffer offset:route_indices_view.offset atIndex:2];
    [encoder setBuffer:route_weights_view.buffer offset:route_weights_view.offset atIndex:3];
    [encoder setBuffer:out_view.buffer offset:out_view.offset atIndex:4];
    [encoder setBytes:&params length:sizeof(params) atIndex:5];
    [encoder dispatchThreadgroups:MTLSizeMake((output_cols + 3u) / 4u, routes, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    if (!out_view.is_private) {
        metal_command_stream_track_shared_write(ctx, out->data);
    }
    metal_command_stream_flush(ctx, false);

    [weight_view.buffer release];
    [input_view.buffer release];
    [route_indices_view.buffer release];
    [route_weights_view.buffer release];
    [out_view.buffer release];
    [pipeline release];
    if (out_view.is_private) {
        metal_residency_mark_dirty(ctx, out, out->dtype);
    }
    return MARMOT_SUCCESS;
}

static marmot_error_t metal_moe_expert_down_glu_indexed(
    metal_context_t *ctx, const marmot_tensor_t *gate_batch, const marmot_tensor_t *up_batch,
    const marmot_tensor_t *down_weight, const marmot_tensor_t *route_indices, const marmot_tensor_t *route_weights,
    marmot_ffn_type_t ffn_type, marmot_tensor_t *out
) {
    if (ctx == nullptr || gate_batch == nullptr || up_batch == nullptr || down_weight == nullptr ||
        route_indices == nullptr || route_weights == nullptr || out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Metal indexed MoE GLU down requires valid tensors");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const size_t routes = gate_batch->shape.shape[0];
    const size_t input_cols = gate_batch->shape.shape[1];
    const size_t output_cols = out->shape.shape[1];
    const size_t output_rows = out->shape.shape[0];
    if (routes == 0 || input_cols == 0 || output_cols == 0 || output_rows == 0) {
        return MARMOT_SUCCESS;
    }
    if (routes > UINT32_MAX || input_cols > UINT32_MAX || output_cols > UINT32_MAX || output_rows > UINT32_MAX ||
        gate_batch->shape.strides[0] > UINT32_MAX || out->shape.strides[0] > UINT32_MAX) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Metal indexed MoE GLU down shape exceeds kernel limits");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    const marmot_quant_kind_traits_t *traits = marmot_get_quant_kind_traits(down_weight->quant_kind);
    if (traits == nullptr || traits->block_values == 0) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Metal indexed MoE GLU down requires block quant weights");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    const size_t weight_blocks = (input_cols + traits->block_values - 1u) / traits->block_values;
    if (weight_blocks == 0 || weight_blocks > UINT32_MAX) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Metal indexed MoE GLU down weight blocks overflow");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    const char *pipeline_name =
        metal_moe_indexed_glu_down_pipeline_name(down_weight->quant_kind, gate_batch->dtype, out->dtype);
    id<MTLComputePipelineState> pipeline = pipeline_name != nullptr ? metal_pipeline_get(ctx, pipeline_name) : nil;
    if (pipeline == nil) {
        marmot_set_error(MARMOT_ERROR_BACKEND_INIT_FAILED, "Metal indexed MoE GLU down pipeline initialization failed");
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    const size_t weight_bytes = marmot_tensor_quant_storage_bytes(down_weight);
    const size_t weight_span = weight_bytes != 0 ? weight_bytes : marmot_tensor_size_bytes(down_weight);
    const size_t gate_bytes = marmot_tensor_size_bytes(gate_batch);
    const size_t up_bytes = marmot_tensor_size_bytes(up_batch);
    const size_t route_indices_bytes = marmot_tensor_size_bytes(route_indices);
    const size_t route_weights_bytes = marmot_tensor_size_bytes(route_weights);
    const size_t out_bytes = marmot::metal::tensor_span_bytes(out);
    metal_tensor_buffer_t weight_view = metal_buffer_acquire_view(ctx, down_weight, down_weight->dtype, weight_span);
    metal_tensor_buffer_t gate_view = metal_buffer_acquire_view(ctx, gate_batch, gate_batch->dtype, gate_bytes);
    metal_tensor_buffer_t up_view = metal_buffer_acquire_view(ctx, up_batch, up_batch->dtype, up_bytes);
    metal_tensor_buffer_t route_indices_view =
        metal_buffer_acquire_view(ctx, route_indices, route_indices->dtype, route_indices_bytes);
    metal_tensor_buffer_t route_weights_view =
        metal_buffer_acquire_view(ctx, route_weights, route_weights->dtype, route_weights_bytes);
    metal_tensor_buffer_t out_view = metal_buffer_acquire_view(ctx, out, out->dtype, out_bytes);
    if (weight_view.buffer == nil || gate_view.buffer == nil || up_view.buffer == nil ||
        route_indices_view.buffer == nil || route_weights_view.buffer == nil || out_view.buffer == nil) {
        if (weight_view.buffer != nil) {
            [weight_view.buffer release];
        }
        if (gate_view.buffer != nil) {
            [gate_view.buffer release];
        }
        if (up_view.buffer != nil) {
            [up_view.buffer release];
        }
        if (route_indices_view.buffer != nil) {
            [route_indices_view.buffer release];
        }
        if (route_weights_view.buffer != nil) {
            [route_weights_view.buffer release];
        }
        if (out_view.buffer != nil) {
            [out_view.buffer release];
        }
        [pipeline release];
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Metal indexed MoE GLU down buffer acquisition failed");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
    if (encoder == nil) {
        [weight_view.buffer release];
        [gate_view.buffer release];
        [up_view.buffer release];
        [route_indices_view.buffer release];
        [route_weights_view.buffer release];
        [out_view.buffer release];
        [pipeline release];
        marmot_set_error(MARMOT_ERROR_BACKEND_INIT_FAILED, "Metal indexed MoE GLU down encoder acquisition failed");
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    metal_moe_indexed_down_uniforms_t params = {
        .routes = (uint32_t)routes,
        .input_cols = (uint32_t)input_cols,
        .output_cols = (uint32_t)output_cols,
        .input_stride = (uint32_t)gate_batch->shape.strides[0],
        .output_stride = (uint32_t)out->shape.strides[0],
        .weight_blocks = (uint32_t)weight_blocks,
        .activation =
            (uint32_t)(ffn_type == MARMOT_FFN_GEGLU ? MARMOT_DEVICE_BINARY_GEGLU : MARMOT_DEVICE_BINARY_SWIGLU),
        .output_rows = (uint32_t)output_rows,
    };

    [encoder setBuffer:weight_view.buffer offset:weight_view.offset atIndex:0];
    [encoder setBuffer:gate_view.buffer offset:gate_view.offset atIndex:1];
    [encoder setBuffer:up_view.buffer offset:up_view.offset atIndex:2];
    [encoder setBuffer:route_indices_view.buffer offset:route_indices_view.offset atIndex:3];
    [encoder setBuffer:route_weights_view.buffer offset:route_weights_view.offset atIndex:4];
    [encoder setBuffer:out_view.buffer offset:out_view.offset atIndex:5];
    [encoder setBytes:&params length:sizeof(params) atIndex:6];
    [encoder dispatchThreadgroups:MTLSizeMake((output_cols + 3u) / 4u, routes, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    if (!out_view.is_private) {
        metal_command_stream_track_shared_write(ctx, out->data);
    }
    metal_command_stream_flush(ctx, false);

    [weight_view.buffer release];
    [gate_view.buffer release];
    [up_view.buffer release];
    [route_indices_view.buffer release];
    [route_weights_view.buffer release];
    [out_view.buffer release];
    [pipeline release];
    if (out_view.is_private) {
        metal_residency_mark_dirty(ctx, out, out->dtype);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t metal_topk_impl(const void *device_ctx, const marmot_topk_desc_t *desc) {
    metal_context_t *ctx = (metal_context_t *)device_ctx;
    if (ctx == nullptr || desc == nullptr || desc->x == nullptr || desc->values_out == nullptr ||
        desc->indices_out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "TopK descriptor is incomplete");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (!metal_moe_value_dtype_supported(desc->x->dtype) || desc->values_out->dtype != desc->x->dtype ||
        desc->indices_out->dtype != MARMOT_DTYPE_INT32) {
        marmot_set_error(
            MARMOT_ERROR_UNSUPPORTED_DTYPE,
            "Metal TopK currently supports FLOAT16/FLOAT32 values with matching output dtype"
        );
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (desc->x->shape.ndim != 2 || desc->values_out->shape.ndim != 2 || desc->indices_out->shape.ndim != 2) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "TopK expects 2D tensors");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (desc->axis != 1 && desc->axis != -1) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "TopK currently supports only the last axis");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    const size_t rows = desc->x->shape.shape[0];
    const size_t cols = desc->x->shape.shape[1];
    const size_t k = desc->k;
    if (rows == 0) {
        return MARMOT_SUCCESS;
    }
    if (k == 0 || k > cols || desc->values_out->shape.shape[0] != rows || desc->indices_out->shape.shape[0] != rows ||
        desc->values_out->shape.shape[1] != k || desc->indices_out->shape.shape[1] != k) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "TopK shapes do not match");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (rows > UINT32_MAX || cols > UINT32_MAX || desc->x->shape.strides[0] > UINT32_MAX ||
        desc->values_out->shape.strides[0] > UINT32_MAX || desc->indices_out->shape.strides[0] > UINT32_MAX ||
        k > UINT32_MAX) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Metal TopK shape exceeds current kernel limits");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    const size_t input_bytes = marmot::metal::tensor_span_bytes(desc->x);
    const size_t values_bytes = marmot::metal::tensor_span_bytes(desc->values_out);
    const size_t indices_bytes = marmot::metal::tensor_span_bytes(desc->indices_out);
    if (input_bytes == 0 || values_bytes == 0 || indices_bytes == 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "TopK tensor span is invalid");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    metal_routing_log_decision(ctx, METAL_ROUTING_CATEGORY_MISC, "topk", input_bytes, true, "gpu");

    metal_tensor_buffer_t input_view = metal_buffer_acquire_view(ctx, desc->x, desc->x->dtype, input_bytes);
    metal_tensor_buffer_t values_view =
        metal_buffer_acquire_view(ctx, desc->values_out, desc->values_out->dtype, values_bytes);
    metal_tensor_buffer_t indices_view =
        metal_buffer_acquire_view(ctx, desc->indices_out, desc->indices_out->dtype, indices_bytes);
    if (input_view.buffer == nil || values_view.buffer == nil || indices_view.buffer == nil) {
        if (input_view.buffer != nil) {
            [input_view.buffer release];
        }
        if (values_view.buffer != nil) {
            [values_view.buffer release];
        }
        if (indices_view.buffer != nil) {
            [indices_view.buffer release];
        }
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Metal TopK buffer acquisition failed");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    bool use_small_topk = rows <= 8 && cols <= kMetalTopkSmallCols && k <= kMetalTopkSmallK;
    const char *pipeline_name = metal_topk_pipeline_name(desc->x->dtype, use_small_topk);
    id<MTLComputePipelineState> pipeline = pipeline_name != nullptr ? metal_pipeline_get(ctx, pipeline_name) : nil;
    if (pipeline != nil && use_small_topk && pipeline.maxTotalThreadsPerThreadgroup < kMetalTopkSmallCols) {
        [pipeline release];
        use_small_topk = false;
        pipeline_name = metal_topk_pipeline_name(desc->x->dtype, false);
        pipeline = pipeline_name != nullptr ? metal_pipeline_get(ctx, pipeline_name) : nil;
    }
    if (pipeline == nil) {
        [input_view.buffer release];
        [values_view.buffer release];
        [indices_view.buffer release];
        marmot_set_error(MARMOT_ERROR_BACKEND_INIT_FAILED, "Metal TopK pipeline initialization failed");
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
    if (encoder == nil) {
        [pipeline release];
        [input_view.buffer release];
        [values_view.buffer release];
        [indices_view.buffer release];
        marmot_set_error(MARMOT_ERROR_BACKEND_INIT_FAILED, "Metal TopK encoder acquisition failed");
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    metal_topk_uniforms_t params = {
        .rows = (uint32_t)rows,
        .cols = (uint32_t)cols,
        .input_stride = (uint32_t)desc->x->shape.strides[0],
        .values_stride = (uint32_t)desc->values_out->shape.strides[0],
        .indices_stride = (uint32_t)desc->indices_out->shape.strides[0],
        .k = (uint32_t)k,
    };

    metal_profiling_set_label(ctx, "topk");
    metal_profiling_begin(ctx);
    [encoder setBuffer:input_view.buffer offset:input_view.offset atIndex:0];
    [encoder setBuffer:values_view.buffer offset:values_view.offset atIndex:1];
    [encoder setBuffer:indices_view.buffer offset:indices_view.offset atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];

    if (use_small_topk) {
        [encoder setThreadgroupMemoryLength:kMetalTopkSmallCols * sizeof(float) atIndex:0];
        [encoder setThreadgroupMemoryLength:kMetalTopkSmallCols * sizeof(int32_t) atIndex:1];
        MTLSize threadgroups = MTLSizeMake(1, (NSUInteger)rows, 1);
        MTLSize threads = MTLSizeMake((NSUInteger)kMetalTopkSmallCols, 1, 1);
        [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads];
    } else {
        MTLSize grid = MTLSizeMake((NSUInteger)rows, 1, 1);
        NSUInteger threadgroup_size = metal_threadgroup_size_1d(pipeline, (NSUInteger)rows);
        if (threadgroup_size == 0) {
            threadgroup_size = 1;
        }
        MTLSize threads = MTLSizeMake(threadgroup_size, 1, 1);
        [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    }
    metal_profiling_end(ctx);
    if (!values_view.is_private) {
        metal_command_stream_track_shared_write(ctx, desc->values_out->data);
    }
    if (!indices_view.is_private) {
        metal_command_stream_track_shared_write(ctx, desc->indices_out->data);
    }

    metal_command_stream_flush(ctx, false);

    [pipeline release];
    [input_view.buffer release];
    [values_view.buffer release];
    [indices_view.buffer release];

    if (values_view.is_private) {
        metal_residency_mark_dirty(ctx, desc->values_out, desc->values_out->dtype);
        metal_residency_sync_shared_range(ctx, desc->values_out->data, values_bytes);
        metal_command_stream_track_shared_write(ctx, desc->values_out->data);
    } else {
        metal_moe_mark_device_write((marmot_tensor_t *)desc->values_out);
        metal_residency_mark_shared_write(ctx, desc->values_out->data);
    }
    if (indices_view.is_private) {
        metal_residency_mark_dirty(ctx, desc->indices_out, desc->indices_out->dtype);
        metal_residency_sync_shared_range(ctx, desc->indices_out->data, indices_bytes);
        metal_command_stream_track_shared_write(ctx, desc->indices_out->data);
    } else {
        metal_moe_mark_device_write((marmot_tensor_t *)desc->indices_out);
        metal_residency_mark_shared_write(ctx, desc->indices_out->data);
    }

    return MARMOT_SUCCESS;
}

marmot_error_t metal_moe_experts_impl(const void *device_ctx, const marmot_moe_experts_desc_t *desc) {
    if (desc == nullptr || desc->hidden_states == nullptr || desc->gate_exps == nullptr || desc->up_exps == nullptr ||
        desc->down_exps == nullptr || desc->topk_ids == nullptr || desc->topk_weights == nullptr ||
        desc->out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "MoE descriptor is incomplete");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    metal_context_t *ctx = (metal_context_t *)device_ctx;
    const marmot_context_t *src_ctx = desc->hidden_states->ctx != nullptr ? desc->hidden_states->ctx : desc->out->ctx;
    if (ctx == nullptr || src_ctx == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Metal MoE execution requires tensors bound to a context");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const bool quantized_weights = marmot_tensor_is_block_quantized_weight(desc->gate_exps);
    const bool dense_supported = !quantized_weights && metal_moe_value_dtype_supported(desc->hidden_states->dtype) &&
        desc->gate_exps->dtype == desc->hidden_states->dtype && desc->up_exps->dtype == desc->hidden_states->dtype &&
        desc->down_exps->dtype == desc->hidden_states->dtype &&
        desc->topk_weights->dtype == desc->hidden_states->dtype && desc->topk_ids->dtype == MARMOT_DTYPE_INT32 &&
        desc->out->dtype == desc->hidden_states->dtype;
    const bool quantized_supported = quantized_weights && metal_moe_value_dtype_supported(desc->hidden_states->dtype) &&
        desc->topk_weights->dtype == desc->hidden_states->dtype && desc->topk_ids->dtype == MARMOT_DTYPE_INT32 &&
        desc->out->dtype == desc->hidden_states->dtype;
    if (!dense_supported && !quantized_supported) {
        metal_routing_log_decision(ctx, METAL_ROUTING_CATEGORY_MISC, "moe_experts", 0, false, "cpu_fallback");
        return metal_moe_cpu_fallback(desc);
    }

    const marmot_dtype_t value_dtype = desc->hidden_states->dtype;
    const size_t value_bytes = marmot_dtype_size(value_dtype);
    const size_t tokens = desc->hidden_states->shape.shape[0];
    const size_t hidden = desc->hidden_states->shape.shape[1];
    const size_t experts_per_token = desc->topk_ids->shape.shape[1];
    const size_t experts = desc->gate_exps->shape.shape[2];
    const size_t ff_length = desc->gate_exps->shape.shape[1];
    const size_t route_capacity = tokens * experts_per_token;
    size_t route_count = route_capacity;
    const bool use_host_routes = metal_moe_should_use_host_routes(desc);
    if (route_capacity == 0 || hidden == 0) {
        return metal_moe_zero_output(ctx, desc->out);
    }
    if (tokens > UINT32_MAX || hidden > INT32_MAX || ff_length > INT32_MAX || experts > SIZE_MAX / sizeof(size_t) ||
        route_capacity > SIZE_MAX / sizeof(float) || route_capacity > SIZE_MAX / sizeof(int32_t)) {
        metal_routing_log_decision(ctx, METAL_ROUTING_CATEGORY_MISC, "moe_experts", 0, false, "cpu_fallback");
        return metal_moe_cpu_fallback(desc);
    }
    if (desc->hidden_states->data == nullptr || desc->gate_exps->data == nullptr || desc->up_exps->data == nullptr ||
        desc->down_exps->data == nullptr || desc->out->data == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Metal MoE tensors require storage");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    metal_moe_workspace_t *workspace = nullptr;
    size_t *expert_counts = nullptr;
    size_t *expert_offsets = nullptr;
    size_t *expert_order = nullptr;
    size_t prefix = 0;
    size_t max_batch = 0;
    size_t ordered_active_experts = 0;
    size_t route_indices_bytes = 0;
    size_t route_weights_bytes = 0;
    size_t hidden_batch_bytes = 0;
    size_t ff_batch_bytes = 0;
    size_t scratch_rows = 0;
    bool use_grouped_decode = false;
    bool use_expert_batch_prefill = false;
    bool use_grouped_batch = false;
    bool deferred_route_sync = false;
    bool can_expert_batch_early = false;
    bool use_grouped_decode_gate_up = false;
    bool use_grouped_decode_fused_down = false;
    bool need_expert_loop_metadata = false;
    bool grouped_batch_open = false;
    bool route_batch_open = false;
    const size_t route_capacity_shape[] = {route_capacity};
    size_t hidden_shape[] = {0, hidden};
    size_t ff_shape[] = {0, ff_length};
    marmot_tensor_t route_indices_spec = {};
    marmot_tensor_t route_weights_spec = {};
    marmot_tensor_t hidden_batch_spec = {};
    marmot_tensor_t ff_batch_spec = {};
    marmot_tensor_t route_indices = {};
    marmot_tensor_t route_experts = {};
    marmot_tensor_t route_weights = {};
    marmot_tensor_t hidden_batch = {};
    marmot_tensor_t gate_batch = {};
    marmot_tensor_t up_batch = {};
    marmot_tensor_t fused_batch = {};
    marmot_tensor_t down_batch = {};
    bool use_grouped_decode_down = false;
    metal_moe_stage_profile_t stage_profile = {
        .enabled = metal_moe_stage_profiling_enabled(),
    };
    uint64_t route_stage_start_ns = 0;
    uint32_t *route_counts_gpu = nullptr;
    uint32_t *route_offsets_gpu = nullptr;
    uint32_t *route_status_gpu = nullptr;
    metal_moe_route_summary_t *route_summary_gpu = nullptr;
    const marmot_device_binary_op_t glu_op =
        desc->ffn_type == MARMOT_FFN_GEGLU ? MARMOT_DEVICE_BINARY_GEGLU : MARMOT_DEVICE_BINARY_SWIGLU;
    if (stage_profile.enabled) {
        stage_profile.total_start_ns = metal_moe_now_ns();
    }

    metal_profiling_set_label(ctx, "moe_experts");

    metal_routing_log_decision(
        ctx, METAL_ROUTING_CATEGORY_MISC, "moe_experts", route_capacity * (hidden + ff_length) * value_bytes, true,
        quantized_supported ? (value_dtype == MARMOT_DTYPE_FLOAT16 ? "gpu_quant_f16" : "gpu_quant_f32")
                            : (value_dtype == MARMOT_DTYPE_FLOAT16 ? "gpu_dense_f16" : "gpu_dense_f32")
    );

    marmot_error_t status = metal_moe_zero_output(ctx, desc->out);
    if (status != MARMOT_SUCCESS) {
        goto cleanup;
    }
    metal_moe_init_temp_tensor(&route_indices_spec, src_ctx, route_capacity_shape, 1, MARMOT_DTYPE_INT32, nullptr, 0);
    metal_moe_init_temp_tensor(&route_weights_spec, src_ctx, route_capacity_shape, 1, value_dtype, nullptr, 0);
    route_indices_bytes = marmot_tensor_size_bytes(&route_indices_spec);
    if (route_indices_bytes == 0 && marmot_get_last_error() != MARMOT_SUCCESS) {
        status = marmot_get_last_error();
        goto cleanup;
    }
    route_weights_bytes = marmot_tensor_size_bytes(&route_weights_spec);
    if (route_weights_bytes == 0 && marmot_get_last_error() != MARMOT_SUCCESS) {
        status = marmot_get_last_error();
        goto cleanup;
    }

    status = metal_moe_workspace_acquire(
        ctx, experts, route_indices_bytes, route_weights_bytes, value_bytes, value_bytes, &workspace
    );
    if (status != MARMOT_SUCCESS) {
        goto cleanup;
    }

    expert_counts = workspace->expert_counts;
    expert_offsets = workspace->expert_offsets;
    expert_order = workspace->expert_order;
    route_counts_gpu = (uint32_t *)workspace->route_counts_alloc.ptr;
    route_offsets_gpu = (uint32_t *)workspace->route_offsets_alloc.ptr;
    route_status_gpu = (uint32_t *)workspace->route_status_alloc.ptr;
    route_summary_gpu = (metal_moe_route_summary_t *)workspace->route_summary_alloc.ptr;
    if (route_counts_gpu == nullptr || route_offsets_gpu == nullptr || route_status_gpu == nullptr ||
        route_summary_gpu == nullptr) {
        status = MARMOT_ERROR_OUT_OF_MEMORY;
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Metal MoE route workspace allocation is incomplete");
        goto cleanup;
    }

    metal_moe_init_temp_tensor(
        &route_indices, src_ctx, route_capacity_shape, 1, MARMOT_DTYPE_INT32, workspace->route_indices_alloc.ptr,
        workspace->route_indices_alloc.size
    );
    metal_moe_init_temp_tensor(
        &route_weights, src_ctx, route_capacity_shape, 1, value_dtype, workspace->route_weights_alloc.ptr,
        workspace->route_weights_alloc.size
    );
    metal_moe_init_temp_tensor(
        &route_experts, src_ctx, route_capacity_shape, 1, MARMOT_DTYPE_INT32, workspace->route_experts_alloc.ptr,
        workspace->route_experts_alloc.size
    );

    can_expert_batch_early = !use_host_routes && tokens > 1 && metal_moe_can_expert_batch_prefill(desc, tokens);

    route_stage_start_ns = metal_moe_stage_profile_begin(&stage_profile);
    if (use_host_routes) {
        status = metal_moe_build_routes_cpu(
            ctx, desc, experts, expert_counts, expert_offsets, workspace->expert_cursor, expert_order, &route_indices,
            &route_weights, &route_count, &max_batch, &ordered_active_experts
        );
        if (status != MARMOT_SUCCESS) {
            goto cleanup;
        }
    } else {
        memset(route_counts_gpu, 0, experts * sizeof(*route_counts_gpu));
        memset(route_offsets_gpu, 0, experts * sizeof(*route_offsets_gpu));
        *route_status_gpu = 0u;
        memset(route_summary_gpu, 0, sizeof(*route_summary_gpu));
        if (can_expert_batch_early) {
            memset(route_indices.data, 0, route_capacity * sizeof(int32_t));
        }
        metal_command_batch_begin(ctx);
        route_batch_open = true;
        status = metal_moe_route_count(ctx, desc->topk_ids, experts, route_counts_gpu, route_status_gpu);
        if (status != MARMOT_SUCCESS) {
            goto cleanup;
        }
        status = metal_moe_route_prepare(ctx, experts, route_counts_gpu, route_offsets_gpu, route_summary_gpu);
        if (status != MARMOT_SUCCESS) {
            goto cleanup;
        }
        status =
            metal_moe_route_pack(ctx, desc, route_offsets_gpu, experts, &route_indices, &route_weights, &route_experts);
        if (status != MARMOT_SUCCESS) {
            goto cleanup;
        }
        metal_command_batch_end(ctx, true);
        route_batch_open = false;
        if (can_expert_batch_early) {
            route_count = route_capacity;
            max_batch = tokens;
            ordered_active_experts = experts;
            deferred_route_sync = true;
        } else {
            if (!metal_command_stream_wait_for_shared_read(ctx, route_summary_gpu, sizeof(*route_summary_gpu))) {
                metal_command_stream_flush(ctx, true);
            }
            if (!metal_command_stream_wait_for_shared_read(ctx, route_status_gpu, sizeof(*route_status_gpu))) {
                metal_command_stream_flush(ctx, true);
            }
            if (*route_status_gpu != 0u) {
                status = MARMOT_ERROR_INVALID_ARGUMENT;
                marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "MoE expert id is out of range");
                goto cleanup;
            }
            route_count = route_summary_gpu->route_count;
            max_batch = route_summary_gpu->max_batch;
            ordered_active_experts = route_summary_gpu->active_experts;
            if (route_count > route_capacity) {
                status = MARMOT_ERROR_INVALID_OPERATION;
                marmot_set_error(
                    MARMOT_ERROR_INVALID_OPERATION, "Metal MoE GPU routing produced an invalid route count"
                );
                goto cleanup;
            }
        }
    }
    metal_moe_stage_profile_end(&stage_profile, ctx, route_stage_start_ns, &stage_profile.route_ns);

    if (!deferred_route_sync && max_batch == 0) {
        goto cleanup;
    }
    route_indices.shape.shape[0] = route_count;
    route_weights.shape.shape[0] = route_count;
    route_experts.shape.shape[0] = route_count;

    use_grouped_decode = metal_moe_should_use_grouped_decode(desc, tokens, route_count, max_batch);
    use_expert_batch_prefill = !use_grouped_decode && metal_moe_can_expert_batch_prefill(desc, tokens) &&
        route_count > max_batch && ordered_active_experts > 1;
    use_grouped_batch = use_grouped_decode || use_expert_batch_prefill;
    scratch_rows = use_grouped_batch ? route_count : max_batch;
    hidden_shape[0] = scratch_rows;
    ff_shape[0] = scratch_rows;
    metal_moe_init_temp_tensor(&hidden_batch_spec, src_ctx, hidden_shape, 2, value_dtype, nullptr, 0);
    metal_moe_init_temp_tensor(&ff_batch_spec, src_ctx, ff_shape, 2, value_dtype, nullptr, 0);
    hidden_batch_bytes = marmot_tensor_size_bytes(&hidden_batch_spec);
    if (hidden_batch_bytes == 0 && marmot_get_last_error() != MARMOT_SUCCESS) {
        status = marmot_get_last_error();
        goto cleanup;
    }
    ff_batch_bytes = marmot_tensor_size_bytes(&ff_batch_spec);
    if (ff_batch_bytes == 0 && marmot_get_last_error() != MARMOT_SUCCESS) {
        status = marmot_get_last_error();
        goto cleanup;
    }
    status = metal_moe_workspace_ensure_allocation(ctx, &workspace->hidden_batch_alloc, hidden_batch_bytes);
    if (status == MARMOT_SUCCESS) {
        status = metal_moe_workspace_ensure_allocation(ctx, &workspace->gate_batch_alloc, ff_batch_bytes);
    }
    if (status == MARMOT_SUCCESS) {
        status = metal_moe_workspace_ensure_allocation(ctx, &workspace->up_batch_alloc, ff_batch_bytes);
    }
    if (status == MARMOT_SUCCESS) {
        status = metal_moe_workspace_ensure_allocation(ctx, &workspace->fused_batch_alloc, ff_batch_bytes);
    }
    if (status == MARMOT_SUCCESS) {
        status = metal_moe_workspace_ensure_allocation(ctx, &workspace->down_batch_alloc, hidden_batch_bytes);
    }
    if (status != MARMOT_SUCCESS) {
        goto cleanup;
    }

    metal_moe_init_temp_tensor(
        &hidden_batch, src_ctx, hidden_shape, 2, value_dtype, workspace->hidden_batch_alloc.ptr,
        workspace->hidden_batch_alloc.size
    );
    metal_moe_init_temp_tensor(
        &gate_batch, src_ctx, ff_shape, 2, value_dtype, workspace->gate_batch_alloc.ptr,
        workspace->gate_batch_alloc.size
    );
    metal_moe_init_temp_tensor(
        &up_batch, src_ctx, ff_shape, 2, value_dtype, workspace->up_batch_alloc.ptr, workspace->up_batch_alloc.size
    );
    metal_moe_init_temp_tensor(
        &fused_batch, src_ctx, ff_shape, 2, value_dtype, workspace->fused_batch_alloc.ptr,
        workspace->fused_batch_alloc.size
    );
    metal_moe_init_temp_tensor(
        &down_batch, src_ctx, hidden_shape, 2, value_dtype, workspace->down_batch_alloc.ptr,
        workspace->down_batch_alloc.size
    );

    if (use_grouped_decode) {
        use_grouped_decode_gate_up =
            metal_moe_can_grouped_decode_gate_up(desc, desc->hidden_states, &route_experts, &gate_batch, &up_batch);
        use_grouped_decode_down = metal_moe_can_grouped_decode_down(desc, &fused_batch, &route_experts, &route_weights);
        use_grouped_decode_fused_down = use_grouped_decode_gate_up && use_grouped_decode_down;
    }
    need_expert_loop_metadata = !use_grouped_batch || !use_expert_batch_prefill ||
        (use_grouped_decode && !(use_grouped_decode_gate_up && use_grouped_decode_down));
    if (!use_host_routes && need_expert_loop_metadata && !use_expert_batch_prefill) {
        if (!metal_command_stream_wait_for_shared_read(ctx, route_counts_gpu, experts * sizeof(*route_counts_gpu))) {
            metal_command_stream_flush(ctx, true);
        }
        ordered_active_experts = metal_moe_build_route_metadata_from_counts(
            route_counts_gpu, experts, expert_counts, expert_offsets, expert_order, &prefix, &max_batch
        );
        if (prefix != route_count) {
            status = MARMOT_ERROR_INVALID_OPERATION;
            marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "Metal MoE GPU route metadata mismatch");
            goto cleanup;
        }
    }
    if ((use_grouped_decode_gate_up || use_grouped_decode_down) && use_host_routes) {
        status = metal_moe_fill_route_experts(
            &route_experts, expert_counts, expert_offsets, expert_order, ordered_active_experts
        );
        if (status != MARMOT_SUCCESS) {
            goto cleanup;
        }
        metal_residency_mark_shared_write(ctx, route_experts.data);
    }

    if (use_grouped_batch) {
        // Ensure GPU route kernels have completed before reading route_counts_gpu/route_offsets_gpu
        // via setBytes in expert_batch_matmul (these are shared-memory buffers written by GPU).
        if (use_expert_batch_prefill && deferred_route_sync) {
            if (!metal_command_stream_wait_for_shared_read(
                    ctx, route_counts_gpu, experts * sizeof(*route_counts_gpu)
                )) {
                metal_command_stream_flush(ctx, true);
            }
            if (!metal_command_stream_wait_for_shared_read(ctx, route_summary_gpu, sizeof(*route_summary_gpu))) {
                metal_command_stream_flush(ctx, true);
            }
            if (!metal_command_stream_wait_for_shared_read(ctx, route_status_gpu, sizeof(*route_status_gpu))) {
                metal_command_stream_flush(ctx, true);
            }
            if (*route_status_gpu != 0u) {
                status = MARMOT_ERROR_INVALID_ARGUMENT;
                marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "MoE expert id is out of range");
                goto cleanup;
            }
            route_count = route_summary_gpu->route_count;
            max_batch = route_summary_gpu->max_batch;
            ordered_active_experts = route_summary_gpu->active_experts;
            if (route_count > route_capacity) {
                status = MARMOT_ERROR_INVALID_OPERATION;
                marmot_set_error(
                    MARMOT_ERROR_INVALID_OPERATION, "Metal MoE GPU routing produced an invalid route count"
                );
                goto cleanup;
            }
            route_indices.shape.shape[0] = route_count;
            route_weights.shape.shape[0] = route_count;
            route_experts.shape.shape[0] = route_count;
            deferred_route_sync = false;
        }
        metal_command_batch_begin(ctx);
        grouped_batch_open = true;
        if (status == MARMOT_SUCCESS && use_expert_batch_prefill) {
            uint32_t eb_active_ids_stack[256];
            uint32_t *eb_active_ids_heap = experts > 256 ? (uint32_t *)malloc(experts * sizeof(uint32_t)) : nullptr;
            uint32_t *eb_active_ids = experts <= 256 ? eb_active_ids_stack : eb_active_ids_heap;
            if (eb_active_ids == nullptr) {
                status = MARMOT_ERROR_OUT_OF_MEMORY;
                marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Metal MoE active expert ids allocation failed");
                goto cleanup;
            }
            size_t eb_active_count = 0;
            if (use_host_routes) {
                for (size_t e = 0; e < experts; ++e) {
                    route_counts_gpu[e] = (uint32_t)expert_counts[e];
                    route_offsets_gpu[e] = (uint32_t)expert_offsets[e];
                    if (expert_counts[e] > 0) {
                        eb_active_ids[eb_active_count++] = (uint32_t)e;
                    }
                }
            } else {
                for (size_t e = 0; e < experts; ++e) {
                    if (route_counts_gpu[e] > 0) {
                        eb_active_ids[eb_active_count++] = (uint32_t)e;
                    }
                }
            }

            uint64_t gather_stage_start_ns = metal_moe_stage_profile_begin(&stage_profile);
            status = metal_gather_rows(device_ctx, desc->hidden_states, &route_indices, &hidden_batch);
            metal_moe_stage_profile_end(&stage_profile, ctx, gather_stage_start_ns, &stage_profile.gather_ns);

            if (status == MARMOT_SUCCESS) {
                uint64_t gate_up_stage_start_ns = metal_moe_stage_profile_begin(&stage_profile);
                marmot_error_t fused_status = metal_moe_expert_batch_fused_gate_up(
                    ctx, desc->gate_exps, desc->up_exps, &hidden_batch, &gate_batch, &up_batch, route_counts_gpu,
                    route_offsets_gpu, experts, max_batch, eb_active_ids, eb_active_count
                );
                if (fused_status == MARMOT_SUCCESS) {
                    status = MARMOT_SUCCESS;
                } else {
                    status = metal_moe_expert_batch_matmul(
                        ctx, desc->gate_exps, &hidden_batch, &gate_batch, route_counts_gpu, route_offsets_gpu, experts,
                        max_batch, MARMOT_QUANT_KIND_Q4_K, eb_active_ids, eb_active_count
                    );
                    if (status == MARMOT_SUCCESS) {
                        status = metal_moe_expert_batch_matmul(
                            ctx, desc->up_exps, &hidden_batch, &up_batch, route_counts_gpu, route_offsets_gpu, experts,
                            max_batch, MARMOT_QUANT_KIND_Q4_K, eb_active_ids, eb_active_count
                        );
                    }
                }
                metal_moe_stage_profile_end(&stage_profile, ctx, gate_up_stage_start_ns, &stage_profile.gate_up_ns);
            }

            if (status == MARMOT_SUCCESS) {
                uint64_t down_stage_start_ns = metal_moe_stage_profile_begin(&stage_profile);
                status = metal_moe_expert_batch_glu_down(
                    ctx, desc->down_exps, &gate_batch, &up_batch, &down_batch, route_counts_gpu, route_offsets_gpu,
                    experts, max_batch, desc->ffn_type, eb_active_ids, eb_active_count
                );
                metal_moe_stage_profile_end(&stage_profile, ctx, down_stage_start_ns, &stage_profile.down_ns);
            }
            free(eb_active_ids_heap);
        }
        if (status == MARMOT_SUCCESS && use_grouped_decode_gate_up) {
            uint64_t gate_up_stage_start_ns = metal_moe_stage_profile_begin(&stage_profile);
            status = metal_moe_decode_gate_up_grouped(
                ctx, desc->hidden_states, desc->gate_exps, desc->up_exps, &route_experts, &gate_batch, &up_batch
            );
            metal_moe_stage_profile_end(&stage_profile, ctx, gate_up_stage_start_ns, &stage_profile.gate_up_ns);
        }
        if (status == MARMOT_SUCCESS && (use_grouped_decode_gate_up && !use_grouped_decode_fused_down)) {
            uint64_t glu_stage_start_ns = metal_moe_stage_profile_begin(&stage_profile);
            status = metal_elementwise_binary_impl(device_ctx, glu_op, &gate_batch, &up_batch, &fused_batch);
            metal_moe_stage_profile_end(&stage_profile, ctx, glu_stage_start_ns, &stage_profile.glu_ns);
        }
    }

    for (size_t ordered_idx = 0; status == MARMOT_SUCCESS &&
         !((use_grouped_decode_gate_up && use_grouped_decode_down) || use_expert_batch_prefill) &&
         ordered_idx < ordered_active_experts;
         ++ordered_idx) {
        const size_t expert = expert_order[ordered_idx];
        const size_t count = expert_counts[expert];
        if (count == 0) {
            continue;
        }

        marmot_tensor_t index_view;
        marmot_tensor_t weight_view;
        marmot_tensor_t hidden_view;
        marmot_tensor_t gate_view;
        marmot_tensor_t up_view;
        marmot_tensor_t fused_view;
        marmot_tensor_t down_view;
        marmot_tensor_t gate_weight_view;
        marmot_tensor_t up_weight_view;
        marmot_tensor_t down_weight_view;
        const size_t offset = expert_offsets[expert];
        bool fused_gate_up = false;
        bool indexed_down = false;
        bool indexed_glu_down = false;
        bool direct_hidden_alias = false;
        bool quant_hints_pushed = false;
        uint32_t quant_hints = 0;
        metal_moe_init_1d_slice_alias(&index_view, &route_indices, count, offset);
        metal_moe_init_1d_slice_alias(&weight_view, &route_weights, count, offset);
        if (!use_grouped_batch) {
            metal_command_batch_begin(ctx);
        }
        if (use_grouped_batch) {
            metal_moe_init_2d_slice_alias(&hidden_view, &hidden_batch, count, hidden, offset);
            metal_moe_init_2d_slice_alias(&gate_view, &gate_batch, count, ff_length, offset);
            metal_moe_init_2d_slice_alias(&up_view, &up_batch, count, ff_length, offset);
            metal_moe_init_2d_slice_alias(&fused_view, &fused_batch, count, ff_length, offset);
            metal_moe_init_2d_slice_alias(&down_view, &down_batch, count, hidden, offset);
        } else {
            metal_moe_init_2d_alias(&hidden_view, &hidden_batch, count, hidden);
            metal_moe_init_2d_alias(&gate_view, &gate_batch, count, ff_length);
            metal_moe_init_2d_alias(&up_view, &up_batch, count, ff_length);
            metal_moe_init_2d_alias(&fused_view, &fused_batch, count, ff_length);
            metal_moe_init_2d_alias(&down_view, &down_batch, count, hidden);
        }
        metal_moe_init_expert_view(&gate_weight_view, desc->gate_exps, ff_length, hidden, expert);
        metal_moe_init_expert_view(&up_weight_view, desc->up_exps, ff_length, hidden, expert);
        metal_moe_init_expert_view(&down_weight_view, desc->down_exps, hidden, ff_length, expert);
        fused_gate_up = !use_grouped_decode_gate_up && metal_moe_should_fused_gate_up(count, hidden, ff_length) &&
            metal_moe_can_fused_gate_up(
                &hidden_view, &gate_weight_view, &up_weight_view, &gate_view, &up_view, &fused_view
            );
        indexed_glu_down = !use_grouped_batch &&
            metal_moe_can_indexed_glu_down(
                &down_weight_view, &gate_view, &up_view, &index_view, &weight_view, desc->ffn_type, desc->out
            );
        indexed_down = !indexed_glu_down && !use_grouped_batch &&
            metal_moe_can_indexed_down(&down_weight_view, &fused_view, &index_view, &weight_view, desc->out);
        quant_hints = !use_grouped_decode_gate_up ? metal_moe_quant_matmul_hints(&gate_weight_view, count) : 0;
        if (quant_hints != 0) {
            metal_matmul_quant_push_hints(quant_hints);
            quant_hints_pushed = true;
        }
        if (!use_grouped_decode_gate_up) {
            {
                direct_hidden_alias = tokens == 1 && count == 1 && marmot_tensor_is_contiguous(desc->hidden_states);
                if (direct_hidden_alias) {
                    metal_moe_init_2d_alias(&hidden_view, desc->hidden_states, 1, hidden);
                    status = MARMOT_SUCCESS;
                } else {
                    uint64_t gather_stage_start_ns = metal_moe_stage_profile_begin(&stage_profile);
                    status = metal_gather_rows(device_ctx, desc->hidden_states, &index_view, &hidden_view);
                    metal_moe_stage_profile_end(&stage_profile, ctx, gather_stage_start_ns, &stage_profile.gather_ns);
                }
            }
            if (status != MARMOT_SUCCESS) {
                if (quant_hints_pushed) {
                    metal_matmul_quant_pop_hints(quant_hints);
                }
                if (!use_grouped_batch) {
                    metal_command_batch_end(ctx, false);
                }
                break;
            }
            uint64_t gate_up_stage_start_ns = metal_moe_stage_profile_begin(&stage_profile);
            if (fused_gate_up) {
                status = metal_moe_project_gate_up(
                    src_ctx, &hidden_view, &gate_weight_view, &up_weight_view, &gate_view, &up_view, &fused_view
                );
            } else {
                status = metal_moe_project(device_ctx, ctx, &hidden_view, &gate_weight_view, &gate_view);
                if (status == MARMOT_SUCCESS) {
                    status = metal_moe_project(device_ctx, ctx, &hidden_view, &up_weight_view, &up_view);
                }
            }
            metal_moe_stage_profile_end(&stage_profile, ctx, gate_up_stage_start_ns, &stage_profile.gate_up_ns);
            if (status != MARMOT_SUCCESS) {
                if (quant_hints_pushed) {
                    metal_matmul_quant_pop_hints(quant_hints);
                }
                if (!use_grouped_batch) {
                    metal_command_batch_end(ctx, false);
                }
                break;
            }
            if (!indexed_glu_down) {
                uint64_t glu_stage_start_ns = metal_moe_stage_profile_begin(&stage_profile);
                status = metal_elementwise_binary_impl(device_ctx, glu_op, &gate_view, &up_view, &fused_view);
                metal_moe_stage_profile_end(&stage_profile, ctx, glu_stage_start_ns, &stage_profile.glu_ns);
                if (status != MARMOT_SUCCESS) {
                    if (quant_hints_pushed) {
                        metal_matmul_quant_pop_hints(quant_hints);
                    }
                    if (!use_grouped_batch) {
                        metal_command_batch_end(ctx, false);
                    }
                    break;
                }
            }
        }
        if (!use_grouped_decode_down) {
            uint64_t down_stage_start_ns = metal_moe_stage_profile_begin(&stage_profile);
            if (indexed_glu_down) {
                status = metal_moe_expert_down_glu_indexed(
                    ctx, &gate_view, &up_view, &down_weight_view, &index_view, &weight_view, desc->ffn_type, desc->out
                );
            } else if (indexed_down) {
                status = metal_moe_expert_down_indexed(
                    ctx, &fused_view, &down_weight_view, &index_view, &weight_view, desc->out
                );
            } else {
                status = metal_moe_project(device_ctx, ctx, &fused_view, &down_weight_view, &down_view);
            }
            metal_moe_stage_profile_end(&stage_profile, ctx, down_stage_start_ns, &stage_profile.down_ns);
            if (status != MARMOT_SUCCESS) {
                if (quant_hints_pushed) {
                    metal_matmul_quant_pop_hints(quant_hints);
                }
                if (!use_grouped_batch) {
                    metal_command_batch_end(ctx, false);
                }
                break;
            }
        }
        if (quant_hints_pushed) {
            metal_matmul_quant_pop_hints(quant_hints);
        }
        if (!use_grouped_batch && !indexed_down && !indexed_glu_down) {
            uint64_t scatter_stage_start_ns = metal_moe_stage_profile_begin(&stage_profile);
            status = metal_moe_scatter_add(ctx, &down_view, &index_view, &weight_view, desc->out);
            metal_moe_stage_profile_end(&stage_profile, ctx, scatter_stage_start_ns, &stage_profile.scatter_ns);
        }
        if (!use_grouped_batch) {
            metal_command_batch_end(ctx, status == MARMOT_SUCCESS);
        }
        if (status != MARMOT_SUCCESS) {
            break;
        }
    }

    if (status == MARMOT_SUCCESS && use_grouped_decode_down) {
        uint64_t down_stage_start_ns = metal_moe_stage_profile_begin(&stage_profile);
        if (use_grouped_decode_fused_down) {
            status = metal_moe_decode_down_glu_grouped(
                ctx, &gate_batch, &up_batch, desc->down_exps, &route_experts, &route_weights, desc->ffn_type, desc->out
            );
        } else {
            status = metal_moe_decode_down_grouped(
                ctx, &fused_batch, desc->down_exps, &route_experts, &route_weights, desc->out
            );
        }
        metal_moe_stage_profile_end(&stage_profile, ctx, down_stage_start_ns, &stage_profile.down_ns);
    }
    if (status == MARMOT_SUCCESS && deferred_route_sync) {
        if (!metal_command_stream_wait_for_shared_read(ctx, route_summary_gpu, sizeof(*route_summary_gpu))) {
            metal_command_stream_flush(ctx, true);
        }
        if (!metal_command_stream_wait_for_shared_read(ctx, route_status_gpu, sizeof(*route_status_gpu))) {
            metal_command_stream_flush(ctx, true);
        }
        if (*route_status_gpu != 0u) {
            status = MARMOT_ERROR_INVALID_ARGUMENT;
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "MoE expert id is out of range");
        }
        if (status == MARMOT_SUCCESS) {
            route_count = route_summary_gpu->route_count;
            max_batch = route_summary_gpu->max_batch;
            ordered_active_experts = route_summary_gpu->active_experts;
            if (route_count > route_capacity) {
                status = MARMOT_ERROR_INVALID_OPERATION;
                marmot_set_error(
                    MARMOT_ERROR_INVALID_OPERATION, "Metal MoE GPU routing produced an invalid route count"
                );
            }
        }
        if (status == MARMOT_SUCCESS) {
            route_indices.shape.shape[0] = route_count;
            route_weights.shape.shape[0] = route_count;
            down_batch.shape.shape[0] = route_count;
        }
    }
    if (status == MARMOT_SUCCESS && use_grouped_batch && !use_grouped_decode_down) {
        uint64_t scatter_stage_start_ns = metal_moe_stage_profile_begin(&stage_profile);
        // Use atomic scatter when expert batch path batches ALL routes together —
        // multiple routes targeting the same token (top_k > 1) would race on output.
        const bool need_atomic = use_expert_batch_prefill;
        status = metal_moe_scatter_add(ctx, &down_batch, &route_indices, &route_weights, desc->out, need_atomic);
        metal_moe_stage_profile_end(&stage_profile, ctx, scatter_stage_start_ns, &stage_profile.scatter_ns);
    }
    if (grouped_batch_open) {
        metal_command_batch_end(ctx, status == MARMOT_SUCCESS);
        grouped_batch_open = false;
    }

cleanup:
    if (route_batch_open) {
        metal_command_batch_end(ctx, false);
    }
    if (grouped_batch_open) {
        metal_command_batch_end(ctx, false);
    }
    metal_moe_stage_profile_emit_details(
        &stage_profile, desc, quantized_supported, ordered_active_experts, use_grouped_decode,
        use_grouped_decode_gate_up, use_grouped_decode_down, use_expert_batch_prefill
    );
    metal_moe_stage_profile_emit(
        &stage_profile, tokens, route_count, experts, max_batch, use_host_routes, use_grouped_decode,
        use_grouped_decode_gate_up || use_expert_batch_prefill, use_grouped_decode_down || use_expert_batch_prefill
    );
    metal_moe_workspace_release(ctx, workspace);
    return status;
}

#endif
