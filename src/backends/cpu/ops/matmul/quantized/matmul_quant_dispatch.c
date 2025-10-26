#include "marmot/dispatch.h"
#include "marmot/ops/quantization.h"
#include "marmot/quant_block.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include <float.h>
#include <string.h>

#include "core/helpers/matmul.h"
#include "core/helpers/quant.h"
#include "cpu_backend_internal.h"
#include "ops/matmul/matmul_epilogue.h"
#include "ops/matmul/quantized/matmul_quant_activation.h"
#include "ops/matmul/quantized/matmul_quant_internal.h"
#include "ops/matmul/quantized/matmul_quant_kernels.h"
#include "quantization/format_metadata.h"

static uint8_t g_cpu_quant_dummy_storage = 0;

static inline bool cpu_matmul_quant_kernel_available(const cpu_matmul_quant_kernel_t *kernel) {
    return kernel != nullptr &&
        (kernel->ops.dot_q8_0 != nullptr || kernel->ops.dot_q8_k != nullptr || kernel->ops.dot_fp16 != nullptr);
}

static void cpu_quant_matmul_make_vec_tensor_view(
    marmot_tensor_t *view, size_t K, marmot_quant_kind_t kind, marmot_backend_type_t backend
) {
    const marmot_quant_kind_traits_t *traits = marmot_get_quant_kind_traits(kind);
    memset(view, 0, sizeof(*view));
    view->shape.ndim = 2;
    view->shape.shape[0] = 1;
    view->shape.shape[1] = K;
    view->shape.strides[1] = 1;
    view->shape.strides[0] = K;
    view->dtype = traits != nullptr ? traits->storage_dtype : MARMOT_DTYPE_UINT8;
    view->data = &g_cpu_quant_dummy_storage;
    view->capacity_bytes = 0;
    view->owns_data = false;
    view->quant_kind = kind;
    view->quant_layout = traits != nullptr ? traits->layout : MARMOT_QUANT_LAYOUT_GGUF;
    view->backend = backend;
}

static marmot_error_t cpu_quant_matmul_build_vec_dot_template(
    size_t K, marmot_quant_kind_t weight_kind, marmot_quant_kind_t activation_kind,
    marmot_vec_dot_descriptor_t *out_desc
) {
    if (out_desc == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    marmot_tensor_t weight_vec = {0};
    marmot_tensor_t activation_vec = {0};
    cpu_quant_matmul_make_vec_tensor_view(&weight_vec, K, weight_kind, MARMOT_BACKEND_CPU);
    cpu_quant_matmul_make_vec_tensor_view(&activation_vec, K, activation_kind, MARMOT_BACKEND_CPU);
    marmot_error_t err = marmot_vec_dot_descriptor_from_tensors(&activation_vec, &weight_vec, out_desc);
    if (err == MARMOT_SUCCESS) {
        out_desc->weights = nullptr;
        out_desc->activations = nullptr;
    }
    return err;
}

static const uint8_t *
cpu_quant_matmul_get_packed_weights(const marmot_tensor_t *weight, size_t rows, size_t row_bytes) {
    const size_t bytes = rows * row_bytes;
    marmot_tensor_t *mutable_weight = (marmot_tensor_t *)weight;
    if (mutable_weight->packed_data != nullptr && mutable_weight->packed_src_data == weight->data &&
        mutable_weight->packed_bytes == bytes && mutable_weight->packed_row_bytes == row_bytes &&
        mutable_weight->packed_rows == rows) {
        return (const uint8_t *)mutable_weight->packed_data;
    }

    if (mutable_weight->packed_data != nullptr) {
        free(mutable_weight->packed_data);
        mutable_weight->packed_data = nullptr;
        mutable_weight->packed_src_data = nullptr;
        mutable_weight->packed_bytes = 0;
        mutable_weight->packed_row_bytes = 0;
        mutable_weight->packed_rows = 0;
    }

    uint8_t *buf = (uint8_t *)marmot_aligned_alloc(64, bytes);
    if (buf == nullptr) {
        return (const uint8_t *)weight->data;
    }
    memcpy(buf, weight->data, bytes);

    mutable_weight->packed_data = buf;
    mutable_weight->packed_src_data = weight->data;
    mutable_weight->packed_bytes = bytes;
    mutable_weight->packed_row_bytes = row_bytes;
    mutable_weight->packed_rows = rows;
    return buf;
}

typedef struct {
    const void *activations;
    size_t stride_k;
    size_t stride_n;
    size_t n0;
    size_t K;
    size_t blocks_per_row;
    size_t column_stride_bytes;
    void *activation_blocks;
    cpu_matmul_quant_pack_fp32_fn pack_f32;
    cpu_matmul_quant_pack_fp16_fn pack_f16;
    bool input_is_fp16;
} cpu_quant_matmul_pack_ctx_t;

static void cpu_quant_matmul_pack_column(void *ctx, size_t column_offset) {
    cpu_quant_matmul_pack_ctx_t *pack_ctx = (cpu_quant_matmul_pack_ctx_t *)ctx;
    void *col_blocks = (uint8_t *)pack_ctx->activation_blocks + column_offset * pack_ctx->column_stride_bytes;
    const size_t column_index = pack_ctx->n0 + column_offset;
    if (pack_ctx->input_is_fp16) {
        const marmot_float16_t *activations = (const marmot_float16_t *)pack_ctx->activations;
        pack_ctx->pack_f16(
            activations, pack_ctx->stride_k, pack_ctx->stride_n, column_index, pack_ctx->K, pack_ctx->blocks_per_row,
            col_blocks
        );
        return;
    }

    const float *activations = (const float *)pack_ctx->activations;
    pack_ctx->pack_f32(
        activations, pack_ctx->stride_k, pack_ctx->stride_n, column_index, pack_ctx->K, pack_ctx->blocks_per_row,
        col_blocks
    );
}

static inline void cpu_matmul_quant_repack_activation_panel_q8_k(
    const marmot_q8_k_block_t *src, marmot_q8_k_block_t *dst, size_t blocks_per_row, size_t cols_in_tile
) {
    if (dst == nullptr) {
        return;
    }
    for (size_t b = 0; b < blocks_per_row; ++b) {
        for (size_t c = 0; c < cols_in_tile; ++c) {
            dst[b * cols_in_tile + c] = src[c * blocks_per_row + b];
        }
    }
}

static marmot_error_t cpu_matmul_quantized_impl(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out, const cpu_matmul_quant_kernel_t *kernel
) {
    VALIDATE_TENSORS_3(input, weight, out);

    if (device_ctx == nullptr) {
        marmot_set_error(MARMOT_ERROR_DEVICE_NOT_AVAILABLE, "CPU context is not initialised");
        return MARMOT_ERROR_DEVICE_NOT_AVAILABLE;
    }

    marmot_matmul_dims_t dims = {0};
    marmot_matmul_activation_profile_t profile = {0};
    const marmot_quant_kind_traits_t *quant_traits = nullptr;
    marmot_error_t status = marmot_matmul_validate_quantized(input, weight, out, &dims, &profile, &quant_traits);
    if (status != MARMOT_SUCCESS) {
        return status;
    }

    const bool input_is_fp16 = profile.input_is_fp16;
    const bool out_is_fp32 = profile.output_is_fp32;
    const bool out_is_fp16 = profile.output_is_fp16;

    const size_t N = dims.N;
    const size_t K = dims.K;
    const size_t M = dims.M;

    const cpu_context_t *cpu_ctx = (const cpu_context_t *)device_ctx;

    const cpu_matmul_quant_kernel_t *selected_kernel = kernel;
    const cpu_quant_format_info_t *format = selected_kernel != nullptr ? selected_kernel->format : nullptr;
    if (format == nullptr || format->kind != weight->quant_kind) {
        format = cpu_quant_format_info(weight->quant_kind);
    }
    if (!cpu_matmul_quant_kernel_available(selected_kernel) || format == nullptr) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Quantized matmul kernel not properly configured");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (format->layout != weight->quant_layout) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Quantized matmul layout mismatch");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (input_is_fp16 && !selected_kernel->supports_fp16_input) {
        marmot_set_error(
            MARMOT_ERROR_UNSUPPORTED_DTYPE, "Quantized matmul FP16 activations are not supported for this quant format"
        );
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    const size_t block_values = format->block_values;
    const size_t block_bytes = format->block_bytes;
    const size_t blocks_per_row = (K + block_values - 1) / block_values;
    const uint8_t *weight_bytes = (const uint8_t *)weight->data;
    const size_t row_bytes = block_bytes * blocks_per_row;

    size_t thread_cap = 1;
    if (cpu_ctx != nullptr && cpu_ctx->num_threads > 1) {
        thread_cap = (size_t)cpu_ctx->num_threads;
    }
    if (thread_cap > M) {
        thread_cap = M;
    }
    if (thread_cap == 0) {
        thread_cap = 1;
    }
    const size_t out_stride_m = out->shape.strides[0];
    const size_t out_stride_n = out->shape.strides[1];

    const bool backend_force_hint =
        cpu_ctx->force_q8_activations && cpu_ctx->quant_activation_mode == MARMOT_QUANT_ACTIVATION_AUTO;
    const bool force_pack = marmot_matmul_quant_should_force_pack(cpu_ctx->quant_activation_mode, backend_force_hint);
    const bool pack_activations = !input_is_fp16 || force_pack;
    const bool use_q8_k_activation = format->activation_packer == MARMOT_MATMUL_ACTIVATION_PACKER_Q8_K;
    if (pack_activations) {
        if (use_q8_k_activation && selected_kernel->ops.dot_q8_k == nullptr) {
            marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Quantized matmul kernel missing Q8_K activation support");
            return MARMOT_ERROR_NOT_IMPLEMENTED;
        }
        if (!use_q8_k_activation && selected_kernel->ops.dot_q8_0 == nullptr) {
            marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Quantized matmul kernel missing Q8_0 activation support");
            return MARMOT_ERROR_NOT_IMPLEMENTED;
        }
    }

    const marmot_quant_kind_t activation_quant_kind =
        use_q8_k_activation ? MARMOT_QUANT_KIND_Q8_K : MARMOT_QUANT_KIND_Q8_0;
    marmot_vec_dot_descriptor_t vec_dot_template = {0};
    marmot_error_t vec_desc_err =
        cpu_quant_matmul_build_vec_dot_template(K, weight->quant_kind, activation_quant_kind, &vec_dot_template);
    if (vec_desc_err != MARMOT_SUCCESS) {
        return vec_desc_err;
    }
    if (vec_dot_template.num_blocks != blocks_per_row) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Vec dot descriptor does not match packed block count");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (input_is_fp16 && selected_kernel->supports_fp16_input && !force_pack) {
        const marmot_float16_t *activation_data_f16 = (const marmot_float16_t *)input->data;
        const size_t act_stride_k = input->shape.strides[1];
        const size_t act_stride_n = input->shape.strides[0];
        float *out_data_f32 = out_is_fp32 ? (float *)out->data : nullptr;
        marmot_float16_t *out_data_f16 = out_is_fp16 ? (marmot_float16_t *)out->data : nullptr;

        const marmot_float16_t *row_ptrs[CPU_QUANT_MATMUL_TILE_COLS];

        for (size_t n0 = 0; n0 < N; n0 += CPU_QUANT_MATMUL_TILE_COLS) {
            const size_t cols_in_tile = (n0 + CPU_QUANT_MATMUL_TILE_COLS <= N) ? CPU_QUANT_MATMUL_TILE_COLS : (N - n0);
            for (size_t c = 0; c < cols_in_tile; ++c) {
                row_ptrs[c] = activation_data_f16 + (n0 + c) * act_stride_n;
            }
            cpu_quant_matmul_dispatch_fp16_tile(
                selected_kernel, weight_bytes, row_bytes, blocks_per_row, row_ptrs, act_stride_k, cols_in_tile, n0,
                out_data_f32, out_data_f16, out_stride_m, out_stride_n, M, K, thread_cap
            );
        }

        return cpu_matmul_apply_epilogue(device_ctx, out, epilogue);
    }

    marmot_q8_0_block_t *activation_blocks_q8_0 = nullptr;
    marmot_q8_k_block_t *activation_blocks_q8_k = nullptr;
    marmot_q8_k_block_t *activation_panel_q8_k = nullptr;
    if (use_q8_k_activation) {
        const size_t scratch_blocks = blocks_per_row * CPU_QUANT_MATMUL_TILE_COLS;
        activation_blocks_q8_k = (marmot_q8_k_block_t *)malloc(sizeof(marmot_q8_k_block_t) * scratch_blocks);
        if (activation_blocks_q8_k == nullptr) {
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to allocate Q8_K activation scratch buffer");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        activation_panel_q8_k = (marmot_q8_k_block_t *)malloc(sizeof(marmot_q8_k_block_t) * scratch_blocks);
        if (activation_panel_q8_k == nullptr) {
            free(activation_blocks_q8_k);
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to allocate Q8_K activation panel buffer");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
    } else {
        activation_blocks_q8_0 =
            (marmot_q8_0_block_t *)malloc(sizeof(marmot_q8_0_block_t) * blocks_per_row * CPU_QUANT_MATMUL_TILE_COLS);
        if (activation_blocks_q8_0 == nullptr) {
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to allocate activation scratch buffer");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
    }

    const size_t act_stride_k = input->shape.strides[1];
    const size_t act_stride_n = input->shape.strides[0];
    float *out_data_f32 = out_is_fp32 ? (float *)out->data : nullptr;
    marmot_float16_t *out_data_f16 = out_is_fp16 ? (marmot_float16_t *)out->data : nullptr;
    const float *activation_data_f32 = (const float *)input->data;
    const marmot_float16_t *activation_data_f16 = (const marmot_float16_t *)input->data;

    cpu_matmul_quant_pack_fp32_fn pack_f32_default =
        use_q8_k_activation ? cpu_matmul_quant_pack_q8_k_f32 : cpu_matmul_quant_pack_q8_0_f32;
    cpu_matmul_quant_pack_fp16_fn pack_f16_default =
        use_q8_k_activation ? cpu_matmul_quant_pack_q8_k_f16 : cpu_matmul_quant_pack_q8_0_f16;

    cpu_matmul_quant_pack_fp32_fn pack_f32 = selected_kernel->ops.pack_activations_f32 != nullptr
        ? selected_kernel->ops.pack_activations_f32
        : pack_f32_default;
    cpu_matmul_quant_pack_fp16_fn pack_f16 = selected_kernel->ops.pack_activations_f16 != nullptr
        ? selected_kernel->ops.pack_activations_f16
        : pack_f16_default;

    const uint8_t *packed_weights = cpu_quant_matmul_get_packed_weights(weight, M, row_bytes);
    const size_t activation_column_stride_bytes =
        blocks_per_row * (use_q8_k_activation ? sizeof(marmot_q8_k_block_t) : sizeof(marmot_q8_0_block_t));
    void *activation_blocks = use_q8_k_activation ? (void *)activation_blocks_q8_k : (void *)activation_blocks_q8_0;

    for (size_t n0 = 0; n0 < N; n0 += CPU_QUANT_MATMUL_TILE_COLS) {
        const size_t cols_in_tile = (n0 + CPU_QUANT_MATMUL_TILE_COLS <= N) ? CPU_QUANT_MATMUL_TILE_COLS : (N - n0);

        cpu_quant_matmul_pack_ctx_t pack_ctx = {
            .activations = input_is_fp16 ? (const void *)activation_data_f16 : (const void *)activation_data_f32,
            .stride_k = act_stride_k,
            .stride_n = act_stride_n,
            .n0 = n0,
            .K = K,
            .blocks_per_row = blocks_per_row,
            .column_stride_bytes = activation_column_stride_bytes,
            .activation_blocks = activation_blocks,
            .pack_f32 = pack_f32,
            .pack_f16 = pack_f16,
            .input_is_fp16 = input_is_fp16,
        };
        const bool pack_parallel = cols_in_tile >= 4 && cpu_ctx != nullptr && cpu_ctx->num_threads > 1;
        if (pack_parallel) {
            marmot_dispatch_parallel_for(
                MARMOT_DISPATCH_PRIORITY_HIGH, cols_in_tile, &pack_ctx, cpu_quant_matmul_pack_column
            );
        } else {
            for (size_t c = 0; c < cols_in_tile; ++c) {
                cpu_quant_matmul_pack_column(&pack_ctx, c);
            }
        }

        const marmot_q8_0_block_t *tile_q8_0 = use_q8_k_activation ? nullptr : activation_blocks_q8_0;
        const marmot_q8_k_block_t *tile_q8_k = use_q8_k_activation ? activation_blocks_q8_k : nullptr;
        marmot_q8_k_block_t *tile_q8_k_panel = nullptr;
        if (use_q8_k_activation) {
            cpu_matmul_quant_repack_activation_panel_q8_k(
                tile_q8_k, activation_panel_q8_k, blocks_per_row, cols_in_tile
            );
            tile_q8_k_panel = activation_panel_q8_k;
        }

        cpu_quant_matmul_dispatch_quant_tile(
            selected_kernel, packed_weights, row_bytes, blocks_per_row, tile_q8_0, tile_q8_k, tile_q8_k_panel,
            cols_in_tile, cols_in_tile, n0, out_data_f32, out_data_f16, out_stride_m, out_stride_n, 0, M, thread_cap,
            weight->quant_kind
        );
    }

    marmot_error_t ep_status = cpu_matmul_apply_epilogue(device_ctx, out, epilogue);
    free(activation_blocks_q8_k);
    free(activation_panel_q8_k);
    free(activation_blocks_q8_0);
    return ep_status;
}

marmot_error_t cpu_matmul_quant_prepack(const void *device_ctx, const marmot_tensor_t *weight) {
    if (device_ctx == nullptr || weight == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (!marmot_quant_kind_is_block_quantized(weight->quant_kind)) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const size_t K = weight->shape.shape[1];
    const cpu_matmul_quant_kernel_t *kernel = cpu_matmul_quant_select_kernel(device_ctx, weight->quant_kind);
    const cpu_quant_format_info_t *format = kernel != nullptr ? kernel->format : nullptr;
    if (format == nullptr || format->kind != weight->quant_kind) {
        format = cpu_quant_format_info(weight->quant_kind);
    }
    if (!cpu_matmul_quant_kernel_available(kernel) || format == nullptr) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    const bool use_q8_k_activation = format->activation_packer == MARMOT_MATMUL_ACTIVATION_PACKER_Q8_K;
    if (use_q8_k_activation && kernel->ops.dot_q8_k == nullptr) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (!use_q8_k_activation && kernel->ops.dot_q8_0 == nullptr) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    const size_t blocks_per_row = (K + format->block_values - 1) / format->block_values;
    const size_t row_bytes = format->block_bytes * blocks_per_row;
    const size_t rows = weight->shape.shape[0];
    cpu_quant_matmul_get_packed_weights(weight, rows, row_bytes);
    return MARMOT_SUCCESS;
}

#define DEFINE_CPU_MATMUL_QUANTIZED_DIRECT(suffix, quant_kind)                                                         \
    marmot_error_t cpu_matmul_quantized_##suffix(                                                                      \
        const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,                           \
        const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out                                                 \
    ) {                                                                                                                \
        const cpu_matmul_quant_kernel_t *kernel = cpu_matmul_quant_select_kernel(device_ctx, quant_kind);              \
        if (kernel == nullptr) {                                                                                       \
            marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "No kernel for quant kind " #suffix);                       \
            return MARMOT_ERROR_NOT_IMPLEMENTED;                                                                       \
        }                                                                                                              \
        return cpu_matmul_quantized_impl(device_ctx, input, weight, epilogue, out, kernel);                            \
    }

DEFINE_CPU_MATMUL_QUANTIZED_DIRECT(q2_k, MARMOT_QUANT_KIND_Q2_K)
DEFINE_CPU_MATMUL_QUANTIZED_DIRECT(q3_k, MARMOT_QUANT_KIND_Q3_K)
DEFINE_CPU_MATMUL_QUANTIZED_DIRECT(q4_k, MARMOT_QUANT_KIND_Q4_K)
DEFINE_CPU_MATMUL_QUANTIZED_DIRECT(q5_k, MARMOT_QUANT_KIND_Q5_K)
DEFINE_CPU_MATMUL_QUANTIZED_DIRECT(q6_k, MARMOT_QUANT_KIND_Q6_K)
DEFINE_CPU_MATMUL_QUANTIZED_DIRECT(q8_k, MARMOT_QUANT_KIND_Q8_K)
DEFINE_CPU_MATMUL_QUANTIZED_DIRECT(q4_0, MARMOT_QUANT_KIND_Q4_0)
DEFINE_CPU_MATMUL_QUANTIZED_DIRECT(q4_1, MARMOT_QUANT_KIND_Q4_1)
DEFINE_CPU_MATMUL_QUANTIZED_DIRECT(q5_0, MARMOT_QUANT_KIND_Q5_0)
DEFINE_CPU_MATMUL_QUANTIZED_DIRECT(q5_1, MARMOT_QUANT_KIND_Q5_1)
DEFINE_CPU_MATMUL_QUANTIZED_DIRECT(q8_0, MARMOT_QUANT_KIND_Q8_0)
DEFINE_CPU_MATMUL_QUANTIZED_DIRECT(q8_1, MARMOT_QUANT_KIND_Q8_1)

#undef DEFINE_CPU_MATMUL_QUANTIZED_DIRECT

marmot_error_t cpu_matmul_quantized(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
) {
    const cpu_matmul_quant_kernel_t *kernel = cpu_matmul_quant_select_kernel(device_ctx, weight->quant_kind);
    if (kernel == nullptr) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "No kernel for weight quant kind");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    return cpu_matmul_quantized_impl(device_ctx, input, weight, epilogue, out, kernel);
}
