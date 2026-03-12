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

static thread_local size_t g_cpu_quant_thread_cap_override = 0;

void cpu_quant_matmul_set_thread_cap_override(size_t thread_cap) {
    g_cpu_quant_thread_cap_override = thread_cap;
}

marmot_error_t
cpu_matmul_quant_pin_range(const void *device_ctx, const void *src, size_t bytes, size_t row_bytes, size_t rows) {
    if (device_ctx == nullptr || src == nullptr || bytes == 0 || row_bytes == 0 || rows == 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    cpu_context_t *cpu_ctx = (cpu_context_t *)device_ctx;
    return cpu_pinned_weight_cache_pin(cpu_ctx, src, bytes, row_bytes, rows) ? MARMOT_SUCCESS
                                                                             : MARMOT_ERROR_OUT_OF_MEMORY;
}

static size_t cpu_quant_matmul_resolve_thread_cap(const cpu_context_t *cpu_ctx, size_t row_count) {
    size_t thread_cap = g_cpu_quant_thread_cap_override;
    if (thread_cap == 0 && cpu_ctx != nullptr && cpu_ctx->num_threads > 1) {
        thread_cap = (size_t)cpu_ctx->num_threads;
    }
    if (thread_cap == 0) {
        thread_cap = 1;
    }
    if (thread_cap > row_count) {
        thread_cap = row_count;
    }
    if (thread_cap == 0) {
        thread_cap = 1;
    }
    return thread_cap;
}

static inline bool cpu_matmul_quant_kernel_available(const cpu_matmul_quant_kernel_t *kernel) {
    return kernel != nullptr &&
        (kernel->ops.dot_q8_0 != nullptr || kernel->ops.dot_q8_k != nullptr || kernel->ops.dot_fp16 != nullptr);
}

static const uint8_t *cpu_quant_matmul_get_packed_weights(
    cpu_context_t *cpu_ctx, const marmot_tensor_t *weight, size_t rows, size_t row_bytes, bool prefer_raw
) {
    const size_t bytes = rows * row_bytes;
    marmot_tensor_t *mutable_weight = (marmot_tensor_t *)weight;
    if (!prefer_raw && mutable_weight->packed_data != nullptr && mutable_weight->packed_src_data == weight->data &&
        mutable_weight->packed_bytes == bytes && mutable_weight->packed_row_bytes == row_bytes &&
        mutable_weight->packed_rows == rows) {
        return (const uint8_t *)mutable_weight->packed_data;
    }

    if (cpu_ctx != nullptr) {
        if (prefer_raw) {
            const uint8_t *prepacked = cpu_prepacked_weight_lookup(cpu_ctx, weight->data, bytes, row_bytes, rows);
            if (prepacked != nullptr) {
                return prepacked;
            }
            const cpu_packed_weight_view_t prepacked_range = cpu_prepacked_weight_lookup_packed_range(
                cpu_ctx, weight->data, rows, row_bytes, 0, 0, 0, CPU_PACKED_WEIGHT_LAYOUT_RAW
            );
            if (prepacked_range.data != nullptr && prepacked_range.data != (const uint8_t *)weight->data) {
                return prepacked_range.data;
            }
            const uint8_t *pinned = cpu_pinned_weight_cache_lookup(cpu_ctx, weight->data, bytes, row_bytes, rows);
            if (pinned != nullptr) {
                return pinned;
            }
            pinned = cpu_pinned_weight_cache_lookup_range(cpu_ctx, weight->data, bytes);
            if (pinned != nullptr) {
                return pinned;
            }
            return (const uint8_t *)weight->data;
        }

        const cpu_packed_weight_view_t prepacked_range = cpu_prepacked_weight_lookup_packed_range(
            cpu_ctx, weight->data, rows, row_bytes, 0, 0, 0, CPU_PACKED_WEIGHT_LAYOUT_RAW
        );
        if (prepacked_range.data != nullptr && prepacked_range.data != (const uint8_t *)weight->data) {
            return prepacked_range.data;
        }

        const cpu_packed_weight_view_t packed_range = cpu_packed_weight_cache_lookup_packed_range(
            cpu_ctx, weight->data, rows, row_bytes, 0, 0, 0, CPU_PACKED_WEIGHT_LAYOUT_RAW
        );
        if (packed_range.data != nullptr && packed_range.data != (const uint8_t *)weight->data) {
            return packed_range.data;
        }

        const uint8_t *pinned = cpu_pinned_weight_cache_lookup(cpu_ctx, weight->data, bytes, row_bytes, rows);
        if (pinned != nullptr) {
            return pinned;
        }
        pinned = cpu_pinned_weight_cache_lookup_range(cpu_ctx, weight->data, bytes);
        if (pinned != nullptr) {
            return pinned;
        }

        return cpu_packed_weight_cache_get(cpu_ctx, weight->data, bytes, row_bytes, rows);
    }

    if (prefer_raw) {
        return (const uint8_t *)weight->data;
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

static inline bool
cpu_quant_matmul_should_use_row_panel_decode(size_t N, bool use_q8_k_activation, marmot_quant_kind_t quant_kind) {
#if MARMOT_ENABLE_NEON && defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
    return N > 2 && N <= 4 && use_q8_k_activation &&
        (quant_kind == MARMOT_QUANT_KIND_Q4_K || quant_kind == MARMOT_QUANT_KIND_Q6_K);
#else
    (void)N;
    (void)use_q8_k_activation;
    (void)quant_kind;
    return false;
#endif
}

static inline bool cpu_quant_matmul_should_use_q4_k_decoded_row_panel(size_t row_count) {
    return row_count >= 16384;
}

static inline bool
cpu_quant_matmul_should_use_q6_k_decoded_row_panel(size_t N, size_t row_count, bool use_q8_k_activation) {
#if MARMOT_ENABLE_NEON && defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
    return use_q8_k_activation && N == 1 && row_count >= 8192;
#else
    (void)N;
    (void)row_count;
    (void)use_q8_k_activation;
    return false;
#endif
}

static inline bool cpu_quant_matmul_should_force_output_projection_q6_k_decoded_row_panel(
    uint32_t hints, marmot_quant_kind_t quant_kind, size_t N, size_t row_count, bool use_q8_k_activation
) {
#if MARMOT_ENABLE_NEON && defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
    return (hints & CPU_QUANT_MATMUL_HINT_OUTPUT_PROJECTION) != 0 && quant_kind == MARMOT_QUANT_KIND_Q6_K &&
        use_q8_k_activation && N > 1 && N <= 4 && row_count >= 32768;
#else
    (void)hints;
    (void)quant_kind;
    (void)N;
    (void)row_count;
    (void)use_q8_k_activation;
    return false;
#endif
}

static cpu_packed_weight_view_t cpu_quant_matmul_lookup_row_panel_weights(
    cpu_context_t *cpu_ctx, const marmot_tensor_t *weight, size_t rows, size_t row_bytes, size_t block_bytes,
    size_t blocks_per_row, const uint8_t *packed_weights, bool use_q4_k_decoded_row_panel,
    bool use_q6_k_decoded_row_panel, bool use_row_panel_decode
) {
    if (!use_row_panel_decode) {
        return (cpu_packed_weight_view_t){
            .data = packed_weights,
            .row_bytes = row_bytes,
            .rows = rows,
            .block_bytes = block_bytes,
            .blocks_per_row = blocks_per_row,
            .panel_rows = CPU_QUANT_DECODE_PANEL_ROWS,
            .packed_bytes = rows * row_bytes,
            .layout = CPU_PACKED_WEIGHT_LAYOUT_RAW,
        };
    }

    cpu_packed_weight_view_t cached = {0};
    if (use_q4_k_decoded_row_panel) {
        cached = cpu_prepacked_weight_lookup_packed_range(
            cpu_ctx, weight->data, rows, row_bytes, 0, blocks_per_row, CPU_QUANT_DECODE_PANEL_ROWS,
            CPU_PACKED_WEIGHT_LAYOUT_Q4_K_ROW_PANEL_DECODED
        );
        if (cached.layout == CPU_PACKED_WEIGHT_LAYOUT_Q4_K_ROW_PANEL_DECODED) {
            return cached;
        }
        cached = cpu_packed_weight_cache_lookup_packed_range(
            cpu_ctx, weight->data, rows, row_bytes, 0, blocks_per_row, CPU_QUANT_DECODE_PANEL_ROWS,
            CPU_PACKED_WEIGHT_LAYOUT_Q4_K_ROW_PANEL_DECODED
        );
        if (cached.layout == CPU_PACKED_WEIGHT_LAYOUT_Q4_K_ROW_PANEL_DECODED) {
            return cached;
        }
        return cpu_packed_weight_cache_get_q4_k_row_panel_decoded(
            cpu_ctx, weight->data, rows, row_bytes, blocks_per_row, CPU_QUANT_DECODE_PANEL_ROWS
        );
    }

    if (use_q6_k_decoded_row_panel) {
        cached = cpu_prepacked_weight_lookup_packed_range(
            cpu_ctx, weight->data, rows, row_bytes, 0, blocks_per_row, CPU_QUANT_DECODE_PANEL_ROWS,
            CPU_PACKED_WEIGHT_LAYOUT_Q6_K_ROW_PANEL_DECODED
        );
        if (cached.layout == CPU_PACKED_WEIGHT_LAYOUT_Q6_K_ROW_PANEL_DECODED) {
            return cached;
        }
        cached = cpu_packed_weight_cache_lookup_packed_range(
            cpu_ctx, weight->data, rows, row_bytes, 0, blocks_per_row, CPU_QUANT_DECODE_PANEL_ROWS,
            CPU_PACKED_WEIGHT_LAYOUT_Q6_K_ROW_PANEL_DECODED
        );
        if (cached.layout == CPU_PACKED_WEIGHT_LAYOUT_Q6_K_ROW_PANEL_DECODED) {
            return cached;
        }
        return cpu_packed_weight_cache_get_q6_k_row_panel_decoded(
            cpu_ctx, weight->data, rows, row_bytes, blocks_per_row, CPU_QUANT_DECODE_PANEL_ROWS
        );
    }

    cached = cpu_prepacked_weight_lookup_packed_range(
        cpu_ctx, weight->data, rows, row_bytes, block_bytes, blocks_per_row, CPU_QUANT_DECODE_PANEL_ROWS,
        CPU_PACKED_WEIGHT_LAYOUT_ROW_PANEL
    );
    if (cached.layout == CPU_PACKED_WEIGHT_LAYOUT_ROW_PANEL) {
        return cached;
    }
    cached = cpu_packed_weight_cache_lookup_packed_range(
        cpu_ctx, weight->data, rows, row_bytes, block_bytes, blocks_per_row, CPU_QUANT_DECODE_PANEL_ROWS,
        CPU_PACKED_WEIGHT_LAYOUT_ROW_PANEL
    );
    if (cached.layout == CPU_PACKED_WEIGHT_LAYOUT_ROW_PANEL) {
        return cached;
    }
    return cpu_packed_weight_cache_get_row_panel(
        cpu_ctx, weight->data, rows, row_bytes, block_bytes, blocks_per_row, CPU_QUANT_DECODE_PANEL_ROWS
    );
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

static inline const marmot_q8_k_block_t *cpu_quant_matmul_prepare_activation_panel_q8_k(
    const marmot_q8_k_block_t *tile_q8_k, marmot_q8_k_block_t *activation_panel_q8_k, size_t blocks_per_row,
    size_t cols_in_tile
) {
    if (tile_q8_k == nullptr || cols_in_tile == 0) {
        return nullptr;
    }
    if (cols_in_tile == 1) {
        return tile_q8_k;
    }
    cpu_matmul_quant_repack_activation_panel_q8_k(tile_q8_k, activation_panel_q8_k, blocks_per_row, cols_in_tile);
    return activation_panel_q8_k;
}

static marmot_error_t cpu_quant_matmul_acquire_workspace(
    cpu_context_t *cpu_ctx, bool use_q8_k_activation, size_t blocks_per_row, cpu_quant_workspace_slot_t **out_slot,
    marmot_q8_0_block_t **out_activation_blocks_q8_0, marmot_q8_k_block_t **out_activation_blocks_q8_k,
    marmot_q8_k_block_t **out_activation_panel_q8_k
) {
    if (out_slot == nullptr || out_activation_blocks_q8_0 == nullptr || out_activation_blocks_q8_k == nullptr ||
        out_activation_panel_q8_k == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Quantized matmul workspace outputs must not be null");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    *out_slot = nullptr;
    *out_activation_blocks_q8_0 = nullptr;
    *out_activation_blocks_q8_k = nullptr;
    *out_activation_panel_q8_k = nullptr;

    cpu_quant_workspace_slot_t *slot = cpu_quant_workspace_acquire(cpu_ctx);
    if (slot == nullptr) {
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to acquire quantized matmul workspace");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    const size_t scratch_blocks = blocks_per_row * CPU_QUANT_MATMUL_TILE_COLS;
    const size_t activation_blocks_bytes =
        scratch_blocks * (use_q8_k_activation ? sizeof(marmot_q8_k_block_t) : sizeof(marmot_q8_0_block_t));
    const size_t activation_panel_bytes = use_q8_k_activation ? scratch_blocks * sizeof(marmot_q8_k_block_t) : 0;
    if (!cpu_quant_workspace_ensure_buffers(slot, activation_blocks_bytes, activation_panel_bytes)) {
        cpu_quant_workspace_release(slot);
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to grow quantized matmul workspace");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    *out_slot = slot;
    if (use_q8_k_activation) {
        *out_activation_blocks_q8_k = (marmot_q8_k_block_t *)slot->activation_blocks;
        *out_activation_panel_q8_k = (marmot_q8_k_block_t *)slot->activation_panel;
    } else {
        *out_activation_blocks_q8_0 = (marmot_q8_0_block_t *)slot->activation_blocks;
    }
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_matmul_quantized_impl(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out, const cpu_matmul_quant_kernel_t *kernel,
    uint32_t hints
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

    const size_t thread_cap = cpu_quant_matmul_resolve_thread_cap(cpu_ctx, M);
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

    cpu_quant_workspace_slot_t *workspace = nullptr;
    marmot_q8_0_block_t *activation_blocks_q8_0 = nullptr;
    marmot_q8_k_block_t *activation_blocks_q8_k = nullptr;
    marmot_q8_k_block_t *activation_panel_q8_k = nullptr;
    status = cpu_quant_matmul_acquire_workspace(
        (cpu_context_t *)device_ctx, use_q8_k_activation, blocks_per_row, &workspace, &activation_blocks_q8_0,
        &activation_blocks_q8_k, &activation_panel_q8_k
    );
    if (status != MARMOT_SUCCESS) {
        return status;
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

    const bool prefer_raw = (hints & CPU_QUANT_MATMUL_HINT_PREFER_RAW) != 0;
    const uint8_t *packed_weights =
        cpu_quant_matmul_get_packed_weights((cpu_context_t *)cpu_ctx, weight, M, row_bytes, prefer_raw);
    const bool use_q4_k_decoded_row_panel = !prefer_raw && weight->quant_kind == MARMOT_QUANT_KIND_Q4_K &&
        cpu_quant_matmul_should_use_row_panel_decode(N, use_q8_k_activation, weight->quant_kind) &&
        cpu_quant_matmul_should_use_q4_k_decoded_row_panel(M);
    const bool force_output_projection_q6_k_decoded_row_panel = !prefer_raw &&
        cpu_quant_matmul_should_force_output_projection_q6_k_decoded_row_panel(
            hints, weight->quant_kind, N, M, use_q8_k_activation
        );
    const bool use_q6_k_decoded_row_panel = !prefer_raw && weight->quant_kind == MARMOT_QUANT_KIND_Q6_K &&
        (force_output_projection_q6_k_decoded_row_panel ||
         cpu_quant_matmul_should_use_q6_k_decoded_row_panel(N, M, use_q8_k_activation));
    const bool use_row_panel_decode = !prefer_raw &&
        (use_q4_k_decoded_row_panel || use_q6_k_decoded_row_panel ||
         cpu_quant_matmul_should_use_row_panel_decode(N, use_q8_k_activation, weight->quant_kind));
    const cpu_packed_weight_view_t row_panel_weights = cpu_quant_matmul_lookup_row_panel_weights(
        (cpu_context_t *)cpu_ctx, weight, M, row_bytes, block_bytes, blocks_per_row, packed_weights,
        use_q4_k_decoded_row_panel, use_q6_k_decoded_row_panel, use_row_panel_decode
    );
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
        const marmot_q8_k_block_t *tile_q8_k_panel = use_q8_k_activation
            ? cpu_quant_matmul_prepare_activation_panel_q8_k(
                  tile_q8_k, activation_panel_q8_k, blocks_per_row, cols_in_tile
              )
            : nullptr;

        if (use_row_panel_decode && row_panel_weights.layout == CPU_PACKED_WEIGHT_LAYOUT_Q4_K_ROW_PANEL_DECODED) {
            cpu_quant_matmul_dispatch_quant_tile_q4_k_row_panel_decoded(
                selected_kernel, row_panel_weights.data, M, blocks_per_row, tile_q8_k, tile_q8_k_panel, cols_in_tile,
                cols_in_tile, n0, out_data_f32, out_data_f16, out_stride_m, out_stride_n, row_panel_weights.panel_rows
            );
        } else if (use_row_panel_decode &&
                   row_panel_weights.layout == CPU_PACKED_WEIGHT_LAYOUT_Q6_K_ROW_PANEL_DECODED) {
            cpu_quant_matmul_dispatch_quant_tile_q6_k_row_panel_decoded(
                selected_kernel, row_panel_weights.data, M, blocks_per_row, tile_q8_k, tile_q8_k_panel, cols_in_tile,
                cols_in_tile, n0, out_data_f32, out_data_f16, out_stride_m, out_stride_n, row_panel_weights.panel_rows
            );
        } else if (use_row_panel_decode && row_panel_weights.layout == CPU_PACKED_WEIGHT_LAYOUT_ROW_PANEL) {
            cpu_quant_matmul_dispatch_quant_tile_row_panel(
                selected_kernel, row_panel_weights.data, M, row_bytes, blocks_per_row, tile_q8_k, tile_q8_k_panel,
                cols_in_tile, cols_in_tile, n0, out_data_f32, out_data_f16, out_stride_m, out_stride_n,
                row_panel_weights.panel_rows, weight->quant_kind
            );
        } else {
            cpu_quant_matmul_dispatch_quant_tile(
                selected_kernel, packed_weights, row_bytes, blocks_per_row, tile_q8_0, tile_q8_k, tile_q8_k_panel,
                cols_in_tile, cols_in_tile, n0, out_data_f32, out_data_f16, out_stride_m, out_stride_n, 0, M,
                thread_cap, weight->quant_kind
            );
        }
    }

    marmot_error_t ep_status = cpu_matmul_apply_epilogue(device_ctx, out, epilogue);
    cpu_quant_workspace_release(workspace);
    return ep_status;
}

static marmot_error_t cpu_matmul_quantized_dual_output_impl(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight_a,
    const marmot_tensor_t *weight_b, marmot_tensor_t *out_a, marmot_tensor_t *out_b
) {
    if (input == nullptr || weight_a == nullptr || weight_b == nullptr || out_a == nullptr || out_b == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Tensor argument must not be null");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (device_ctx == nullptr) {
        marmot_set_error(MARMOT_ERROR_DEVICE_NOT_AVAILABLE, "CPU context is not initialised");
        return MARMOT_ERROR_DEVICE_NOT_AVAILABLE;
    }

    marmot_matmul_dims_t dims_a = {0};
    marmot_matmul_dims_t dims_b = {0};
    marmot_matmul_activation_profile_t profile_a = {0};
    marmot_matmul_activation_profile_t profile_b = {0};
    const marmot_quant_kind_traits_t *quant_traits_a = nullptr;
    const marmot_quant_kind_traits_t *quant_traits_b = nullptr;
    marmot_error_t status =
        marmot_matmul_validate_quantized(input, weight_a, out_a, &dims_a, &profile_a, &quant_traits_a);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    status = marmot_matmul_validate_quantized(input, weight_b, out_b, &dims_b, &profile_b, &quant_traits_b);
    if (status != MARMOT_SUCCESS) {
        return status;
    }

    if (dims_a.N != dims_b.N || dims_a.K != dims_b.K || dims_a.M != dims_b.M) {
        marmot_set_error(
            MARMOT_ERROR_DIMENSION_MISMATCH, "Dual quantized matmul expects matching weight/output shapes"
        );
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (profile_a.input_is_fp32 != profile_b.input_is_fp32 || profile_a.input_is_fp16 != profile_b.input_is_fp16 ||
        profile_a.output_is_fp32 != profile_b.output_is_fp32 || profile_a.output_is_fp16 != profile_b.output_is_fp16) {
        marmot_set_error(
            MARMOT_ERROR_UNSUPPORTED_DTYPE, "Dual quantized matmul requires matching activation/output dtypes"
        );
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (weight_a->quant_kind != weight_b->quant_kind || weight_a->quant_layout != weight_b->quant_layout ||
        weight_a->dtype != weight_b->dtype) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Dual quantized matmul requires matching weight formats");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (out_a->shape.strides[0] != out_b->shape.strides[0] || out_a->shape.strides[1] != out_b->shape.strides[1]) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Dual quantized matmul requires matching output strides");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const bool input_is_fp16 = profile_a.input_is_fp16;
    const bool out_is_fp32 = profile_a.output_is_fp32;
    const bool out_is_fp16 = profile_a.output_is_fp16;
    const size_t N = dims_a.N;
    const size_t K = dims_a.K;
    const size_t M = dims_a.M;

    const cpu_context_t *cpu_ctx = (const cpu_context_t *)device_ctx;
    const cpu_matmul_quant_kernel_t *selected_kernel = cpu_matmul_quant_select_kernel(device_ctx, weight_a->quant_kind);
    const cpu_quant_format_info_t *format = selected_kernel != nullptr ? selected_kernel->format : nullptr;
    if (format == nullptr || format->kind != weight_a->quant_kind) {
        format = cpu_quant_format_info(weight_a->quant_kind);
    }
    if (!cpu_matmul_quant_kernel_available(selected_kernel) || format == nullptr) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Quantized matmul kernel not properly configured");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (format->layout != weight_a->quant_layout || format->layout != weight_b->quant_layout) {
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
    const size_t row_bytes = block_bytes * blocks_per_row;
    const uint8_t *weight_bytes_a = (const uint8_t *)weight_a->data;
    const uint8_t *weight_bytes_b = (const uint8_t *)weight_b->data;

    const size_t thread_cap = cpu_quant_matmul_resolve_thread_cap(cpu_ctx, M);

    const size_t out_stride_m = out_a->shape.strides[0];
    const size_t out_stride_n = out_a->shape.strides[1];
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

    if (input_is_fp16 && selected_kernel->supports_fp16_input && !force_pack) {
        const marmot_float16_t *activation_data_f16 = (const marmot_float16_t *)input->data;
        const size_t act_stride_k = input->shape.strides[1];
        const size_t act_stride_n = input->shape.strides[0];
        float *out_a_data_f32 = out_is_fp32 ? (float *)out_a->data : nullptr;
        marmot_float16_t *out_a_data_f16 = out_is_fp16 ? (marmot_float16_t *)out_a->data : nullptr;
        float *out_b_data_f32 = out_is_fp32 ? (float *)out_b->data : nullptr;
        marmot_float16_t *out_b_data_f16 = out_is_fp16 ? (marmot_float16_t *)out_b->data : nullptr;
        const marmot_float16_t *row_ptrs[CPU_QUANT_MATMUL_TILE_COLS];

        for (size_t n0 = 0; n0 < N; n0 += CPU_QUANT_MATMUL_TILE_COLS) {
            const size_t cols_in_tile = (n0 + CPU_QUANT_MATMUL_TILE_COLS <= N) ? CPU_QUANT_MATMUL_TILE_COLS : (N - n0);
            for (size_t c = 0; c < cols_in_tile; ++c) {
                row_ptrs[c] = activation_data_f16 + (n0 + c) * act_stride_n;
            }
            cpu_quant_matmul_dispatch_fp16_tile_dual(
                selected_kernel, weight_bytes_a, weight_bytes_b, row_bytes, blocks_per_row, row_ptrs, act_stride_k,
                cols_in_tile, n0, out_a_data_f32, out_a_data_f16, out_b_data_f32, out_b_data_f16, out_stride_m,
                out_stride_n, M, K, thread_cap
            );
        }
        return MARMOT_SUCCESS;
    }

    cpu_quant_workspace_slot_t *workspace = nullptr;
    marmot_q8_0_block_t *activation_blocks_q8_0 = nullptr;
    marmot_q8_k_block_t *activation_blocks_q8_k = nullptr;
    marmot_q8_k_block_t *activation_panel_q8_k = nullptr;
    status = cpu_quant_matmul_acquire_workspace(
        (cpu_context_t *)device_ctx, use_q8_k_activation, blocks_per_row, &workspace, &activation_blocks_q8_0,
        &activation_blocks_q8_k, &activation_panel_q8_k
    );
    if (status != MARMOT_SUCCESS) {
        return status;
    }

    const size_t act_stride_k = input->shape.strides[1];
    const size_t act_stride_n = input->shape.strides[0];
    float *out_a_data_f32 = out_is_fp32 ? (float *)out_a->data : nullptr;
    marmot_float16_t *out_a_data_f16 = out_is_fp16 ? (marmot_float16_t *)out_a->data : nullptr;
    float *out_b_data_f32 = out_is_fp32 ? (float *)out_b->data : nullptr;
    marmot_float16_t *out_b_data_f16 = out_is_fp16 ? (marmot_float16_t *)out_b->data : nullptr;
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

    const uint8_t *packed_weights_a =
        cpu_quant_matmul_get_packed_weights((cpu_context_t *)cpu_ctx, weight_a, M, row_bytes, false);
    const uint8_t *packed_weights_b =
        cpu_quant_matmul_get_packed_weights((cpu_context_t *)cpu_ctx, weight_b, M, row_bytes, false);
    const bool use_q4_k_decoded_row_panel = weight_a->quant_kind == MARMOT_QUANT_KIND_Q4_K &&
        cpu_quant_matmul_should_use_row_panel_decode(N, use_q8_k_activation, weight_a->quant_kind) &&
        cpu_quant_matmul_should_use_q4_k_decoded_row_panel(M);
    const bool use_q6_k_decoded_row_panel = weight_a->quant_kind == MARMOT_QUANT_KIND_Q6_K &&
        cpu_quant_matmul_should_use_q6_k_decoded_row_panel(N, M, use_q8_k_activation);
    const bool use_row_panel_decode =
        (use_q4_k_decoded_row_panel || use_q6_k_decoded_row_panel ||
         cpu_quant_matmul_should_use_row_panel_decode(N, use_q8_k_activation, weight_a->quant_kind));
    const cpu_packed_weight_view_t row_panel_weights_a = cpu_quant_matmul_lookup_row_panel_weights(
        (cpu_context_t *)cpu_ctx, weight_a, M, row_bytes, block_bytes, blocks_per_row, packed_weights_a,
        use_q4_k_decoded_row_panel, use_q6_k_decoded_row_panel, use_row_panel_decode
    );
    const cpu_packed_weight_view_t row_panel_weights_b = cpu_quant_matmul_lookup_row_panel_weights(
        (cpu_context_t *)cpu_ctx, weight_b, M, row_bytes, block_bytes, blocks_per_row, packed_weights_b,
        use_q4_k_decoded_row_panel, use_q6_k_decoded_row_panel, use_row_panel_decode
    );
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
        const marmot_q8_k_block_t *tile_q8_k_panel = use_q8_k_activation
            ? cpu_quant_matmul_prepare_activation_panel_q8_k(
                  tile_q8_k, activation_panel_q8_k, blocks_per_row, cols_in_tile
              )
            : nullptr;

        if (use_row_panel_decode && row_panel_weights_a.layout == CPU_PACKED_WEIGHT_LAYOUT_Q4_K_ROW_PANEL_DECODED &&
            row_panel_weights_b.layout == CPU_PACKED_WEIGHT_LAYOUT_Q4_K_ROW_PANEL_DECODED) {
            cpu_quant_matmul_dispatch_quant_tile_q4_k_row_panel_decoded_dual(
                selected_kernel, row_panel_weights_a.data, row_panel_weights_b.data, M, blocks_per_row, tile_q8_k,
                tile_q8_k_panel, cols_in_tile, cols_in_tile, n0, out_a_data_f32, out_a_data_f16, out_b_data_f32,
                out_b_data_f16, out_stride_m, out_stride_n, row_panel_weights_a.panel_rows
            );
        } else if (use_row_panel_decode &&
                   row_panel_weights_a.layout == CPU_PACKED_WEIGHT_LAYOUT_Q6_K_ROW_PANEL_DECODED &&
                   row_panel_weights_b.layout == CPU_PACKED_WEIGHT_LAYOUT_Q6_K_ROW_PANEL_DECODED) {
            cpu_quant_matmul_dispatch_quant_tile_q6_k_row_panel_decoded_dual(
                selected_kernel, row_panel_weights_a.data, row_panel_weights_b.data, M, blocks_per_row, tile_q8_k,
                tile_q8_k_panel, cols_in_tile, cols_in_tile, n0, out_a_data_f32, out_a_data_f16, out_b_data_f32,
                out_b_data_f16, out_stride_m, out_stride_n, row_panel_weights_a.panel_rows
            );
        } else if (use_row_panel_decode && row_panel_weights_a.layout == CPU_PACKED_WEIGHT_LAYOUT_ROW_PANEL &&
                   row_panel_weights_b.layout == CPU_PACKED_WEIGHT_LAYOUT_ROW_PANEL) {
            cpu_quant_matmul_dispatch_quant_tile_row_panel_dual(
                selected_kernel, row_panel_weights_a.data, row_panel_weights_b.data, M, row_bytes, blocks_per_row,
                tile_q8_k, tile_q8_k_panel, cols_in_tile, cols_in_tile, n0, out_a_data_f32, out_a_data_f16,
                out_b_data_f32, out_b_data_f16, out_stride_m, out_stride_n, row_panel_weights_a.panel_rows,
                weight_a->quant_kind
            );
        } else {
            cpu_quant_matmul_dispatch_quant_tile_dual(
                selected_kernel, packed_weights_a, packed_weights_b, row_bytes, blocks_per_row, tile_q8_0, tile_q8_k,
                tile_q8_k_panel, cols_in_tile, cols_in_tile, n0, out_a_data_f32, out_a_data_f16, out_b_data_f32,
                out_b_data_f16, out_stride_m, out_stride_n, 0, M, thread_cap, weight_a->quant_kind
            );
        }
    }

    cpu_quant_workspace_release(workspace);
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_matmul_quantized_dual_output(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight_a,
    const marmot_tensor_t *weight_b, marmot_tensor_t *out_a, marmot_tensor_t *out_b
) {
    return cpu_matmul_quantized_dual_output_impl(device_ctx, input, weight_a, weight_b, out_a, out_b);
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
    cpu_context_t *cpu_ctx = (cpu_context_t *)device_ctx;
    (void)cpu_pinned_weight_cache_pin(cpu_ctx, weight->data, rows * row_bytes, row_bytes, rows);
    (void)cpu_prepacked_weight_store_put_raw(cpu_ctx, weight->data, rows * row_bytes, row_bytes, rows);
    if (weight->quant_kind == MARMOT_QUANT_KIND_Q4_K && cpu_quant_matmul_should_use_q4_k_decoded_row_panel(rows)) {
        (void)cpu_prepacked_weight_store_put_q4_k_row_panel_decoded(
            cpu_ctx, weight->data, rows, row_bytes, blocks_per_row, CPU_QUANT_DECODE_PANEL_ROWS
        );
    } else if (weight->quant_kind == MARMOT_QUANT_KIND_Q6_K &&
               cpu_quant_matmul_should_use_q6_k_decoded_row_panel(1, rows, use_q8_k_activation)) {
        (void)cpu_prepacked_weight_store_put_q6_k_row_panel_decoded(
            cpu_ctx, weight->data, rows, row_bytes, blocks_per_row, CPU_QUANT_DECODE_PANEL_ROWS
        );
    } else if (weight->quant_kind == MARMOT_QUANT_KIND_Q6_K) {
        (void)cpu_prepacked_weight_store_put_row_panel(
            cpu_ctx, weight->data, rows, row_bytes, format->block_bytes, blocks_per_row, CPU_QUANT_DECODE_PANEL_ROWS
        );
    }
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
        return cpu_matmul_quantized_impl(device_ctx, input, weight, epilogue, out, kernel, 0);                         \
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

marmot_error_t cpu_matmul_quantized_with_hints(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out, uint32_t hints
) {
    if (weight == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Quantized matmul weight must not be null");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const cpu_matmul_quant_kernel_t *kernel = cpu_matmul_quant_select_kernel(device_ctx, weight->quant_kind);
    if (kernel == nullptr) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "No kernel for quantized matmul hint dispatch");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    return cpu_matmul_quantized_impl(device_ctx, input, weight, epilogue, out, kernel, hints);
}

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
    return cpu_matmul_quantized_impl(device_ctx, input, weight, epilogue, out, kernel, 0);
}
