#include "core/bytecode/bytecode_compile.h"

#include "marmot/error.h"
#include "marmot/op_metadata.gen.h"
#include "marmot/op_signature_hash.gen.h"
#include "marmot/types.h"

#include <stdalign.h>
#include <stdlib.h>

#include <string.h>

#include "backends/cpu/dispatch/bytecode_exec_cpu.gen.h"
#include "backends/cpu/dispatch/bytecode_tables_cpu.gen.h"
#include "core/dispatch/kernel_query.h"
#include "graph/kernel_dispatch_args.gen.h"

#if MARMOT_ENABLE_METAL
#include "backends/metal/ops/bytecode_exec_metal.gen.h"
#include "backends/metal/ops/bytecode_tables_metal.gen.h"
#endif

enum { MARMOT_BC_MAX_REGS = 64 };
enum { MARMOT_BC_COMPILE_CACHE_SIZE = 64 };

typedef struct marmot_bc_compile_cache_entry {
    uint64_t sig_hash;
    uint64_t caps_hash;
    marmot_backend_type_t backend;
    bool allow_fallback;
    marmot_bc_selection_t selection;
    bool valid;
} marmot_bc_compile_cache_entry_t;

typedef struct marmot_bc_compile_cache {
    marmot_bc_compile_cache_entry_t entries[MARMOT_BC_COMPILE_CACHE_SIZE];
    size_t count;
    size_t next_slot;
} marmot_bc_compile_cache_t;

static thread_local marmot_bc_compile_cache_t g_bc_compile_cache = {0};

static bool marmot_bc_emit_tensor_reg(
    marmot_bc_builder_t *builder, marmot_tensor_t **regs, uint16_t *reg_count, const marmot_tensor_t *tensor
) {
    if (builder == nullptr || regs == nullptr || reg_count == nullptr) {
        return false;
    }
    uint16_t idx = MARMOT_BC_REG_INVALID;
    if (tensor != nullptr) {
        if (*reg_count >= MARMOT_BC_MAX_REGS) {
            return false;
        }
        idx = *reg_count;
        regs[idx] = (marmot_tensor_t *)tensor;
        *reg_count = (uint16_t)(*reg_count + 1);
    }
    return marmot_bc_builder_emit_u16(builder, idx);
}

static bool marmot_bc_assign_tensor_reg(
    marmot_tensor_t **regs, uint16_t *reg_count, const marmot_tensor_t *tensor, uint16_t *out_idx
) {
    if (regs == nullptr || reg_count == nullptr || out_idx == nullptr) {
        return false;
    }
    uint16_t idx = MARMOT_BC_REG_INVALID;
    if (tensor != nullptr) {
        if (*reg_count >= MARMOT_BC_MAX_REGS) {
            return false;
        }
        idx = *reg_count;
        regs[idx] = (marmot_tensor_t *)tensor;
        *reg_count = (uint16_t)(*reg_count + 1);
    }
    *out_idx = idx;
    return true;
}

static uint32_t
marmot_bc_add_const_data(marmot_bc_builder_t *builder, const void *data, size_t size, size_t alignment) {
    if (data == nullptr || size == 0) {
        return MARMOT_BC_INVALID_OFFSET;
    }
    return marmot_bc_builder_add_const(builder, data, size, alignment);
}

static uint32_t marmot_bc_add_const_ptr_value(marmot_bc_builder_t *builder, const void *ptr) {
    if (ptr == nullptr) {
        return MARMOT_BC_INVALID_OFFSET;
    }
    return marmot_bc_builder_add_const(builder, &ptr, sizeof(ptr), alignof(void *));
}

static bool marmot_bc_infer_transpose_perm(
    const marmot_tensor_t *input, const marmot_tensor_t *output, int *perm_out, size_t *ndim_out
) {
    if (input == nullptr || output == nullptr || perm_out == nullptr || ndim_out == nullptr) {
        return false;
    }
    const size_t ndim = input->shape.ndim;
    if (ndim != output->shape.ndim || ndim > MARMOT_MAX_DIMS) {
        return false;
    }
    bool used[MARMOT_MAX_DIMS] = {false};
    for (size_t i = 0; i < ndim; ++i) {
        const size_t target = output->shape.shape[i];
        int found = -1;
        for (size_t j = 0; j < ndim; ++j) {
            if (used[j] || input->shape.shape[j] != target) {
                continue;
            }
            if (found != -1) {
                return false;
            }
            found = (int)j;
        }
        if (found < 0) {
            return false;
        }
        perm_out[i] = found;
        used[(size_t)found] = true;
    }
    *ndim_out = ndim;
    return true;
}

bool marmot_bc_get_tables(marmot_backend_type_t backend, marmot_bc_tables_t *out) {
    if (out == nullptr) {
        return false;
    }
    switch (backend) {
    case MARMOT_BACKEND_CPU:
        *out = (marmot_bc_tables_t){
            .imm_size = marmot_cpu_bc_imm_size,
            .exec_table = marmot_cpu_bc_exec_table,
            .schema_id = marmot_cpu_bc_schema_id,
            .op_count = MARMOT_CPU_BC_OP_COUNT,
        };
        return true;
    case MARMOT_BACKEND_METAL:
#if MARMOT_ENABLE_METAL
        *out = (marmot_bc_tables_t){
            .imm_size = marmot_metal_bc_imm_size,
            .exec_table = marmot_metal_bc_exec_table,
            .schema_id = marmot_metal_bc_schema_id,
            .op_count = MARMOT_METAL_BC_OP_COUNT,
        };
        return true;
#else
        return false;
#endif
    default:
        return false;
    }
}

bool marmot_bc_exec_supported(marmot_backend_type_t backend, uint16_t op_index) {
    marmot_bc_tables_t tables = {0};
    if (!marmot_bc_get_tables(backend, &tables)) {
        return false;
    }
    if (op_index >= tables.op_count) {
        return false;
    }
    return tables.exec_table[op_index] != nullptr;
}

static uint64_t marmot_bc_hash_caps(const marmot_device_caps_t *caps) {
    if (caps == nullptr) {
        return 0;
    }
    const uint8_t *bytes = (const uint8_t *)caps;
    size_t len = sizeof(*caps);
    uint64_t h = 14695981039346656037ULL;
    const uint64_t prime = 1099511628211ULL;
    for (size_t i = 0; i < len; ++i) {
        h ^= (uint64_t)bytes[i];
        h *= prime;
    }
    return h;
}

static bool marmot_bc_cache_lookup(
    const marmot_bc_compile_cache_t *cache, uint64_t sig_hash, uint64_t caps_hash, marmot_backend_type_t backend,
    bool allow_fallback, marmot_bc_selection_t *out
) {
    if (cache == nullptr || out == nullptr) {
        return false;
    }
    for (size_t i = 0; i < cache->count; ++i) {
        const marmot_bc_compile_cache_entry_t *entry = &cache->entries[i];
        if (!entry->valid) {
            continue;
        }
        if (entry->sig_hash == sig_hash && entry->caps_hash == caps_hash && entry->backend == backend &&
            entry->allow_fallback == allow_fallback) {
            *out = entry->selection;
            return true;
        }
    }
    return false;
}

static void marmot_bc_cache_insert(
    marmot_bc_compile_cache_t *cache, uint64_t sig_hash, uint64_t caps_hash, marmot_backend_type_t backend,
    bool allow_fallback, const marmot_bc_selection_t *selection
) {
    if (cache == nullptr || selection == nullptr) {
        return;
    }
    size_t slot = cache->next_slot;
    cache->entries[slot] = (marmot_bc_compile_cache_entry_t){
        .sig_hash = sig_hash,
        .caps_hash = caps_hash,
        .backend = backend,
        .allow_fallback = allow_fallback,
        .selection = *selection,
        .valid = true,
    };
    cache->next_slot = (slot + 1) % MARMOT_BC_COMPILE_CACHE_SIZE;
    if (cache->count < MARMOT_BC_COMPILE_CACHE_SIZE) {
        cache->count++;
    }
}

marmot_bc_selection_t marmot_bc_compile_signature(const marmot_context_t *ctx, const marmot_op_signature_t *sig) {
    marmot_bc_selection_t result = {
        .supported = false,
        .op_index = MARMOT_BC_OP_INVALID,
        .resolved_sig = {0},
        .reason = "Invalid arguments",
    };
    if (ctx == nullptr || sig == nullptr) {
        return result;
    }
    return marmot_bc_compile_signature_with_caps(ctx->backend_type, &ctx->device_caps, sig, true);
}

marmot_bc_selection_t marmot_bc_compile_signature_with_caps(
    marmot_backend_type_t backend, const marmot_device_caps_t *caps, const marmot_op_signature_t *sig,
    bool allow_fallback
) {
    marmot_bc_selection_t result = {
        .supported = false,
        .op_index = MARMOT_BC_OP_INVALID,
        .resolved_sig = {0},
        .reason = "Invalid arguments",
    };
    if (sig == nullptr || caps == nullptr) {
        return result;
    }

    uint64_t sig_hash = marmot_hash_op_signature(sig);
    uint64_t caps_hash = marmot_bc_hash_caps(caps);
    if (marmot_bc_cache_lookup(&g_bc_compile_cache, sig_hash, caps_hash, backend, allow_fallback, &result)) {
        return result;
    }

    marmot_op_signature_t resolved_sig = *sig;
    marmot_kernel_selection_t selection = allow_fallback
        ? marmot_backend_query_kernel_with_fallback(backend, sig, caps, &resolved_sig)
        : marmot_backend_query_kernel(backend, sig, caps);
    if (!selection.supported) {
        result.reason = selection.fallback_reason != nullptr ? selection.fallback_reason : "Kernel not supported";
        return result;
    }

    uint16_t op_index = selection.op_index;
    if (op_index == MARMOT_KERNEL_OP_INDEX_INVALID) {
        result.reason = "Kernel missing from bytecode table";
        return result;
    }
    if (!marmot_bc_exec_supported(backend, op_index)) {
        result.reason = "Bytecode executor not available";
        return result;
    }

    result.supported = true;
    result.op_index = op_index;
    result.resolved_sig = resolved_sig;
    result.reason = nullptr;
    marmot_bc_cache_insert(&g_bc_compile_cache, sig_hash, caps_hash, backend, allow_fallback, &result);
    return result;
}

marmot_error_t marmot_bc_execute_op(
    marmot_backend_type_t backend, uint16_t op_index, const marmot_bc_exec_ctx_t *exec_ctx, const void *args
) {
    if (exec_ctx == nullptr || exec_ctx->device_ctx == nullptr || args == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null arguments for bytecode execution");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_bc_tables_t tables = {0};
    if (!marmot_bc_get_tables(backend, &tables)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (op_index == MARMOT_BC_OP_INVALID || op_index >= tables.op_count) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (tables.exec_table[op_index] == nullptr) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    if (tables.schema_id == nullptr) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    marmot_bc_schema_id_t schema_id = tables.schema_id[op_index];
    if (schema_id == MARMOT_BC_SCHEMA_INVALID) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    marmot_bc_builder_t builder = {0};
    if (!marmot_bc_builder_init(&builder)) {
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to init bytecode builder");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    marmot_tensor_t *regs[MARMOT_BC_MAX_REGS] = {nullptr};
    uint16_t reg_count = 0;

    if (!marmot_bc_builder_emit_u16(&builder, op_index)) {
        marmot_bc_builder_reset(&builder);
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to emit bytecode op");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    switch (schema_id) {
    case MARMOT_BC_SCHEMA_UNARY: {
        const marmot_kernel_args_unary_t *packed = args;
        const marmot_tensor_t *input = packed->input;
        marmot_tensor_t *output = packed->output;
        const marmot_activation_params_t *params = packed->params;
        if (input == nullptr || output == nullptr) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null tensors for unary bytecode execution");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        if (!marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, input) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, output)) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode unary operands");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        uint32_t params_offset = marmot_bc_add_const_data(
            &builder, params, params != nullptr ? sizeof(*params) : 0, alignof(marmot_activation_params_t)
        );
        if (!marmot_bc_builder_emit_u32(&builder, params_offset)) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode unary params");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        break;
    }
    case MARMOT_BC_SCHEMA_BINARY: {
        const marmot_kernel_args_binary_t *packed = args;
        const marmot_tensor_t *input_a = packed->input_a;
        const marmot_tensor_t *input_b = packed->input_b;
        marmot_tensor_t *output = packed->output;
        if (input_a == nullptr || input_b == nullptr || output == nullptr) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null tensors for binary bytecode execution");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        if (!marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, input_a) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, input_b) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, output)) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode binary operands");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        break;
    }
    case MARMOT_BC_SCHEMA_TERNARY: {
        const marmot_kernel_args_ternary_t *packed = args;
        const marmot_tensor_t *input_a = packed->input_a;
        const marmot_tensor_t *input_b = packed->input_b;
        const marmot_tensor_t *input_c = packed->input_c;
        marmot_tensor_t *output = packed->output;
        if (input_a == nullptr || input_b == nullptr || input_c == nullptr || output == nullptr) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null tensors for ternary bytecode execution");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        if (!marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, input_a) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, input_b) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, input_c) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, output)) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode ternary operands");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        break;
    }
    case MARMOT_BC_SCHEMA_REDUCTION: {
        const marmot_kernel_args_reduction_t *packed = args;
        const marmot_tensor_t *input = packed->input;
        marmot_tensor_t *out_values = packed->out_values;
        marmot_tensor_t *out_indices = packed->out_indices;
        const marmot_reduction_params_t *params = packed->params;
        if (input == nullptr || out_values == nullptr) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null tensors for reduction bytecode execution");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        if (!marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, input) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, out_values) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, out_indices)) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode reduction operands");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        uint32_t axes_offset = MARMOT_BC_INVALID_OFFSET;
        uint64_t num_axes = 0;
        uint8_t keepdims = 0;
        uint8_t unbiased = 0;
        float epsilon = 0.0f;
        if (params != nullptr) {
            num_axes = (uint64_t)params->num_axes;
            keepdims = params->keepdims ? 1 : 0;
            unbiased = params->unbiased ? 1 : 0;
            epsilon = params->epsilon;
            if (params->axes != nullptr && params->num_axes > 0) {
                axes_offset = marmot_bc_add_const_data(
                    &builder, params->axes, params->num_axes * sizeof(int32_t), alignof(int32_t)
                );
            }
        }
        if (!marmot_bc_builder_emit_u32(&builder, axes_offset) || !marmot_bc_builder_emit_u64(&builder, num_axes) ||
            !marmot_bc_builder_emit_u8(&builder, keepdims) || !marmot_bc_builder_emit_u8(&builder, unbiased) ||
            !marmot_bc_builder_emit_f32(&builder, epsilon)) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode reduction params");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        break;
    }
    case MARMOT_BC_SCHEMA_SOFTMAX: {
        const marmot_kernel_args_softmax_t *packed = args;
        const marmot_tensor_t *input = packed->input;
        marmot_tensor_t *output = packed->output;
        int32_t axis = packed->axis;
        if (input == nullptr || output == nullptr) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null args for softmax bytecode execution");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        if (!marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, input) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, output) ||
            !marmot_bc_builder_emit_u32(&builder, (uint32_t)axis)) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode softmax args");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        break;
    }
    case MARMOT_BC_SCHEMA_TOPK: {
        const marmot_kernel_args_topk_t *packed = args;
        const marmot_tensor_t *input = packed->input;
        marmot_tensor_t *values_out = packed->values_out;
        marmot_tensor_t *indices_out = packed->indices_out;
        if (input == nullptr || values_out == nullptr || indices_out == nullptr) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null args for TopK bytecode execution");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        if (!marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, input) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, values_out) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, indices_out) ||
            !marmot_bc_builder_emit_u32(&builder, (uint32_t)packed->axis) ||
            !marmot_bc_builder_emit_u32(&builder, packed->k)) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode TopK args");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        break;
    }
    case MARMOT_BC_SCHEMA_MOE_EXPERTS: {
        const marmot_kernel_args_moe_experts_t *packed = args;
        if (packed->hidden_states == nullptr || packed->gate_exps == nullptr || packed->up_exps == nullptr ||
            packed->down_exps == nullptr || packed->topk_ids == nullptr || packed->topk_weights == nullptr ||
            packed->out == nullptr) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null args for MoE experts bytecode execution");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        if (!marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, packed->hidden_states) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, packed->gate_exps) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, packed->up_exps) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, packed->down_exps) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, packed->topk_ids) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, packed->topk_weights) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, packed->out) ||
            !marmot_bc_builder_emit_u32(&builder, (uint32_t)packed->ffn_type) ||
            !marmot_bc_builder_emit_f32(&builder, packed->weights_scale) ||
            !marmot_bc_builder_emit_u32(&builder, (uint32_t)packed->router_weight_policy)) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode MoE experts args");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        break;
    }
    case MARMOT_BC_SCHEMA_LAYERNORM: {
        const marmot_kernel_args_layernorm_t *packed = args;
        const marmot_tensor_t *input = packed->input;
        const marmot_tensor_t *weight = packed->weight;
        const marmot_tensor_t *bias = packed->bias;
        const marmot_tensor_t *residual = packed->residual;
        marmot_tensor_t *output = packed->output;
        float eps = packed->eps;
        if (input == nullptr || output == nullptr) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null args for layernorm bytecode execution");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        if (!marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, input) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, weight) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, bias) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, residual) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, output) ||
            !marmot_bc_builder_emit_f32(&builder, eps)) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode layernorm args");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        break;
    }
    case MARMOT_BC_SCHEMA_RMS_NORM: {
        const marmot_kernel_args_rms_norm_t *packed = args;
        const marmot_tensor_t *input = packed->input;
        const marmot_tensor_t *weight = packed->weight;
        const marmot_tensor_t *residual = packed->residual;
        marmot_tensor_t *output = packed->output;
        float eps = packed->eps;
        if (input == nullptr || output == nullptr) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null args for rms norm bytecode execution");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        if (!marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, input) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, weight) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, residual) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, output) ||
            !marmot_bc_builder_emit_f32(&builder, eps)) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode rms norm args");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        break;
    }
    case MARMOT_BC_SCHEMA_PAGED_ATTENTION: {
        const marmot_kernel_args_paged_attention_t *packed = args;
        const marmot_tensor_t *token_meta = packed->token_meta;
        const marmot_tensor_t *q = packed->q;
        const marmot_tensor_t *k_new = packed->k_new;
        const marmot_tensor_t *v_new = packed->v_new;
        marmot_tensor_t *kv_k = packed->kv_k;
        marmot_tensor_t *kv_v = packed->kv_v;
        const marmot_tensor_t *block_table = packed->block_table;
        marmot_tensor_t *kv_k_scale = packed->kv_k_scale;
        marmot_tensor_t *kv_v_scale = packed->kv_v_scale;
        marmot_tensor_t *out = packed->out;
        uint32_t token_count = packed->token_count;
        uint32_t layer_idx = packed->layer_idx;
        uint32_t num_q_heads = packed->num_q_heads;
        uint32_t num_kv_heads = packed->num_kv_heads;
        uint32_t head_dim = packed->head_dim;
        uint32_t block_size = packed->block_size;
        float scale = packed->scale;
        if (token_meta == nullptr || q == nullptr || k_new == nullptr || v_new == nullptr || kv_k == nullptr ||
            kv_v == nullptr || block_table == nullptr || out == nullptr) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null args for paged attention bytecode execution");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        if (!marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, token_meta) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, q) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, k_new) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, v_new) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, kv_k) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, kv_v) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, block_table) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, kv_k_scale) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, kv_v_scale) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, out) ||
            !marmot_bc_builder_emit_u32(&builder, token_count) || !marmot_bc_builder_emit_u32(&builder, layer_idx) ||
            !marmot_bc_builder_emit_u32(&builder, num_q_heads) || !marmot_bc_builder_emit_u32(&builder, num_kv_heads) ||
            !marmot_bc_builder_emit_u32(&builder, head_dim) || !marmot_bc_builder_emit_u32(&builder, block_size) ||
            !marmot_bc_builder_emit_f32(&builder, scale)) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode paged attention args");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        break;
    }
    case MARMOT_BC_SCHEMA_ROPE: {
        const marmot_kernel_args_rope_t *packed = args;
        const marmot_tensor_t *input = packed->input;
        marmot_tensor_t *output = packed->output;
        const marmot_rope_params_t *params = packed->rope_params;
        uint32_t n_past = packed->n_past;
        uint32_t n_rot = packed->n_rot;
        if (input == nullptr || output == nullptr) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null args for rope bytecode execution");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        if (!marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, input) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, output) ||
            !marmot_bc_builder_emit_u16(&builder, MARMOT_BC_REG_INVALID)) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode rope tensors");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        uint32_t params_offset = marmot_bc_add_const_data(
            &builder, params, params != nullptr ? sizeof(*params) : 0, alignof(marmot_rope_params_t)
        );
        if (!marmot_bc_builder_emit_u32(&builder, params_offset) || !marmot_bc_builder_emit_u32(&builder, n_past) ||
            !marmot_bc_builder_emit_u32(&builder, n_rot)) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode rope params");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        break;
    }
    case MARMOT_BC_SCHEMA_MATMUL: {
        const marmot_kernel_args_matmul_t *packed = args;
        const marmot_tensor_t *input = packed->input;
        const marmot_tensor_t *weight = packed->weight;
        const marmot_matmul_epilogue_t *epilogue = packed->epilogue;
        marmot_tensor_t *output = packed->output;
        if (input == nullptr || weight == nullptr || output == nullptr) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null args for matmul bytecode execution");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        if (!marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, input) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, weight)) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode matmul tensors");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        uint16_t bias_reg = MARMOT_BC_REG_INVALID;
        marmot_matmul_epilogue_t epilogue_copy = {0};
        const marmot_matmul_epilogue_t *epilogue_ptr = epilogue;
        if (epilogue != nullptr) {
            if (!marmot_bc_assign_tensor_reg(regs, &reg_count, epilogue->bias, &bias_reg)) {
                marmot_bc_builder_reset(&builder);
                marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode matmul bias");
                return MARMOT_ERROR_OUT_OF_MEMORY;
            }
            epilogue_copy = *epilogue;
            epilogue_copy.bias = nullptr;
            epilogue_ptr = &epilogue_copy;
        }
        uint32_t epilogue_offset = marmot_bc_add_const_data(
            &builder, epilogue_ptr, epilogue_ptr != nullptr ? sizeof(*epilogue_ptr) : 0,
            alignof(marmot_matmul_epilogue_t)
        );
        if (!marmot_bc_builder_emit_u16(&builder, bias_reg) || !marmot_bc_builder_emit_u32(&builder, epilogue_offset) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, output)) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode matmul args");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        break;
    }
    case MARMOT_BC_SCHEMA_QKV: {
        const marmot_kernel_args_qkv_t *packed = args;
        const marmot_tensor_t *input = packed->input;
        const marmot_tensor_t *wq = packed->wq;
        const marmot_tensor_t *wk = packed->wk;
        const marmot_tensor_t *wv = packed->wv;
        const marmot_tensor_t *bq = packed->bq;
        const marmot_tensor_t *bk = packed->bk;
        const marmot_tensor_t *bv = packed->bv;
        const marmot_matmul_epilogue_t *epilogue = packed->epilogue;
        const marmot_rope_params_t *rope_params = packed->rope_params;
        marmot_tensor_t *out_q = packed->out_q;
        marmot_tensor_t *out_k = packed->out_k;
        marmot_tensor_t *out_v = packed->out_v;
        if (input == nullptr || wq == nullptr || wk == nullptr || wv == nullptr || out_q == nullptr ||
            out_k == nullptr || out_v == nullptr) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null args for qkv bytecode execution");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        if (!marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, input) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, wq) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, wk) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, wv) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, bq) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, bk) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, bv)) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode qkv tensors");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        uint32_t epilogue_offset = marmot_bc_add_const_data(
            &builder, epilogue, epilogue != nullptr ? sizeof(*epilogue) : 0, alignof(marmot_matmul_epilogue_t)
        );
        uint32_t rope_offset = marmot_bc_add_const_data(
            &builder, rope_params, rope_params != nullptr ? sizeof(*rope_params) : 0, alignof(marmot_rope_params_t)
        );
        if (!marmot_bc_builder_emit_u32(&builder, epilogue_offset) ||
            !marmot_bc_builder_emit_u32(&builder, rope_offset) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, out_q) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, out_k) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, out_v)) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode qkv args");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        break;
    }
    case MARMOT_BC_SCHEMA_RESHAPE: {
        const marmot_kernel_args_reshape_t *packed = args;
        const marmot_tensor_t *input = packed->input;
        marmot_tensor_t *output = packed->output;
        const size_t *new_shape = packed->new_shape;
        size_t new_ndim = packed->new_ndim;
        if (input == nullptr || output == nullptr || new_shape == nullptr) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null args for reshape bytecode execution");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        if (!marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, input) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, output)) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode reshape tensors");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        size_t ndim = new_ndim;
        uint32_t shape_offset =
            marmot_bc_add_const_data(&builder, new_shape, ndim > 0 ? ndim * sizeof(size_t) : 0, alignof(size_t));
        if (!marmot_bc_builder_emit_u32(&builder, shape_offset) ||
            !marmot_bc_builder_emit_u64(&builder, (uint64_t)ndim)) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode reshape params");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        break;
    }
    case MARMOT_BC_SCHEMA_VIEW: {
        const marmot_kernel_args_view_t *packed = args;
        const marmot_tensor_t *input = packed->input;
        marmot_tensor_t *output = packed->output;
        size_t byte_offset = packed->byte_offset;
        if (input == nullptr || output == nullptr) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null args for view bytecode execution");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        if (!marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, input) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, output) ||
            !marmot_bc_builder_emit_u64(&builder, (uint64_t)byte_offset)) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode view args");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        break;
    }
    case MARMOT_BC_SCHEMA_CONTIGUOUS: {
        const marmot_kernel_args_contiguous_t *packed = args;
        const marmot_tensor_t *input = packed->input;
        marmot_tensor_t *output = packed->output;
        if (input == nullptr || output == nullptr) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null args for contiguous bytecode execution");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        if (!marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, input) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, output)) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode contiguous args");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        break;
    }
    case MARMOT_BC_SCHEMA_TRANSPOSE: {
        const marmot_kernel_args_transpose_t *packed = args;
        const marmot_tensor_t *input = packed->input;
        marmot_tensor_t *output = packed->output;
        const int *perm = packed->perm;
        if (input == nullptr || output == nullptr) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null tensors for transpose bytecode execution");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        int inferred_perm[MARMOT_MAX_DIMS] = {0};
        size_t ndim = input->shape.ndim;
        const int *perm_ptr = perm;
        if (perm_ptr == nullptr) {
            if (!marmot_bc_infer_transpose_perm(input, output, inferred_perm, &ndim)) {
                marmot_bc_builder_reset(&builder);
                marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Failed to infer transpose permutation");
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            perm_ptr = inferred_perm;
        }
        if (!marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, input) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, output)) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode transpose tensors");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        uint32_t perm_offset =
            marmot_bc_add_const_data(&builder, perm_ptr, ndim > 0 ? ndim * sizeof(int) : 0, alignof(int));
        if (!marmot_bc_builder_emit_u32(&builder, perm_offset)) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode transpose perm");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        break;
    }
    case MARMOT_BC_SCHEMA_CONCAT: {
        const marmot_kernel_args_concat_t *packed = args;
        const marmot_tensor_t *const *inputs = packed->inputs;
        size_t num_inputs = packed->num_inputs;
        marmot_tensor_t *output = packed->output;
        int axis = packed->axis;
        if (inputs == nullptr || output == nullptr) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null args for concat bytecode execution");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        if (!marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, output)) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode concat output");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        size_t count = num_inputs;
        uint32_t inputs_offset = MARMOT_BC_INVALID_OFFSET;
        if (count > 0) {
            uint16_t *temp = malloc(count * sizeof(*temp));
            if (temp == nullptr) {
                marmot_bc_builder_reset(&builder);
                marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to allocate concat inputs");
                return MARMOT_ERROR_OUT_OF_MEMORY;
            }
            for (size_t i = 0; i < count; ++i) {
                if (!marmot_bc_assign_tensor_reg(regs, &reg_count, inputs[i], &temp[i])) {
                    free(temp);
                    marmot_bc_builder_reset(&builder);
                    marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to assign concat inputs");
                    return MARMOT_ERROR_OUT_OF_MEMORY;
                }
            }
            inputs_offset = marmot_bc_add_const_data(&builder, temp, count * sizeof(*temp), alignof(uint16_t));
            free(temp);
        }
        if (!marmot_bc_builder_emit_u32(&builder, inputs_offset) ||
            !marmot_bc_builder_emit_u64(&builder, (uint64_t)count) ||
            !marmot_bc_builder_emit_u32(&builder, (uint32_t)axis)) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode concat args");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        break;
    }
    case MARMOT_BC_SCHEMA_SLICE: {
        const marmot_kernel_args_slice_t *packed = args;
        const marmot_tensor_t *input = packed->input;
        marmot_tensor_t *output = packed->output;
        const size_t *starts = packed->starts;
        const size_t *sizes = packed->sizes;
        if (input == nullptr || output == nullptr || starts == nullptr || sizes == nullptr) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null args for slice bytecode execution");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        size_t ndim = output->shape.ndim;
        if (!marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, input) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, output)) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode slice tensors");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        uint32_t starts_offset =
            marmot_bc_add_const_data(&builder, starts, ndim > 0 ? ndim * sizeof(size_t) : 0, alignof(size_t));
        uint32_t sizes_offset =
            marmot_bc_add_const_data(&builder, sizes, ndim > 0 ? ndim * sizeof(size_t) : 0, alignof(size_t));
        if (!marmot_bc_builder_emit_u32(&builder, starts_offset) ||
            !marmot_bc_builder_emit_u32(&builder, sizes_offset)) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode slice params");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        break;
    }
    case MARMOT_BC_SCHEMA_GATHER_ROWS: {
        const marmot_kernel_args_gather_rows_t *packed = args;
        const marmot_tensor_t *input = packed->input;
        const marmot_tensor_t *indices = packed->indices;
        marmot_tensor_t *output = packed->output;
        if (input == nullptr || indices == nullptr || output == nullptr) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null args for gather rows bytecode execution");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        if (!marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, input) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, indices) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, output)) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode gather rows args");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        break;
    }
    case MARMOT_BC_SCHEMA_QUANTIZE: {
        const marmot_kernel_args_quantize_t *packed = args;
        const marmot_tensor_t *input = packed->input;
        const marmot_quant_params_t *params = packed->quant_params;
        marmot_tensor_t *output = packed->output;
        marmot_quant_kind_t kind = packed->kind;
        marmot_quant_layout_t layout = packed->layout;
        if (input == nullptr || output == nullptr) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null args for quantize bytecode execution");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        if (!marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, input)) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode quantize input");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        uint32_t params_offset = marmot_bc_add_const_data(
            &builder, params, params != nullptr ? sizeof(*params) : 0, alignof(marmot_quant_params_t)
        );
        if (!marmot_bc_builder_emit_u32(&builder, params_offset) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, output) ||
            !marmot_bc_builder_emit_u32(&builder, (uint32_t)kind) ||
            !marmot_bc_builder_emit_u32(&builder, (uint32_t)layout)) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode quantize args");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        break;
    }
    case MARMOT_BC_SCHEMA_DEQUANTIZE: {
        const marmot_kernel_args_dequantize_t *packed = args;
        const marmot_tensor_t *input = packed->input;
        marmot_tensor_t *output = packed->output;
        marmot_quant_kind_t kind = packed->kind;
        marmot_quant_layout_t layout = packed->layout;
        if (input == nullptr || output == nullptr) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null args for dequantize bytecode execution");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        if (!marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, input) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, output) ||
            !marmot_bc_builder_emit_u32(&builder, (uint32_t)kind) ||
            !marmot_bc_builder_emit_u32(&builder, (uint32_t)layout)) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode dequantize args");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        break;
    }
    case MARMOT_BC_SCHEMA_COMPUTE_QPARAMS: {
        const marmot_kernel_args_compute_qparams_t *packed = args;
        const marmot_tensor_t *tensor = packed->tensor;
        marmot_dtype_t target_dtype = packed->target_dtype;
        size_t block_size = packed->block_size;
        marmot_quant_params_t *out_params = packed->out_params;
        if (tensor == nullptr || out_params == nullptr) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null args for compute qparams bytecode execution");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        if (!marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, tensor)) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode compute qparams tensor");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        uint32_t out_params_offset = marmot_bc_add_const_ptr_value(&builder, out_params);
        if (!marmot_bc_builder_emit_u32(&builder, (uint32_t)target_dtype) ||
            !marmot_bc_builder_emit_u64(&builder, (uint64_t)block_size) ||
            !marmot_bc_builder_emit_u32(&builder, out_params_offset)) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode compute qparams args");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        break;
    }
    case MARMOT_BC_SCHEMA_EMBEDDING: {
        const marmot_kernel_args_embedding_t *packed = args;
        const marmot_tensor_t *weights = packed->weights;
        const marmot_tensor_t *token_ids = packed->token_ids;
        marmot_tensor_t *out = packed->out;
        marmot_dtype_t dtype_out = packed->dtype_out;
        float scale = packed->scale;
        int32_t padding_id = packed->padding_id;
        bool bounds_check = packed->bounds_check;
        bool prefer_gpu = packed->prefer_gpu_private;
        bool allow_decode = packed->allow_quant_decode_on_the_fly;
        if (weights == nullptr || token_ids == nullptr || out == nullptr) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null args for embedding bytecode execution");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        if (!marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, weights) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, token_ids) ||
            !marmot_bc_emit_tensor_reg(&builder, regs, &reg_count, out) ||
            !marmot_bc_builder_emit_u32(&builder, (uint32_t)dtype_out) ||
            !marmot_bc_builder_emit_f32(&builder, scale) ||
            !marmot_bc_builder_emit_u32(&builder, (uint32_t)padding_id) ||
            !marmot_bc_builder_emit_u8(&builder, bounds_check ? 1 : 0) ||
            !marmot_bc_builder_emit_u8(&builder, prefer_gpu ? 1 : 0) ||
            !marmot_bc_builder_emit_u8(&builder, allow_decode ? 1 : 0)) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode embedding args");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        break;
    }
    case MARMOT_BC_SCHEMA_CONVERT: {
        const marmot_kernel_args_convert_t *packed = args;
        void *dst = packed->dst;
        const void *src = packed->src;
        size_t n = packed->n;
        marmot_dtype_t dst_dtype = packed->dst_dtype;
        marmot_dtype_t src_dtype = packed->src_dtype;
        if (dst == nullptr || src == nullptr) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null args for convert bytecode execution");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        uint32_t dst_offset = marmot_bc_add_const_ptr_value(&builder, dst);
        uint32_t src_offset = marmot_bc_add_const_ptr_value(&builder, src);
        if (!marmot_bc_builder_emit_u32(&builder, dst_offset) || !marmot_bc_builder_emit_u32(&builder, src_offset) ||
            !marmot_bc_builder_emit_u64(&builder, (uint64_t)n) ||
            !marmot_bc_builder_emit_u32(&builder, (uint32_t)dst_dtype) ||
            !marmot_bc_builder_emit_u32(&builder, (uint32_t)src_dtype)) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode convert args");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        break;
    }
    case MARMOT_BC_SCHEMA_VEC_DOT: {
        const marmot_kernel_args_vec_dot_t *packed = args;
        const marmot_vec_dot_descriptor_t *desc = packed->desc;
        float *result = packed->result;
        if (desc == nullptr || result == nullptr) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null args for vec dot bytecode execution");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        uint32_t desc_offset = marmot_bc_add_const_data(
            &builder, desc, desc != nullptr ? sizeof(*desc) : 0, alignof(marmot_vec_dot_descriptor_t)
        );
        uint32_t result_offset = marmot_bc_add_const_ptr_value(&builder, result);
        if (!marmot_bc_builder_emit_u16(&builder, MARMOT_BC_REG_INVALID) ||
            !marmot_bc_builder_emit_u16(&builder, MARMOT_BC_REG_INVALID) ||
            !marmot_bc_builder_emit_u16(&builder, MARMOT_BC_REG_INVALID) ||
            !marmot_bc_builder_emit_u32(&builder, desc_offset) ||
            !marmot_bc_builder_emit_u32(&builder, result_offset)) {
            marmot_bc_builder_reset(&builder);
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode vec dot args");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        break;
    }
    default:
        marmot_bc_builder_reset(&builder);
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    if (!marmot_bc_builder_emit_u16(&builder, MARMOT_BC_END)) {
        marmot_bc_builder_reset(&builder);
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to finalize bytecode");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    marmot_bc_program_t program = {0};
    if (!marmot_bc_builder_finish(&builder, &program, tables.imm_size, tables.exec_table, reg_count, tables.op_count)) {
        marmot_bc_builder_reset(&builder);
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to finalize bytecode program");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    marmot_error_t status = marmot_bc_execute(&program, exec_ctx, regs);
    marmot_bc_program_destroy(&program);
    return status;
}

marmot_error_t
marmot_bc_try_execute_signature(const marmot_context_t *ctx, const marmot_op_signature_t *sig, const void *args) {
    if (ctx == nullptr || sig == nullptr || args == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null arguments for bytecode selection");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_bc_selection_t selection = marmot_bc_compile_signature(ctx, sig);
    if (!selection.supported) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    marmot_bc_exec_ctx_t exec_ctx = {
        .ctx = ctx,
        .device_ctx = ctx->device_ctx,
    };
    return marmot_bc_execute_op(ctx->backend_type, selection.op_index, &exec_ctx, args);
}
