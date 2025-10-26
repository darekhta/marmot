#include "graph_executor.hpp"

#include "marmot/allocator.h"
#include "marmot/device.h"
#include "marmot/error.h"
#include "marmot/graph/graph.hpp"
#include "marmot/op_metadata.gen.h"
#include "marmot/ops/manipulation.h"
#include "marmot/ops/matmul.h"
#include "marmot/ops/paged_attention.h"
#include "marmot/ops/quantization.h"
#include "marmot/tensor.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ranges>
#include <span>
#include <string_view>
#include <vector>

#include "binding_table.hpp"
#include "core/bytecode/bytecode_compile.h"
#include "core/dispatch/dispatch_build.h"
#include "core/dispatch/fusion_detection.h"
#include "execution_session.hpp"
#include "graph_impl.hpp"
#include "kernel_dispatch_args.gen.h"
#include "tensor_alloc.hpp"
#include "utils/dtype_ref.h"

namespace {

bool graph_trace_enabled() {
    static bool enabled = [] {
        const char *env = std::getenv("MARMOT_GRAPH_TRACE");
        return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

bool graph_nan_check_enabled() {
    static bool enabled = [] {
        const char *env = std::getenv("MARMOT_GRAPH_NAN_CHECK");
        return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

bool tensor_has_nan(const marmot_context_t *ctx, marmot_tensor_t *tensor);
void log_tensor_stats(const marmot_context_t *ctx, marmot_tensor_t *tensor, const char *label);

// Extract (total_seqs, seq_len) from RoPE input tensor shape.
// Shape is [..., seq_len, dim] -> total_seqs = numel / (seq_len * dim).
static bool infer_rope_broadcast_params(const marmot_tensor_t *input, size_t &total_seqs_out, size_t &seq_len_out) {
    if (input == nullptr || input->shape.ndim < 2) {
        return false;
    }
    const size_t ndim = input->shape.ndim;
    const size_t dim = input->shape.shape[ndim - 1];
    const size_t seq_len = input->shape.shape[ndim - 2];
    if (dim == 0 || seq_len == 0) {
        return false;
    }
    const size_t total_tokens = marmot_tensor_num_elements(input);
    if (total_tokens % (seq_len * dim) != 0) {
        return false;
    }
    total_seqs_out = total_tokens / (seq_len * dim);
    seq_len_out = seq_len;
    return total_seqs_out > 0;
}

static bool graph_node_is_qkv(const marmot::graph::GraphNode &node) {
    return node.signature.op_id == MARMOT_OP_QKV_SHARED_INPUT || node.signature.op_id == MARMOT_OP_QKV_PROJECTION ||
        node.signature.op_id == MARMOT_OP_QKV_ROPE;
}

static bool graph_node_uses_qkv_rope(const marmot::graph::GraphNode &node) {
    return graph_node_is_qkv(node) && (node.signature.epilogue_flags & MARMOT_EPILOGUE_ROPE) != 0;
}

static bool graph_node_uses_rope(const marmot::graph::GraphNode &node) {
    return node.signature.op_id == MARMOT_OP_ROPE || graph_node_uses_qkv_rope(node);
}

struct GraphBcHookContext {
    const std::vector<marmot::graph::GraphNode> *nodes;
    const std::vector<uint32_t> *bc_instr_nodes;
    marmot::graph::ExecutionSession *session;
    const marmot_context_t *ctx;
    std::span<marmot_tensor_t *> bindings;
    bool trace;
    bool nan_check;
};

struct GraphBcOpState {
    uint32_t node_index;
    marmot_tensor_t *rope_restore_ptr;
    marmot_value_id_t rope_restore_id;
};

static void graph_bc_on_start(
    const marmot_bc_program_t *program, const void *backend_exec_ctx, marmot_tensor_t **regs, void *user_data
) {
    (void)backend_exec_ctx;
    (void)regs;
    auto *ctx = static_cast<GraphBcHookContext *>(user_data);
    if (ctx == nullptr || ctx->nodes == nullptr || ctx->bc_instr_nodes == nullptr || !ctx->trace) {
        return;
    }
    if (program != nullptr && program->code_size >= sizeof(uint16_t)) {
        uint16_t first_op = 0;
        memcpy(&first_op, program->code, sizeof(first_op));
        fprintf(
            stderr, "[graph trace] bytecode first op=%u op_count=%u code_size=%zu\n", first_op, program->op_count,
            program->code_size
        );
    }
    if (!ctx->bc_instr_nodes->empty()) {
        uint32_t first_node = (*ctx->bc_instr_nodes)[0];
        if (first_node < ctx->nodes->size()) {
            fprintf(
                stderr, "[graph trace] first node %u bc_op_index=%u op_id=%d\n", first_node,
                (*ctx->nodes)[first_node].bc_op_index, (*ctx->nodes)[first_node].signature.op_id
            );
        }
    }
}

static marmot_error_t graph_bc_before_op(const marmot_bc_hook_info_t *info, void *user_data, void *op_state) {
    if (info == nullptr || user_data == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Invalid bytecode hook arguments");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    auto *ctx = static_cast<GraphBcHookContext *>(user_data);
    auto *state = static_cast<GraphBcOpState *>(op_state);
    if (state != nullptr) {
        state->node_index = UINT32_MAX;
        state->rope_restore_ptr = nullptr;
        state->rope_restore_id = MARMOT_VALUE_ID_INVALID;
    }
    if (ctx == nullptr || ctx->nodes == nullptr || ctx->bc_instr_nodes == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Invalid bytecode hook context");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    uint32_t node_index =
        info->instr_index < ctx->bc_instr_nodes->size() ? (*ctx->bc_instr_nodes)[info->instr_index] : UINT32_MAX;
    if (state != nullptr) {
        state->node_index = node_index;
    }

    if (ctx->trace && node_index != UINT32_MAX && node_index < ctx->nodes->size()) {
        const auto &node = (*ctx->nodes)[node_index];
        fprintf(
            stderr, "[graph trace] executing node %u: %s (op_id=%d)\n", node_index, node.op_name.c_str(),
            node.signature.op_id
        );
    }

    if (ctx->session != nullptr && node_index != UINT32_MAX && node_index < ctx->nodes->size()) {
        const auto &node = (*ctx->nodes)[node_index];
        if (node.signature.op_id == MARMOT_OP_ROPE && node.inputs.size() >= 2) {
            const marmot_value_id_t input_id = node.inputs[0];
            const marmot_value_id_t positions_id = node.inputs[1];
            if (input_id >= ctx->bindings.size() || positions_id >= ctx->bindings.size()) {
                marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "RoPE bindings out of range");
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            marmot_tensor_t *input = ctx->bindings[input_id];
            marmot_tensor_t *positions = ctx->bindings[positions_id];
            if (input == nullptr || positions == nullptr) {
                marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "RoPE input or positions not bound");
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            const size_t ndim = input->shape.ndim;
            if (ndim < 2) {
                marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "RoPE input must be at least 2D");
                return MARMOT_ERROR_DIMENSION_MISMATCH;
            }
            const size_t dim = input->shape.shape[ndim - 1];
            const size_t seq_len = input->shape.shape[ndim - 2];
            if (dim == 0 || seq_len == 0) {
                marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "RoPE input has zero-sized dimensions");
                return MARMOT_ERROR_DIMENSION_MISMATCH;
            }
            const size_t total_tokens = marmot_tensor_num_elements(input);
            const size_t denom = seq_len * dim;
            if (denom == 0 || (total_tokens % denom) != 0) {
                marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "RoPE input shape is not compatible");
                return MARMOT_ERROR_DIMENSION_MISMATCH;
            }
            const size_t total_seqs = total_tokens / denom;
            marmot_tensor_t *broadcast = ctx->session->broadcast_rope_positions(positions, total_seqs, seq_len);
            if (broadcast == nullptr) {
                if (ctx->trace) {
                    const size_t positions_elems = marmot_tensor_num_elements(positions);
                    fprintf(
                        stderr,
                        "[graph trace] RoPE broadcast failed: input_ndim=%zu "
                        "seq_len=%zu dim=%zu total_tokens=%zu total_seqs=%zu "
                        "positions_elems=%zu positions_dtype=%d\n",
                        ndim, seq_len, dim, total_tokens, total_seqs, positions_elems, (int)positions->dtype
                    );
                }
                marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "RoPE positions broadcast failed");
                return MARMOT_ERROR_DIMENSION_MISMATCH;
            }
            if (broadcast != positions) {
                if (state != nullptr) {
                    state->rope_restore_ptr = positions;
                    state->rope_restore_id = positions_id;
                }
                ctx->bindings[positions_id] = broadcast;
            }
        }
        if (graph_node_uses_qkv_rope(node)) {
            if (node.inputs.size() < 5) {
                marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "QKV RoPE node missing positions input");
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            const marmot_value_id_t input_id = node.inputs[0];
            const marmot_value_id_t positions_id = node.inputs.back();
            if (input_id >= ctx->bindings.size() || positions_id >= ctx->bindings.size()) {
                marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "QKV RoPE bindings out of range");
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            marmot_tensor_t *input = ctx->bindings[input_id];
            marmot_tensor_t *positions = ctx->bindings[positions_id];
            if (input == nullptr || positions == nullptr) {
                marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "QKV RoPE input or positions not bound");
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            size_t total_seqs = 0;
            size_t seq_len = 0;
            if (!infer_rope_broadcast_params(input, total_seqs, seq_len)) {
                marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "QKV RoPE input shape is not compatible");
                return MARMOT_ERROR_DIMENSION_MISMATCH;
            }
            marmot_tensor_t *broadcast = ctx->session->broadcast_rope_positions(positions, total_seqs, seq_len);
            if (broadcast == nullptr) {
                marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "QKV RoPE positions broadcast failed");
                return MARMOT_ERROR_DIMENSION_MISMATCH;
            }
            if (broadcast != positions) {
                if (state != nullptr) {
                    state->rope_restore_ptr = positions;
                    state->rope_restore_id = positions_id;
                }
                ctx->bindings[positions_id] = broadcast;
                positions = broadcast;
            }

            if (node.rope_params_offset == MARMOT_BC_INVALID_OFFSET || info->program == nullptr ||
                info->program->const_pool == nullptr ||
                node.rope_params_offset + sizeof(marmot_rope_params_t) > info->program->const_pool_size) {
                marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "QKV RoPE params offset invalid");
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            marmot_rope_params_t *params =
                (marmot_rope_params_t *)(info->program->const_pool + node.rope_params_offset);
            *params = *ctx->session->rope_params();
            params->positions = positions;
        }
    }

    return MARMOT_SUCCESS;
}

static marmot_error_t
graph_bc_after_op(const marmot_bc_hook_info_t *info, marmot_error_t status, void *user_data, void *op_state) {
    if (info == nullptr || user_data == nullptr) {
        return status;
    }
    auto *ctx = static_cast<GraphBcHookContext *>(user_data);
    auto *state = static_cast<GraphBcOpState *>(op_state);
    if (ctx == nullptr || ctx->nodes == nullptr || ctx->bc_instr_nodes == nullptr) {
        return status;
    }

    uint32_t node_index = UINT32_MAX;
    if (state != nullptr) {
        node_index = state->node_index;
        if (state->rope_restore_ptr != nullptr && state->rope_restore_id < ctx->bindings.size()) {
            ctx->bindings[state->rope_restore_id] = state->rope_restore_ptr;
        }
    } else if (info->instr_index < ctx->bc_instr_nodes->size()) {
        node_index = (*ctx->bc_instr_nodes)[info->instr_index];
    }

    if (status != MARMOT_SUCCESS) {
        if (ctx->trace && node_index != UINT32_MAX && node_index < ctx->nodes->size()) {
            const auto &node = (*ctx->nodes)[node_index];
            fprintf(
                stderr, "[graph trace] FAILED at node %u: %s (status=%d)\n", node_index, node.op_name.c_str(), status
            );
        }
        if (ctx->trace) {
            const char *detail = marmot_get_last_error_detail();
            fprintf(
                stderr, "[graph trace] op fail instr=%zu op=%u node=%u status=%d detail=%s\n", info->instr_index,
                info->op, node_index, status, detail != nullptr ? detail : ""
            );
        }
        const char *detail = marmot_get_last_error_detail();
        if (detail == nullptr || detail[0] == '\0') {
            char msg[160];
            if (node_index != UINT32_MAX && node_index < ctx->nodes->size()) {
                const auto &node = (*ctx->nodes)[node_index];
                snprintf(msg, sizeof(msg), "Bytecode op failed at node %u (%s)", node_index, node.op_name.c_str());
            } else {
                snprintf(msg, sizeof(msg), "Bytecode op failed at instr %zu", info->instr_index);
            }
            marmot_set_error(status, msg);
        }
        return status;
    }

    if (ctx->nan_check && node_index != UINT32_MAX && node_index < ctx->nodes->size()) {
        const auto &node = (*ctx->nodes)[node_index];
        for (marmot_value_id_t output_id : node.outputs) {
            if (output_id >= ctx->bindings.size()) {
                continue;
            }
            marmot_tensor_t *out = ctx->bindings[output_id];
            if (tensor_has_nan(ctx->ctx, out)) {
                fprintf(
                    stderr, "[graph nan] node %u: %s (op_id=%d) produced NaN/Inf\n", node_index, node.op_name.c_str(),
                    node.signature.op_id
                );
                if (node.signature.op_id == MARMOT_OP_GEGLU || node.signature.op_id == MARMOT_OP_SWIGLU) {
                    for (size_t i = 0; i < node.inputs.size(); ++i) {
                        marmot_value_id_t input_id = node.inputs[i];
                        if (input_id < ctx->bindings.size()) {
                            log_tensor_stats(ctx->ctx, ctx->bindings[input_id], i == 0 ? "glu_in0" : "glu_in1");
                        }
                    }
                }
                log_tensor_stats(ctx->ctx, out, "glu_out");
                return MARMOT_ERROR_INVALID_OPERATION;
            }
        }
    }

    return status;
}

bool tensor_has_nan(const marmot_context_t *ctx, marmot_tensor_t *tensor) {
    if (ctx == nullptr || tensor == nullptr) {
        return false;
    }
    switch (tensor->dtype) {
    case MARMOT_DTYPE_FLOAT32: {
        const float *data = marmot_tensor_data_f32(ctx, tensor);
        if (data == nullptr) {
            return true;
        }
        const size_t count = marmot_tensor_num_elements(tensor);
        for (size_t i = 0; i < count; ++i) {
            if (!std::isfinite(data[i])) {
                return true;
            }
        }
        return false;
    }
    case MARMOT_DTYPE_FLOAT16: {
        const marmot_float16_t *data = marmot_tensor_data_f16(ctx, tensor);
        if (data == nullptr) {
            return true;
        }
        const size_t count = marmot_tensor_num_elements(tensor);
        for (size_t i = 0; i < count; ++i) {
            float v = marmot_f16_to_f32_ref(data[i]);
            if (!std::isfinite(v)) {
                return true;
            }
        }
        return false;
    }
    case MARMOT_DTYPE_BFLOAT16: {
        const marmot_bfloat16_t *data = marmot_tensor_data_bf16(ctx, tensor);
        if (data == nullptr) {
            return true;
        }
        const size_t count = marmot_tensor_num_elements(tensor);
        for (size_t i = 0; i < count; ++i) {
            float v = marmot_bf16_to_f32_ref(data[i]);
            if (!std::isfinite(v)) {
                return true;
            }
        }
        return false;
    }
    default:
        return false;
    }
}

void log_tensor_stats(const marmot_context_t *ctx, marmot_tensor_t *tensor, const char *label) {
    if (ctx == nullptr || tensor == nullptr || label == nullptr) {
        return;
    }
    size_t count = marmot_tensor_num_elements(tensor);
    size_t nan_count = 0;
    size_t inf_count = 0;
    float minv = 0.0f;
    float maxv = 0.0f;
    bool initialized = false;

    auto update = [&](float v) {
        if (!initialized) {
            minv = v;
            maxv = v;
            initialized = true;
        } else {
            if (v < minv)
                minv = v;
            if (v > maxv)
                maxv = v;
        }
    };

    if (tensor->dtype == MARMOT_DTYPE_FLOAT32) {
        const float *data = marmot_tensor_data_f32(ctx, tensor);
        if (!data) {
            fprintf(stderr, "[graph nan] %s: failed to read data\n", label);
            return;
        }
        for (size_t i = 0; i < count; ++i) {
            float v = data[i];
            if (std::isnan(v)) {
                nan_count++;
                continue;
            }
            if (std::isinf(v)) {
                inf_count++;
                continue;
            }
            update(v);
        }
    } else if (tensor->dtype == MARMOT_DTYPE_FLOAT16) {
        const marmot_float16_t *data = marmot_tensor_data_f16(ctx, tensor);
        if (!data) {
            fprintf(stderr, "[graph nan] %s: failed to read data\n", label);
            return;
        }
        for (size_t i = 0; i < count; ++i) {
            float v = marmot_f16_to_f32_ref(data[i]);
            if (std::isnan(v)) {
                nan_count++;
                continue;
            }
            if (std::isinf(v)) {
                inf_count++;
                continue;
            }
            update(v);
        }
    } else if (tensor->dtype == MARMOT_DTYPE_BFLOAT16) {
        const marmot_bfloat16_t *data = marmot_tensor_data_bf16(ctx, tensor);
        if (!data) {
            fprintf(stderr, "[graph nan] %s: failed to read data\n", label);
            return;
        }
        for (size_t i = 0; i < count; ++i) {
            float v = marmot_bf16_to_f32_ref(data[i]);
            if (std::isnan(v)) {
                nan_count++;
                continue;
            }
            if (std::isinf(v)) {
                inf_count++;
                continue;
            }
            update(v);
        }
    } else {
        return;
    }

    if (!initialized) {
        minv = 0.0f;
        maxv = 0.0f;
    }
    fprintf(stderr, "[graph nan] %s: min=%g max=%g nan=%zu inf=%zu\n", label, minv, maxv, nan_count, inf_count);
}

marmot_error_t dispatch_node_bytecode(
    const marmot::graph::GraphNode &node, const marmot_context_t *ctx, const marmot_op_signature_t *signature,
    const void *args, const char *op_name
) {
    if (signature == &node.signature && node.bc_op_index != MARMOT_BC_OP_INVALID) {
        marmot_bc_exec_ctx_t exec_ctx = {
            .ctx = ctx,
            .device_ctx = ctx->device_ctx,
        };
        return marmot_bc_execute_op(ctx->backend_type, node.bc_op_index, &exec_ctx, args);
    }

    marmot_bc_selection_t selection = marmot_bc_compile_signature(ctx, signature);
    if (!selection.supported) {
        const char *label = op_name != nullptr ? op_name : "operation";
        const char *reason = selection.reason != nullptr ? selection.reason : "Bytecode not supported";
        char msg[192];
        snprintf(msg, sizeof(msg), "%s not available for bytecode dispatch: %s", label, reason);
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, msg);
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    marmot_bc_exec_ctx_t exec_ctx = {
        .ctx = ctx,
        .device_ctx = ctx->device_ctx,
    };
    return marmot_bc_execute_op(ctx->backend_type, selection.op_index, &exec_ctx, args);
}

static bool infer_permutation(const marmot_tensor_t *input, const marmot_tensor_t *output, std::vector<int> &perm_out) {
    if (input->shape.ndim != output->shape.ndim)
        return false;

    const size_t ndim = input->shape.ndim;
    perm_out.assign(ndim, -1);

    std::vector<bool> used(ndim, false);
    for (size_t i = 0; i < ndim; ++i) {
        const size_t target = output->shape.shape[i];
        int found = -1;
        for (size_t j = 0; j < ndim; ++j) {
            if (used[j] || input->shape.shape[j] != target)
                continue;
            if (found != -1) {
                return false; // Ambiguous permutation when dimensions repeat
            }
            found = static_cast<int>(j);
        }
        if (found == -1) {
            return false;
        }
        perm_out[i] = found;
        used[static_cast<size_t>(found)] = true;
    }

    return true;
}

static bool
infer_concat_axis(const std::vector<const marmot_tensor_t *> &inputs, const marmot_tensor_t *output, int &axis_out) {
    if (inputs.empty() || output == nullptr)
        return false;

    const size_t ndim = output->shape.ndim;
    for (const auto *input : inputs) {
        if (input == nullptr || input->shape.ndim != ndim)
            return false;
    }

    axis_out = -1;
    for (size_t dim = 0; dim < ndim; ++dim) {
        size_t out_dim = output->shape.shape[dim];
        size_t sum_dim = 0;
        bool all_same = true;
        const size_t first_dim = inputs.front()->shape.shape[dim];
        for (const auto *input : inputs) {
            const size_t current = input->shape.shape[dim];
            if (current != first_dim) {
                all_same = false;
            }
            sum_dim += current;
        }

        if (out_dim == sum_dim && (!all_same || inputs.size() > 1)) {
            if (axis_out != -1) {
                return false;
            }
            axis_out = static_cast<int>(dim);
            continue;
        }

        if (out_dim != first_dim) {
            return false;
        }
    }

    return axis_out != -1;
}

static bool infer_reduction_axes(
    const marmot_tensor_t *input, const marmot_tensor_t *output, bool &keepdims, std::vector<int32_t> &axes_out
) {
    if (input == nullptr || output == nullptr)
        return false;

    const size_t in_ndim = input->shape.ndim;
    const size_t out_ndim = output->shape.ndim;
    if (in_ndim == 0 || out_ndim == 0 || in_ndim > MARMOT_MAX_DIMS || out_ndim > MARMOT_MAX_DIMS)
        return false;

    axes_out.clear();
    if (in_ndim == out_ndim) {
        keepdims = true;
        for (size_t i = 0; i < in_ndim; ++i) {
            const size_t in_dim = input->shape.shape[i];
            const size_t out_dim = output->shape.shape[i];
            if (in_dim != out_dim) {
                if (out_dim != 1)
                    return false;
                axes_out.push_back((int32_t)i);
            }
        }
        return !axes_out.empty();
    }

    if (out_ndim > in_ndim)
        return false;

    keepdims = false;
    size_t out_idx = 0;
    for (size_t in_idx = 0; in_idx < in_ndim; ++in_idx) {
        if (out_idx < out_ndim && input->shape.shape[in_idx] == output->shape.shape[out_idx]) {
            ++out_idx;
            continue;
        }
        axes_out.push_back((int32_t)in_idx);
    }

    return out_idx == out_ndim && !axes_out.empty();
}

} // namespace

namespace marmot::graph {

marmot_error_t Executor::update_qkv_rope_params(std::span<marmot_tensor_t *> bindings) {
    if (session_ == nullptr || impl_.rope_nodes.empty()) {
        return MARMOT_SUCCESS;
    }
    if (impl_.program.const_pool == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Missing bytecode const pool");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const marmot_rope_params_t *base = session_->rope_params();
    if (base == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Missing rope params");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    for (const GraphRopeInfo &rope : impl_.rope_nodes) {
        if (rope.node_index >= impl_.nodes.size()) {
            continue;
        }
        const GraphNode &node = impl_.nodes[rope.node_index];
        if (!graph_node_uses_qkv_rope(node)) {
            continue;
        }
        if (node.rope_params_offset == MARMOT_BC_INVALID_OFFSET ||
            node.rope_params_offset + sizeof(marmot_rope_params_t) > impl_.program.const_pool_size) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "QKV RoPE params offset invalid");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        if (rope.positions_id >= bindings.size()) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "QKV RoPE positions binding out of range");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        marmot_tensor_t *positions = bindings[rope.positions_id];
        if (positions == nullptr) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "QKV RoPE positions not bound");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        marmot_rope_params_t *params = (marmot_rope_params_t *)(impl_.program.const_pool + node.rope_params_offset);
        *params = *base;
        params->positions = positions;
    }
    return MARMOT_SUCCESS;
}

Executor::Executor(Graph::Impl &graph_impl, ExecutionSession *session) : impl_(graph_impl), session_(session) {}

marmot_error_t Executor::execute_bound(const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings) {
    if (ctx == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null context");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (!impl_.finalized) {
        return MARMOT_ERROR_INVALID_OPERATION;
    }
    if (ctx->backend_type != impl_.backend) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Context backend mismatch");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_error_t batch_begin = marmot_graph_batch_begin(ctx);
    if (batch_begin != MARMOT_SUCCESS) {
        return batch_begin;
    }

    const bool trace = graph_trace_enabled();
    const bool nan_check = graph_nan_check_enabled();
    marmot_error_t status = MARMOT_SUCCESS;

    if (status == MARMOT_SUCCESS) {
        const marmot_bc_program_t &program = impl_.program;
        marmot_bc_exec_ctx_t exec_ctx = {
            .ctx = ctx,
            .device_ctx = ctx->device_ctx,
        };

        // RoPE pre-broadcast optimization: broadcast positions once before execution
        // instead of using per-instruction hooks. Falls back to hooks if pre-broadcast
        // isn't safe (e.g., positions is computed, or RoPE nodes have different shapes).
        marmot_value_id_t rope_restore_id = MARMOT_VALUE_ID_INVALID;
        marmot_tensor_t *rope_restore_ptr = nullptr;
        bool rope_needs_hooks = false;

        const bool has_rope = session_ != nullptr && !impl_.rope_nodes.empty();
        const bool can_try_prebroadcast = has_rope && !trace && !nan_check;

        if (can_try_prebroadcast) {
            const GraphRopeInfo &first_rope = impl_.rope_nodes.front();
            const marmot_value_id_t positions_id = first_rope.positions_id;

            // Check 1: positions must be a graph input or constant (not computed)
            bool valid = positions_id < bindings.size() && positions_id < impl_.values.size();
            if (valid) {
                const GraphValue &val = impl_.values[positions_id];
                valid = val.is_input || val.is_constant;
            }

            // Check 2: all uses of positions must be RoPE-compatible ops
            if (valid) {
                for (uint32_t use_idx : impl_.values[positions_id].uses) {
                    if (use_idx >= impl_.nodes.size()) {
                        valid = false;
                        break;
                    }
                    const GraphNode &node = impl_.nodes[use_idx];
                    if (!node.skip && !graph_node_uses_rope(node)) {
                        valid = false;
                        break;
                    }
                }
            }

            // Check 3: all RoPE nodes must share same positions and have compatible shapes
            size_t total_seqs = 0, seq_len = 0;
            if (valid && first_rope.input_id < bindings.size()) {
                valid = infer_rope_broadcast_params(bindings[first_rope.input_id], total_seqs, seq_len);
            }
            if (valid) {
                for (const GraphRopeInfo &rope : impl_.rope_nodes) {
                    if (rope.positions_id != positions_id || rope.input_id >= bindings.size()) {
                        valid = false;
                        break;
                    }
                    size_t ts = 0, sl = 0;
                    if (!infer_rope_broadcast_params(bindings[rope.input_id], ts, sl) || ts != total_seqs ||
                        sl != seq_len) {
                        valid = false;
                        break;
                    }
                }
            }

            // Perform pre-broadcast or fall back to hooks
            if (valid) {
                marmot_tensor_t *positions = bindings[positions_id];
                if (positions == nullptr) {
                    marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "RoPE positions not bound");
                    return MARMOT_ERROR_INVALID_ARGUMENT;
                }
                marmot_tensor_t *broadcast = session_->broadcast_rope_positions(positions, total_seqs, seq_len);
                if (broadcast == nullptr) {
                    marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "RoPE positions broadcast failed");
                    return MARMOT_ERROR_INVALID_OPERATION;
                }
                if (broadcast != positions) {
                    rope_restore_id = positions_id;
                    rope_restore_ptr = positions;
                    bindings[positions_id] = broadcast;
                }
            } else {
                rope_needs_hooks = true;
            }
        } else if (has_rope) {
            rope_needs_hooks = true;
        }

        if (status == MARMOT_SUCCESS) {
            marmot_error_t rope_patch_status = update_qkv_rope_params(bindings);
            if (rope_patch_status != MARMOT_SUCCESS) {
                status = rope_patch_status;
            }
        }

        const bool needs_hooks = trace || nan_check || rope_needs_hooks;
        if (status == MARMOT_SUCCESS && !needs_hooks) {
            status = marmot_bc_execute(&program, &exec_ctx, bindings.data());
        } else if (status == MARMOT_SUCCESS) {
            GraphBcHookContext hook_ctx = {
                .nodes = &impl_.nodes,
                .bc_instr_nodes = &impl_.bc_instr_nodes,
                .session = session_,
                .ctx = ctx,
                .bindings = bindings,
                .trace = trace,
                .nan_check = nan_check,
            };
            GraphBcOpState op_state = {
                .node_index = UINT32_MAX,
                .rope_restore_ptr = nullptr,
                .rope_restore_id = MARMOT_VALUE_ID_INVALID,
            };
            marmot_bc_hooks_t hooks = {
                .user_data = &hook_ctx,
                .op_state = &op_state,
                .op_state_size = sizeof(op_state),
                .on_start = graph_bc_on_start,
                .before_op = graph_bc_before_op,
                .after_op = graph_bc_after_op,
                .on_finish = nullptr,
            };
            status = marmot_bc_execute_with_hooks(&program, &exec_ctx, bindings.data(), &hooks);
        }

        if (rope_restore_ptr != nullptr && rope_restore_id < bindings.size()) {
            bindings[rope_restore_id] = rope_restore_ptr;
        }
    }
    marmot_error_t batch_end = marmot_graph_batch_end(ctx, status == MARMOT_SUCCESS);
    if (status == MARMOT_SUCCESS) {
        return batch_end;
    }
    return status;
}

marmot_error_t Executor::execute_node(
    uint32_t node_index, const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings
) {
    // All operations use universal dispatch which routes to the correct backend based on ctx->backend_type
    if (node.signature.op_id == MARMOT_OP_QUANTIZE) {
        return execute_quantize(node, ctx, bindings);
    }
    if (node.signature.op_id == MARMOT_OP_DEQUANTIZE) {
        return execute_dequantize(node, ctx, bindings);
    }
    if (node.signature.op_id == MARMOT_OP_VEC_DOT) {
        return execute_vec_dot(node, ctx, bindings);
    }
    if (node.signature.op_id == MARMOT_OP_REDUCTION_SUM || node.signature.op_id == MARMOT_OP_REDUCTION_MEAN ||
        node.signature.op_id == MARMOT_OP_REDUCTION_MAX || node.signature.op_id == MARMOT_OP_REDUCTION_MIN ||
        node.signature.op_id == MARMOT_OP_REDUCTION_VARIANCE || node.signature.op_id == MARMOT_OP_REDUCTION_STD ||
        node.signature.op_id == MARMOT_OP_REDUCTION_NORM_L1 || node.signature.op_id == MARMOT_OP_REDUCTION_NORM_L2 ||
        node.signature.op_id == MARMOT_OP_REDUCTION_PROD || node.signature.op_id == MARMOT_OP_REDUCTION_ARGMAX ||
        node.signature.op_id == MARMOT_OP_REDUCTION_ARGMIN || node.signature.op_id == MARMOT_OP_REDUCTION_ANY ||
        node.signature.op_id == MARMOT_OP_REDUCTION_ALL) {
        return execute_reduction(node, ctx, bindings);
    }
    if (node.signature.op_id == MARMOT_OP_EMBEDDING) {
        return execute_embedding(node, ctx, bindings);
    }
    if (node.signature.op_id == MARMOT_OP_MATMUL || node.signature.op_id == MARMOT_OP_MATMUL_BIAS ||
        node.signature.op_id == MARMOT_OP_MATMUL_BIAS_RELU || node.signature.op_id == MARMOT_OP_MATMUL_BIAS_GELU ||
        node.signature.op_id == MARMOT_OP_MATMUL_BIAS_SILU) {
        return execute_matmul(node, ctx, bindings);
    }
    if (node.signature.op_id == MARMOT_OP_QKV_ROPE || node.signature.op_id == MARMOT_OP_QKV_SHARED_INPUT ||
        node.signature.op_id == MARMOT_OP_QKV_PROJECTION) {
        return execute_qkv(node, ctx, bindings);
    }
    if (node.signature.op_id == MARMOT_OP_RMS_NORM || node.signature.op_id == MARMOT_OP_RMS_NORM_GEMMA) {
        return execute_rms_norm(node, ctx, bindings);
    }
    if (node.signature.op_id == MARMOT_OP_LAYERNORM) {
        return execute_layernorm(node, ctx, bindings);
    }
    if (node.signature.op_id == MARMOT_OP_SOFTMAX) {
        return execute_softmax(node, ctx, bindings);
    }
    if (node.signature.op_id == MARMOT_OP_PAGED_ATTENTION) {
        return execute_paged_attention(node, ctx, bindings);
    }
    if (node.signature.op_id == MARMOT_OP_ROPE) {
        return execute_rope(node, ctx, bindings);
    }
    if (node.signature.op_id == MARMOT_OP_RESHAPE) {
        return execute_reshape(node, ctx, bindings);
    }
    if (node.signature.op_id == MARMOT_OP_TRANSPOSE) {
        return execute_transpose(node, ctx, bindings);
    }
    if (node.signature.op_id == MARMOT_OP_CONCAT) {
        return execute_concat(node, ctx, bindings);
    }
    if (node.signature.op_id == MARMOT_OP_SLICE) {
        return execute_slice(node, ctx, bindings);
    }
    if (node.signature.op_id == MARMOT_OP_GATHER_ROWS) {
        return execute_gather_rows(node, ctx, bindings);
    }
    if (node.signature.op_id == MARMOT_OP_VIEW) {
        return execute_view(node, ctx, bindings);
    }
    if (node.signature.op_id == MARMOT_OP_CONTIGUOUS) {
        return execute_contiguous(node, ctx, bindings);
    }
    // Binary ops (including explicit fused op IDs)
    if (node.signature.op_id == MARMOT_OP_ADD || node.signature.op_id == MARMOT_OP_ADD_RELU ||
        node.signature.op_id == MARMOT_OP_ADD_GELU || node.signature.op_id == MARMOT_OP_ADD_SILU ||
        node.signature.op_id == MARMOT_OP_SUB || node.signature.op_id == MARMOT_OP_MUL ||
        node.signature.op_id == MARMOT_OP_DIV || node.signature.op_id == MARMOT_OP_SWIGLU ||
        node.signature.op_id == MARMOT_OP_GEGLU) {
        return execute_binary(node, ctx, bindings);
    }
    if (node.signature.op_id == MARMOT_OP_SILU || node.signature.op_id == MARMOT_OP_RELU ||
        node.signature.op_id == MARMOT_OP_GELU || node.signature.op_id == MARMOT_OP_GELU_TANH ||
        node.signature.op_id == MARMOT_OP_SIGMOID || node.signature.op_id == MARMOT_OP_TANH ||
        node.signature.op_id == MARMOT_OP_MISH || node.signature.op_id == MARMOT_OP_ELU ||
        node.signature.op_id == MARMOT_OP_SELU || node.signature.op_id == MARMOT_OP_LEAKY_RELU ||
        node.signature.op_id == MARMOT_OP_PRELU || node.signature.op_id == MARMOT_OP_ABS ||
        node.signature.op_id == MARMOT_OP_NEG || node.signature.op_id == MARMOT_OP_SIGN ||
        node.signature.op_id == MARMOT_OP_SQRT || node.signature.op_id == MARMOT_OP_EXP ||
        node.signature.op_id == MARMOT_OP_LOG || node.signature.op_id == MARMOT_OP_BITWISE_NOT) {
        return execute_unary(node, ctx, bindings);
    }
    marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Op not supported");
    return MARMOT_ERROR_NOT_IMPLEMENTED;
}

marmot_error_t
Executor::execute_matmul(const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings) {
    if (node.inputs.size() < 2 || node.outputs.empty())
        return MARMOT_ERROR_INVALID_ARGUMENT;

    auto *input = bindings[node.inputs[0]];
    auto *weight = bindings[node.inputs[1]];
    auto *output = bindings[node.outputs[0]];

    if (!input || !weight)
        return MARMOT_ERROR_INVALID_ARGUMENT;

    // Output should have been allocated by bind_outputs or passed in
    if (!output) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const bool wants_bias = (node.signature.epilogue_flags & MARMOT_EPILOGUE_BIAS) != 0;

    const marmot_tensor_t *bias = nullptr;

    if (wants_bias) {
        if (node.inputs.size() <= 2) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Matmul epilogue requires bias input");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        bias = bindings[node.inputs[2]];
        if (bias == nullptr) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Matmul bias input not bound");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
    }

    marmot_matmul_epilogue_t epilogue{};
    const marmot_matmul_epilogue_t *epilogue_ptr = nullptr;
    if (wants_bias) {
        epilogue.bias = bias;
        epilogue.enable_output_cast = false;
        epilogue.output_dtype = node.signature.output_dtype;
        epilogue_ptr = &epilogue;
    }

    marmot_kernel_args_matmul_t packed =
        {.ctx = ctx, .input = input, .weight = weight, .epilogue = epilogue_ptr, .output = output};
    return dispatch_node_bytecode(node, ctx, &node.signature, &packed, node.op_name.c_str());
}

marmot_error_t
Executor::execute_qkv(const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings) {
    if (node.inputs.size() < 4 || node.outputs.size() < 3)
        return MARMOT_ERROR_INVALID_ARGUMENT;

    auto *input = bindings[node.inputs[0]];
    auto *wq = bindings[node.inputs[1]];
    auto *wk = bindings[node.inputs[2]];
    auto *wv = bindings[node.inputs[3]];
    auto *out_q = bindings[node.outputs[0]];
    auto *out_k = bindings[node.outputs[1]];
    auto *out_v = bindings[node.outputs[2]];

    if (!input || !wq || !wk || !wv || !out_q || !out_k || !out_v)
        return MARMOT_ERROR_INVALID_ARGUMENT;

    const bool wants_bias = (node.signature.epilogue_flags & MARMOT_EPILOGUE_BIAS) != 0;
    const bool wants_rope = (node.signature.epilogue_flags & MARMOT_EPILOGUE_ROPE) != 0;

    const marmot_tensor_t *bq = nullptr;
    const marmot_tensor_t *bk = nullptr;
    const marmot_tensor_t *bv = nullptr;

    size_t bias_base = 4;
    if (wants_bias) {
        if (node.inputs.size() <= bias_base + 2)
            return MARMOT_ERROR_INVALID_ARGUMENT;
        bq = bindings[node.inputs[bias_base]];
        bk = bindings[node.inputs[bias_base + 1]];
        bv = bindings[node.inputs[bias_base + 2]];
        if (bq == nullptr || bk == nullptr || bv == nullptr)
            return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const marmot_matmul_epilogue_t *ep_ptr = nullptr;
    marmot_rope_params_t rope_params = marmot_rope_params_default();
    const marmot_rope_params_t *rope_ptr = nullptr;

    if (wants_rope) {
        if (session_ == nullptr || node.inputs.empty()) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        const marmot_value_id_t positions_id = node.inputs.back();
        if (positions_id >= bindings.size()) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        marmot_tensor_t *positions = bindings[positions_id];
        if (positions == nullptr) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        rope_params = *session_->rope_params();
        rope_params.positions = positions;
        rope_ptr = &rope_params;
    }

    marmot_kernel_args_qkv_t packed = {
        .ctx = ctx,
        .input = input,
        .wq = wq,
        .wk = wk,
        .wv = wv,
        .bq = bq,
        .bk = bk,
        .bv = bv,
        .epilogue = ep_ptr,
        .rope_params = rope_ptr,
        .out_q = out_q,
        .out_k = out_k,
        .out_v = out_v,
    };
    return dispatch_node_bytecode(node, ctx, &node.signature, &packed, node.op_name.c_str());
}

marmot_error_t
Executor::execute_quantize(const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings) {
    if (node.inputs.empty() || node.outputs.empty())
        return MARMOT_ERROR_INVALID_ARGUMENT;

    const marmot_tensor_t *input = bindings[node.inputs[0]];
    marmot_tensor_t *output = bindings[node.outputs[0]];
    if (input == nullptr || output == nullptr)
        return MARMOT_ERROR_INVALID_ARGUMENT;

    marmot_quant_kind_t kind = output->quant_kind;
    if (kind == MARMOT_QUANT_KIND_GENERIC && input != nullptr && input->quant_kind != MARMOT_QUANT_KIND_GENERIC) {
        kind = input->quant_kind;
    }
    if (kind == MARMOT_QUANT_KIND_GENERIC) {
        kind = marmot_op_metadata_quant_kind_from_qscheme(node.signature.qscheme_id);
    }

    marmot_op_signature_t sig{};
    marmot_quant_layout_t layout = MARMOT_QUANT_LAYOUT_GENERIC;
    marmot_kernel_args_quantize_t packed = {};
    marmot_error_t build_status = marmot_quantize_build(ctx, kind, input, nullptr, output, &sig, &packed, &layout);
    if (build_status != MARMOT_SUCCESS) {
        return build_status;
    }

    marmot_error_t status = dispatch_node_bytecode(node, ctx, &sig, &packed, node.op_name.c_str());
    if (status == MARMOT_SUCCESS) {
        output->quant_kind = kind;
        output->quant_layout = layout;
    }
    return status;
}

marmot_error_t Executor::execute_dequantize(
    const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings
) {
    if (node.inputs.empty() || node.outputs.empty())
        return MARMOT_ERROR_INVALID_ARGUMENT;

    const marmot_tensor_t *input = bindings[node.inputs[0]];
    marmot_tensor_t *output = bindings[node.outputs[0]];
    if (input == nullptr || output == nullptr)
        return MARMOT_ERROR_INVALID_ARGUMENT;

    marmot_quant_kind_t kind = input->quant_kind;
    if (kind == MARMOT_QUANT_KIND_GENERIC) {
        kind = marmot_op_metadata_quant_kind_from_qscheme(node.signature.qscheme_id);
    }

    marmot_op_signature_t sig{};
    marmot_quant_layout_t layout = MARMOT_QUANT_LAYOUT_GENERIC;
    marmot_kernel_args_dequantize_t packed = {};
    marmot_error_t build_status = marmot_dequantize_build(ctx, kind, input, output, &sig, &packed, &layout);
    if (build_status != MARMOT_SUCCESS) {
        return build_status;
    }

    return dispatch_node_bytecode(node, ctx, &sig, &packed, node.op_name.c_str());
}

marmot_error_t
Executor::execute_embedding(const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings) {
    if (node.inputs.size() < 2 || node.outputs.empty())
        return MARMOT_ERROR_INVALID_ARGUMENT;

    const marmot_tensor_t *weights = bindings[node.inputs[0]];
    const marmot_tensor_t *token_ids = bindings[node.inputs[1]];
    marmot_tensor_t *output = bindings[node.outputs[0]];

    if (weights == nullptr || token_ids == nullptr || output == nullptr)
        return MARMOT_ERROR_INVALID_ARGUMENT;

    marmot_embedding_gather_desc_t desc = marmot_embedding_gather_desc_default();
    desc.weights = weights;
    desc.token_ids = token_ids;
    desc.out = output;
    desc.dtype_out = output->dtype;

    marmot_op_signature_t sig{};
    marmot_dtype_t resolved_dtype = (marmot_dtype_t)MARMOT_DTYPE_COUNT;
    marmot_kernel_args_embedding_t packed = {};
    marmot_error_t build_status = marmot_embedding_gather_build(ctx, &desc, &sig, &packed, &resolved_dtype);
    if (build_status != MARMOT_SUCCESS) {
        return build_status;
    }

    return dispatch_node_bytecode(node, ctx, &sig, &packed, node.op_name.c_str());
}

marmot_error_t Executor::execute(
    const marmot_context_t *ctx, std::span<const marmot_tensor_t *const> inputs,
    std::span<marmot_tensor_t *const> outputs
) {
    if (!impl_.finalized)
        return MARMOT_ERROR_INVALID_OPERATION;
    if (ctx->backend_type != impl_.backend) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Context backend mismatch");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    // Validate counts
    auto is_runtime_input = [](const auto &v) { return v.is_input && !v.is_constant; };
    auto is_graph_output = [](const auto &v) { return !v.is_input && v.uses.empty(); };

    size_t expected_inputs = std::ranges::count_if(impl_.values, is_runtime_input);
    size_t expected_outputs = std::ranges::count_if(impl_.values, is_graph_output);

    if (inputs.size() != expected_inputs || outputs.size() != expected_outputs) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Input/Output count mismatch");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    BindingTable table(impl_.values.size());
    auto bindings = table.bindings();

    // Bind inputs
    size_t input_idx = 0;
    for (size_t i = 0; i < impl_.values.size(); ++i) {
        const auto &val = impl_.values[i];
        if (!val.is_input)
            continue;
        if (val.is_constant && val.constant_tensor) {
            table.set(i, val.constant_tensor);
        } else {
            if (!inputs[input_idx])
                return MARMOT_ERROR_INVALID_ARGUMENT;
            table.set(i, const_cast<marmot_tensor_t *>(inputs[input_idx++]));
        }
    }

    // Bind outputs
    size_t output_idx = 0;
    for (size_t i = 0; i < impl_.values.size(); ++i) {
        const auto &val = impl_.values[i];
        if (val.is_input || !val.uses.empty())
            continue;
        if (!outputs[output_idx])
            return MARMOT_ERROR_INVALID_ARGUMENT;
        table.set(i, outputs[output_idx++]);
    }

    // Allocate intermediates
    for (size_t i = 0; i < impl_.values.size(); ++i) {
        if (bindings[i])
            continue; // Already bound
        const auto &val = impl_.values[i];
        // Allocate
        marmot_tensor_t *t = allocate_tensor_for_desc(val.desc, impl_.backend);
        if (auto err = table.emplace_owned(i, t); err != MARMOT_SUCCESS)
            return err;
    }

    return execute_bound(ctx, bindings);
}

marmot_error_t
Executor::execute_rms_norm(const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings) {
    if (node.inputs.size() < 2 || node.outputs.empty())
        return MARMOT_ERROR_INVALID_ARGUMENT;

    auto *input = bindings[node.inputs[0]];
    auto *weight = bindings[node.inputs[1]];
    auto *output = bindings[node.outputs[0]];

    if (!input || !weight || !output)
        return MARMOT_ERROR_INVALID_ARGUMENT;

    float eps = session_ != nullptr ? session_->rms_norm_eps() : 1e-6f;

    marmot_kernel_args_rms_norm_t packed = {.ctx = ctx, .input = input, .weight = weight, .output = output, .eps = eps};
    return dispatch_node_bytecode(node, ctx, &node.signature, &packed, node.op_name.c_str());
}

marmot_error_t
Executor::execute_rope(const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings) {
    if (node.inputs.size() < 2 || node.outputs.empty())
        return MARMOT_ERROR_INVALID_ARGUMENT;

    auto *input = bindings[node.inputs[0]];
    auto *positions = bindings[node.inputs[1]];
    auto *output = bindings[node.outputs[0]];

    if (!input || !positions || !output)
        return MARMOT_ERROR_INVALID_ARGUMENT;

    marmot_rope_params_t rope_params = marmot_rope_params_default();
    if (session_ != nullptr) {
        rope_params = *session_->rope_params();
    }

    const marmot_tensor_t *rope_input = input;

    const size_t ndim = input->shape.ndim;
    if (ndim < 2) {
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    const size_t dim = input->shape.shape[ndim - 1];
    const size_t seq_len = input->shape.shape[ndim - 2];
    if (dim == 0 || seq_len == 0) {
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    const size_t total_tokens = marmot_tensor_num_elements(input);
    const size_t denom = seq_len * dim;
    if (denom == 0 || (total_tokens % denom) != 0) {
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    const size_t total_seqs = total_tokens / denom;

    marmot_tensor_t *positions_effective = positions;
    if (session_ != nullptr) {
        marmot_tensor_t *broadcast = session_->broadcast_rope_positions(positions, total_seqs, seq_len);
        if (broadcast == nullptr) {
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
        positions_effective = broadcast;
    }
    rope_params.positions = positions_effective;

    marmot_kernel_args_rope_t packed = {
        .ctx = ctx,
        .input = rope_input,
        .output = output,
        .rope_params = &rope_params,
        .n_past = 0,
        .n_rot = 0,
    };
    return dispatch_node_bytecode(node, ctx, &node.signature, &packed, node.op_name.c_str());
}

marmot_error_t
Executor::execute_reshape(const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings) {
    if (node.inputs.empty() || node.outputs.empty())
        return MARMOT_ERROR_INVALID_ARGUMENT;

    const marmot_tensor_t *input = bindings[node.inputs[0]];
    marmot_tensor_t *output = bindings[node.outputs[0]];

    if (input == nullptr || output == nullptr)
        return MARMOT_ERROR_INVALID_ARGUMENT;

    if (!impl_.is_valid_id(node.outputs[0])) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const marmot_graph_tensor_desc_t &desc = impl_.values[node.outputs[0]].desc;
    if (desc.dtype != input->dtype) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Reshape dtype mismatch");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    size_t input_elems = marmot_tensor_num_elements(input);
    size_t output_elems = 1;
    for (uint32_t i = 0; i < desc.ndim; ++i) {
        output_elems *= desc.shape[i];
    }
    if (input_elems != output_elems) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Reshape element count mismatch");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (output->owns_data && output->data != nullptr && output->data != input->data) {
        free(output->data);
    }

    output->shape.ndim = desc.ndim;
    for (uint32_t i = 0; i < desc.ndim; ++i) {
        output->shape.shape[i] = desc.shape[i];
        output->shape.strides[i] = desc.strides[i];
    }
    output->dtype = desc.dtype;
    output->data = input->data;
    output->capacity_bytes = input->capacity_bytes;
    output->owns_data = false;
    output->quant_params = input->quant_params;
    output->quant_kind = input->quant_kind;
    output->quant_layout = input->quant_layout;
    output->backend = input->backend;
    output->memory_location = input->memory_location;
    output->needs_sync = input->needs_sync;
    output->packed_data = nullptr;
    output->packed_src_data = nullptr;
    output->packed_bytes = 0;
    output->packed_row_bytes = 0;
    output->packed_rows = 0;

    (void)ctx;
    return MARMOT_SUCCESS;
}

marmot_error_t
Executor::execute_view(const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings) {
    if (node.inputs.empty() || node.outputs.empty())
        return MARMOT_ERROR_INVALID_ARGUMENT;

    const marmot_tensor_t *input = bindings[node.inputs[0]];
    marmot_tensor_t *output = bindings[node.outputs[0]];

    if (input == nullptr || output == nullptr)
        return MARMOT_ERROR_INVALID_ARGUMENT;

    if (!impl_.is_valid_id(node.outputs[0])) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const marmot_graph_tensor_desc_t &desc = impl_.values[node.outputs[0]].desc;

    if (output->owns_data && output->data != nullptr) {
        free(output->data);
    }

    output->shape.ndim = desc.ndim;
    for (uint32_t i = 0; i < desc.ndim; ++i) {
        output->shape.shape[i] = desc.shape[i];
        output->shape.strides[i] = desc.strides[i];
    }
    output->dtype = desc.dtype;
    output->data = static_cast<uint8_t *>(input->data) + node.view_byte_offset;
    output->capacity_bytes =
        input->capacity_bytes > node.view_byte_offset ? input->capacity_bytes - node.view_byte_offset : 0;
    output->owns_data = false;
    output->quant_params = input->quant_params;
    output->quant_kind = input->quant_kind;
    output->quant_layout = input->quant_layout;
    output->backend = input->backend;
    output->memory_location = input->memory_location;
    output->needs_sync = input->needs_sync;
    output->packed_data = nullptr;
    output->packed_src_data = nullptr;
    output->packed_bytes = 0;
    output->packed_row_bytes = 0;
    output->packed_rows = 0;

    (void)ctx;
    return MARMOT_SUCCESS;
}

marmot_error_t Executor::execute_contiguous(
    const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings
) {
    if (node.inputs.empty() || node.outputs.empty())
        return MARMOT_ERROR_INVALID_ARGUMENT;

    const marmot_tensor_t *input = bindings[node.inputs[0]];
    marmot_tensor_t *output = bindings[node.outputs[0]];

    if (input == nullptr || output == nullptr)
        return MARMOT_ERROR_INVALID_ARGUMENT;

    marmot_op_signature_t sig{};
    marmot_kernel_args_contiguous_t packed = {};
    marmot_error_t build_status = marmot_contiguous_build(ctx, input, output, &sig, &packed);
    if (build_status != MARMOT_SUCCESS) {
        return build_status;
    }

    return dispatch_node_bytecode(node, ctx, &sig, &packed, node.op_name.c_str());
}

marmot_error_t
Executor::execute_transpose(const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings) {
    if (node.inputs.empty() || node.outputs.empty())
        return MARMOT_ERROR_INVALID_ARGUMENT;

    const marmot_tensor_t *input = bindings[node.inputs[0]];
    marmot_tensor_t *output = bindings[node.outputs[0]];

    if (input == nullptr || output == nullptr)
        return MARMOT_ERROR_INVALID_ARGUMENT;

    std::vector<int> perm;
    if (!infer_permutation(input, output, perm)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Unable to infer transpose permutation from shapes");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_op_signature_t sig{};
    marmot_kernel_args_transpose_t packed = {};
    marmot_error_t build_status = marmot_transpose_build(ctx, input, output, perm.data(), &sig, &packed);
    if (build_status != MARMOT_SUCCESS) {
        return build_status;
    }

    return dispatch_node_bytecode(node, ctx, &sig, &packed, node.op_name.c_str());
}

marmot_error_t
Executor::execute_concat(const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings) {
    if (node.inputs.size() < 2 || node.outputs.empty())
        return MARMOT_ERROR_INVALID_ARGUMENT;

    std::vector<const marmot_tensor_t *> inputs;
    inputs.reserve(node.inputs.size());
    for (auto id : node.inputs) {
        if (bindings[id] == nullptr)
            return MARMOT_ERROR_INVALID_ARGUMENT;
        inputs.push_back(bindings[id]);
    }

    marmot_tensor_t *output = bindings[node.outputs[0]];
    if (output == nullptr)
        return MARMOT_ERROR_INVALID_ARGUMENT;

    int axis = -1;
    if (!infer_concat_axis(inputs, output, axis)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Unable to infer concat axis from shapes");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_op_signature_t sig{};
    marmot_kernel_args_concat_t packed = {};
    marmot_error_t build_status = marmot_concat_build(ctx, inputs.data(), inputs.size(), output, axis, &sig, &packed);
    if (build_status != MARMOT_SUCCESS) {
        return build_status;
    }

    return dispatch_node_bytecode(node, ctx, &sig, &packed, node.op_name.c_str());
}

marmot_error_t
Executor::execute_slice(const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings) {
    if (node.inputs.empty() || node.outputs.empty())
        return MARMOT_ERROR_INVALID_ARGUMENT;

    const marmot_tensor_t *input = bindings[node.inputs[0]];
    marmot_tensor_t *output = bindings[node.outputs[0]];
    if (input == nullptr || output == nullptr)
        return MARMOT_ERROR_INVALID_ARGUMENT;

    const size_t ndim = output->shape.ndim;
    if (ndim > MARMOT_MAX_DIMS)
        return MARMOT_ERROR_INVALID_ARGUMENT;

    std::array<size_t, MARMOT_MAX_DIMS> starts{};
    std::array<size_t, MARMOT_MAX_DIMS> sizes{};
    for (size_t i = 0; i < ndim; ++i) {
        starts[i] = node.slice_starts[i];
        sizes[i] = output->shape.shape[i];
    }

    marmot_op_signature_t sig{};
    marmot_kernel_args_slice_t packed = {};
    marmot_error_t build_status = marmot_slice_build(ctx, input, output, starts.data(), sizes.data(), &sig, &packed);
    if (build_status != MARMOT_SUCCESS) {
        return build_status;
    }

    return dispatch_node_bytecode(node, ctx, &sig, &packed, node.op_name.c_str());
}

marmot_error_t Executor::execute_gather_rows(
    const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings
) {
    if (node.inputs.size() < 2 || node.outputs.empty())
        return MARMOT_ERROR_INVALID_ARGUMENT;

    const marmot_tensor_t *input = bindings[node.inputs[0]];
    const marmot_tensor_t *indices = bindings[node.inputs[1]];
    marmot_tensor_t *output = bindings[node.outputs[0]];
    if (input == nullptr || indices == nullptr || output == nullptr)
        return MARMOT_ERROR_INVALID_ARGUMENT;

    marmot_op_signature_t sig{};
    marmot_kernel_args_gather_rows_t packed = {};
    marmot_error_t build_status = marmot_gather_rows_build(ctx, input, indices, output, &sig, &packed);
    if (build_status != MARMOT_SUCCESS) {
        return build_status;
    }

    return dispatch_node_bytecode(node, ctx, &sig, &packed, node.op_name.c_str());
}

marmot_error_t
Executor::execute_unary(const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings) {
    if (node.inputs.empty() || node.outputs.empty())
        return MARMOT_ERROR_INVALID_ARGUMENT;

    auto *input = bindings[node.inputs[0]];
    auto *output = bindings[node.outputs[0]];

    if (!input || !output)
        return MARMOT_ERROR_INVALID_ARGUMENT;

    marmot_kernel_args_unary_t packed = {.ctx = ctx, .input = input, .params = nullptr, .output = output};
    return dispatch_node_bytecode(node, ctx, &node.signature, &packed, node.op_name.c_str());
}

marmot_error_t
Executor::execute_binary(const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings) {
    if (node.inputs.size() < 2 || node.outputs.empty())
        return MARMOT_ERROR_INVALID_ARGUMENT;

    auto *input_a = bindings[node.inputs[0]];
    auto *input_b = bindings[node.inputs[1]];
    auto *output = bindings[node.outputs[0]];

    if (!input_a || !input_b || !output)
        return MARMOT_ERROR_INVALID_ARGUMENT;

    marmot_kernel_args_binary_t packed = {.ctx = ctx, .input_a = input_a, .input_b = input_b, .output = output};
    return dispatch_node_bytecode(node, ctx, &node.signature, &packed, node.op_name.c_str());
}

marmot_error_t
Executor::execute_softmax(const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings) {
    if (node.inputs.empty() || node.outputs.empty())
        return MARMOT_ERROR_INVALID_ARGUMENT;

    auto *input = bindings[node.inputs[0]];
    auto *output = bindings[node.outputs[0]];

    if (!input || !output)
        return MARMOT_ERROR_INVALID_ARGUMENT;

    // Default axis is -1 (last dimension)
    int32_t axis = -1;

    marmot_kernel_args_softmax_t packed = {.ctx = ctx, .input = input, .output = output, .axis = axis};
    return dispatch_node_bytecode(node, ctx, &node.signature, &packed, "softmax");
}

marmot_error_t
Executor::execute_layernorm(const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings) {
    if (node.inputs.size() < 2 || node.outputs.empty())
        return MARMOT_ERROR_INVALID_ARGUMENT;

    auto *input = bindings[node.inputs[0]];
    auto *weight = bindings[node.inputs[1]];
    auto *output = bindings[node.outputs[0]];

    if (!input || !weight || !output)
        return MARMOT_ERROR_INVALID_ARGUMENT;

    const bool wants_bias = (node.signature.epilogue_flags & MARMOT_EPILOGUE_BIAS) != 0;
    const bool wants_residual = (node.signature.epilogue_flags & MARMOT_EPILOGUE_RESIDUAL) != 0;

    const marmot_tensor_t *bias = nullptr;
    const marmot_tensor_t *residual = nullptr;

    size_t next_input = 2;
    if (wants_bias && node.inputs.size() > next_input) {
        bias = bindings[node.inputs[next_input++]];
    }
    if (wants_residual && node.inputs.size() > next_input) {
        residual = bindings[node.inputs[next_input++]];
    }

    float eps = 1e-5f;

    marmot_kernel_args_layernorm_t packed = {
        .ctx = ctx,
        .input = input,
        .weight = weight,
        .bias = bias,
        .residual = residual,
        .output = output,
        .eps = eps
    };
    return dispatch_node_bytecode(node, ctx, &node.signature, &packed, "layernorm");
}

marmot_error_t Executor::execute_paged_attention(
    const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings
) {
    if (node.inputs.size() < 7 || node.outputs.empty()) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const marmot_tensor_t *token_meta = bindings[node.inputs[0]];
    const marmot_tensor_t *q = bindings[node.inputs[1]];
    const marmot_tensor_t *k_new = bindings[node.inputs[2]];
    const marmot_tensor_t *v_new = bindings[node.inputs[3]];
    marmot_tensor_t *kv_k = bindings[node.inputs[4]];
    marmot_tensor_t *kv_v = bindings[node.inputs[5]];
    const marmot_tensor_t *block_table = bindings[node.inputs[6]];
    const bool has_scales = node.inputs.size() >= 9;
    marmot_tensor_t *kv_k_scale = has_scales ? bindings[node.inputs[7]] : nullptr;
    marmot_tensor_t *kv_v_scale = has_scales ? bindings[node.inputs[8]] : nullptr;
    marmot_tensor_t *out = bindings[node.outputs[0]];

    if (!token_meta || !q || !k_new || !v_new || !kv_k || !kv_v || !block_table || !out) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (has_scales && (!kv_k_scale || !kv_v_scale)) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (token_meta->shape.ndim < 2 || token_meta->shape.shape[1] != 4 || q->shape.ndim < 3 || k_new->shape.ndim < 3 ||
        v_new->shape.ndim < 3 || out->shape.ndim < 3 || kv_k->shape.ndim < 4) {
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    const size_t token_count = token_meta->shape.shape[0];
    if (token_count > q->shape.shape[0] || token_count > k_new->shape.shape[0] || token_count > v_new->shape.shape[0] ||
        token_count > out->shape.shape[0]) {
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    const size_t num_q_heads = q->shape.shape[1];
    const size_t head_dim = q->shape.shape[2];
    const size_t num_kv_heads = k_new->shape.shape[1];
    const size_t block_size = kv_k->shape.shape[3];
    if (token_count > UINT32_MAX || num_q_heads > UINT32_MAX || num_kv_heads > UINT32_MAX || head_dim > UINT32_MAX ||
        block_size > UINT32_MAX) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
#if MARMOT_ENABLE_FP8
    if (kv_k->dtype == MARMOT_DTYPE_FLOAT8_E4M3 && (!kv_k_scale || !kv_v_scale)) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
#endif

    const float scale = head_dim > 0 ? 1.0f / std::sqrt((float)head_dim) : 1.0f;

    marmot_kernel_args_paged_attention_t packed = {
        .ctx = ctx,
        .token_meta = token_meta,
        .q = q,
        .k_new = k_new,
        .v_new = v_new,
        .kv_k = kv_k,
        .kv_v = kv_v,
        .block_table = block_table,
        .kv_k_scale = kv_k_scale,
        .kv_v_scale = kv_v_scale,
        .out = out,
        .token_count = static_cast<uint32_t>(token_count),
        .layer_idx = node.paged_attention_layer_idx,
        .num_q_heads = static_cast<uint32_t>(num_q_heads),
        .num_kv_heads = static_cast<uint32_t>(num_kv_heads),
        .head_dim = static_cast<uint32_t>(head_dim),
        .block_size = static_cast<uint32_t>(block_size),
        .scale = scale,
    };
    marmot_error_t status = dispatch_node_bytecode(node, ctx, &node.signature, &packed, node.op_name.c_str());
    return status;
}

marmot_error_t
Executor::execute_vec_dot(const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings) {
    if (node.inputs.size() < 2 || node.outputs.empty())
        return MARMOT_ERROR_INVALID_ARGUMENT;

    const marmot_tensor_t *input = bindings[node.inputs[0]];
    const marmot_tensor_t *weight = bindings[node.inputs[1]];
    marmot_tensor_t *output = bindings[node.outputs[0]];

    if (input == nullptr || weight == nullptr || output == nullptr)
        return MARMOT_ERROR_INVALID_ARGUMENT;
    if (output->dtype != MARMOT_DTYPE_FLOAT32 || marmot_tensor_num_elements(output) != 1) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Vec dot output must be a scalar float32 tensor");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_vec_dot_descriptor_t desc{};
    if (marmot_vec_dot_descriptor_from_tensors(input, weight, &desc) != MARMOT_SUCCESS) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    float *result = (float *)output->data;
    marmot_op_signature_t sig{};
    marmot_kernel_args_vec_dot_t packed = {};
    marmot_error_t build_status = marmot_vec_dot_build(ctx, &desc, result, &sig, &packed);
    if (build_status != MARMOT_SUCCESS) {
        return build_status;
    }

    return dispatch_node_bytecode(node, ctx, &sig, &packed, node.op_name.c_str());
}

marmot_error_t
Executor::execute_reduction(const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings) {
    if (node.inputs.empty() || node.outputs.empty())
        return MARMOT_ERROR_INVALID_ARGUMENT;

    const marmot_tensor_t *input = bindings[node.inputs[0]];
    marmot_tensor_t *out_values = bindings[node.outputs[0]];
    marmot_tensor_t *out_indices = nullptr;
    if (node.outputs.size() > 1) {
        out_indices = bindings[node.outputs[1]];
    }

    if (input == nullptr || out_values == nullptr)
        return MARMOT_ERROR_INVALID_ARGUMENT;

    bool keepdims = false;
    std::vector<int32_t> axes;
    if (!infer_reduction_axes(input, out_values, keepdims, axes)) {
        // If inference fails, fall back to reducing all dimensions.
        keepdims = false;
        axes.clear();
    }

    marmot_reduction_params_t params = {
        .axes = axes.empty() ? nullptr : axes.data(),
        .num_axes = axes.size(),
        .keepdims = keepdims,
        .unbiased = false,
        .epsilon = 0.0f,
    };

    const marmot_device_reduction_op_t op = marmot_op_metadata_reduction_from_op_id(node.signature.op_id);
    if (op == MARMOT_DEVICE_REDUCTION_COUNT) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Unknown reduction op");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    marmot_op_signature_t sig{};
    marmot_kernel_args_reduction_t packed = {};
    marmot_error_t build_status =
        marmot_reduction_build(ctx, op, input, out_values, out_indices, &params, node.op_name.c_str(), &sig, &packed);
    if (build_status != MARMOT_SUCCESS) {
        return build_status;
    }

    return dispatch_node_bytecode(node, ctx, &sig, &packed, node.op_name.c_str());
}

marmot_error_t Graph::execute(
    const marmot_context_t *ctx, std::span<const marmot_tensor_t *const> inputs,
    std::span<marmot_tensor_t *const> outputs
) {
    const bool trace = graph_trace_enabled();
    marmot_allocator_usage_t usage_before{};
    marmot_allocator_usage_t usage_after{};
    const bool have_usage = trace && ctx != nullptr && marmot_allocator_get_usage(ctx, &usage_before) == MARMOT_SUCCESS;

    if (impl_ == nullptr || !impl_->finalized) {
        return MARMOT_ERROR_INVALID_OPERATION;
    }
    if (ctx == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (session_ == nullptr || !session_->compatible(ctx)) {
        session_ = std::make_unique<ExecutionSession>(*impl_);
        marmot_error_t init_status = session_->initialize(ctx);
        if (init_status != MARMOT_SUCCESS) {
            session_.reset();
            return init_status;
        }
    }

    marmot_error_t status = session_->execute(ctx, inputs, outputs);

    if (have_usage && marmot_allocator_get_usage(ctx, &usage_after) == MARMOT_SUCCESS) {
        const int64_t delta_current = (int64_t)usage_after.current_bytes - (int64_t)usage_before.current_bytes;
        const int64_t delta_peak = (int64_t)usage_after.peak_bytes - (int64_t)usage_before.peak_bytes;
        const int64_t delta_allocs = (int64_t)usage_after.active_allocations - (int64_t)usage_before.active_allocations;
        fprintf(
            stderr, "[graph trace] alloc: current=%lldB peak=%lldB active_allocs=%lld\n", (long long)delta_current,
            (long long)delta_peak, (long long)delta_allocs
        );
    }

    return status;
}

} // namespace marmot::graph
