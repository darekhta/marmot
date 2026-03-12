#include "marmot/graph/graph.hpp"

#include "marmot/config.h"
#include "marmot/error.h"
#include "marmot/op_metadata.gen.h"
#include "marmot/ops/matmul.h"
#include "marmot/stride_utils.h"
#include "marmot/tensor.h"

#include <stdexcept>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/bytecode/bytecode_compile.h"
#include "core/dispatch/fusion_flags.h"
#include "core/dispatch/kernel_query.h"
#include "core/helpers/elementwise.h"
#include "core/helpers/quant.h"
#include "execution_session.hpp"
#include "graph_handle.hpp"
#include "graph_impl.hpp"
#include "graph_naming.hpp"
#include "graph_signature.hpp"
#include "graph_validator.hpp"
#include "kernel_query.hpp"
#include "yyjson.h"

namespace marmot::graph {

namespace {

marmot_op_signature_t default_signature(marmot_op_id_t op_id) {
    marmot_op_signature_t sig = {
        .op_id = op_id,
        .profile_id = MARMOT_PROFILE_INVALID,
        .matmul_layout = MARMOT_MATMUL_LAYOUT_INVALID,
        .input_dtype = MARMOT_DTYPE_COUNT,
        .weight_dtype = MARMOT_DTYPE_COUNT,
        .output_dtype = MARMOT_DTYPE_COUNT,
        .accum_dtype = MARMOT_DTYPE_COUNT,
        .qscheme_id = MARMOT_QSCHEME_NONE,
        .quant_block =
            {
                .block_size = 0,
                .group_size = 0,
                .scale_dtype = MARMOT_DTYPE_COUNT,
                .zero_point_dtype = MARMOT_DTYPE_COUNT,
            },
        .weight_layout = MARMOT_WEIGHT_LAYOUT_INVALID,
        .stride_mode = MARMOT_STRIDE_MODE_ANY,
        .epilogue_flags = MARMOT_EPILOGUE_NONE,
        .activation = MARMOT_DEVICE_UNARY_IDENTITY,
        .variant_flags = MARMOT_FUSION_NONE,
    };
    return sig;
}

bool graph_desc_is_contiguous(const marmot_graph_tensor_desc_t &desc) {
    if (desc.ndim == 0) {
        return true;
    }
    size_t expected_stride = 1;
    for (size_t i = desc.ndim; i > 0; --i) {
        const size_t dim_idx = i - 1;
        if (desc.strides[dim_idx] != expected_stride) {
            return false;
        }
        expected_stride *= desc.shape[dim_idx];
    }
    return true;
}

bool make_contiguous_desc(marmot_graph_tensor_desc_t &desc) {
    for (uint32_t i = 0; i < desc.ndim; ++i) {
        desc.strides[i] = 0;
    }
    return ensure_strides(desc);
}

const char *stride_mode_name(marmot_stride_mode_t mode) {
    switch (mode) {
    case MARMOT_STRIDE_MODE_CONTIGUOUS:
        return "contiguous";
    case MARMOT_STRIDE_MODE_ROW_STRIDED:
        return "row_strided";
    case MARMOT_STRIDE_MODE_STRIDED:
        return "strided";
    case MARMOT_STRIDE_MODE_ANY:
    default:
        return "any";
    }
}

const char *matmul_layout_name(marmot_matmul_layout_t layout) {
    switch (layout) {
    case MARMOT_MATMUL_LAYOUT_NN:
        return "nn";
    case MARMOT_MATMUL_LAYOUT_NT:
        return "nt";
    case MARMOT_MATMUL_LAYOUT_TN:
        return "tn";
    case MARMOT_MATMUL_LAYOUT_TT:
        return "tt";
    case MARMOT_MATMUL_LAYOUT_INVALID:
    default:
        return "invalid";
    }
}

std::string format_stride_modes(const std::vector<GraphValue> &values, const std::vector<marmot_value_id_t> &ids) {
    std::string out;
    out.push_back('[');
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i > 0) {
            out.append(", ");
        }
        marmot_value_id_t id = ids[i];
        if (id >= values.size()) {
            out.append("invalid");
            continue;
        }
        const auto &desc = values[id].desc;
        marmot_stride_mode_t mode = marmot_stride_mode_from_layout(desc.ndim, desc.shape, desc.strides);
        out.append(stride_mode_name(mode));
    }
    out.push_back(']');
    return out;
}

static bool bc_emit_reg(marmot_bc_builder_t *builder, marmot_value_id_t id, size_t value_count) {
    if (builder == nullptr) {
        return false;
    }
    if (id == MARMOT_VALUE_ID_INVALID) {
        return marmot_bc_builder_emit_u16(builder, MARMOT_BC_REG_INVALID);
    }
    if (id >= value_count || id > UINT16_MAX) {
        return false;
    }
    return marmot_bc_builder_emit_u16(builder, (uint16_t)id);
}

static uint32_t bc_add_const_data(marmot_bc_builder_t *builder, const void *data, size_t size, size_t alignment) {
    if (builder == nullptr || data == nullptr || size == 0) {
        return MARMOT_BC_INVALID_OFFSET;
    }
    return marmot_bc_builder_add_const(builder, data, size, alignment);
}

static bool infer_permutation_desc(
    const marmot_graph_tensor_desc_t &input, const marmot_graph_tensor_desc_t &output, std::vector<int> &perm_out
) {
    if (input.ndim != output.ndim) {
        return false;
    }
    const size_t ndim = input.ndim;
    perm_out.assign(ndim, -1);
    std::vector<bool> used(ndim, false);
    for (size_t i = 0; i < ndim; ++i) {
        const size_t target = output.shape[i];
        int found = -1;
        for (size_t j = 0; j < ndim; ++j) {
            if (used[j] || input.shape[j] != target) {
                continue;
            }
            if (found != -1) {
                return false;
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

static bool infer_concat_axis_desc(
    const std::vector<marmot_graph_tensor_desc_t> &inputs, const marmot_graph_tensor_desc_t &output, int &axis_out
) {
    if (inputs.empty()) {
        return false;
    }
    const size_t ndim = output.ndim;
    for (const auto &input : inputs) {
        if (input.ndim != ndim) {
            return false;
        }
    }
    axis_out = -1;
    for (size_t dim = 0; dim < ndim; ++dim) {
        size_t out_dim = output.shape[dim];
        size_t sum_dim = 0;
        bool all_same = true;
        const size_t first_dim = inputs.front().shape[dim];
        for (const auto &input : inputs) {
            const size_t current = input.shape[dim];
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
    }
    return axis_out != -1;
}

static bool infer_reduction_axes_desc(
    const marmot_graph_tensor_desc_t &input, const marmot_graph_tensor_desc_t &output, bool &keepdims,
    std::vector<int32_t> &axes_out
) {
    const size_t in_ndim = input.ndim;
    const size_t out_ndim = output.ndim;
    axes_out.clear();
    if (in_ndim == out_ndim) {
        keepdims = true;
        for (size_t i = 0; i < in_ndim; ++i) {
            const size_t in_dim = input.shape[i];
            const size_t out_dim = output.shape[i];
            if (in_dim != out_dim) {
                if (out_dim != 1) {
                    return false;
                }
                axes_out.push_back((int32_t)i);
            }
        }
        return !axes_out.empty();
    }

    if (out_ndim > in_ndim) {
        return false;
    }

    keepdims = false;
    size_t out_idx = 0;
    for (size_t in_idx = 0; in_idx < in_ndim; ++in_idx) {
        if (out_idx < out_ndim && input.shape[in_idx] == output.shape[out_idx]) {
            ++out_idx;
            continue;
        }
        axes_out.push_back((int32_t)in_idx);
    }
    return out_idx == out_ndim && !axes_out.empty();
}

} // namespace

float Graph::estimated_total_us(const Impl &impl) {
    float total = 0.0f;
    for (const auto &node : impl.nodes) {
        if (!node.skip) {
            total += node.estimated_us;
        }
    }
    return total;
}

marmot_error_t Graph::finalize_impl(Impl &impl, marmot_backend_type_t backend, bool emit_errors) {
    auto rebuild_value_links = [](Impl &impl) {
        for (auto &value : impl.values) {
            value.uses.clear();
            if (!value.is_input) {
                value.defining_node = kNodeIndexInvalid;
            }
        }

        for (uint32_t i = 0; i < impl.nodes.size(); ++i) {
            const auto &node = impl.nodes[i];
            for (auto out_id : node.outputs) {
                if (out_id < impl.values.size()) {
                    impl.values[out_id].defining_node = i;
                }
            }
            for (auto in_id : node.inputs) {
                if (in_id < impl.values.size()) {
                    impl.values[in_id].uses.push_back(i);
                }
            }
        }
    };

    auto encode_node_bytecode = [&](Impl &impl, GraphNode &node, const marmot_bc_tables_t &tables,
                                    const marmot_backend_preferences_t *prefs, marmot_bc_builder_t *builder,
                                    bool emit_errors) -> marmot_error_t {
        if (builder == nullptr) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        node.rope_params_offset = MARMOT_BC_INVALID_OFFSET;
        if (node.bc_op_index == MARMOT_BC_OP_INVALID || node.bc_op_index >= tables.op_count) {
            if (emit_errors) {
                marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Invalid bytecode opcode");
            }
            return MARMOT_ERROR_NOT_IMPLEMENTED;
        }
        if (tables.schema_id == nullptr) {
            if (emit_errors) {
                marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Missing bytecode schema table");
            }
            return MARMOT_ERROR_NOT_IMPLEMENTED;
        }
        const marmot_bc_schema_id_t schema_id = tables.schema_id[node.bc_op_index];
        const size_t value_count = impl.values.size();
        const size_t op_start = builder->code_size;
        if (!marmot_bc_builder_emit_u16(builder, node.bc_op_index)) {
            if (emit_errors) {
                marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to emit opcode");
            }
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }

        switch (schema_id) {
        case MARMOT_BC_SCHEMA_UNARY: {
            if (node.inputs.empty() || node.outputs.empty()) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Unary node missing inputs or outputs");
                }
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            uint32_t params_offset = MARMOT_BC_INVALID_OFFSET;
            marmot_device_unary_op_t device_op = marmot_op_metadata_unary_from_op_id(node.signature.op_id);
            if (device_op != MARMOT_DEVICE_UNARY_COUNT) {
                marmot_activation_params_t params = {};
                marmot_error_t prep_status = marmot_unary_prepare_activation_params(device_op, nullptr, &params);
                if (prep_status != MARMOT_SUCCESS) {
                    if (emit_errors) {
                        marmot_set_error(prep_status, "Failed to prepare unary params");
                    }
                    return prep_status;
                }
                params_offset =
                    bc_add_const_data(builder, &params, sizeof(params), alignof(marmot_activation_params_t));
            }
            if (!bc_emit_reg(builder, node.inputs[0], value_count) ||
                !bc_emit_reg(builder, node.outputs[0], value_count) ||
                !marmot_bc_builder_emit_u32(builder, params_offset)) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode unary bytecode");
                }
                return MARMOT_ERROR_OUT_OF_MEMORY;
            }
            break;
        }
        case MARMOT_BC_SCHEMA_BINARY: {
            if (node.inputs.size() < 2 || node.outputs.empty()) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Binary node missing inputs or outputs");
                }
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            if (!bc_emit_reg(builder, node.inputs[0], value_count) ||
                !bc_emit_reg(builder, node.inputs[1], value_count) ||
                !bc_emit_reg(builder, node.outputs[0], value_count)) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode binary bytecode");
                }
                return MARMOT_ERROR_OUT_OF_MEMORY;
            }
            break;
        }
        case MARMOT_BC_SCHEMA_TERNARY: {
            if (node.inputs.size() < 3 || node.outputs.empty()) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Ternary node missing inputs or outputs");
                }
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            if (!bc_emit_reg(builder, node.inputs[0], value_count) ||
                !bc_emit_reg(builder, node.inputs[1], value_count) ||
                !bc_emit_reg(builder, node.inputs[2], value_count) ||
                !bc_emit_reg(builder, node.outputs[0], value_count)) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode ternary bytecode");
                }
                return MARMOT_ERROR_OUT_OF_MEMORY;
            }
            break;
        }
        case MARMOT_BC_SCHEMA_REDUCTION: {
            if (node.inputs.empty() || node.outputs.empty()) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Reduction node missing inputs or outputs");
                }
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            const auto &input_desc = impl.values[node.inputs[0]].desc;
            const auto &output_desc = impl.values[node.outputs[0]].desc;
            bool keepdims = false;
            std::vector<int32_t> axes;
            if (!infer_reduction_axes_desc(input_desc, output_desc, keepdims, axes)) {
                keepdims = false;
                axes.clear();
            }
            uint32_t axes_offset = MARMOT_BC_INVALID_OFFSET;
            if (!axes.empty()) {
                axes_offset = bc_add_const_data(builder, axes.data(), axes.size() * sizeof(int32_t), alignof(int32_t));
            }
            const uint64_t num_axes = (uint64_t)axes.size();
            const uint8_t keepdims_u8 = keepdims ? 1 : 0;
            const uint8_t unbiased_u8 = 0;
            const float epsilon = 0.0f;
            marmot_value_id_t out_indices = node.outputs.size() > 1 ? node.outputs[1] : MARMOT_VALUE_ID_INVALID;
            if (!bc_emit_reg(builder, node.inputs[0], value_count) ||
                !bc_emit_reg(builder, node.outputs[0], value_count) ||
                !bc_emit_reg(builder, out_indices, value_count) || !marmot_bc_builder_emit_u32(builder, axes_offset) ||
                !marmot_bc_builder_emit_u64(builder, num_axes) || !marmot_bc_builder_emit_u8(builder, keepdims_u8) ||
                !marmot_bc_builder_emit_u8(builder, unbiased_u8) || !marmot_bc_builder_emit_f32(builder, epsilon)) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode reduction bytecode");
                }
                return MARMOT_ERROR_OUT_OF_MEMORY;
            }
            break;
        }
        case MARMOT_BC_SCHEMA_SOFTMAX: {
            if (node.inputs.empty() || node.outputs.empty()) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Softmax node missing inputs or outputs");
                }
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            const uint32_t axis = UINT32_MAX;
            if (!bc_emit_reg(builder, node.inputs[0], value_count) ||
                !bc_emit_reg(builder, node.outputs[0], value_count) || !marmot_bc_builder_emit_u32(builder, axis)) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode softmax bytecode");
                }
                return MARMOT_ERROR_OUT_OF_MEMORY;
            }
            break;
        }
        case MARMOT_BC_SCHEMA_TOPK: {
            if (node.inputs.empty() || node.outputs.size() < 2) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "TopK node missing inputs or outputs");
                }
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            const auto &values_desc = impl.values[node.outputs[0]].desc;
            if (values_desc.ndim != 2 || values_desc.shape[1] > UINT32_MAX) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "TopK output shape mismatch");
                }
                return MARMOT_ERROR_DIMENSION_MISMATCH;
            }
            const uint32_t axis = UINT32_MAX;
            const uint32_t k = (uint32_t)values_desc.shape[1];
            if (!bc_emit_reg(builder, node.inputs[0], value_count) ||
                !bc_emit_reg(builder, node.outputs[0], value_count) ||
                !bc_emit_reg(builder, node.outputs[1], value_count) || !marmot_bc_builder_emit_u32(builder, axis) ||
                !marmot_bc_builder_emit_u32(builder, k)) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode TopK bytecode");
                }
                return MARMOT_ERROR_OUT_OF_MEMORY;
            }
            break;
        }
        case MARMOT_BC_SCHEMA_MOE_EXPERTS: {
            if (node.inputs.size() < 6 || node.outputs.empty()) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "MoE experts node missing inputs or outputs");
                }
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            if (!bc_emit_reg(builder, node.inputs[0], value_count) ||
                !bc_emit_reg(builder, node.inputs[1], value_count) ||
                !bc_emit_reg(builder, node.inputs[2], value_count) ||
                !bc_emit_reg(builder, node.inputs[3], value_count) ||
                !bc_emit_reg(builder, node.inputs[4], value_count) ||
                !bc_emit_reg(builder, node.inputs[5], value_count) ||
                !bc_emit_reg(builder, node.outputs[0], value_count) ||
                !marmot_bc_builder_emit_u32(builder, (uint32_t)node.moe_ffn_type) ||
                !marmot_bc_builder_emit_f32(builder, node.moe_weights_scale) ||
                !marmot_bc_builder_emit_u32(builder, (uint32_t)node.moe_router_weight_policy)) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode MoE experts bytecode");
                }
                return MARMOT_ERROR_OUT_OF_MEMORY;
            }
            break;
        }
        case MARMOT_BC_SCHEMA_LAYERNORM: {
            if (node.inputs.size() < 2 || node.outputs.empty()) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Layernorm node missing inputs or outputs");
                }
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            const bool wants_bias = (node.signature.epilogue_flags & MARMOT_EPILOGUE_BIAS) != 0;
            const bool wants_residual = (node.signature.epilogue_flags & MARMOT_EPILOGUE_RESIDUAL) != 0;
            marmot_value_id_t bias_id = MARMOT_VALUE_ID_INVALID;
            marmot_value_id_t residual_id = MARMOT_VALUE_ID_INVALID;
            size_t next_input = 2;
            if (wants_bias && node.inputs.size() > next_input) {
                bias_id = node.inputs[next_input++];
            }
            if (wants_residual && node.inputs.size() > next_input) {
                residual_id = node.inputs[next_input++];
            }
            const float eps = 1e-5f;
            if (!bc_emit_reg(builder, node.inputs[0], value_count) ||
                !bc_emit_reg(builder, node.inputs[1], value_count) || !bc_emit_reg(builder, bias_id, value_count) ||
                !bc_emit_reg(builder, residual_id, value_count) ||
                !bc_emit_reg(builder, node.outputs[0], value_count) || !marmot_bc_builder_emit_f32(builder, eps)) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode layernorm bytecode");
                }
                return MARMOT_ERROR_OUT_OF_MEMORY;
            }
            break;
        }
        case MARMOT_BC_SCHEMA_RMS_NORM: {
            if (node.inputs.size() < 2 || node.outputs.empty()) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "RMS norm node missing inputs or outputs");
                }
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            marmot_value_id_t residual_id = node.inputs.size() > 2 ? node.inputs[2] : MARMOT_VALUE_ID_INVALID;
            const float eps = impl.inference.rms_norm_eps;
            if (!bc_emit_reg(builder, node.inputs[0], value_count) ||
                !bc_emit_reg(builder, node.inputs[1], value_count) || !bc_emit_reg(builder, residual_id, value_count) ||
                !bc_emit_reg(builder, node.outputs[0], value_count) || !marmot_bc_builder_emit_f32(builder, eps)) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode rms norm bytecode");
                }
                return MARMOT_ERROR_OUT_OF_MEMORY;
            }
            break;
        }
        case MARMOT_BC_SCHEMA_PAGED_ATTENTION: {
            if (node.inputs.size() < 7 || node.outputs.empty()) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Paged attention node missing inputs or outputs");
                }
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            const auto &token_meta = impl.values[node.inputs[0]].desc;
            const auto &q_desc = impl.values[node.inputs[1]].desc;
            const auto &k_new_desc = impl.values[node.inputs[2]].desc;
            const auto &v_new_desc = impl.values[node.inputs[3]].desc;
            const auto &kv_k_desc = impl.values[node.inputs[4]].desc;
            const auto &block_table_desc = impl.values[node.inputs[6]].desc;
            const auto &out_desc = impl.values[node.outputs[0]].desc;
            if (token_meta.ndim < 2 || token_meta.shape[1] != 4 || q_desc.ndim < 3 || k_new_desc.ndim < 3 ||
                v_new_desc.ndim < 3 || out_desc.ndim < 3 || kv_k_desc.ndim < 4 || block_table_desc.ndim < 1) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Paged attention shape mismatch");
                }
                return MARMOT_ERROR_DIMENSION_MISMATCH;
            }
            const size_t token_capacity = token_meta.shape[0];
            const size_t num_q_heads = q_desc.shape[1];
            const size_t head_dim = q_desc.shape[2];
            const size_t num_kv_heads = k_new_desc.shape[1];
            const size_t block_size = kv_k_desc.shape[3];
            if (token_capacity > UINT32_MAX || num_q_heads > UINT32_MAX || num_kv_heads > UINT32_MAX ||
                head_dim > UINT32_MAX || block_size > UINT32_MAX) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Paged attention parameter overflow");
                }
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            const float scale = head_dim > 0 ? 1.0f / std::sqrt((float)head_dim) : 0.0f;
            const uint32_t token_count = MARMOT_BC_U32_RUNTIME;
            marmot_value_id_t kv_k_scale_id = MARMOT_VALUE_ID_INVALID;
            marmot_value_id_t kv_v_scale_id = MARMOT_VALUE_ID_INVALID;
            if (node.inputs.size() >= 9) {
                kv_k_scale_id = node.inputs[7];
                kv_v_scale_id = node.inputs[8];
            }
            if (!bc_emit_reg(builder, node.inputs[0], value_count) ||
                !bc_emit_reg(builder, node.inputs[1], value_count) ||
                !bc_emit_reg(builder, node.inputs[2], value_count) ||
                !bc_emit_reg(builder, node.inputs[3], value_count) ||
                !bc_emit_reg(builder, node.inputs[4], value_count) ||
                !bc_emit_reg(builder, node.inputs[5], value_count) ||
                !bc_emit_reg(builder, node.inputs[6], value_count) ||
                !bc_emit_reg(builder, kv_k_scale_id, value_count) ||
                !bc_emit_reg(builder, kv_v_scale_id, value_count) ||
                !bc_emit_reg(builder, node.outputs[0], value_count) ||
                !marmot_bc_builder_emit_u32(builder, (uint32_t)token_count) ||
                !marmot_bc_builder_emit_u32(builder, node.paged_attention_layer_idx) ||
                !marmot_bc_builder_emit_u32(builder, (uint32_t)num_q_heads) ||
                !marmot_bc_builder_emit_u32(builder, (uint32_t)num_kv_heads) ||
                !marmot_bc_builder_emit_u32(builder, (uint32_t)head_dim) ||
                !marmot_bc_builder_emit_u32(builder, (uint32_t)block_size) ||
                !marmot_bc_builder_emit_f32(builder, scale)) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode paged attention bytecode");
                }
                return MARMOT_ERROR_OUT_OF_MEMORY;
            }
            break;
        }
        case MARMOT_BC_SCHEMA_ROPE: {
            if (node.inputs.size() < 2 || node.outputs.empty()) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "RoPE node missing inputs or outputs");
                }
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            marmot_rope_params_t params = marmot_rope_params_default();
            params.positions = nullptr;
            params.scaling_type = impl.inference.rope_scaling_type;
            params.rope_type = impl.inference.rope_type;
            params.theta = impl.inference.rope_theta;
            params.freq_scale = impl.inference.rope_freq_scale;
            params.ext_factor = impl.inference.rope_ext_factor;
            params.attn_factor = impl.inference.rope_attn_factor;
            params.beta_fast = impl.inference.rope_beta_fast;
            params.beta_slow = impl.inference.rope_beta_slow;
            params.orig_ctx_len = impl.inference.rope_orig_ctx_len;
            params.head_dim = impl.inference.rope_head_dim;
            uint32_t params_offset = bc_add_const_data(builder, &params, sizeof(params), alignof(marmot_rope_params_t));
            if (!bc_emit_reg(builder, node.inputs[0], value_count) ||
                !bc_emit_reg(builder, node.outputs[0], value_count) ||
                !bc_emit_reg(builder, node.inputs[1], value_count) ||
                !marmot_bc_builder_emit_u32(builder, params_offset) || !marmot_bc_builder_emit_u32(builder, 0) ||
                !marmot_bc_builder_emit_u32(builder, 0)) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode rope bytecode");
                }
                return MARMOT_ERROR_OUT_OF_MEMORY;
            }
            break;
        }
        case MARMOT_BC_SCHEMA_MATMUL: {
            if (node.inputs.size() < 2 || node.outputs.empty()) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Matmul node missing inputs or outputs");
                }
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            const bool wants_bias = (node.signature.epilogue_flags & MARMOT_EPILOGUE_BIAS) != 0;
            marmot_value_id_t bias_id = MARMOT_VALUE_ID_INVALID;
            if (wants_bias && node.inputs.size() > 2) {
                bias_id = node.inputs[2];
            }
            uint32_t epilogue_offset = MARMOT_BC_INVALID_OFFSET;
            if (bias_id != MARMOT_VALUE_ID_INVALID) {
                marmot_matmul_epilogue_t epilogue = {
                    .bias = nullptr,
                    .enable_output_cast = false,
                    .output_dtype = node.signature.output_dtype,
                };
                epilogue_offset =
                    bc_add_const_data(builder, &epilogue, sizeof(epilogue), alignof(marmot_matmul_epilogue_t));
            }
            if (!bc_emit_reg(builder, node.inputs[0], value_count) ||
                !bc_emit_reg(builder, node.inputs[1], value_count) || !bc_emit_reg(builder, bias_id, value_count) ||
                !marmot_bc_builder_emit_u32(builder, epilogue_offset) ||
                !bc_emit_reg(builder, node.outputs[0], value_count)) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode matmul bytecode");
                }
                return MARMOT_ERROR_OUT_OF_MEMORY;
            }
            break;
        }
        case MARMOT_BC_SCHEMA_QKV: {
            if (node.inputs.size() < 4 || node.outputs.size() < 3) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "QKV node missing inputs or outputs");
                }
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            marmot_value_id_t bq_id = MARMOT_VALUE_ID_INVALID;
            marmot_value_id_t bk_id = MARMOT_VALUE_ID_INVALID;
            marmot_value_id_t bv_id = MARMOT_VALUE_ID_INVALID;
            if (node.inputs.size() > 6) {
                bq_id = node.inputs[4];
                bk_id = node.inputs[5];
                bv_id = node.inputs[6];
            }
            const uint32_t epilogue_offset = MARMOT_BC_INVALID_OFFSET;
            uint32_t rope_offset = MARMOT_BC_INVALID_OFFSET;
            const bool wants_rope = (node.signature.epilogue_flags & MARMOT_EPILOGUE_ROPE) != 0;
            if (wants_rope) {
                marmot_rope_params_t params = marmot_rope_params_default();
                params.positions = nullptr;
                params.scaling_type = impl.inference.rope_scaling_type;
                params.rope_type = impl.inference.rope_type;
                params.theta = impl.inference.rope_theta;
                params.freq_scale = impl.inference.rope_freq_scale;
                params.ext_factor = impl.inference.rope_ext_factor;
                params.attn_factor = impl.inference.rope_attn_factor;
                params.beta_fast = impl.inference.rope_beta_fast;
                params.beta_slow = impl.inference.rope_beta_slow;
                params.orig_ctx_len = impl.inference.rope_orig_ctx_len;
                params.head_dim = impl.inference.rope_head_dim;
                rope_offset = bc_add_const_data(builder, &params, sizeof(params), alignof(marmot_rope_params_t));
                node.rope_params_offset = rope_offset;
            }
            if (!bc_emit_reg(builder, node.inputs[0], value_count) ||
                !bc_emit_reg(builder, node.inputs[1], value_count) ||
                !bc_emit_reg(builder, node.inputs[2], value_count) ||
                !bc_emit_reg(builder, node.inputs[3], value_count) || !bc_emit_reg(builder, bq_id, value_count) ||
                !bc_emit_reg(builder, bk_id, value_count) || !bc_emit_reg(builder, bv_id, value_count) ||
                !marmot_bc_builder_emit_u32(builder, epilogue_offset) ||
                !marmot_bc_builder_emit_u32(builder, rope_offset) ||
                !bc_emit_reg(builder, node.outputs[0], value_count) ||
                !bc_emit_reg(builder, node.outputs[1], value_count) ||
                !bc_emit_reg(builder, node.outputs[2], value_count)) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode qkv bytecode");
                }
                return MARMOT_ERROR_OUT_OF_MEMORY;
            }
            break;
        }
        case MARMOT_BC_SCHEMA_RESHAPE: {
            if (node.inputs.empty() || node.outputs.empty()) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Reshape node missing inputs or outputs");
                }
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            const auto &out_desc = impl.values[node.outputs[0]].desc;
            const uint64_t new_ndim = (uint64_t)out_desc.ndim;
            uint32_t shape_offset =
                bc_add_const_data(builder, out_desc.shape, out_desc.ndim * sizeof(size_t), alignof(size_t));
            if (!bc_emit_reg(builder, node.inputs[0], value_count) ||
                !bc_emit_reg(builder, node.outputs[0], value_count) ||
                !marmot_bc_builder_emit_u32(builder, shape_offset) || !marmot_bc_builder_emit_u64(builder, new_ndim)) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode reshape bytecode");
                }
                return MARMOT_ERROR_OUT_OF_MEMORY;
            }
            break;
        }
        case MARMOT_BC_SCHEMA_VIEW: {
            if (node.inputs.empty() || node.outputs.empty()) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "View node missing inputs or outputs");
                }
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            const uint64_t byte_offset = (uint64_t)node.view_byte_offset;
            if (!bc_emit_reg(builder, node.inputs[0], value_count) ||
                !bc_emit_reg(builder, node.outputs[0], value_count) ||
                !marmot_bc_builder_emit_u64(builder, byte_offset)) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode view bytecode");
                }
                return MARMOT_ERROR_OUT_OF_MEMORY;
            }
            break;
        }
        case MARMOT_BC_SCHEMA_CONTIGUOUS: {
            if (node.inputs.empty() || node.outputs.empty()) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Contiguous node missing inputs or outputs");
                }
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            if (!bc_emit_reg(builder, node.inputs[0], value_count) ||
                !bc_emit_reg(builder, node.outputs[0], value_count)) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode contiguous bytecode");
                }
                return MARMOT_ERROR_OUT_OF_MEMORY;
            }
            break;
        }
        case MARMOT_BC_SCHEMA_TRANSPOSE: {
            if (node.inputs.empty() || node.outputs.empty()) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Transpose node missing inputs or outputs");
                }
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            const auto &input_desc = impl.values[node.inputs[0]].desc;
            const auto &output_desc = impl.values[node.outputs[0]].desc;
            std::vector<int> perm;
            if (!infer_permutation_desc(input_desc, output_desc, perm)) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Unable to infer transpose permutation");
                }
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            uint32_t perm_offset = bc_add_const_data(builder, perm.data(), perm.size() * sizeof(int), alignof(int));
            if (!bc_emit_reg(builder, node.inputs[0], value_count) ||
                !bc_emit_reg(builder, node.outputs[0], value_count) ||
                !marmot_bc_builder_emit_u32(builder, perm_offset)) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode transpose bytecode");
                }
                return MARMOT_ERROR_OUT_OF_MEMORY;
            }
            break;
        }
        case MARMOT_BC_SCHEMA_CONCAT: {
            if (node.inputs.empty() || node.outputs.empty()) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Concat node missing inputs or outputs");
                }
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            std::vector<marmot_graph_tensor_desc_t> input_descs;
            input_descs.reserve(node.inputs.size());
            std::vector<uint16_t> input_regs;
            input_regs.reserve(node.inputs.size());
            for (marmot_value_id_t id : node.inputs) {
                if (id >= impl.values.size() || id > UINT16_MAX) {
                    if (emit_errors) {
                        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Concat input id out of range");
                    }
                    return MARMOT_ERROR_INVALID_ARGUMENT;
                }
                input_descs.push_back(impl.values[id].desc);
                input_regs.push_back((uint16_t)id);
            }
            const auto &output_desc = impl.values[node.outputs[0]].desc;
            int axis = -1;
            if (!infer_concat_axis_desc(input_descs, output_desc, axis)) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Unable to infer concat axis");
                }
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            uint32_t inputs_offset =
                bc_add_const_data(builder, input_regs.data(), input_regs.size() * sizeof(uint16_t), alignof(uint16_t));
            const uint64_t count = (uint64_t)input_regs.size();
            if (!bc_emit_reg(builder, node.outputs[0], value_count) ||
                !marmot_bc_builder_emit_u32(builder, inputs_offset) || !marmot_bc_builder_emit_u64(builder, count) ||
                !marmot_bc_builder_emit_u32(builder, (uint32_t)axis)) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode concat bytecode");
                }
                return MARMOT_ERROR_OUT_OF_MEMORY;
            }
            break;
        }
        case MARMOT_BC_SCHEMA_SLICE: {
            if (node.inputs.empty() || node.outputs.empty()) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Slice node missing inputs or outputs");
                }
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            const auto &output_desc = impl.values[node.outputs[0]].desc;
            const size_t ndim = output_desc.ndim;
            std::vector<size_t> starts(ndim, 0);
            std::vector<size_t> sizes(ndim, 0);
            for (size_t i = 0; i < ndim; ++i) {
                starts[i] = node.slice_starts[i];
                sizes[i] = output_desc.shape[i];
            }
            uint32_t starts_offset =
                bc_add_const_data(builder, starts.data(), starts.size() * sizeof(size_t), alignof(size_t));
            uint32_t sizes_offset =
                bc_add_const_data(builder, sizes.data(), sizes.size() * sizeof(size_t), alignof(size_t));
            if (!bc_emit_reg(builder, node.inputs[0], value_count) ||
                !bc_emit_reg(builder, node.outputs[0], value_count) ||
                !marmot_bc_builder_emit_u32(builder, starts_offset) ||
                !marmot_bc_builder_emit_u32(builder, sizes_offset)) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode slice bytecode");
                }
                return MARMOT_ERROR_OUT_OF_MEMORY;
            }
            break;
        }
        case MARMOT_BC_SCHEMA_GATHER_ROWS: {
            if (node.inputs.size() < 2 || node.outputs.empty()) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Gather rows node missing inputs or outputs");
                }
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            if (!bc_emit_reg(builder, node.inputs[0], value_count) ||
                !bc_emit_reg(builder, node.inputs[1], value_count) ||
                !bc_emit_reg(builder, node.outputs[0], value_count)) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode gather rows bytecode");
                }
                return MARMOT_ERROR_OUT_OF_MEMORY;
            }
            break;
        }
        case MARMOT_BC_SCHEMA_QUANTIZE: {
            if (node.inputs.empty() || node.outputs.empty()) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Quantize node missing inputs or outputs");
                }
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            marmot_quant_kind_t kind = MARMOT_QUANT_KIND_GENERIC;
            const GraphValue &out_val = impl.values[node.outputs[0]];
            if (out_val.constant_tensor != nullptr &&
                out_val.constant_tensor->quant_kind != MARMOT_QUANT_KIND_GENERIC) {
                kind = out_val.constant_tensor->quant_kind;
            }
            if (kind == MARMOT_QUANT_KIND_GENERIC) {
                kind = marmot_op_metadata_quant_kind_from_qscheme(node.signature.qscheme_id);
            }
            marmot_quant_layout_t layout = marmot_quant_kind_to_layout(kind);
            if (!bc_emit_reg(builder, node.inputs[0], value_count) ||
                !marmot_bc_builder_emit_u32(builder, MARMOT_BC_INVALID_OFFSET) ||
                !bc_emit_reg(builder, node.outputs[0], value_count) ||
                !marmot_bc_builder_emit_u32(builder, (uint32_t)kind) ||
                !marmot_bc_builder_emit_u32(builder, (uint32_t)layout)) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode quantize bytecode");
                }
                return MARMOT_ERROR_OUT_OF_MEMORY;
            }
            break;
        }
        case MARMOT_BC_SCHEMA_DEQUANTIZE: {
            if (node.inputs.empty() || node.outputs.empty()) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Dequantize node missing inputs or outputs");
                }
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            marmot_quant_kind_t kind = MARMOT_QUANT_KIND_GENERIC;
            const GraphValue &in_val = impl.values[node.inputs[0]];
            if (in_val.constant_tensor != nullptr && in_val.constant_tensor->quant_kind != MARMOT_QUANT_KIND_GENERIC) {
                kind = in_val.constant_tensor->quant_kind;
            }
            if (kind == MARMOT_QUANT_KIND_GENERIC) {
                kind = marmot_op_metadata_quant_kind_from_qscheme(node.signature.qscheme_id);
            }
            marmot_quant_layout_t layout = marmot_quant_kind_to_layout(kind);
            if (!bc_emit_reg(builder, node.inputs[0], value_count) ||
                !bc_emit_reg(builder, node.outputs[0], value_count) ||
                !marmot_bc_builder_emit_u32(builder, (uint32_t)kind) ||
                !marmot_bc_builder_emit_u32(builder, (uint32_t)layout)) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode dequantize bytecode");
                }
                return MARMOT_ERROR_OUT_OF_MEMORY;
            }
            break;
        }
        case MARMOT_BC_SCHEMA_EMBEDDING: {
            if (node.inputs.size() < 2 || node.outputs.empty()) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Embedding node missing inputs or outputs");
                }
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            const marmot_dtype_t dtype_out = impl.values[node.outputs[0]].desc.dtype;
            const float scale = 1.0f;
            const int32_t padding_id = -1;
            const bool bounds_check = true;
            bool prefer_gpu = false;
            bool allow_decode = true;
            if (prefs != nullptr) {
                prefer_gpu = prefs->policy.embedding_prefer_gpu_private;
                allow_decode = prefs->policy.embedding_allow_quant_decode_on_the_fly;
            }
            if (!bc_emit_reg(builder, node.inputs[0], value_count) ||
                !bc_emit_reg(builder, node.inputs[1], value_count) ||
                !bc_emit_reg(builder, node.outputs[0], value_count) ||
                !marmot_bc_builder_emit_u32(builder, (uint32_t)dtype_out) ||
                !marmot_bc_builder_emit_f32(builder, scale) ||
                !marmot_bc_builder_emit_u32(builder, (uint32_t)padding_id) ||
                !marmot_bc_builder_emit_u8(builder, bounds_check ? 1 : 0) ||
                !marmot_bc_builder_emit_u8(builder, prefer_gpu ? 1 : 0) ||
                !marmot_bc_builder_emit_u8(builder, allow_decode ? 1 : 0)) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode embedding bytecode");
                }
                return MARMOT_ERROR_OUT_OF_MEMORY;
            }
            break;
        }
        case MARMOT_BC_SCHEMA_CONVERT:
        case MARMOT_BC_SCHEMA_COMPUTE_QPARAMS: {
            if (emit_errors) {
                marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Bytecode schema not supported in graphs");
            }
            return MARMOT_ERROR_NOT_IMPLEMENTED;
        }
        case MARMOT_BC_SCHEMA_VEC_DOT: {
            if (node.inputs.size() < 2 || node.outputs.empty()) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Vec dot node missing inputs or outputs");
                }
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            if (!bc_emit_reg(builder, node.inputs[0], value_count) ||
                !bc_emit_reg(builder, node.inputs[1], value_count) ||
                !bc_emit_reg(builder, node.outputs[0], value_count) ||
                !marmot_bc_builder_emit_u32(builder, MARMOT_BC_INVALID_OFFSET) ||
                !marmot_bc_builder_emit_u32(builder, MARMOT_BC_INVALID_OFFSET)) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to encode vec dot bytecode");
                }
                return MARMOT_ERROR_OUT_OF_MEMORY;
            }
            break;
        }
        default:
            if (emit_errors) {
                marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Unsupported bytecode schema");
            }
            return MARMOT_ERROR_NOT_IMPLEMENTED;
        }

        if (tables.imm_size != nullptr) {
            const size_t imm_bytes = builder->code_size - op_start - sizeof(uint16_t);
            const uint16_t expected = tables.imm_size[node.bc_op_index];
            if (imm_bytes != expected) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Bytecode immediate size mismatch");
                }
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
        }

        return MARMOT_SUCCESS;
    };

    auto legalize_layouts = [&](Impl &impl, marmot_backend_type_t backend, bool emit_errors) -> marmot_error_t {
        std::vector<GraphNode> legalized_nodes;
        legalized_nodes.reserve(impl.nodes.size());
        std::unordered_map<marmot_value_id_t, marmot_value_id_t> contiguous_cache;

        for (size_t i = 0; i < impl.nodes.size(); ++i) {
            GraphNode node = impl.nodes[i];

            if (!populate_signature(impl.values, node)) {
                if (emit_errors) {
                    marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Unable to infer signature");
                }
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            if (node.signature.op_id == MARMOT_OP_LINEAR) {
                node.signature.op_id = MARMOT_OP_MATMUL;
            }

            auto selection = query_backend_for_node(backend, &node.signature);
            std::vector<GraphNode> output_copy_nodes;
            if (!selection.supported) {
                bool inserted_inputs = false;
                for (size_t input_idx = 0; input_idx < node.inputs.size(); ++input_idx) {
                    marmot_value_id_t input_id = node.inputs[input_idx];
                    if (input_id >= impl.values.size()) {
                        if (emit_errors) {
                            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Invalid input id");
                        }
                        return MARMOT_ERROR_INVALID_ARGUMENT;
                    }

                    const auto &input_desc = impl.values[input_id].desc;
                    if (graph_desc_is_contiguous(input_desc)) {
                        continue;
                    }

                    auto cached = contiguous_cache.find(input_id);
                    if (cached != contiguous_cache.end()) {
                        node.inputs[input_idx] = cached->second;
                        inserted_inputs = true;
                        continue;
                    }

                    marmot_graph_tensor_desc_t copy_desc = input_desc;
                    if (!make_contiguous_desc(copy_desc)) {
                        if (emit_errors) {
                            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Unable to legalize contiguous layout");
                        }
                        return MARMOT_ERROR_INVALID_ARGUMENT;
                    }

                    GraphValue copy_value{};
                    copy_value.desc = copy_desc;
                    copy_value.name = generate_output_name(legalized_nodes.size(), 0);
                    impl.values.push_back(std::move(copy_value));
                    marmot_value_id_t copy_id = static_cast<marmot_value_id_t>(impl.values.size() - 1);

                    GraphNode copy_node{};
                    copy_node.op_name = "contiguous";
                    copy_node.signature = default_signature(MARMOT_OP_CONTIGUOUS);
                    copy_node.inputs = {input_id};
                    copy_node.outputs = {copy_id};

                    legalized_nodes.push_back(std::move(copy_node));
                    contiguous_cache.emplace(input_id, copy_id);
                    node.inputs[input_idx] = copy_id;
                    inserted_inputs = true;
                }

                if (inserted_inputs) {
                    node.signature.stride_mode = MARMOT_STRIDE_MODE_ANY;
                    if (!populate_signature(impl.values, node)) {
                        if (emit_errors) {
                            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Unable to infer signature");
                        }
                        return MARMOT_ERROR_INVALID_ARGUMENT;
                    }
                    if (node.signature.op_id == MARMOT_OP_LINEAR) {
                        node.signature.op_id = MARMOT_OP_MATMUL;
                    }
                    selection = query_backend_for_node(backend, &node.signature);
                }

                if (!selection.supported) {
                    bool inserted_outputs = false;
                    for (size_t output_idx = 0; output_idx < node.outputs.size(); ++output_idx) {
                        marmot_value_id_t output_id = node.outputs[output_idx];
                        if (output_id >= impl.values.size()) {
                            if (emit_errors) {
                                marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Invalid output id");
                            }
                            return MARMOT_ERROR_INVALID_ARGUMENT;
                        }

                        const auto &output_desc = impl.values[output_id].desc;
                        if (graph_desc_is_contiguous(output_desc)) {
                            continue;
                        }

                        marmot_graph_tensor_desc_t copy_desc = output_desc;
                        if (!make_contiguous_desc(copy_desc)) {
                            if (emit_errors) {
                                marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Unable to legalize output layout");
                            }
                            return MARMOT_ERROR_INVALID_ARGUMENT;
                        }

                        GraphValue copy_value{};
                        copy_value.desc = copy_desc;
                        copy_value.name = generate_output_name(legalized_nodes.size(), output_idx);
                        impl.values.push_back(std::move(copy_value));
                        marmot_value_id_t copy_id = static_cast<marmot_value_id_t>(impl.values.size() - 1);

                        node.outputs[output_idx] = copy_id;

                        GraphNode copy_node{};
                        copy_node.op_name = "contiguous";
                        copy_node.signature = default_signature(MARMOT_OP_CONTIGUOUS);
                        copy_node.inputs = {copy_id};
                        copy_node.outputs = {output_id};
                        output_copy_nodes.push_back(std::move(copy_node));
                        inserted_outputs = true;
                    }

                    if (inserted_outputs) {
                        node.signature.stride_mode = MARMOT_STRIDE_MODE_ANY;
                        if (!populate_signature(impl.values, node)) {
                            if (emit_errors) {
                                marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Unable to infer signature");
                            }
                            return MARMOT_ERROR_INVALID_ARGUMENT;
                        }
                        if (node.signature.op_id == MARMOT_OP_LINEAR) {
                            node.signature.op_id = MARMOT_OP_MATMUL;
                        }
                        selection = query_backend_for_node(backend, &node.signature);
                    }
                }

                if (!selection.supported) {
                    if (emit_errors) {
                        const char *reason = selection.fallback_reason ? selection.fallback_reason : "No kernel";
                        std::string inputs = format_stride_modes(impl.values, node.inputs);
                        std::string outputs = format_stride_modes(impl.values, node.outputs);
                        fprintf(
                            stderr,
                            "Graph finalize: missing kernel for op %d profile %d layout %s (variant 0x%x, epilogue "
                            "0x%x): %s "
                            "(stride %s inputs %s outputs %s)\n",
                            (int)node.signature.op_id, (int)node.signature.profile_id,
                            matmul_layout_name(node.signature.matmul_layout), node.signature.variant_flags,
                            node.signature.epilogue_flags, reason, stride_mode_name(node.signature.stride_mode),
                            inputs.c_str(), outputs.c_str()
                        );
                        std::string detail = "Kernel missing for op " + std::to_string(node.signature.op_id) +
                            " profile " + std::to_string(node.signature.profile_id) + " layout " +
                            matmul_layout_name(node.signature.matmul_layout) + ": " + reason + " (stride " +
                            stride_mode_name(node.signature.stride_mode) + " inputs " + inputs + " outputs " + outputs +
                            ")";
                        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, detail.c_str());
                    }
                    return MARMOT_ERROR_NOT_IMPLEMENTED;
                }
            }

            legalized_nodes.push_back(std::move(node));
            for (auto &copy_node : output_copy_nodes) {
                legalized_nodes.push_back(std::move(copy_node));
            }
        }

        impl.nodes = std::move(legalized_nodes);
        rebuild_value_links(impl);
        return MARMOT_SUCCESS;
    };

    marmot_error_t legalize_status = legalize_layouts(impl, backend, emit_errors);
    if (legalize_status != MARMOT_SUCCESS) {
        return legalize_status;
    }

    for (auto &node : impl.nodes) {
        if (!populate_signature(impl.values, node)) {
            if (emit_errors) {
                marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Unable to infer signature");
            }
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        if (node.signature.op_id == MARMOT_OP_LINEAR) {
            node.signature.op_id = MARMOT_OP_MATMUL;
        }
    }

    apply_fusion_pass(impl, backend);

    for (auto &node : impl.nodes) {
        if (node.signature.op_id != MARMOT_OP_RESHAPE && node.signature.op_id != MARMOT_OP_VIEW) {
            continue;
        }
        if (node.inputs.size() != 1 || node.outputs.size() != 1) {
            continue;
        }
        node.skip = true;
    }

    marmot_device_caps_t caps = {};
    if (!marmot_backend_detect_default_caps(backend, &caps)) {
        if (emit_errors) {
            marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Backend capabilities not available");
        }
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    for (size_t i = 0; i < impl.nodes.size(); ++i) {
        auto &node = impl.nodes[i];
        if (node.skip)
            continue;

        auto selection = query_backend_for_node(backend, &node.signature);
        if (!selection.supported) {
            if (emit_errors) {
                char detail[128];
                const char *reason = selection.fallback_reason ? selection.fallback_reason : "No kernel";
                fprintf(
                    stderr,
                    "Graph finalize: missing kernel for op %d profile %d layout %s (fusion 0x%x, epilogue 0x%x): %s\n",
                    (int)node.signature.op_id, (int)node.signature.profile_id,
                    matmul_layout_name(node.signature.matmul_layout), node.signature.variant_flags,
                    node.signature.epilogue_flags, reason
                );
                snprintf(
                    detail, sizeof(detail), "Kernel missing for op %d profile %d layout %s: %s",
                    (int)node.signature.op_id, (int)node.signature.profile_id,
                    matmul_layout_name(node.signature.matmul_layout), reason
                );
                marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, detail);
            }
            return MARMOT_ERROR_NOT_IMPLEMENTED;
        }
        node.kernel_id = selection.kernel_id;
        node.estimated_us = selection.estimated_us;
        marmot_bc_selection_t bc_sel = marmot_bc_compile_signature_with_caps(backend, &caps, &node.signature, false);
        if (!bc_sel.supported) {
            if (emit_errors) {
                char detail[128];
                const char *reason = bc_sel.reason != nullptr ? bc_sel.reason : "Bytecode not supported";
                snprintf(detail, sizeof(detail), "Bytecode missing for op %d: %s", (int)node.signature.op_id, reason);
                marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, detail);
            }
            return MARMOT_ERROR_NOT_IMPLEMENTED;
        }
        node.bc_op_index = bc_sel.op_index;
    }

    impl.plan.clear();
    impl.plan.reserve(impl.nodes.size());
    for (uint32_t i = 0; i < impl.nodes.size(); ++i) {
        if (impl.nodes[i].skip)
            continue;
        impl.plan.push_back({ExecutionCommand::Kind::Launch, i});
    }

    marmot_bc_program_destroy(&impl.program);
    impl.bc_instr_nodes.clear();
    impl.rope_nodes.clear();

    marmot_bc_tables_t tables = {};
    if (!marmot_bc_get_tables(backend, &tables)) {
        if (emit_errors) {
            marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Bytecode tables not available for backend");
        }
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (impl.values.size() > UINT16_MAX) {
        if (emit_errors) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Graph has too many value slots for bytecode");
        }
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_backend_preferences_t prefs = {};
    bool prefs_ok = marmot_backend_get_default_preferences(backend, &caps, &prefs);

    marmot_bc_builder_t builder = {};
    if (!marmot_bc_builder_init(&builder)) {
        if (emit_errors) {
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to init bytecode builder");
        }
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    for (const auto &cmd : impl.plan) {
        if (cmd.kind != ExecutionCommand::Kind::Launch) {
            continue;
        }
        auto &node = impl.nodes[cmd.node_index];
        if (node.signature.op_id == MARMOT_OP_ROPE && node.inputs.size() >= 2) {
            impl.rope_nodes.push_back(
                GraphRopeInfo{.node_index = cmd.node_index, .input_id = node.inputs[0], .positions_id = node.inputs[1]}
            );
        }
        if ((node.signature.epilogue_flags & MARMOT_EPILOGUE_ROPE) != 0 &&
            (node.signature.op_id == MARMOT_OP_QKV_SHARED_INPUT || node.signature.op_id == MARMOT_OP_QKV_PROJECTION ||
             node.signature.op_id == MARMOT_OP_QKV_ROPE) &&
            node.inputs.size() >= 5) {
            impl.rope_nodes.push_back(
                GraphRopeInfo{
                    .node_index = cmd.node_index,
                    .input_id = node.inputs[0],
                    .positions_id = node.inputs.back()
                }
            );
        }
        marmot_error_t encode_status =
            encode_node_bytecode(impl, node, tables, prefs_ok ? &prefs : nullptr, &builder, emit_errors);
        if (encode_status != MARMOT_SUCCESS) {
            marmot_bc_builder_reset(&builder);
            return encode_status;
        }
        impl.bc_instr_nodes.push_back(cmd.node_index);
    }

    if (!marmot_bc_builder_emit_u16(&builder, MARMOT_BC_END)) {
        marmot_bc_builder_reset(&builder);
        if (emit_errors) {
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to finalize bytecode");
        }
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    const uint16_t reg_count = (uint16_t)impl.values.size();
    if (!marmot_bc_builder_finish(
            &builder, &impl.program, tables.imm_size, tables.exec_table, reg_count, tables.op_count
        )) {
        marmot_bc_builder_reset(&builder);
        if (emit_errors) {
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to build bytecode program");
        }
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    impl.backend = backend;
    impl.finalized = true;
    return MARMOT_SUCCESS;
}

namespace {

const char *backend_name(marmot_backend_type_t backend) {
    switch (backend) {
    case MARMOT_BACKEND_CPU:
        return "cpu";
    case MARMOT_BACKEND_METAL:
        return "metal";
    case MARMOT_BACKEND_CUDA:
        return "cuda";
    default:
        return "unknown";
    }
}

const char *dtype_name(marmot_dtype_t dtype) {
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT32:
        return "float32";
    case MARMOT_DTYPE_FLOAT16:
        return "float16";
    case MARMOT_DTYPE_BFLOAT16:
        return "bfloat16";
    case MARMOT_DTYPE_INT32:
        return "int32";
    case MARMOT_DTYPE_INT16:
        return "int16";
    case MARMOT_DTYPE_INT8:
        return "int8";
    case MARMOT_DTYPE_UINT8:
        return "uint8";
    case MARMOT_DTYPE_UINT16:
        return "uint16";
    case MARMOT_DTYPE_UINT32:
        return "uint32";
    case MARMOT_DTYPE_UINT64:
        return "uint64";
    case MARMOT_DTYPE_FLOAT64:
        return "float64";
    default:
        return "invalid";
    }
}

const char *storage_class(const GraphValue &v) {
    if (v.is_constant)
        return "constant";
    if (v.is_input)
        return "input";
    if (v.uses.empty())
        return "output";
    return "intermediate";
}

const char *weight_layout_name(marmot_weight_layout_t layout) {
    switch (layout) {
    case MARMOT_WEIGHT_LAYOUT_SEPARATE:
        return "separate";
    case MARMOT_WEIGHT_LAYOUT_PACKED_3MK:
        return "packed_3mk";
    default:
        return "invalid";
    }
}

const char *activation_name(marmot_device_unary_op_t op) {
    switch (op) {
    case MARMOT_DEVICE_UNARY_RELU:
        return "relu";
    case MARMOT_DEVICE_UNARY_GELU:
        return "gelu";
    case MARMOT_DEVICE_UNARY_GELU_TANH:
        return "gelu_tanh";
    case MARMOT_DEVICE_UNARY_SILU:
        return "silu";
    case MARMOT_DEVICE_UNARY_SIGMOID:
        return "sigmoid";
    case MARMOT_DEVICE_UNARY_TANH:
        return "tanh";
    case MARMOT_DEVICE_UNARY_MISH:
        return "mish";
    case MARMOT_DEVICE_UNARY_ELU:
        return "elu";
    case MARMOT_DEVICE_UNARY_SELU:
        return "selu";
    case MARMOT_DEVICE_UNARY_LEAKY_RELU:
        return "leaky_relu";
    case MARMOT_DEVICE_UNARY_PRELU:
        return "prelu";
    case MARMOT_DEVICE_UNARY_ABS:
        return "abs";
    case MARMOT_DEVICE_UNARY_NEG:
        return "neg";
    case MARMOT_DEVICE_UNARY_SIGN:
        return "sign";
    case MARMOT_DEVICE_UNARY_SQRT:
        return "sqrt";
    case MARMOT_DEVICE_UNARY_EXP:
        return "exp";
    case MARMOT_DEVICE_UNARY_LOG:
        return "log";
    case MARMOT_DEVICE_UNARY_BITWISE_NOT:
        return "bitwise_not";
    case MARMOT_DEVICE_UNARY_IDENTITY:
        return "identity";
    default:
        return "unknown";
    }
}

} // namespace

// Constructor & Destructor
Graph::Graph() : impl_(std::make_unique<Impl>()) {}
Graph::~Graph() = default;
Graph::Graph(Graph &&) noexcept = default;
Graph &Graph::operator=(Graph &&) noexcept = default;

marmot_backend_type_t Graph::backend() const {
    return impl_->backend;
}

std::expected<marmot_value_id_t, marmot_error_t> Graph::add_input(const marmot_graph_tensor_desc_t &desc) {
    if (impl_->finalized)
        return std::unexpected(MARMOT_ERROR_INVALID_OPERATION);
    if (!validate_desc(desc))
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);

    GraphValue value{};
    value.desc = desc;
    if (!ensure_strides(value.desc)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Unable to compute input strides");
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }
    value.is_input = true;
    value.name = generate_input_name(static_cast<marmot_value_id_t>(impl_->values.size()));

    try {
        impl_->values.push_back(std::move(value));
    } catch (...) {
        return std::unexpected(MARMOT_ERROR_OUT_OF_MEMORY);
    }

    return static_cast<marmot_value_id_t>(impl_->values.size() - 1);
}

std::expected<std::vector<marmot_value_id_t>, marmot_error_t> Graph::add_op(
    std::string_view op_name, const marmot_op_signature_t *signature, std::span<const marmot_value_id_t> inputs,
    std::span<const marmot_graph_tensor_desc_t> output_descs
) {
    if (impl_->finalized)
        return std::unexpected(MARMOT_ERROR_INVALID_OPERATION);

    for (auto id : inputs) {
        if (!impl_->is_valid_id(id))
            return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }

    // Check inputs are ready
    for (auto id : inputs) {
        const auto &val = impl_->values[id];
        if (!val.is_input && (val.defining_node == kNodeIndexInvalid || val.defining_node >= impl_->nodes.size())) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Graph value has no defining node");
            return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
        }
    }

    GraphNode node{};
    node.op_name = std::string(op_name);
    if (signature) {
        node.signature = *signature;
    } else {
        // Default empty signature
        node.signature.profile_id = MARMOT_PROFILE_INVALID;
        node.signature.input_dtype = MARMOT_DTYPE_COUNT;
        node.signature.weight_dtype = MARMOT_DTYPE_COUNT;
        node.signature.output_dtype = MARMOT_DTYPE_COUNT;
        node.signature.accum_dtype = MARMOT_DTYPE_COUNT;
        node.signature.qscheme_id = MARMOT_QSCHEME_NONE;
        node.signature.weight_layout = MARMOT_WEIGHT_LAYOUT_INVALID;
        node.signature.epilogue_flags = MARMOT_EPILOGUE_NONE;
        node.signature.activation = MARMOT_DEVICE_UNARY_IDENTITY;
        node.signature.variant_flags = MARMOT_FUSION_NONE;
    }

    if (node.signature.op_id == MARMOT_OP_INVALID) {
        node.signature.op_id = marmot_string_to_op_id(node.op_name.c_str());
    }
    if (node.signature.op_id == MARMOT_OP_INVALID)
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);

    try {
        node.inputs.assign(inputs.begin(), inputs.end());
    } catch (...) {
        return std::unexpected(MARMOT_ERROR_OUT_OF_MEMORY);
    }

    const size_t node_index = impl_->nodes.size();

    std::vector<GraphValue> new_values;
    new_values.reserve(output_descs.size());
    for (size_t i = 0; i < output_descs.size(); ++i) {
        if (!validate_desc(output_descs[i])) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Invalid output descriptor");
            return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
        }
        GraphValue value{};
        value.desc = output_descs[i];
        if (!ensure_strides(value.desc)) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Unable to compute output strides");
            return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
        }
        value.defining_node = static_cast<uint32_t>(node_index);
        value.name = generate_output_name(node_index, i);
        new_values.push_back(std::move(value));
    }

    // Record uses (stage for rollback)
    std::vector<size_t> original_use_counts;
    original_use_counts.reserve(inputs.size());
    for (auto id : inputs) {
        original_use_counts.push_back(impl_->values[id].uses.size());
        try {
            impl_->values[id].uses.push_back(static_cast<uint32_t>(node_index));
        } catch (...) {
            // Rollback uses
            for (size_t i = 0; i < original_use_counts.size(); ++i) {
                if (i < impl_->values[inputs[i]].uses.size()) // Check if pushed
                    impl_->values[inputs[i]].uses.resize(original_use_counts[i]);
            }
            return std::unexpected(MARMOT_ERROR_OUT_OF_MEMORY);
        }
    }

    std::vector<marmot_value_id_t> out_ids;
    out_ids.resize(output_descs.size());

    try {
        for (size_t i = 0; i < new_values.size(); ++i) {
            impl_->values.push_back(std::move(new_values[i]));
            marmot_value_id_t vid = static_cast<marmot_value_id_t>(impl_->values.size() - 1);
            node.outputs.push_back(vid);
            out_ids[i] = vid;
        }
        impl_->nodes.push_back(std::move(node));
    } catch (...) {
        // Roll back uses and any appended values
        for (size_t i = 0; i < original_use_counts.size(); ++i) {
            if (inputs[i] < impl_->values.size()) {
                impl_->values[inputs[i]].uses.resize(original_use_counts[i]);
            }
        }
        while (!impl_->values.empty() && impl_->values.back().defining_node == node_index) {
            impl_->values.pop_back();
        }
        return std::unexpected(MARMOT_ERROR_OUT_OF_MEMORY);
    }

    return out_ids;
}

marmot_error_t Graph::set_constant(marmot_value_id_t id, marmot_tensor_t *tensor) {
    if (impl_->finalized)
        return MARMOT_ERROR_INVALID_OPERATION;
    if (!impl_->is_valid_id(id))
        return MARMOT_ERROR_INVALID_ARGUMENT;
    if (tensor == nullptr)
        return MARMOT_ERROR_INVALID_ARGUMENT;

    auto &val = impl_->values[id];
    if (!val.is_input)
        return MARMOT_ERROR_INVALID_ARGUMENT;

    val.is_constant = true;
    val.constant_tensor = tensor;
    return MARMOT_SUCCESS;
}

marmot_error_t Graph::set_name(marmot_value_id_t id, std::string_view name) {
    if (impl_->finalized)
        return MARMOT_ERROR_INVALID_OPERATION;
    if (!impl_->is_valid_id(id))
        return MARMOT_ERROR_INVALID_ARGUMENT;
    impl_->values[id].name = name;
    return MARMOT_SUCCESS;
}

marmot_error_t
Graph::set_inference_hints(size_t max_seq_len, const marmot_rope_params_t *rope_params, float rms_norm_eps) {
    impl_->inference.max_seq_len = max_seq_len;
    if (rope_params != nullptr) {
        impl_->inference.rope_theta = rope_params->theta;
        impl_->inference.rope_scaling_type = rope_params->scaling_type;
        impl_->inference.rope_type = rope_params->rope_type;
        impl_->inference.rope_freq_scale = rope_params->freq_scale;
        impl_->inference.rope_ext_factor = rope_params->ext_factor;
        impl_->inference.rope_attn_factor = rope_params->attn_factor;
        impl_->inference.rope_beta_fast = rope_params->beta_fast;
        impl_->inference.rope_beta_slow = rope_params->beta_slow;
        impl_->inference.rope_orig_ctx_len = rope_params->orig_ctx_len;
        impl_->inference.rope_head_dim = rope_params->head_dim;
    }
    impl_->inference.rms_norm_eps = rms_norm_eps;
    if (session_) {
        session_.reset();
    }
    return MARMOT_SUCCESS;
}

std::expected<marmot_graph_tensor_desc_t, marmot_error_t> Graph::get_value_desc(marmot_value_id_t id) const {
    if (!impl_->is_valid_id(id))
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    return impl_->values[id].desc;
}

marmot_error_t Graph::finalize(marmot_backend_type_t backend) {
    if (impl_->finalized)
        return MARMOT_ERROR_INVALID_OPERATION;

    Impl draft = *impl_;
    marmot_error_t status = finalize_impl(draft, backend, true);
    if (status != MARMOT_SUCCESS)
        return status;
    *impl_ = std::move(draft);
    return MARMOT_SUCCESS;
}

marmot_error_t Graph::finalize_auto(marmot_backend_type_t *out_backend) {
    return finalize_auto_with_policy(MARMOT_ROUTING_AUTO, out_backend);
}

marmot_error_t Graph::finalize_auto_with_policy(marmot_routing_policy_t policy, marmot_backend_type_t *out_backend) {
    if (impl_->finalized)
        return MARMOT_ERROR_INVALID_OPERATION;

    std::vector<marmot_backend_type_t> candidates;
    candidates.reserve(2);

    switch (policy) {
    case MARMOT_ROUTING_ALWAYS_CPU:
        candidates.push_back(MARMOT_BACKEND_CPU);
        break;
    case MARMOT_ROUTING_ALWAYS_GPU:
#if MARMOT_ENABLE_METAL
        candidates.push_back(MARMOT_BACKEND_METAL);
#else
        marmot_set_error(MARMOT_ERROR_DEVICE_NOT_AVAILABLE, "GPU routing requested but no GPU backend is available");
        return MARMOT_ERROR_DEVICE_NOT_AVAILABLE;
#endif
        break;
    case MARMOT_ROUTING_AUTO:
        // Prefer GPU over CPU when both support the graph
#if MARMOT_ENABLE_METAL
        candidates.push_back(MARMOT_BACKEND_METAL);
#endif
        candidates.push_back(MARMOT_BACKEND_CPU);
        break;
    default:
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Invalid routing policy");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    struct Candidate {
        bool ok{false};
        marmot_backend_type_t backend{MARMOT_BACKEND_CPU};
        float total_us{std::numeric_limits<float>::infinity()};
    };

    Candidate best{};
    Candidate metal{};

    auto consider = [&](marmot_backend_type_t backend) {
        Impl draft = *impl_;
        if (finalize_impl(draft, backend, false) != MARMOT_SUCCESS)
            return;
        float total = estimated_total_us(draft);
        Candidate candidate = {
            .ok = true,
            .backend = backend,
            .total_us = total,
        };
        if (backend == MARMOT_BACKEND_METAL) {
            metal = candidate;
        }
        if (!best.ok || total < best.total_us) {
            best = candidate;
        }
    };

    for (auto backend : candidates) {
        consider(backend);
    }

    if (!best.ok) {
        marmot_backend_type_t fallback = candidates.empty() ? MARMOT_BACKEND_CPU : candidates[0];
        Impl draft = *impl_;
        return finalize_impl(draft, fallback, true);
    }

    const marmot_backend_type_t chosen_backend =
        (policy == MARMOT_ROUTING_AUTO && metal.ok) ? metal.backend : best.backend;

    Impl draft = *impl_;
    marmot_error_t status = finalize_impl(draft, chosen_backend, true);
    if (status != MARMOT_SUCCESS)
        return status;
    *impl_ = std::move(draft);
    if (out_backend != nullptr) {
        *out_backend = chosen_backend;
    }
    return MARMOT_SUCCESS;
}

marmot_error_t Graph::dump_json(const char *path) const {
    yyjson_mut_doc *doc = yyjson_mut_doc_new(nullptr);
    if (!doc)
        return MARMOT_ERROR_OUT_OF_MEMORY;
    auto *root = yyjson_mut_obj(doc);
    if (!root) {
        yyjson_mut_doc_free(doc);
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }
    yyjson_mut_doc_set_root(doc, root);

    yyjson_mut_obj_add_str(doc, root, "version", "1.0");
    yyjson_mut_obj_add_str(doc, root, "backend", backend_name(impl_->backend));
    yyjson_mut_obj_add_uint(doc, root, "num_nodes", impl_->nodes.size());
    yyjson_mut_obj_add_uint(doc, root, "num_values", impl_->values.size());

    auto *inputs_arr = yyjson_mut_arr(doc);
    auto *outputs_arr = yyjson_mut_arr(doc);
    auto *nodes_arr = yyjson_mut_arr(doc);
    auto *values_arr = yyjson_mut_arr(doc);
    auto *tensors_arr = yyjson_mut_arr(doc);
    if (!inputs_arr || !outputs_arr || !nodes_arr || !values_arr || !tensors_arr) {
        yyjson_mut_doc_free(doc);
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    yyjson_mut_obj_add_val(doc, root, "inputs", inputs_arr);
    yyjson_mut_obj_add_val(doc, root, "outputs", outputs_arr);
    yyjson_mut_obj_add_val(doc, root, "nodes", nodes_arr);
    yyjson_mut_obj_add_val(doc, root, "values", values_arr);
    yyjson_mut_obj_add_val(doc, root, "tensors", tensors_arr);

    // Values metadata
    for (size_t i = 0; i < impl_->values.size(); ++i) {
        const auto &v = impl_->values[i];
        auto *val_obj = yyjson_mut_obj(doc);
        if (!val_obj) {
            yyjson_mut_doc_free(doc);
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        yyjson_mut_obj_add_uint(doc, val_obj, "id", i);
        yyjson_mut_obj_add_str(doc, val_obj, "name", v.name.c_str());
        yyjson_mut_obj_add_str(doc, val_obj, "dtype", dtype_name(v.desc.dtype));
        auto *shape_arr = yyjson_mut_arr(doc);
        if (!shape_arr) {
            yyjson_mut_doc_free(doc);
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        auto *strides_arr = yyjson_mut_arr(doc);
        if (!strides_arr) {
            yyjson_mut_doc_free(doc);
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        for (uint32_t d = 0; d < v.desc.ndim; ++d)
            yyjson_mut_arr_add_uint(doc, shape_arr, v.desc.shape[d]);
        for (uint32_t d = 0; d < v.desc.ndim; ++d)
            yyjson_mut_arr_add_uint(doc, strides_arr, v.desc.strides[d]);
        yyjson_mut_obj_add_val(doc, val_obj, "shape", shape_arr);
        yyjson_mut_obj_add_val(doc, val_obj, "strides", strides_arr);
        yyjson_mut_obj_add_bool(doc, val_obj, "is_input", v.is_input);
        yyjson_mut_obj_add_bool(doc, val_obj, "is_constant", v.is_constant);
        yyjson_mut_obj_add_str(doc, val_obj, "storage", storage_class(v));
        yyjson_mut_arr_add_val(values_arr, val_obj);
        auto *tensor_obj = yyjson_mut_obj(doc);
        if (!tensor_obj) {
            yyjson_mut_doc_free(doc);
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        yyjson_mut_obj_add_uint(doc, tensor_obj, "id", i);
        auto *tensor_shape = yyjson_mut_arr(doc);
        auto *tensor_strides = yyjson_mut_arr(doc);
        if (!tensor_shape || !tensor_strides) {
            yyjson_mut_doc_free(doc);
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        for (uint32_t d = 0; d < v.desc.ndim; ++d) {
            yyjson_mut_arr_add_uint(doc, tensor_shape, v.desc.shape[d]);
            yyjson_mut_arr_add_uint(doc, tensor_strides, v.desc.strides[d]);
        }
        yyjson_mut_obj_add_val(doc, tensor_obj, "shape", tensor_shape);
        yyjson_mut_obj_add_val(doc, tensor_obj, "strides", tensor_strides);
        yyjson_mut_obj_add_str(doc, tensor_obj, "dtype", dtype_name(v.desc.dtype));
        yyjson_mut_obj_add_str(doc, tensor_obj, "storage", storage_class(v));
        yyjson_mut_arr_add_val(tensors_arr, tensor_obj);

        if (v.is_input && !v.is_constant) {
            yyjson_mut_arr_add_uint(doc, inputs_arr, i);
        } else if (!v.is_input && v.uses.empty()) {
            yyjson_mut_arr_add_uint(doc, outputs_arr, i);
        }
    }

    // Nodes
    for (size_t i = 0; i < impl_->nodes.size(); ++i) {
        const auto &node = impl_->nodes[i];
        auto *node_obj = yyjson_mut_obj(doc);
        if (!node_obj) {
            yyjson_mut_doc_free(doc);
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        yyjson_mut_obj_add_uint(doc, node_obj, "id", i);
        yyjson_mut_obj_add_str(doc, node_obj, "op", node.op_name.c_str());
        yyjson_mut_obj_add_uint(doc, node_obj, "op_id", node.signature.op_id);
        yyjson_mut_obj_add_uint(doc, node_obj, "kernel_id", node.kernel_id);
        yyjson_mut_obj_add_str(doc, node_obj, "kernel_name", marmot_kernel_id_to_string(node.kernel_id));
        yyjson_mut_obj_add_real(doc, node_obj, "estimated_us", node.estimated_us);

        auto *inputs = yyjson_mut_arr(doc);
        auto *outputs = yyjson_mut_arr(doc);
        auto *weights = yyjson_mut_arr(doc);
        if (!inputs || !outputs) {
            yyjson_mut_doc_free(doc);
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        for (auto id : node.inputs) {
            yyjson_mut_arr_add_uint(doc, inputs, id);
            if (id < impl_->values.size() && impl_->values[id].is_constant)
                yyjson_mut_arr_add_uint(doc, weights, id);
        }
        for (auto id : node.outputs)
            yyjson_mut_arr_add_uint(doc, outputs, id);
        yyjson_mut_obj_add_val(doc, node_obj, "inputs", inputs);
        yyjson_mut_obj_add_val(doc, node_obj, "outputs", outputs);
        yyjson_mut_obj_add_val(doc, node_obj, "weights", weights ? weights : yyjson_mut_arr(doc));

        // Signature summary
        auto *sig = yyjson_mut_obj(doc);
        if (!sig) {
            yyjson_mut_doc_free(doc);
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        yyjson_mut_obj_add_str(doc, sig, "profile", marmot_profile_id_to_string(node.signature.profile_id));
        yyjson_mut_obj_add_str(doc, sig, "matmul_layout", matmul_layout_name(node.signature.matmul_layout));
        yyjson_mut_obj_add_str(doc, sig, "input_dtype", dtype_name(node.signature.input_dtype));
        yyjson_mut_obj_add_str(doc, sig, "weight_dtype", dtype_name(node.signature.weight_dtype));
        yyjson_mut_obj_add_str(doc, sig, "output_dtype", dtype_name(node.signature.output_dtype));
        yyjson_mut_obj_add_str(doc, sig, "accum_dtype", dtype_name(node.signature.accum_dtype));
        yyjson_mut_obj_add_str(doc, sig, "qscheme", marmot_qscheme_id_to_string(node.signature.qscheme_id));
        yyjson_mut_obj_add_uint(doc, sig, "epilogue_flags", node.signature.epilogue_flags);
        yyjson_mut_obj_add_uint(doc, sig, "variant_flags", node.signature.variant_flags);
        yyjson_mut_obj_add_str(doc, sig, "weight_layout", weight_layout_name(node.signature.weight_layout));
        yyjson_mut_obj_add_str(doc, sig, "activation", activation_name(node.signature.activation));

        auto *quant = yyjson_mut_obj(doc);
        if (!quant) {
            yyjson_mut_doc_free(doc);
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        yyjson_mut_obj_add_uint(doc, quant, "block_size", node.signature.quant_block.block_size);
        yyjson_mut_obj_add_uint(doc, quant, "group_size", node.signature.quant_block.group_size);
        yyjson_mut_obj_add_str(doc, quant, "scale_dtype", dtype_name(node.signature.quant_block.scale_dtype));
        yyjson_mut_obj_add_str(doc, quant, "zero_point_dtype", dtype_name(node.signature.quant_block.zero_point_dtype));
        yyjson_mut_obj_add_val(doc, sig, "quant_block", quant);

        if (node.signature.op_id == MARMOT_OP_MATMUL || node.signature.op_id == MARMOT_OP_LINEAR ||
            node.signature.op_id == MARMOT_OP_MATMUL_BIAS || node.signature.op_id == MARMOT_OP_MATMUL_BIAS_RELU ||
            node.signature.op_id == MARMOT_OP_MATMUL_BIAS_GELU || node.signature.op_id == MARMOT_OP_MATMUL_BIAS_SILU) {
            auto *dims = yyjson_mut_obj(doc);
            if (!dims) {
                yyjson_mut_doc_free(doc);
                return MARMOT_ERROR_OUT_OF_MEMORY;
            }
            yyjson_mut_obj_add_uint(doc, dims, "n", node.signature.dims.matmul.N);
            yyjson_mut_obj_add_uint(doc, dims, "m", node.signature.dims.matmul.M);
            yyjson_mut_obj_add_uint(doc, dims, "k", node.signature.dims.matmul.K);
            yyjson_mut_obj_add_val(doc, sig, "dims", dims);
        }
        yyjson_mut_obj_add_val(doc, node_obj, "signature", sig);

        // Epilogue/attrs placeholders to match schema
        auto *attrs_obj = yyjson_mut_obj(doc);
        auto *epilogue_obj = yyjson_mut_obj(doc);
        auto *ep_flags = yyjson_mut_arr(doc);
        auto *ep_flag_names = yyjson_mut_arr(doc);
        if (!attrs_obj || !epilogue_obj || !ep_flags) {
            yyjson_mut_doc_free(doc);
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        if (node.signature.epilogue_flags & MARMOT_EPILOGUE_BIAS)
            yyjson_mut_arr_add_str(doc, ep_flag_names, "bias");
        if (node.signature.epilogue_flags & MARMOT_EPILOGUE_ACTIVATION)
            yyjson_mut_arr_add_str(doc, ep_flag_names, "activation");
        if (node.signature.epilogue_flags & MARMOT_EPILOGUE_RESIDUAL)
            yyjson_mut_arr_add_str(doc, ep_flag_names, "residual");
        if (node.signature.epilogue_flags & MARMOT_EPILOGUE_ROPE)
            yyjson_mut_arr_add_str(doc, ep_flag_names, "rope");
        yyjson_mut_obj_add_val(doc, epilogue_obj, "flags", ep_flags);
        yyjson_mut_obj_add_val(doc, epilogue_obj, "flag_names", ep_flag_names);
        yyjson_mut_obj_add_str(doc, epilogue_obj, "activation", activation_name(node.signature.activation));
        yyjson_mut_obj_add_val(doc, node_obj, "attrs", attrs_obj);
        yyjson_mut_obj_add_val(doc, node_obj, "epilogue", epilogue_obj);

        yyjson_mut_arr_add_val(nodes_arr, node_obj);
    }

    size_t len = 0;
    char *json = yyjson_mut_write(doc, 0, &len);
    if (!json) {
        yyjson_mut_doc_free(doc);
        return MARMOT_ERROR_INVALID_OPERATION;
    }

    marmot_error_t status = MARMOT_SUCCESS;
    if (path) {
        std::ofstream file(path, std::ios::out | std::ios::binary);
        if (!file.is_open()) {
            status = MARMOT_ERROR_INVALID_ARGUMENT;
        } else {
            file.write(json, static_cast<std::streamsize>(len));
            if (!file.good())
                status = MARMOT_ERROR_INVALID_OPERATION;
        }
    } else {
        std::cout.write(json, static_cast<std::streamsize>(len));
        std::cout.put('\n');
    }

    free(json);
    yyjson_mut_doc_free(doc);
    return status;
}

float Graph::estimated_total_us() const {
    float total = 0.0f;
    for (const auto &node : impl_->nodes) {
        if (!node.skip) {
            total += node.estimated_us;
        }
    }
    return total;
}

size_t Graph::node_count() const {
    size_t count = 0;
    for (const auto &node : impl_->nodes) {
        if (!node.skip) {
            ++count;
        }
    }
    return count;
}

size_t Graph::fused_node_count() const {
    size_t count = 0;
    for (const auto &node : impl_->nodes) {
        const bool has_explicit_fused =
            (node.signature.op_id == MARMOT_OP_ADD_RELU || node.signature.op_id == MARMOT_OP_ADD_GELU ||
             node.signature.op_id == MARMOT_OP_ADD_SILU || node.signature.op_id == MARMOT_OP_MATMUL_BIAS ||
             node.signature.op_id == MARMOT_OP_MATMUL_BIAS_RELU || node.signature.op_id == MARMOT_OP_MATMUL_BIAS_GELU ||
             node.signature.op_id == MARMOT_OP_MATMUL_BIAS_SILU);
        if (!node.skip && (node.signature.variant_flags != 0 || has_explicit_fused)) {
            ++count;
        }
    }
    return count;
}

void Graph::reset_session() {
    session_.reset();
}

void Graph::set_last_node_view_byte_offset(size_t byte_offset) {
    if (!impl_->nodes.empty()) {
        impl_->nodes.back().view_byte_offset = byte_offset;
    }
}

void Graph::set_last_node_slice_starts(const size_t *starts, size_t ndim) {
    if (starts == nullptr || ndim == 0 || ndim > MARMOT_MAX_DIMS) {
        return;
    }
    if (!impl_->nodes.empty()) {
        auto &node = impl_->nodes.back();
        for (size_t i = 0; i < ndim; ++i) {
            node.slice_starts[i] = starts[i];
        }
    }
}

void Graph::set_last_node_paged_attention_layer(uint32_t layer_idx) {
    if (!impl_->nodes.empty()) {
        impl_->nodes.back().paged_attention_layer_idx = layer_idx;
    }
}

[[nodiscard]] marmot_error_t Graph::set_last_node_fast_hint_checked(
    marmot_fast_stage_hint_t stage_hint, marmot_fast_node_role_t role, uint32_t block_id
) {
    if (impl_ == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "Graph implementation is not initialized");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
    if (impl_->finalized) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "Graph is already finalized");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
    if (impl_->nodes.empty()) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "Graph has no nodes");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
    if (stage_hint >= MARMOT_FAST_STAGE_HINT_COUNT || role >= MARMOT_FAST_NODE_ROLE_COUNT ||
        (stage_hint == MARMOT_FAST_STAGE_HINT_NONE) != (role == MARMOT_FAST_NODE_ROLE_NONE)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Invalid fast-path node hint");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    GraphNode &node = impl_->nodes.back();
    node.fast_stage_hint = stage_hint;
    node.fast_node_role = role;
    node.fast_block_id = block_id;
    return MARMOT_SUCCESS;
}

[[nodiscard]] marmot_error_t Graph::set_last_node_moe_params_checked(
    marmot_ffn_type_t ffn_type, float weights_scale, marmot_router_weight_policy_t router_weight_policy
) {
    if (impl_ == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "Graph implementation is not initialized");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
    if (impl_->finalized) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "Graph is already finalized");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
    if (impl_->nodes.empty()) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "Graph has no nodes");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
    if (ffn_type >= MARMOT_FFN_COUNT || !std::isfinite(weights_scale) ||
        router_weight_policy >= MARMOT_ROUTER_WEIGHT_POLICY_COUNT) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Invalid MoE node parameters");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    GraphNode &node = impl_->nodes.back();
    if (node.signature.op_id != MARMOT_OP_MOE_EXPERTS) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "Last node is not MOE_EXPERTS");
        return MARMOT_ERROR_INVALID_OPERATION;
    }

    node.moe_ffn_type = ffn_type;
    node.moe_weights_scale = weights_scale;
    node.moe_router_weight_policy = router_weight_policy;
    return MARMOT_SUCCESS;
}

void Graph::set_last_node_fast_hint(
    marmot_fast_stage_hint_t stage_hint, marmot_fast_node_role_t role, uint32_t block_id
) {
    if (impl_ == nullptr || impl_->finalized || impl_->nodes.empty() || stage_hint >= MARMOT_FAST_STAGE_HINT_COUNT ||
        role >= MARMOT_FAST_NODE_ROLE_COUNT ||
        ((stage_hint == MARMOT_FAST_STAGE_HINT_NONE) != (role == MARMOT_FAST_NODE_ROLE_NONE))) {
        return;
    }

    GraphNode &node = impl_->nodes.back();
    node.fast_stage_hint = stage_hint;
    node.fast_node_role = role;
    node.fast_block_id = block_id;
}

[[nodiscard]] marmot_error_t Graph::set_last_node_moe_params_checked(marmot_ffn_type_t ffn_type, float weights_scale) {
    return set_last_node_moe_params_checked(ffn_type, weights_scale, MARMOT_ROUTER_WEIGHT_POLICY_SOFTMAX_SELECTED);
}

void Graph::set_last_node_moe_params(
    marmot_ffn_type_t ffn_type, float weights_scale, marmot_router_weight_policy_t router_weight_policy
) {
    if (impl_ == nullptr || impl_->finalized || impl_->nodes.empty() || ffn_type >= MARMOT_FFN_COUNT ||
        !std::isfinite(weights_scale) || router_weight_policy >= MARMOT_ROUTER_WEIGHT_POLICY_COUNT) {
        return;
    }

    GraphNode &node = impl_->nodes.back();
    if (node.signature.op_id != MARMOT_OP_MOE_EXPERTS) {
        return;
    }

    node.moe_ffn_type = ffn_type;
    node.moe_weights_scale = weights_scale;
    node.moe_router_weight_policy = router_weight_policy;
}

void Graph::set_last_node_moe_params(marmot_ffn_type_t ffn_type, float weights_scale) {
    set_last_node_moe_params(ffn_type, weights_scale, MARMOT_ROUTER_WEIGHT_POLICY_SOFTMAX_SELECTED);
}

} // namespace marmot::graph
