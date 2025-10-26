#include "execution_session.hpp"

#include "marmot/device.h"
#include "marmot/error.h"
#include "marmot/tensor.h"

#include <cstdlib>
#include <cstring>

#include "graph_executor.hpp"
#include "tensor_alloc.hpp"

namespace marmot::graph {

static bool is_graph_output_value(const GraphValue &value) {
    return !value.is_input && value.uses.empty();
}

static marmot_tensor_t *allocate_view_tensor(const marmot_graph_tensor_desc_t &desc, marmot_backend_type_t backend) {
    marmot_tensor_t *tensor = (marmot_tensor_t *)calloc(1, sizeof(*tensor));
    if (tensor == nullptr) {
        return nullptr;
    }

    tensor->shape.ndim = desc.ndim;
    for (uint32_t i = 0; i < desc.ndim; ++i) {
        tensor->shape.shape[i] = desc.shape[i];
        tensor->shape.strides[i] = desc.strides[i];
    }

    tensor->dtype = desc.dtype;
    tensor->data = nullptr;
    tensor->capacity_bytes = 0;
    tensor->owns_data = false;
    tensor->quant_params = nullptr;
    tensor->quant_kind = MARMOT_QUANT_KIND_GENERIC;
    tensor->quant_layout = MARMOT_QUANT_LAYOUT_GENERIC;
    tensor->backend = backend;
    tensor->memory_location = MARMOT_MEMORY_UNKNOWN;
    tensor->needs_sync = false;
    tensor->packed_data = nullptr;
    tensor->packed_src_data = nullptr;
    tensor->packed_bytes = 0;
    tensor->packed_row_bytes = 0;
    tensor->packed_rows = 0;

    return tensor;
}

static bool apply_desc_shape(marmot_tensor_t *tensor, const marmot_graph_tensor_desc_t &desc) {
    if (tensor == nullptr || desc.ndim == 0 || desc.ndim > MARMOT_MAX_DIMS) {
        return false;
    }

    bool has_stride = false;
    bool has_zero = false;
    tensor->shape.ndim = desc.ndim;
    for (uint32_t i = 0; i < desc.ndim; ++i) {
        tensor->shape.shape[i] = desc.shape[i];
        tensor->shape.strides[i] = desc.strides[i];
        if (desc.strides[i] == 0) {
            has_zero = true;
        } else {
            has_stride = true;
        }
    }

    if (!has_stride) {
        tensor->shape.strides[desc.ndim - 1] = 1;
        for (uint32_t i = desc.ndim - 1; i-- > 0;) {
            tensor->shape.strides[i] = tensor->shape.strides[i + 1] * tensor->shape.shape[i + 1];
        }
        return true;
    }

    return !has_zero;
}

static void free_tensor_data(const marmot_context_t *ctx, marmot_tensor_t *tensor, const void *avoid_ptr) {
    if (tensor == nullptr || !tensor->owns_data || tensor->data == nullptr) {
        return;
    }
    if (avoid_ptr != nullptr && tensor->data == avoid_ptr) {
        return;
    }
    if (ctx != nullptr && ctx->ops != nullptr && ctx->ops->on_host_ptr_freed != nullptr) {
        ctx->ops->on_host_ptr_freed(ctx->device_ctx, tensor->data);
    }
    free(tensor->data);
    tensor->data = nullptr;
}

static marmot_error_t apply_reshape_alias(
    const marmot_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output,
    const marmot_graph_tensor_desc_t &desc
) {
    if (input == nullptr || output == nullptr || desc.ndim == 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null pointer in reshape");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (!apply_desc_shape(output, desc)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Invalid reshape strides");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    size_t input_elems = marmot_tensor_num_elements(input);
    size_t output_elems = 1;
    for (uint32_t i = 0; i < desc.ndim; ++i) {
        output_elems *= desc.shape[i];
    }
    if (input_elems != output_elems) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Reshape element count mismatch");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    free_tensor_data(ctx, output, input->data);
    if (output->packed_data != nullptr) {
        free(output->packed_data);
        output->packed_data = nullptr;
    }
    if (output->quant_params != nullptr && output->quant_params != input->quant_params) {
        free(output->quant_params);
    }

    output->dtype = input->dtype;
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
    return MARMOT_SUCCESS;
}

static marmot_error_t apply_view_alias(
    const marmot_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output, size_t byte_offset
) {
    if (input == nullptr || output == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null pointer in view");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (input->dtype != output->dtype) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "View requires matching dtypes");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (input->backend != output->backend) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "View requires matching backends");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const size_t elem_size = marmot_dtype_size(input->dtype);
    if (elem_size != 0 && (byte_offset % elem_size) != 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "View byte offset must align to dtype size");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_tensor_t out_probe = *output;
    out_probe.quant_kind = input->quant_kind;
    out_probe.quant_layout = input->quant_layout;
    size_t out_bytes = marmot_tensor_size_bytes(&out_probe);
    if (input->capacity_bytes < byte_offset) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "View byte offset exceeds input capacity");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (out_bytes > input->capacity_bytes - byte_offset) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "View exceeds input capacity");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (out_bytes != 0 && input->data == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "View requires non-null input data");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    free_tensor_data(ctx, output, nullptr);
    if (output->packed_data != nullptr) {
        free(output->packed_data);
        output->packed_data = nullptr;
    }
    if (output->quant_params != nullptr && output->quant_params != input->quant_params) {
        free(output->quant_params);
    }

    output->dtype = input->dtype;
    output->data = static_cast<uint8_t *>(input->data) + byte_offset;
    output->capacity_bytes = input->capacity_bytes > byte_offset ? input->capacity_bytes - byte_offset : 0;
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
    return MARMOT_SUCCESS;
}

ExecutionSession::ExecutionSession(Graph::Impl &impl) : impl_(impl), table_(impl.values.size()) {}

ExecutionSession::~ExecutionSession() {
    release_rope_positions();
}

marmot_error_t ExecutionSession::initialize(const marmot_context_t *ctx) {
    if (ctx == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    device_ctx_ = ctx->device_ctx;
    max_seq_len_ = impl_.inference.max_seq_len;
    rope_params_.positions = nullptr;
    rope_params_.scaling_type = impl_.inference.rope_scaling_type;
    rope_params_.rope_type = impl_.inference.rope_type;
    rope_params_.theta = impl_.inference.rope_theta;
    rope_params_.freq_scale = impl_.inference.rope_freq_scale;
    rope_params_.ext_factor = impl_.inference.rope_ext_factor;
    rope_params_.attn_factor = impl_.inference.rope_attn_factor;
    rope_params_.beta_fast = impl_.inference.rope_beta_fast;
    rope_params_.beta_slow = impl_.inference.rope_beta_slow;
    rope_params_.orig_ctx_len = impl_.inference.rope_orig_ctx_len;
    rope_params_.head_dim = impl_.inference.rope_head_dim;
    rope_params_.apply_to_q = true;
    rope_params_.apply_to_k = true;
    rms_norm_eps_ = impl_.inference.rms_norm_eps;

    runtime_input_ids_.clear();
    graph_output_ids_.clear();
    for (size_t idx = 0; idx < impl_.values.size(); ++idx) {
        marmot_value_id_t id = (marmot_value_id_t)idx;
        const auto &val = impl_.values[idx];
        if (val.is_input && !val.is_constant) {
            runtime_input_ids_.push_back(id);
        }
        if (is_graph_output_value(val)) {
            graph_output_ids_.push_back(id);
        }
    }

    marmot_error_t bindings_status = allocate_persistent_bindings(ctx);
    if (bindings_status != MARMOT_SUCCESS) {
        return bindings_status;
    }

    build_view_aliases();

    return MARMOT_SUCCESS;
}

bool ExecutionSession::compatible(const marmot_context_t *ctx) const {
    return ctx != nullptr && ctx->device_ctx == device_ctx_ && ctx->backend_type == impl_.backend;
}

marmot_error_t ExecutionSession::allocate_persistent_bindings(const marmot_context_t *ctx) {
    if (ctx == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    table_ = BindingTable(impl_.values.size());

    for (size_t idx = 0; idx < impl_.values.size(); ++idx) {
        marmot_value_id_t id = (marmot_value_id_t)idx;
        const auto &val = impl_.values[idx];
        if (!val.is_constant || val.constant_tensor == nullptr) {
            continue;
        }
        table_.set(id, val.constant_tensor);
    }

    for (size_t idx = 0; idx < impl_.values.size(); ++idx) {
        marmot_value_id_t id = (marmot_value_id_t)idx;
        if (table_.bindings()[idx] != nullptr) {
            continue;
        }

        const auto &val = impl_.values[idx];
        if (val.is_input && !val.is_constant) {
            continue;
        }
        if (is_graph_output_value(val)) {
            continue;
        }

        bool is_view_op = false;
        if (val.defining_node != kNodeIndexInvalid && val.defining_node < impl_.nodes.size()) {
            const auto &def_node = impl_.nodes[val.defining_node];
            is_view_op = def_node.signature.op_id == MARMOT_OP_RESHAPE || def_node.signature.op_id == MARMOT_OP_VIEW;
        }

        marmot_tensor_t *tensor = nullptr;
        if (is_view_op) {
            tensor = allocate_view_tensor(val.desc, impl_.backend);
        } else {
            tensor = allocate_tensor_for_desc(val.desc, impl_.backend);
        }
        if (tensor == nullptr) {
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        // Set ctx so Metal residency gets invalidated when tensor is destroyed
        tensor->ctx = const_cast<marmot_context_t *>(ctx);
        marmot_error_t err = table_.emplace_owned(id, tensor);
        if (err != MARMOT_SUCCESS) {
            return err;
        }
        if (!is_view_op && tensor->data != nullptr && tensor->owns_data && impl_.backend != MARMOT_BACKEND_CPU) {
            marmot_error_t upload_status = marmot_tensor_to_device(ctx, tensor);
            if (upload_status != MARMOT_SUCCESS) {
                return upload_status;
            }
        }
    }

    return MARMOT_SUCCESS;
}

void ExecutionSession::release_rope_positions() {
    if (rope_positions_ != nullptr) {
        marmot_tensor_destroy(rope_positions_);
    }
    rope_positions_ = nullptr;
    rope_positions_capacity_ = 0;
}

void ExecutionSession::build_view_aliases() {
    view_aliases_.clear();
    view_aliases_.reserve(impl_.nodes.size());
    for (const auto &node : impl_.nodes) {
        if (node.signature.op_id != MARMOT_OP_RESHAPE && node.signature.op_id != MARMOT_OP_VIEW) {
            continue;
        }
        if (node.inputs.size() != 1 || node.outputs.size() != 1) {
            continue;
        }
        ViewAlias alias{};
        alias.input_id = node.inputs[0];
        alias.output_id = node.outputs[0];
        alias.op_id = node.signature.op_id;
        alias.byte_offset = node.view_byte_offset;
        if (alias.output_id < impl_.values.size()) {
            alias.desc = impl_.values[alias.output_id].desc;
        }
        view_aliases_.push_back(alias);
    }
}

marmot_error_t ExecutionSession::apply_view_aliases(const marmot_context_t *ctx) {
    if (view_aliases_.empty()) {
        return MARMOT_SUCCESS;
    }
    auto bindings = table_.bindings();
    for (auto &alias : view_aliases_) {
        if (alias.input_id >= bindings.size() || alias.output_id >= bindings.size()) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "View alias binding out of range");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        marmot_tensor_t *input = bindings[alias.input_id];
        marmot_tensor_t *output = bindings[alias.output_id];
        if (input == nullptr || output == nullptr) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null tensor binding for view alias");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }

        bool needs_update = false;
        if (alias.last_input != input || alias.last_output != output) {
            needs_update = true;
        } else if (alias.last_input_data != input->data || alias.last_input_capacity != input->capacity_bytes ||
                   alias.last_input_dtype != input->dtype || alias.last_input_quant_kind != input->quant_kind ||
                   alias.last_input_quant_layout != input->quant_layout ||
                   alias.last_input_quant_params != input->quant_params || alias.last_input_backend != input->backend ||
                   alias.last_input_memory != input->memory_location ||
                   alias.last_input_needs_sync != input->needs_sync || alias.last_input_ndim != input->shape.ndim) {
            needs_update = true;
        } else if (input->shape.ndim > 0) {
            size_t shape_bytes = input->shape.ndim * sizeof(size_t);
            if (memcmp(alias.last_input_shape.data(), input->shape.shape, shape_bytes) != 0) {
                needs_update = true;
            }
        }

        if (!needs_update) {
            continue;
        }

        marmot_error_t status = MARMOT_SUCCESS;
        if (alias.op_id == MARMOT_OP_RESHAPE) {
            status = apply_reshape_alias(ctx, input, output, alias.desc);
        } else {
            status = apply_view_alias(ctx, input, output, alias.byte_offset);
        }
        if (status != MARMOT_SUCCESS) {
            return status;
        }

        alias.last_input = input;
        alias.last_output = output;
        alias.last_input_data = input->data;
        alias.last_input_capacity = input->capacity_bytes;
        alias.last_input_dtype = input->dtype;
        alias.last_input_quant_kind = input->quant_kind;
        alias.last_input_quant_layout = input->quant_layout;
        alias.last_input_quant_params = input->quant_params;
        alias.last_input_backend = input->backend;
        alias.last_input_memory = input->memory_location;
        alias.last_input_needs_sync = input->needs_sync;
        alias.last_input_ndim = input->shape.ndim;
        alias.last_input_shape.fill(0);
        if (input->shape.ndim > 0) {
            memcpy(alias.last_input_shape.data(), input->shape.shape, input->shape.ndim * sizeof(size_t));
        }
    }
    return MARMOT_SUCCESS;
}

marmot_tensor_t *
ExecutionSession::broadcast_rope_positions(const marmot_tensor_t *positions, size_t total_seqs, size_t seq_len) {
    if (positions == nullptr || total_seqs == 0 || seq_len == 0) {
        return nullptr;
    }

    const size_t positions_elems = marmot_tensor_num_elements(positions);
    const size_t expected_positions = total_seqs * seq_len;
    if (positions_elems == expected_positions && positions->dtype == MARMOT_DTYPE_FLOAT32) {
        return const_cast<marmot_tensor_t *>(positions);
    }

    if (positions->dtype != MARMOT_DTYPE_FLOAT32 && positions->dtype != MARMOT_DTYPE_INT32) {
        return nullptr;
    }
    if (positions_elems != expected_positions && positions_elems != seq_len) {
        return nullptr;
    }

    if (expected_positions > rope_positions_capacity_) {
        release_rope_positions();
        size_t shape[1] = {expected_positions};
        rope_positions_ = marmot_tensor_create(nullptr, shape, 1, MARMOT_DTYPE_FLOAT32);
        if (rope_positions_ == nullptr) {
            return nullptr;
        }
        rope_positions_->backend = impl_.backend;
        rope_positions_capacity_ = expected_positions;
    }

    rope_positions_->shape.ndim = 1;
    rope_positions_->shape.shape[0] = expected_positions;
    rope_positions_->shape.strides[0] = 1;
    rope_positions_->dtype = MARMOT_DTYPE_FLOAT32;

    float *dst = (float *)rope_positions_->data;
    if (positions_elems == expected_positions) {
        if (positions->dtype == MARMOT_DTYPE_FLOAT32) {
            const float *src = (const float *)positions->data;
            for (size_t i = 0; i < expected_positions; ++i) {
                dst[i] = src[i];
            }
        } else {
            const int32_t *src = (const int32_t *)positions->data;
            for (size_t i = 0; i < expected_positions; ++i) {
                dst[i] = (float)src[i];
            }
        }
        return rope_positions_;
    }

    if (positions->dtype == MARMOT_DTYPE_FLOAT32) {
        const float *src = (const float *)positions->data;
        for (size_t i = 0; i < expected_positions; ++i) {
            dst[i] = src[i % seq_len];
        }
        return rope_positions_;
    }

    const int32_t *src = (const int32_t *)positions->data;
    for (size_t i = 0; i < expected_positions; ++i) {
        dst[i] = (float)src[i % seq_len];
    }
    return rope_positions_;
}

float ExecutionSession::rope_theta() const {
    return rope_params_.theta;
}

const marmot_rope_params_t *ExecutionSession::rope_params() const {
    return &rope_params_;
}

float ExecutionSession::rms_norm_eps() const {
    return rms_norm_eps_;
}

void ExecutionSession::bind_runtime_inputs(std::span<const marmot_tensor_t *const> inputs) {
    const size_t count = runtime_input_ids_.size();
    for (size_t i = 0; i < count; ++i) {
        marmot_value_id_t id = runtime_input_ids_[i];
        table_.set(id, const_cast<marmot_tensor_t *>(inputs[i]));
    }
}

void ExecutionSession::bind_graph_outputs(std::span<marmot_tensor_t *const> outputs) {
    const size_t count = graph_output_ids_.size();
    for (size_t i = 0; i < count; ++i) {
        marmot_value_id_t id = graph_output_ids_[i];
        table_.set(id, outputs[i]);
    }
}

marmot_error_t ExecutionSession::execute(
    const marmot_context_t *ctx, std::span<const marmot_tensor_t *const> inputs,
    std::span<marmot_tensor_t *const> outputs
) {
    if (ctx == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (inputs.size() != runtime_input_ids_.size() || outputs.size() != graph_output_ids_.size()) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Input/Output count mismatch");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    for (const marmot_tensor_t *input : inputs) {
        if (input == nullptr) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null graph input");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
    }
    for (marmot_tensor_t *output : outputs) {
        if (output == nullptr) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null graph output");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
    }

    bind_runtime_inputs(inputs);
    bind_graph_outputs(outputs);

    marmot_error_t alias_status = apply_view_aliases(ctx);
    if (alias_status != MARMOT_SUCCESS) {
        return alias_status;
    }

    Executor executor(impl_, this);
    return executor.execute_bound(ctx, table_.bindings());
}

} // namespace marmot::graph
