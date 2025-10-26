#include "graph_builder.hpp"

#include "marmot/error.h"
#include "marmot/graph/architecture.h"
#include "marmot/graph/graph.h"
#include "marmot/graph/graph.hpp"
#include "marmot/macros.h"
#include "marmot/ops/matmul.h"

#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstring>
#include <expected>
#include <format>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/dispatch/fusion_flags.h"
#include "graph/graph_handle.hpp"
#include "graph/graph_validator.hpp"
#include "model.hpp"
#include "utils/dtype_ref.h"

extern "C" marmot_routing_policy_t marmot_routing_policy_from_env(void);

[[nodiscard]] static marmot_error_t marmot_graph_from_model_packed_impl(
    const marmot_gguf_model_t *model, marmot_backend_type_t backend, bool auto_backend,
    const marmot_packed_graph_options_t *packed_opts, marmot_graph_t **out_graph
);

[[nodiscard]] static marmot_error_t marmot_graph_from_model_packed_single_backend_impl(
    const marmot_gguf_model_t *model, marmot_backend_type_t backend, const marmot_packed_graph_options_t *packed_opts,
    marmot_graph_t **out_graph
);

namespace {
void destroy_model(void *ptr);
}

namespace marmot::gguf {

marmot_error_t GraphBuilder::build_from_file(const char *path, marmot_graph_t **out_graph, Error &err) const {
    err.clear();
    if (path == nullptr || out_graph == nullptr) {
        err.set(MARMOT_ERROR_INVALID_ARGUMENT, "invalid graph build arguments");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (auto_backend_) {
        marmot_gguf_model_t *model = nullptr;
        marmot_error_t status = marmot_gguf_model_load(path, MARMOT_BACKEND_CPU, &model);
        if (status != MARMOT_SUCCESS) {
            err.set(status, "marmot_gguf_model_load failed");
            return status;
        }

        marmot_graph_t *graph = nullptr;
        status = marmot_graph_from_model_packed_impl(model, backend_, true, &packed_opts_, &graph);
        if (status != MARMOT_SUCCESS) {
            marmot_gguf_model_destroy(model);
            err.set(status, "marmot_graph_from_model_packed failed");
            return status;
        }

        graph->external_state = model;
        graph->external_cleanup = destroy_model;
        *out_graph = graph;
        return MARMOT_SUCCESS;
    }

    marmot_error_t status = marmot_graph_from_gguf_packed(path, backend_, &packed_opts_, out_graph);
    if (status != MARMOT_SUCCESS) {
        err.set(status, "marmot_graph_from_gguf_packed failed");
    }
    return status;
}

} // namespace marmot::gguf

namespace {

struct GraphDeleter {
    void operator()(marmot_graph_t *graph) const noexcept {
        marmot_graph_destroy(graph);
    }
};

struct ModelDeleter {
    void operator()(marmot_gguf_model_t *model) const noexcept {
        marmot_gguf_model_destroy(model);
    }
};

using GraphOwner = std::unique_ptr<marmot_graph_t, GraphDeleter>;
using ModelOwner = std::unique_ptr<marmot_gguf_model_t, ModelDeleter>;

void destroy_model(void *ptr) {
    marmot_gguf_model_destroy(static_cast<marmot_gguf_model_t *>(ptr));
}

[[nodiscard]] std::expected<GraphOwner, marmot_error_t> make_graph() {
    marmot_graph_t *graph = marmot_graph_create();
    if (graph == nullptr) {
        return std::unexpected(MARMOT_ERROR_OUT_OF_MEMORY);
    }
    return GraphOwner(graph);
}

[[nodiscard]] std::expected<ModelOwner, marmot_error_t> load_model(const char *path, marmot_backend_type_t backend) {
    marmot_gguf_model_t *model = nullptr;
    marmot_error_t status = marmot_gguf_model_load(path, backend, &model);
    if (status != MARMOT_SUCCESS) {
        return std::unexpected(status);
    }
    return ModelOwner(model);
}

[[nodiscard]] std::expected<marmot_graph_tensor_desc_t, marmot_error_t>
hidden_desc(size_t seq_len, size_t n_embd, marmot_dtype_t dtype) {
    if (seq_len == 0 || n_embd == 0) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }
    if (dtype != MARMOT_DTYPE_FLOAT16 && dtype != MARMOT_DTYPE_FLOAT32) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }

    marmot_graph_tensor_desc_t desc{};
    desc.dtype = dtype;
    desc.ndim = 2;
    desc.shape[0] = seq_len;
    desc.shape[1] = n_embd;
    if (!marmot::graph::ensure_strides(desc)) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }
    return desc;
}

[[nodiscard]] std::expected<marmot_graph_tensor_desc_t, marmot_error_t>
desc_from_tensor(const marmot_tensor_t *tensor) {
    if (tensor == nullptr) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }
    if (tensor->shape.ndim > MARMOT_MAX_DIMS) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }

    marmot_graph_tensor_desc_t desc{};
    desc.dtype = tensor->dtype;
    desc.ndim = tensor->shape.ndim;
    for (uint32_t i = 0; i < tensor->shape.ndim; ++i) {
        desc.shape[i] = tensor->shape.shape[i];
    }
    if (!marmot::graph::ensure_strides(desc)) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }
    return desc;
}

[[nodiscard]] bool graph_desc_is_contiguous(const marmot_graph_tensor_desc_t &desc) {
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

[[nodiscard]] bool graph_desc_is_row_strided(const marmot_graph_tensor_desc_t &desc) {
    if (desc.ndim < 2) {
        return graph_desc_is_contiguous(desc);
    }
    if (desc.strides[desc.ndim - 1] != 1) {
        return false;
    }
    size_t min_stride = 1;
    for (size_t i = desc.ndim; i > 0; --i) {
        const size_t dim_idx = i - 1;
        if (desc.strides[dim_idx] < min_stride) {
            return false;
        }
        min_stride *= desc.shape[dim_idx];
    }
    return true;
}

[[nodiscard]] bool graph_desc_is_row_strided_2d(const marmot_graph_tensor_desc_t &desc) {
    if (desc.ndim != 2) {
        return false;
    }
    if (desc.strides[1] != 1) {
        return false;
    }
    return desc.strides[0] >= desc.shape[1];
}

[[nodiscard]] marmot_stride_mode_t
graph_rope_stride_mode(const marmot_graph_tensor_desc_t &input, const marmot_graph_tensor_desc_t &output) {
    if (graph_desc_is_contiguous(input) && graph_desc_is_contiguous(output)) {
        return MARMOT_STRIDE_MODE_CONTIGUOUS;
    }
    if (graph_desc_is_row_strided(input) && graph_desc_is_row_strided(output)) {
        return MARMOT_STRIDE_MODE_ROW_STRIDED;
    }
    return MARMOT_STRIDE_MODE_STRIDED;
}

[[nodiscard]] marmot_stride_mode_t graph_glu_stride_mode(
    const marmot_graph_tensor_desc_t &gate, const marmot_graph_tensor_desc_t &up, const marmot_graph_tensor_desc_t &out
) {
    if (graph_desc_is_contiguous(gate) && graph_desc_is_contiguous(up) && graph_desc_is_contiguous(out)) {
        return MARMOT_STRIDE_MODE_CONTIGUOUS;
    }
    if (graph_desc_is_row_strided_2d(gate) && graph_desc_is_row_strided_2d(up) && graph_desc_is_row_strided_2d(out)) {
        return MARMOT_STRIDE_MODE_ROW_STRIDED;
    }
    return MARMOT_STRIDE_MODE_STRIDED;
}

[[nodiscard]] std::expected<marmot_graph_tensor_desc_t, marmot_error_t>
matmul_output_desc(const marmot_graph_tensor_desc_t &activation_desc, const marmot_tensor_t *weight) {
    if (weight == nullptr) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }
    if (activation_desc.ndim != 2 || weight->shape.ndim != 2) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }
    if (activation_desc.shape[1] != weight->shape.shape[1]) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }

    marmot_graph_tensor_desc_t out{};
    out.dtype = activation_desc.dtype;
    out.ndim = 2;
    out.shape[0] = activation_desc.shape[0];
    out.shape[1] = weight->shape.shape[0];
    if (!marmot::graph::ensure_strides(out)) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }
    return out;
}

[[nodiscard]] std::expected<marmot_value_id_t, marmot_error_t>
add_constant_value(marmot_graph_t *graph, const marmot_gguf_tensor_t *info);

[[nodiscard]] std::expected<marmot_value_id_t, marmot_error_t>
add_constant_value(marmot_graph_t *graph, const marmot_gguf_tensor_t *info) {
    if (graph == nullptr || info == nullptr || info->tensor == nullptr || info->name == nullptr) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }

    auto desc = desc_from_tensor(info->tensor);
    if (!desc) {
        return std::unexpected(desc.error());
    }

    marmot_value_id_t value_id = MARMOT_VALUE_ID_INVALID;
    marmot_error_t status = marmot_graph_add_input(graph, &*desc, &value_id);
    if (status != MARMOT_SUCCESS) {
        return std::unexpected(status);
    }

    status = graph->inner.set_constant(value_id, info->tensor);
    if (status != MARMOT_SUCCESS) {
        return std::unexpected(status);
    }

    status = graph->inner.set_name(value_id, info->name);
    if (status != MARMOT_SUCCESS) {
        return std::unexpected(status);
    }

    return value_id;
}

[[nodiscard]] std::expected<marmot_value_id_t, marmot_error_t> add_matmul(
    marmot_graph_t *graph, marmot_value_id_t activation_id, marmot_value_id_t weight_id, marmot_qscheme_id_t qscheme,
    const marmot_graph_tensor_desc_t &out_desc, std::optional<marmot_value_id_t> bias_id = std::nullopt
) {
    if (graph == nullptr) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }

    auto activation_desc = graph->inner.get_value_desc(activation_id);
    auto weight_desc = graph->inner.get_value_desc(weight_id);
    if (!activation_desc || !weight_desc) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }
    if (bias_id) {
        auto bias_desc = graph->inner.get_value_desc(*bias_id);
        if (!bias_desc) {
            return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
        }
    }

    marmot_op_signature_t sig = {
        .op_id = MARMOT_OP_LINEAR,
        .profile_id = MARMOT_PROFILE_INVALID,
        .matmul_layout = MARMOT_MATMUL_LAYOUT_NT,
        .input_dtype = activation_desc->dtype,
        .weight_dtype = weight_desc->dtype,
        .output_dtype = out_desc.dtype,
        .accum_dtype = MARMOT_DTYPE_FLOAT32,
        .qscheme_id = qscheme,
        .weight_layout = qscheme == MARMOT_QSCHEME_NONE ? MARMOT_WEIGHT_LAYOUT_INVALID : MARMOT_WEIGHT_LAYOUT_SEPARATE,
        .epilogue_flags = bias_id ? MARMOT_EPILOGUE_BIAS : MARMOT_EPILOGUE_NONE,
        .activation = MARMOT_DEVICE_UNARY_IDENTITY,
        .variant_flags = MARMOT_FUSION_NONE,
    };

    marmot_value_id_t inputs[3] = {activation_id, weight_id, bias_id.value_or(MARMOT_VALUE_ID_INVALID)};
    const size_t input_count = bias_id ? 3 : 2;
    marmot_value_id_t out_value_id = MARMOT_VALUE_ID_INVALID;
    marmot_error_t status =
        marmot_graph_add_op(graph, "linear", &sig, inputs, input_count, &out_desc, 1, &out_value_id);
    if (status != MARMOT_SUCCESS) {
        return std::unexpected(status);
    }
    return out_value_id;
}

struct QkvResult {
    marmot_value_id_t q_id;
    marmot_value_id_t k_id;
    marmot_value_id_t v_id;
};

[[nodiscard]] std::expected<QkvResult, marmot_error_t> add_qkv(
    marmot_graph_t *graph, marmot_value_id_t input_id, marmot_value_id_t wq_id, marmot_value_id_t wk_id,
    marmot_value_id_t wv_id, marmot_qscheme_id_t qscheme, const marmot_graph_tensor_desc_t &q_desc,
    const marmot_graph_tensor_desc_t &k_desc, const marmot_graph_tensor_desc_t &v_desc,
    std::optional<marmot_value_id_t> bq_id, std::optional<marmot_value_id_t> bk_id,
    std::optional<marmot_value_id_t> bv_id, std::optional<marmot_value_id_t> positions_id, bool fuse_rope
) {
    if (graph == nullptr) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }

    auto input_desc = graph->inner.get_value_desc(input_id);
    auto wq_desc = graph->inner.get_value_desc(wq_id);
    auto wk_desc = graph->inner.get_value_desc(wk_id);
    auto wv_desc = graph->inner.get_value_desc(wv_id);
    if (!input_desc || !wq_desc || !wk_desc || !wv_desc) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }

    const bool wants_bias = bq_id && bk_id && bv_id;
    const bool any_bias = bq_id || bk_id || bv_id;
    if (any_bias && !wants_bias) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }
    if (bq_id && !graph->inner.get_value_desc(*bq_id)) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }
    if (bk_id && !graph->inner.get_value_desc(*bk_id)) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }
    if (bv_id && !graph->inner.get_value_desc(*bv_id)) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }

    if (fuse_rope && (!positions_id || !graph->inner.get_value_desc(*positions_id))) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }

    marmot_op_signature_t sig = {
        .op_id = fuse_rope ? MARMOT_OP_QKV_SHARED_INPUT : MARMOT_OP_QKV_PROJECTION,
        .profile_id = MARMOT_PROFILE_INVALID,
        .matmul_layout = MARMOT_MATMUL_LAYOUT_NT,
        .input_dtype = input_desc->dtype,
        .weight_dtype = wq_desc->dtype,
        .output_dtype = q_desc.dtype,
        .accum_dtype = MARMOT_DTYPE_FLOAT32,
        .qscheme_id = qscheme,
        .weight_layout = qscheme == MARMOT_QSCHEME_NONE ? MARMOT_WEIGHT_LAYOUT_INVALID : MARMOT_WEIGHT_LAYOUT_SEPARATE,
        .epilogue_flags =
            (wants_bias ? MARMOT_EPILOGUE_BIAS : MARMOT_EPILOGUE_NONE) | (fuse_rope ? MARMOT_EPILOGUE_ROPE : 0u),
        .activation = MARMOT_DEVICE_UNARY_IDENTITY,
        .variant_flags = MARMOT_FUSION_NONE,
    };

    marmot_value_id_t inputs[8] = {
        input_id,
        wq_id,
        wk_id,
        wv_id,
        MARMOT_VALUE_ID_INVALID,
        MARMOT_VALUE_ID_INVALID,
        MARMOT_VALUE_ID_INVALID,
        MARMOT_VALUE_ID_INVALID
    };
    size_t input_count = 4;
    if (wants_bias) {
        inputs[input_count++] = *bq_id;
        inputs[input_count++] = *bk_id;
        inputs[input_count++] = *bv_id;
    }
    if (fuse_rope) {
        inputs[input_count++] = *positions_id;
    }

    marmot_graph_tensor_desc_t out_descs[3] = {q_desc, k_desc, v_desc};
    marmot_value_id_t out_ids[3] = {MARMOT_VALUE_ID_INVALID, MARMOT_VALUE_ID_INVALID, MARMOT_VALUE_ID_INVALID};
    const char *op_name = fuse_rope ? "qkv_rope" : "qkv_projection";
    marmot_error_t status = marmot_graph_add_op(graph, op_name, &sig, inputs, input_count, out_descs, 3, out_ids);
    if (status != MARMOT_SUCCESS) {
        return std::unexpected(status);
    }
    return QkvResult{.q_id = out_ids[0], .k_id = out_ids[1], .v_id = out_ids[2]};
}

[[nodiscard]] std::expected<marmot_value_id_t, marmot_error_t>
add_rms_norm(marmot_graph_t *graph, marmot_value_id_t input_id, marmot_value_id_t weight_id, bool use_gemma_norm) {
    if (graph == nullptr) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }

    auto input_desc = graph->inner.get_value_desc(input_id);
    auto weight_desc = graph->inner.get_value_desc(weight_id);

    if (!input_desc || !weight_desc) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }

    marmot_op_signature_t sig = {
        .op_id = use_gemma_norm ? MARMOT_OP_RMS_NORM_GEMMA : MARMOT_OP_RMS_NORM,
        .profile_id = MARMOT_PROFILE_INVALID,
        .input_dtype = input_desc->dtype,
        .weight_dtype = weight_desc->dtype,
        .output_dtype = input_desc->dtype,
        .accum_dtype = MARMOT_DTYPE_FLOAT32,
        .qscheme_id = MARMOT_QSCHEME_NONE,
        .weight_layout = MARMOT_WEIGHT_LAYOUT_INVALID,
        .epilogue_flags = MARMOT_EPILOGUE_NONE,
        .activation = MARMOT_DEVICE_UNARY_IDENTITY,
        .variant_flags = MARMOT_FUSION_NONE,
    };

    marmot_value_id_t inputs[2] = {input_id, weight_id};
    marmot_value_id_t out_value_id = MARMOT_VALUE_ID_INVALID;
    const char *op_name = use_gemma_norm ? "rms_norm_gemma" : "rms_norm";
    marmot_error_t status = marmot_graph_add_op(graph, op_name, &sig, inputs, 2, &*input_desc, 1, &out_value_id);
    if (status != MARMOT_SUCCESS) {
        return std::unexpected(status);
    }
    return out_value_id;
}

[[nodiscard]] std::expected<marmot_value_id_t, marmot_error_t>
add_add(marmot_graph_t *graph, marmot_value_id_t input_a_id, marmot_value_id_t input_b_id) {
    if (graph == nullptr) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }

    auto input_a_desc = graph->inner.get_value_desc(input_a_id);
    auto input_b_desc = graph->inner.get_value_desc(input_b_id);

    if (!input_a_desc || !input_b_desc) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }

    marmot_op_signature_t sig = {
        .op_id = MARMOT_OP_ADD,
        .profile_id = MARMOT_PROFILE_INVALID,
        .input_dtype = input_a_desc->dtype,
        .weight_dtype = input_a_desc->dtype,
        .output_dtype = input_a_desc->dtype,
        .accum_dtype = MARMOT_DTYPE_FLOAT32,
        .qscheme_id = MARMOT_QSCHEME_NONE,
        .weight_layout = MARMOT_WEIGHT_LAYOUT_INVALID,
        .epilogue_flags = MARMOT_EPILOGUE_NONE,
        .activation = MARMOT_DEVICE_UNARY_IDENTITY,
        .variant_flags = MARMOT_FUSION_NONE,
    };

    marmot_value_id_t inputs[2] = {input_a_id, input_b_id};
    marmot_value_id_t out_value_id = MARMOT_VALUE_ID_INVALID;
    marmot_error_t status = marmot_graph_add_op(graph, "add", &sig, inputs, 2, &*input_a_desc, 1, &out_value_id);
    if (status != MARMOT_SUCCESS) {
        return std::unexpected(status);
    }
    return out_value_id;
}

[[nodiscard]] std::expected<marmot_value_id_t, marmot_error_t>
add_glu(marmot_graph_t *graph, marmot_value_id_t gate_id, marmot_value_id_t up_id, marmot_ffn_type_t ffn_type) {
    if (graph == nullptr) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }

    auto gate_desc = graph->inner.get_value_desc(gate_id);
    auto up_desc = graph->inner.get_value_desc(up_id);

    if (!gate_desc || !up_desc) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }

    const bool is_geglu = (ffn_type == MARMOT_FFN_GEGLU);
    if (!is_geglu && ffn_type != MARMOT_FFN_SWIGLU) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }

    const marmot_op_id_t op_id = is_geglu ? MARMOT_OP_GEGLU : MARMOT_OP_SWIGLU;
    const char *op_name = is_geglu ? "geglu" : "swiglu";

    marmot_graph_tensor_desc_t out_desc{};
    out_desc.dtype = gate_desc->dtype;
    out_desc.ndim = gate_desc->ndim;
    memcpy(out_desc.shape, gate_desc->shape, sizeof(out_desc.shape));
    if (!marmot::graph::ensure_strides(out_desc)) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }

    marmot_op_signature_t sig = {
        .op_id = op_id,
        .profile_id = MARMOT_PROFILE_INVALID,
        .input_dtype = gate_desc->dtype,
        .weight_dtype = gate_desc->dtype,
        .output_dtype = gate_desc->dtype,
        .accum_dtype = MARMOT_DTYPE_FLOAT32,
        .qscheme_id = MARMOT_QSCHEME_NONE,
        .weight_layout = MARMOT_WEIGHT_LAYOUT_INVALID,
        .stride_mode = graph_glu_stride_mode(*gate_desc, *up_desc, out_desc),
        .epilogue_flags = MARMOT_EPILOGUE_NONE,
        .activation = MARMOT_DEVICE_UNARY_IDENTITY,
        .variant_flags = MARMOT_FUSION_NONE,
    };

    marmot_value_id_t inputs[2] = {gate_id, up_id};
    marmot_value_id_t out_value_id = MARMOT_VALUE_ID_INVALID;
    marmot_error_t status = marmot_graph_add_op(graph, op_name, &sig, inputs, 2, &out_desc, 1, &out_value_id);
    if (status != MARMOT_SUCCESS) {
        return std::unexpected(status);
    }
    return out_value_id;
}

[[nodiscard]] std::expected<marmot_value_id_t, marmot_error_t>
add_gelu(marmot_graph_t *graph, marmot_value_id_t input_id) {
    if (graph == nullptr) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }

    auto input_desc = graph->inner.get_value_desc(input_id);

    if (!input_desc) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }

    marmot_op_signature_t sig = {
        .op_id = MARMOT_OP_GELU,
        .profile_id = MARMOT_PROFILE_INVALID,
        .input_dtype = input_desc->dtype,
        .weight_dtype = input_desc->dtype,
        .output_dtype = input_desc->dtype,
        .accum_dtype = MARMOT_DTYPE_FLOAT32,
        .qscheme_id = MARMOT_QSCHEME_NONE,
        .weight_layout = MARMOT_WEIGHT_LAYOUT_INVALID,
        .epilogue_flags = MARMOT_EPILOGUE_NONE,
        .activation = MARMOT_DEVICE_UNARY_IDENTITY,
        .variant_flags = MARMOT_FUSION_NONE,
    };

    marmot_value_id_t out_value_id = MARMOT_VALUE_ID_INVALID;
    marmot_error_t status = marmot_graph_add_op(graph, "gelu", &sig, &input_id, 1, &*input_desc, 1, &out_value_id);
    if (status != MARMOT_SUCCESS) {
        return std::unexpected(status);
    }
    return out_value_id;
}

[[nodiscard]] std::expected<marmot_graph_tensor_desc_t, marmot_error_t> positions_desc(size_t seq_len) {
    if (seq_len == 0) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }

    marmot_graph_tensor_desc_t desc{};
    desc.dtype = MARMOT_DTYPE_FLOAT32;
    desc.ndim = 1;
    desc.shape[0] = seq_len;
    if (!marmot::graph::ensure_strides(desc)) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }
    return desc;
}

[[nodiscard]] std::expected<marmot_value_id_t, marmot_error_t>
add_reshape(marmot_graph_t *graph, marmot_value_id_t input_id, const marmot_graph_tensor_desc_t &out_desc) {
    if (graph == nullptr) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }

    auto input_desc = graph->inner.get_value_desc(input_id);
    if (!input_desc) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }
    if (out_desc.dtype != input_desc->dtype) {
        return std::unexpected(MARMOT_ERROR_UNSUPPORTED_DTYPE);
    }

    marmot_op_signature_t sig = {
        .op_id = MARMOT_OP_RESHAPE,
        .profile_id = MARMOT_PROFILE_INVALID,
        .input_dtype = input_desc->dtype,
        .weight_dtype = input_desc->dtype,
        .output_dtype = out_desc.dtype,
        .accum_dtype = MARMOT_DTYPE_COUNT,
        .qscheme_id = MARMOT_QSCHEME_NONE,
        .weight_layout = MARMOT_WEIGHT_LAYOUT_INVALID,
        .epilogue_flags = MARMOT_EPILOGUE_NONE,
        .activation = MARMOT_DEVICE_UNARY_COUNT,
        .variant_flags = MARMOT_FUSION_NONE,
    };

    marmot_value_id_t out_value_id = MARMOT_VALUE_ID_INVALID;
    marmot_error_t status = marmot_graph_add_op(graph, "reshape", &sig, &input_id, 1, &out_desc, 1, &out_value_id);
    if (status != MARMOT_SUCCESS) {
        return std::unexpected(status);
    }
    return out_value_id;
}

[[nodiscard]] std::expected<marmot_value_id_t, marmot_error_t>
add_rope(marmot_graph_t *graph, marmot_value_id_t input_id, marmot_value_id_t positions_id) {
    if (graph == nullptr) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }

    auto input_desc = graph->inner.get_value_desc(input_id);
    if (!input_desc) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }

    marmot_op_signature_t sig = {
        .op_id = MARMOT_OP_ROPE,
        .profile_id = MARMOT_PROFILE_INVALID,
        .input_dtype = input_desc->dtype,
        .weight_dtype = input_desc->dtype,
        .output_dtype = input_desc->dtype,
        .accum_dtype = MARMOT_DTYPE_COUNT,
        .qscheme_id = MARMOT_QSCHEME_NONE,
        .weight_layout = MARMOT_WEIGHT_LAYOUT_INVALID,
        .stride_mode = graph_rope_stride_mode(*input_desc, *input_desc),
        .epilogue_flags = MARMOT_EPILOGUE_NONE,
        .activation = MARMOT_DEVICE_UNARY_COUNT,
        .variant_flags = MARMOT_FUSION_NONE,
    };

    marmot_value_id_t inputs[2] = {input_id, positions_id};
    marmot_value_id_t out_value_id = MARMOT_VALUE_ID_INVALID;
    marmot_error_t status = marmot_graph_add_op(graph, "rope", &sig, inputs, 2, &*input_desc, 1, &out_value_id);
    if (status != MARMOT_SUCCESS) {
        return std::unexpected(status);
    }
    return out_value_id;
}

[[nodiscard]] std::expected<marmot_value_id_t, marmot_error_t> add_paged_attention(
    marmot_graph_t *graph, marmot_value_id_t token_meta_id, marmot_value_id_t q_id, marmot_value_id_t k_id,
    marmot_value_id_t v_id, marmot_value_id_t kv_k_id, marmot_value_id_t kv_v_id, marmot_value_id_t block_table_id,
    marmot_value_id_t kv_k_scale_id, marmot_value_id_t kv_v_scale_id, const marmot_graph_tensor_desc_t &out_desc,
    uint32_t layer_idx
) {
    if (graph == nullptr) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }

    auto q_desc = graph->inner.get_value_desc(q_id);
    auto kv_desc = graph->inner.get_value_desc(kv_k_id);
    if (!q_desc || !kv_desc) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }

    marmot_op_signature_t sig = {
        .op_id = MARMOT_OP_PAGED_ATTENTION,
        .profile_id = MARMOT_PROFILE_INVALID,
        .input_dtype = q_desc->dtype,
        .weight_dtype = kv_desc->dtype,
        .output_dtype = out_desc.dtype,
        .accum_dtype = MARMOT_DTYPE_FLOAT32,
        .qscheme_id = MARMOT_QSCHEME_NONE,
        .weight_layout = MARMOT_WEIGHT_LAYOUT_INVALID,
        .epilogue_flags = MARMOT_EPILOGUE_NONE,
        .activation = MARMOT_DEVICE_UNARY_COUNT,
        .variant_flags = MARMOT_FUSION_NONE,
    };

    marmot_value_id_t inputs[9] = {token_meta_id, q_id,           k_id,          v_id,         kv_k_id,
                                   kv_v_id,       block_table_id, kv_k_scale_id, kv_v_scale_id};
    size_t input_count = 7;
    if (kv_k_scale_id != MARMOT_VALUE_ID_INVALID && kv_v_scale_id != MARMOT_VALUE_ID_INVALID) {
        input_count = 9;
    }
    marmot_value_id_t out_value_id = MARMOT_VALUE_ID_INVALID;
    marmot_error_t status =
        marmot_graph_add_op(graph, "paged_attention", &sig, inputs, input_count, &out_desc, 1, &out_value_id);
    if (status != MARMOT_SUCCESS) {
        return std::unexpected(status);
    }
    graph->inner.set_last_node_paged_attention_layer(layer_idx);
    return out_value_id;
}

[[nodiscard]] std::expected<marmot_value_id_t, marmot_error_t> add_gather_rows(
    marmot_graph_t *graph, marmot_value_id_t input_id, marmot_value_id_t indices_id,
    const marmot_graph_tensor_desc_t &out_desc
) {
    if (graph == nullptr) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }

    auto input_desc = graph->inner.get_value_desc(input_id);
    auto indices_desc = graph->inner.get_value_desc(indices_id);
    if (!input_desc || !indices_desc) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }

    marmot_op_signature_t sig = {
        .op_id = MARMOT_OP_GATHER_ROWS,
        .profile_id = MARMOT_PROFILE_INVALID,
        .input_dtype = input_desc->dtype,
        .weight_dtype = indices_desc->dtype,
        .output_dtype = out_desc.dtype,
        .accum_dtype = MARMOT_DTYPE_COUNT,
        .qscheme_id = MARMOT_QSCHEME_NONE,
        .weight_layout = MARMOT_WEIGHT_LAYOUT_INVALID,
        .epilogue_flags = MARMOT_EPILOGUE_NONE,
        .activation = MARMOT_DEVICE_UNARY_COUNT,
        .variant_flags = MARMOT_FUSION_NONE,
    };

    marmot_value_id_t inputs[2] = {input_id, indices_id};
    marmot_value_id_t out_value_id = MARMOT_VALUE_ID_INVALID;
    marmot_error_t status = marmot_graph_add_op(graph, "gather_rows", &sig, inputs, 2, &out_desc, 1, &out_value_id);
    if (status != MARMOT_SUCCESS) {
        return std::unexpected(status);
    }
    return out_value_id;
}

// Split a 2D tensor along axis 1 into multiple outputs
// For fused QKV: input shape [seq, q_dim + k_dim + v_dim] -> outputs [seq, q_dim], [seq, k_dim], [seq, v_dim]
struct SplitResult {
    marmot_value_id_t q_id;
    marmot_value_id_t k_id;
    marmot_value_id_t v_id;
};

[[nodiscard]] std::expected<SplitResult, marmot_error_t> add_split_qkv(
    marmot_graph_t *graph, marmot_value_id_t input_id, size_t q_dim, size_t k_dim, size_t v_dim, bool use_views
) {
    if (graph == nullptr) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }

    auto input_desc = graph->inner.get_value_desc(input_id);
    if (!input_desc || input_desc->ndim != 2) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }

    const size_t seq_len = input_desc->shape[0];
    const size_t total_dim = input_desc->shape[1];
    if (total_dim != q_dim + k_dim + v_dim) {
        return std::unexpected(MARMOT_ERROR_DIMENSION_MISMATCH);
    }

    const size_t row_stride = input_desc->strides[0];
    const size_t col_stride = input_desc->strides[1];
    if (use_views && (row_stride == 0 || col_stride == 0)) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }

    const size_t elem_size = marmot_dtype_size(input_desc->dtype);
    SplitResult result{};

    marmot_op_signature_t sig = {
        .op_id = use_views ? MARMOT_OP_VIEW : MARMOT_OP_SLICE,
        .profile_id = MARMOT_PROFILE_INVALID,
        .input_dtype = input_desc->dtype,
        .weight_dtype = input_desc->dtype,
        .output_dtype = input_desc->dtype,
        .accum_dtype = MARMOT_DTYPE_COUNT,
        .qscheme_id = MARMOT_QSCHEME_NONE,
        .weight_layout = MARMOT_WEIGHT_LAYOUT_INVALID,
        .epilogue_flags = MARMOT_EPILOGUE_NONE,
        .activation = MARMOT_DEVICE_UNARY_COUNT,
        .variant_flags = MARMOT_FUSION_NONE,
    };

    auto add_output = [&](const char *name, size_t dim, size_t col_offset, marmot_value_id_t &out_id) {
        marmot_graph_tensor_desc_t desc{};
        desc.dtype = input_desc->dtype;
        desc.ndim = 2;
        desc.shape[0] = seq_len;
        desc.shape[1] = dim;
        if (use_views) {
            desc.strides[0] = row_stride;
            desc.strides[1] = col_stride;
        } else if (!marmot::graph::ensure_strides(desc)) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }

        marmot_error_t status = marmot_graph_add_op(graph, name, &sig, &input_id, 1, &desc, 1, &out_id);
        if (status != MARMOT_SUCCESS) {
            return status;
        }

        if (use_views) {
            graph->inner.set_last_node_view_byte_offset(col_offset * elem_size);
        } else {
            const size_t starts[2] = {0, col_offset};
            graph->inner.set_last_node_slice_starts(starts, 2);
        }
        return MARMOT_SUCCESS;
    };

    marmot_error_t status = add_output(use_views ? "view_q" : "slice_q", q_dim, 0, result.q_id);
    if (status != MARMOT_SUCCESS) {
        return std::unexpected(status);
    }
    status = add_output(use_views ? "view_k" : "slice_k", k_dim, q_dim, result.k_id);
    if (status != MARMOT_SUCCESS) {
        return std::unexpected(status);
    }
    status = add_output(use_views ? "view_v" : "slice_v", v_dim, q_dim + k_dim, result.v_id);
    if (status != MARMOT_SUCCESS) {
        return std::unexpected(status);
    }

    return result;
}

// Split a 2D tensor along axis 1 into two equal parts (for fused gate+up)
struct SplitGateUpResult {
    marmot_value_id_t gate_id;
    marmot_value_id_t up_id;
};

[[nodiscard]] std::expected<SplitGateUpResult, marmot_error_t>
add_split_gate_up(marmot_graph_t *graph, marmot_value_id_t input_id, bool use_views) {
    if (graph == nullptr) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }

    auto input_desc = graph->inner.get_value_desc(input_id);
    if (!input_desc || input_desc->ndim != 2) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }

    const size_t seq_len = input_desc->shape[0];
    const size_t total_dim = input_desc->shape[1];
    if (total_dim % 2 != 0) {
        return std::unexpected(MARMOT_ERROR_DIMENSION_MISMATCH);
    }
    const size_t half_dim = total_dim / 2;
    const size_t row_stride = input_desc->strides[0];
    const size_t col_stride = input_desc->strides[1];
    if (use_views && (row_stride == 0 || col_stride == 0)) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }
    const size_t elem_size = marmot_dtype_size(input_desc->dtype);

    SplitGateUpResult result{};

    marmot_op_signature_t sig = {
        .op_id = use_views ? MARMOT_OP_VIEW : MARMOT_OP_SLICE,
        .profile_id = MARMOT_PROFILE_INVALID,
        .input_dtype = input_desc->dtype,
        .weight_dtype = input_desc->dtype,
        .output_dtype = input_desc->dtype,
        .accum_dtype = MARMOT_DTYPE_COUNT,
        .qscheme_id = MARMOT_QSCHEME_NONE,
        .weight_layout = MARMOT_WEIGHT_LAYOUT_INVALID,
        .epilogue_flags = MARMOT_EPILOGUE_NONE,
        .activation = MARMOT_DEVICE_UNARY_COUNT,
        .variant_flags = MARMOT_FUSION_NONE,
    };

    marmot_graph_tensor_desc_t out_desc{};
    out_desc.dtype = input_desc->dtype;
    out_desc.ndim = 2;
    out_desc.shape[0] = seq_len;
    out_desc.shape[1] = half_dim;
    if (use_views) {
        out_desc.strides[0] = row_stride;
        out_desc.strides[1] = col_stride;
    } else if (!marmot::graph::ensure_strides(out_desc)) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }

    auto add_output = [&](const char *name, size_t col_offset, marmot_value_id_t &out_id) {
        marmot_error_t status = marmot_graph_add_op(graph, name, &sig, &input_id, 1, &out_desc, 1, &out_id);
        if (status != MARMOT_SUCCESS) {
            return status;
        }
        if (use_views) {
            graph->inner.set_last_node_view_byte_offset(col_offset * elem_size);
        } else {
            const size_t starts[2] = {0, col_offset};
            graph->inner.set_last_node_slice_starts(starts, 2);
        }
        return MARMOT_SUCCESS;
    };

    marmot_error_t status = add_output(use_views ? "view_gate" : "slice_gate", 0, result.gate_id);
    if (status != MARMOT_SUCCESS) {
        return std::unexpected(status);
    }
    status = add_output(use_views ? "view_up" : "slice_up", half_dim, result.up_id);
    if (status != MARMOT_SUCCESS) {
        return std::unexpected(status);
    }

    return result;
}

} // namespace

// Build graph from already-loaded model (packed/paged attention only).
// The caller must ensure the model outlives the graph.
[[nodiscard]] static marmot_error_t marmot_graph_from_model_packed_single_backend_impl(
    const marmot_gguf_model_t *model, marmot_backend_type_t backend, const marmot_packed_graph_options_t *packed_opts,
    marmot_graph_t **out_graph
) {
    if (out_graph == nullptr || model == nullptr || packed_opts == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    *out_graph = nullptr;
    if (packed_opts->struct_version != MARMOT_PACKED_GRAPH_OPTIONS_VERSION ||
        packed_opts->struct_size < sizeof(marmot_packed_graph_options_t)) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (packed_opts->token_count == 0 || packed_opts->max_seqs == 0 || packed_opts->max_seq_len == 0 ||
        packed_opts->block_size == 0 || packed_opts->num_kv_blocks == 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (packed_opts->token_count > packed_opts->max_seq_len) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (packed_opts->sample_count > packed_opts->token_count) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (packed_opts->block_size <= 1 || !std::has_single_bit(packed_opts->block_size)) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    try {
        marmot_gguf_model_meta_t meta;
        if (!marmot_gguf_model_metadata(model, &meta)) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }

        const marmot_gguf_t *file = marmot_gguf_model_file(model);
        if (meta.architecture == MARMOT_ARCH_UNKNOWN || marmot_gguf_model_tensor_count(model) == 0) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        const marmot_architecture_traits_t *arch_traits = marmot_get_architecture_traits(meta.architecture);
        if (arch_traits == nullptr) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        const bool use_gemma_norm = arch_traits->uses_gemma_norm;
        bool use_qkv_views = true;
        bool use_gate_up_views = true;
        const size_t seq_len = packed_opts->token_count;
        const size_t sample_count = packed_opts->sample_count;
        const bool emit_logits_actual = sample_count > 0;

        auto graph_result = make_graph();
        if (!graph_result) {
            return graph_result.error();
        }
        GraphOwner graph = std::move(*graph_result);

        marmot_rope_params_t rope_params = marmot_rope_params_default();
        rope_params.theta = meta.rope_freq_base;
        rope_params.scaling_type = meta.rope_scaling_type;
        rope_params.rope_type = meta.rope_type;
        rope_params.freq_scale = meta.rope_freq_scale;
        rope_params.ext_factor = meta.rope_ext_factor;
        rope_params.attn_factor = meta.rope_attn_factor;
        rope_params.beta_fast = meta.rope_beta_fast;
        rope_params.beta_slow = meta.rope_beta_slow;
        rope_params.orig_ctx_len = meta.rope_orig_ctx_len;

        if (meta.n_head == 0 || meta.n_embd == 0 || meta.head_dim == 0) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        if (meta.n_head_kv == 0 || meta.n_head_kv > meta.n_head || (meta.n_head % meta.n_head_kv) != 0) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        const size_t head_dim = meta.head_dim;
        // For models with explicit head_dim (Qwen3+), rope_dimension may not equal computed n_embd/n_head
        // but should still match the actual head dimension used
        if (meta.rope_dimension != head_dim) {
            return MARMOT_ERROR_NOT_IMPLEMENTED;
        }
        rope_params.head_dim = static_cast<uint32_t>(head_dim);

        size_t max_seq_hint = meta.context_length;
        if (packed_opts->max_seq_len > 0) {
            max_seq_hint =
                max_seq_hint > 0 ? std::min(max_seq_hint, packed_opts->max_seq_len) : packed_opts->max_seq_len;
        }
        marmot_error_t status = graph->inner.set_inference_hints(max_seq_hint, &rope_params, meta.rms_norm_eps);
        if (status != MARMOT_SUCCESS) {
            return status;
        }

        const marmot_dtype_t activation_dtype = marmot_activation_dtype_for_architecture(meta.architecture, backend);
        marmot_dtype_t kv_dtype = packed_opts->kv_dtype;
        if ((packed_opts->flags & MARMOT_PACKED_GRAPH_FLAG_KV_DTYPE_AUTO) != 0) {
            kv_dtype = activation_dtype;
        }
        auto kv_dtype_supported = [&](marmot_dtype_t dtype) -> bool {
            if (dtype == MARMOT_DTYPE_FLOAT16 || dtype == MARMOT_DTYPE_FLOAT32 || dtype == MARMOT_DTYPE_BFLOAT16) {
                return true;
            }
#if MARMOT_ENABLE_FP8
            if (dtype == MARMOT_DTYPE_FLOAT8_E4M3 && backend == MARMOT_BACKEND_CPU) {
                return true;
            }
#endif
            return false;
        };
        if (!kv_dtype_supported(kv_dtype)) {
            return MARMOT_ERROR_UNSUPPORTED_DTYPE;
        }
        auto hidden = hidden_desc(seq_len, meta.n_embd, activation_dtype);
        if (!hidden) {
            return hidden.error();
        }

        marmot_value_id_t hidden_id = MARMOT_VALUE_ID_INVALID;
        status = marmot_graph_add_input(graph.get(), &*hidden, &hidden_id);
        if (status != MARMOT_SUCCESS) {
            return status;
        }

        auto positions = positions_desc(hidden->shape[0]);
        if (!positions) {
            return positions.error();
        }
        marmot_value_id_t positions_id = MARMOT_VALUE_ID_INVALID;
        status = marmot_graph_add_input(graph.get(), &*positions, &positions_id);
        if (status != MARMOT_SUCCESS) {
            return status;
        }
        (void)graph->inner.set_name(positions_id, "positions");

        marmot_value_id_t token_meta_id = MARMOT_VALUE_ID_INVALID;
        marmot_value_id_t block_table_id = MARMOT_VALUE_ID_INVALID;
        marmot_value_id_t kv_k_id = MARMOT_VALUE_ID_INVALID;
        marmot_value_id_t kv_v_id = MARMOT_VALUE_ID_INVALID;
        marmot_value_id_t kv_k_scale_id = MARMOT_VALUE_ID_INVALID;
        marmot_value_id_t kv_v_scale_id = MARMOT_VALUE_ID_INVALID;
        marmot_value_id_t sample_indices_id = MARMOT_VALUE_ID_INVALID;

        marmot_graph_tensor_desc_t token_meta_desc{};
        token_meta_desc.dtype = MARMOT_DTYPE_UINT32;
        token_meta_desc.ndim = 2;
        token_meta_desc.shape[0] = seq_len;
        token_meta_desc.shape[1] = 4;
        if (!marmot::graph::ensure_strides(token_meta_desc)) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        status = marmot_graph_add_input(graph.get(), &token_meta_desc, &token_meta_id);
        if (status != MARMOT_SUCCESS) {
            return status;
        }
        (void)graph->inner.set_name(token_meta_id, "token_meta");

        const size_t max_blocks_per_seq =
            (packed_opts->max_seq_len + packed_opts->block_size - 1) / packed_opts->block_size;
        marmot_graph_tensor_desc_t block_table_desc{};
        block_table_desc.dtype = MARMOT_DTYPE_UINT32;
        block_table_desc.ndim = 2;
        block_table_desc.shape[0] = packed_opts->max_seqs;
        block_table_desc.shape[1] = max_blocks_per_seq;
        if (!marmot::graph::ensure_strides(block_table_desc)) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        status = marmot_graph_add_input(graph.get(), &block_table_desc, &block_table_id);
        if (status != MARMOT_SUCCESS) {
            return status;
        }
        (void)graph->inner.set_name(block_table_id, "block_table");

        marmot_graph_tensor_desc_t kv_desc{};
        kv_desc.dtype = kv_dtype;
        kv_desc.ndim = 5;
        kv_desc.shape[0] = packed_opts->num_kv_blocks;
        kv_desc.shape[1] = meta.n_layer;
        kv_desc.shape[2] = meta.n_head_kv;
        kv_desc.shape[3] = packed_opts->block_size;
        kv_desc.shape[4] = head_dim;
        if (!marmot::graph::ensure_strides(kv_desc)) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        status = marmot_graph_add_input(graph.get(), &kv_desc, &kv_k_id);
        if (status != MARMOT_SUCCESS) {
            return status;
        }
        (void)graph->inner.set_name(kv_k_id, "kv_k");
        status = marmot_graph_add_input(graph.get(), &kv_desc, &kv_v_id);
        if (status != MARMOT_SUCCESS) {
            return status;
        }
        (void)graph->inner.set_name(kv_v_id, "kv_v");

        bool use_fp8_kv = false;
#if MARMOT_ENABLE_FP8
        use_fp8_kv = kv_dtype == MARMOT_DTYPE_FLOAT8_E4M3;
#endif
        if (use_fp8_kv) {
            marmot_graph_tensor_desc_t kv_scale_desc{};
            kv_scale_desc.dtype = MARMOT_DTYPE_FLOAT32;
            kv_scale_desc.ndim = 3;
            kv_scale_desc.shape[0] = packed_opts->num_kv_blocks;
            kv_scale_desc.shape[1] = meta.n_layer;
            kv_scale_desc.shape[2] = meta.n_head_kv;
            if (!marmot::graph::ensure_strides(kv_scale_desc)) {
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            status = marmot_graph_add_input(graph.get(), &kv_scale_desc, &kv_k_scale_id);
            if (status != MARMOT_SUCCESS) {
                return status;
            }
            (void)graph->inner.set_name(kv_k_scale_id, "kv_k_scale");
            status = marmot_graph_add_input(graph.get(), &kv_scale_desc, &kv_v_scale_id);
            if (status != MARMOT_SUCCESS) {
                return status;
            }
            (void)graph->inner.set_name(kv_v_scale_id, "kv_v_scale");
        }

        if (emit_logits_actual) {
            marmot_graph_tensor_desc_t sample_desc{};
            sample_desc.dtype = MARMOT_DTYPE_UINT32;
            sample_desc.ndim = 1;
            sample_desc.shape[0] = sample_count;
            if (!marmot::graph::ensure_strides(sample_desc)) {
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            status = marmot_graph_add_input(graph.get(), &sample_desc, &sample_indices_id);
            if (status != MARMOT_SUCCESS) {
                return status;
            }
            (void)graph->inner.set_name(sample_indices_id, "sample_indices");
        }

        const marmot_gguf_tensor_t *output_norm = marmot_gguf_find_tensor(file, "output_norm.weight");
        if (output_norm == nullptr) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        auto output_norm_id = add_constant_value(graph.get(), output_norm);
        if (!output_norm_id) {
            return output_norm_id.error();
        }

        std::optional<marmot_value_id_t> output_weight_id;
        const marmot_gguf_tensor_t *output_weight = nullptr;
        if (emit_logits_actual) {
            output_weight = marmot_gguf_find_tensor(file, "output.weight");
            // Weight tying: fall back to token_embd.weight if output.weight is missing
            if (output_weight == nullptr) {
                output_weight = marmot_gguf_find_tensor(file, "token_embd.weight");
            }
            if (output_weight == nullptr || output_weight->tensor == nullptr) {
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            auto output_weight_id_result = add_constant_value(graph.get(), output_weight);
            if (!output_weight_id_result) {
                return output_weight_id_result.error();
            }
            output_weight_id = *output_weight_id_result;
        }

        marmot_value_id_t current = hidden_id;

        for (size_t layer = 0; layer < meta.n_layer; ++layer) {
            auto attn_norm_name = std::format("blk.{}.attn_norm.weight", layer);
            auto attn_q_name = std::format("blk.{}.attn_q.weight", layer);
            auto attn_k_name = std::format("blk.{}.attn_k.weight", layer);
            auto attn_v_name = std::format("blk.{}.attn_v.weight", layer);
            auto attn_out_name = std::format("blk.{}.attn_output.weight", layer);
            auto ffn_norm_name = std::format("blk.{}.ffn_norm.weight", layer);
            auto ffn_gate_name = std::format("blk.{}.ffn_gate.weight", layer);
            auto ffn_up_name = std::format("blk.{}.ffn_up.weight", layer);
            auto ffn_down_name = std::format("blk.{}.ffn_down.weight", layer);

            const marmot_gguf_tensor_t *attn_norm = marmot_gguf_find_tensor(file, attn_norm_name.c_str());
            const marmot_gguf_tensor_t *attn_q = marmot_gguf_find_tensor(file, attn_q_name.c_str());
            const marmot_gguf_tensor_t *attn_k = marmot_gguf_find_tensor(file, attn_k_name.c_str());
            const marmot_gguf_tensor_t *attn_v = marmot_gguf_find_tensor(file, attn_v_name.c_str());
            const marmot_gguf_tensor_t *attn_out = marmot_gguf_find_tensor(file, attn_out_name.c_str());

            // Check for fused QKV weight (Phi-3 style)
            const marmot_gguf_tensor_t *attn_qkv = nullptr;
            bool use_fused_qkv = false;
            if (attn_q == nullptr && attn_k == nullptr && attn_v == nullptr) {
                auto attn_qkv_name = std::format("blk.{}.attn_qkv.weight", layer);
                attn_qkv = marmot_gguf_find_tensor(file, attn_qkv_name.c_str());
                use_fused_qkv = (attn_qkv != nullptr);
            }
            const marmot_gguf_tensor_t *ffn_norm = marmot_gguf_find_tensor(file, ffn_norm_name.c_str());
            const marmot_gguf_tensor_t *ffn_gate = nullptr;
            if (arch_traits->ffn_type == MARMOT_FFN_SWIGLU || arch_traits->ffn_type == MARMOT_FFN_GEGLU) {
                ffn_gate = marmot_gguf_find_tensor(file, ffn_gate_name.c_str());
            }
            const marmot_gguf_tensor_t *ffn_up = marmot_gguf_find_tensor(file, ffn_up_name.c_str());
            const marmot_gguf_tensor_t *ffn_down = marmot_gguf_find_tensor(file, ffn_down_name.c_str());

            // Check for fused gate+up weight (Phi-3 style)
            // If ffn_gate is missing but ffn_up has 2x the expected size, it's fused
            bool use_fused_gate_up = false;
            if (arch_traits->ffn_type == MARMOT_FFN_SWIGLU && ffn_gate == nullptr && ffn_up != nullptr) {
                // Check if ffn_up shape suggests fused gate+up: [n_embd, 2*ff_length]
                if (ffn_up->tensor != nullptr && ffn_up->tensor->shape.ndim == 2) {
                    size_t up_out_dim = ffn_up->tensor->shape.shape[0];
                    if (up_out_dim == 2 * meta.ff_length) {
                        use_fused_gate_up = true;
                    }
                }
            }

            // Optional attention bias tensors (Qwen2)
            const marmot_gguf_tensor_t *attn_q_bias = nullptr;
            const marmot_gguf_tensor_t *attn_k_bias = nullptr;
            const marmot_gguf_tensor_t *attn_v_bias = nullptr;
            if (arch_traits->has_attention_bias) {
                auto attn_q_bias_name = std::format("blk.{}.attn_q.bias", layer);
                auto attn_k_bias_name = std::format("blk.{}.attn_k.bias", layer);
                auto attn_v_bias_name = std::format("blk.{}.attn_v.bias", layer);
                attn_q_bias = marmot_gguf_find_tensor(file, attn_q_bias_name.c_str());
                attn_k_bias = marmot_gguf_find_tensor(file, attn_k_bias_name.c_str());
                attn_v_bias = marmot_gguf_find_tensor(file, attn_v_bias_name.c_str());
            }

            // Optional Q/K normalization tensors (Qwen3)
            const marmot_gguf_tensor_t *attn_q_norm = nullptr;
            const marmot_gguf_tensor_t *attn_k_norm = nullptr;
            if (arch_traits->has_qk_norm) {
                auto attn_q_norm_name = std::format("blk.{}.attn_q_norm.weight", layer);
                auto attn_k_norm_name = std::format("blk.{}.attn_k_norm.weight", layer);
                attn_q_norm = marmot_gguf_find_tensor(file, attn_q_norm_name.c_str());
                attn_k_norm = marmot_gguf_find_tensor(file, attn_k_norm_name.c_str());
            }

            if (attn_norm == nullptr) {
                auto msg = std::format("Missing tensor: {}", attn_norm_name);
                marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, msg.c_str());
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            // Validate attention weights: either separate Q/K/V or fused QKV
            if (!use_fused_qkv) {
                if (attn_q == nullptr) {
                    auto msg = std::format("Missing tensor: {}", attn_q_name);
                    marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, msg.c_str());
                    return MARMOT_ERROR_INVALID_ARGUMENT;
                }
                if (attn_k == nullptr) {
                    auto msg = std::format("Missing tensor: {}", attn_k_name);
                    marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, msg.c_str());
                    return MARMOT_ERROR_INVALID_ARGUMENT;
                }
                if (attn_v == nullptr) {
                    auto msg = std::format("Missing tensor: {}", attn_v_name);
                    marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, msg.c_str());
                    return MARMOT_ERROR_INVALID_ARGUMENT;
                }
            }
            if (attn_out == nullptr) {
                auto msg = std::format("Missing tensor: {}", attn_out_name);
                marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, msg.c_str());
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            if (ffn_norm == nullptr) {
                auto msg = std::format("Missing tensor: {}", ffn_norm_name);
                marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, msg.c_str());
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            const bool needs_gate =
                arch_traits->ffn_type == MARMOT_FFN_SWIGLU || arch_traits->ffn_type == MARMOT_FFN_GEGLU;
            if (needs_gate && ffn_gate == nullptr && !use_fused_gate_up) {
                auto msg = std::format("Missing tensor: {}", ffn_gate_name);
                marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, msg.c_str());
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            if (ffn_up == nullptr) {
                auto msg = std::format("Missing tensor: {}", ffn_up_name);
                marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, msg.c_str());
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            if (ffn_down == nullptr) {
                auto msg = std::format("Missing tensor: {}", ffn_down_name);
                marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, msg.c_str());
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }

            auto attn_norm_id = add_constant_value(graph.get(), attn_norm);
            if (!attn_norm_id) {
                return attn_norm_id.error();
            }

            // Add attention weight constants (either separate or fused)
            std::optional<marmot_value_id_t> attn_q_id;
            std::optional<marmot_value_id_t> attn_k_id;
            std::optional<marmot_value_id_t> attn_v_id;
            std::optional<marmot_value_id_t> attn_qkv_id;

            if (use_fused_qkv) {
                auto result = add_constant_value(graph.get(), attn_qkv);
                if (!result) {
                    return result.error();
                }
                attn_qkv_id = *result;
            } else {
                auto q_result = add_constant_value(graph.get(), attn_q);
                if (!q_result) {
                    return q_result.error();
                }
                attn_q_id = *q_result;

                auto k_result = add_constant_value(graph.get(), attn_k);
                if (!k_result) {
                    return k_result.error();
                }
                attn_k_id = *k_result;

                auto v_result = add_constant_value(graph.get(), attn_v);
                if (!v_result) {
                    return v_result.error();
                }
                attn_v_id = *v_result;
            }

            // Add attention bias values if present
            std::optional<marmot_value_id_t> attn_q_bias_id;
            std::optional<marmot_value_id_t> attn_k_bias_id;
            std::optional<marmot_value_id_t> attn_v_bias_id;
            if (attn_q_bias != nullptr) {
                auto result = add_constant_value(graph.get(), attn_q_bias);
                if (!result) {
                    return result.error();
                }
                attn_q_bias_id = *result;
            }
            if (attn_k_bias != nullptr) {
                auto result = add_constant_value(graph.get(), attn_k_bias);
                if (!result) {
                    return result.error();
                }
                attn_k_bias_id = *result;
            }
            if (attn_v_bias != nullptr) {
                auto result = add_constant_value(graph.get(), attn_v_bias);
                if (!result) {
                    return result.error();
                }
                attn_v_bias_id = *result;
            }

            // Add Q/K norm values if present (Qwen3)
            std::optional<marmot_value_id_t> attn_q_norm_id;
            std::optional<marmot_value_id_t> attn_k_norm_id;
            if (attn_q_norm != nullptr) {
                auto result = add_constant_value(graph.get(), attn_q_norm);
                if (!result) {
                    return result.error();
                }
                attn_q_norm_id = *result;
            }
            if (attn_k_norm != nullptr) {
                auto result = add_constant_value(graph.get(), attn_k_norm);
                if (!result) {
                    return result.error();
                }
                attn_k_norm_id = *result;
            }

            auto attn_out_id = add_constant_value(graph.get(), attn_out);
            if (!attn_out_id) {
                return attn_out_id.error();
            }
            auto ffn_norm_id = add_constant_value(graph.get(), ffn_norm);
            if (!ffn_norm_id) {
                return ffn_norm_id.error();
            }
            std::optional<marmot_value_id_t> ffn_gate_id;
            if ((arch_traits->ffn_type == MARMOT_FFN_SWIGLU || arch_traits->ffn_type == MARMOT_FFN_GEGLU) &&
                !use_fused_gate_up) {
                auto ffn_gate_id_result = add_constant_value(graph.get(), ffn_gate);
                if (!ffn_gate_id_result) {
                    return ffn_gate_id_result.error();
                }
                ffn_gate_id = *ffn_gate_id_result;
            }
            auto ffn_up_id = add_constant_value(graph.get(), ffn_up);
            if (!ffn_up_id) {
                return ffn_up_id.error();
            }
            auto ffn_down_id = add_constant_value(graph.get(), ffn_down);
            if (!ffn_down_id) {
                return ffn_down_id.error();
            }

            auto norm1_out_id = add_rms_norm(graph.get(), current, *attn_norm_id, use_gemma_norm);
            if (!norm1_out_id) {
                return norm1_out_id.error();
            }
            (void)graph->inner.set_name(*norm1_out_id, std::format("layer.{}.norm1", layer));

            auto norm1_desc = graph->inner.get_value_desc(*norm1_out_id);
            if (!norm1_desc) {
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }

            // Compute Q, K, V projections (either fused or separate)
            std::expected<marmot_value_id_t, marmot_error_t> q_id, k_id, v_id;
            std::expected<marmot_graph_tensor_desc_t, marmot_error_t> q_desc, k_desc, v_desc;
            bool qkv_rope_used = false;

            if (use_fused_qkv) {
                // Fused QKV: single matmul then split
                auto qkv_desc = matmul_output_desc(*norm1_desc, attn_qkv->tensor);
                if (!qkv_desc) {
                    return qkv_desc.error();
                }
                auto qkv_id = add_matmul(graph.get(), *norm1_out_id, *attn_qkv_id, attn_qkv->qscheme_id, *qkv_desc);
                if (!qkv_id) {
                    return qkv_id.error();
                }

                // Split into Q, K, V
                // Phi-3 with MHA: Q, K, V all have same dimension (n_embd)
                // For GQA: Q = n_head * head_dim, K = V = n_head_kv * head_dim
                const size_t q_dim = meta.n_head * head_dim;
                const size_t kv_dim = meta.n_head_kv * head_dim;
                auto split_result = add_split_qkv(graph.get(), *qkv_id, q_dim, kv_dim, kv_dim, use_qkv_views);
                if (!split_result) {
                    return split_result.error();
                }

                q_id = split_result->q_id;
                k_id = split_result->k_id;
                v_id = split_result->v_id;

                if (use_qkv_views) {
                    auto q_view_desc = graph->inner.get_value_desc(*q_id);
                    auto k_view_desc = graph->inner.get_value_desc(*k_id);
                    auto v_view_desc = graph->inner.get_value_desc(*v_id);
                    if (!q_view_desc || !k_view_desc || !v_view_desc) {
                        return MARMOT_ERROR_INVALID_ARGUMENT;
                    }
                    q_desc = *q_view_desc;
                    k_desc = *k_view_desc;
                    v_desc = *v_view_desc;
                } else {
                    // Create descriptors for the split outputs
                    marmot_graph_tensor_desc_t q_d{}, k_d{}, v_d{};
                    q_d.dtype = k_d.dtype = v_d.dtype = norm1_desc->dtype;
                    q_d.ndim = k_d.ndim = v_d.ndim = 2;
                    q_d.shape[0] = k_d.shape[0] = v_d.shape[0] = norm1_desc->shape[0];
                    q_d.shape[1] = q_dim;
                    k_d.shape[1] = kv_dim;
                    v_d.shape[1] = kv_dim;
                    if (!marmot::graph::ensure_strides(q_d) || !marmot::graph::ensure_strides(k_d) ||
                        !marmot::graph::ensure_strides(v_d)) {
                        return MARMOT_ERROR_INVALID_ARGUMENT;
                    }
                    q_desc = q_d;
                    k_desc = k_d;
                    v_desc = v_d;
                }
            } else {
                // Separate Q, K, V projections (optionally fused via QKV kernel)
                q_desc = matmul_output_desc(*norm1_desc, attn_q->tensor);
                if (!q_desc) {
                    return q_desc.error();
                }
                k_desc = matmul_output_desc(*norm1_desc, attn_k->tensor);
                if (!k_desc) {
                    return k_desc.error();
                }
                v_desc = matmul_output_desc(*norm1_desc, attn_v->tensor);
                if (!v_desc) {
                    return v_desc.error();
                }

                const bool any_bias = attn_q_bias_id || attn_k_bias_id || attn_v_bias_id;
                const bool all_bias = attn_q_bias_id && attn_k_bias_id && attn_v_bias_id;
                const bool bias_ok = !any_bias || all_bias;
                const bool same_qscheme =
                    attn_q->qscheme_id == attn_k->qscheme_id && attn_q->qscheme_id == attn_v->qscheme_id;
                const bool same_weight_dtype =
                    attn_q->tensor->dtype == attn_k->tensor->dtype && attn_q->tensor->dtype == attn_v->tensor->dtype;
                const bool same_shape = q_desc->shape[0] == k_desc->shape[0] && q_desc->shape[0] == v_desc->shape[0] &&
                    q_desc->shape[1] == k_desc->shape[1] && q_desc->shape[1] == v_desc->shape[1];
                const bool same_heads = meta.n_head == meta.n_head_kv;
                const bool can_fuse_qkv = bias_ok && same_qscheme && same_weight_dtype && same_shape && same_heads;
                const bool rope_head_dim_ok = (head_dim % 2) == 0;
                const bool can_fuse_rope = can_fuse_qkv && !attn_q_norm_id && !attn_k_norm_id && rope_head_dim_ok;

                if (can_fuse_qkv) {
                    auto qkv_result = add_qkv(
                        graph.get(), *norm1_out_id, *attn_q_id, *attn_k_id, *attn_v_id, attn_q->qscheme_id, *q_desc,
                        *k_desc, *v_desc, all_bias ? attn_q_bias_id : std::nullopt,
                        all_bias ? attn_k_bias_id : std::nullopt, all_bias ? attn_v_bias_id : std::nullopt,
                        can_fuse_rope ? std::optional<marmot_value_id_t>(positions_id) : std::nullopt, can_fuse_rope
                    );
                    if (!qkv_result) {
                        return qkv_result.error();
                    }
                    q_id = qkv_result->q_id;
                    k_id = qkv_result->k_id;
                    v_id = qkv_result->v_id;
                    qkv_rope_used = can_fuse_rope;
                } else {
                    q_id =
                        add_matmul(graph.get(), *norm1_out_id, *attn_q_id, attn_q->qscheme_id, *q_desc, attn_q_bias_id);
                    if (!q_id) {
                        return q_id.error();
                    }

                    k_id =
                        add_matmul(graph.get(), *norm1_out_id, *attn_k_id, attn_k->qscheme_id, *k_desc, attn_k_bias_id);
                    if (!k_id) {
                        return k_id.error();
                    }

                    v_id =
                        add_matmul(graph.get(), *norm1_out_id, *attn_v_id, attn_v->qscheme_id, *v_desc, attn_v_bias_id);
                    if (!v_id) {
                        return v_id.error();
                    }
                }
            }
            (void)graph->inner.set_name(*q_id, std::format("layer.{}.q_proj", layer));
            (void)graph->inner.set_name(*k_id, std::format("layer.{}.k_proj", layer));
            (void)graph->inner.set_name(*v_id, std::format("layer.{}.v_proj", layer));

            if (norm1_desc->shape[0] != q_desc->shape[0]) {
                return MARMOT_ERROR_DIMENSION_MISMATCH;
            }
            const size_t seq_len = q_desc->shape[0];
            const size_t q_embd = meta.n_head * head_dim;
            if (q_desc->shape[1] != q_embd) {
                marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Q projection output shape mismatch");
                return MARMOT_ERROR_DIMENSION_MISMATCH;
            }
            const size_t kv_embd = meta.n_head_kv * head_dim;
            if (k_desc->shape[1] != kv_embd || v_desc->shape[1] != kv_embd) {
                marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "K/V projection output shape mismatch");
                return MARMOT_ERROR_DIMENSION_MISMATCH;
            }

            marmot_value_id_t q_heads_input = *q_id;
            marmot_value_id_t k_heads_input = *k_id;

            if (attn_q_norm_id) {
                marmot_graph_tensor_desc_t q_norm_desc{};
                q_norm_desc.dtype = q_desc->dtype;
                q_norm_desc.ndim = 2;
                q_norm_desc.shape[0] = seq_len * meta.n_head;
                q_norm_desc.shape[1] = head_dim;
                if (!marmot::graph::ensure_strides(q_norm_desc)) {
                    return MARMOT_ERROR_INVALID_ARGUMENT;
                }
                auto q_norm_view = add_reshape(graph.get(), *q_id, q_norm_desc);
                if (!q_norm_view) {
                    return q_norm_view.error();
                }
                auto q_norm_result = add_rms_norm(graph.get(), *q_norm_view, *attn_q_norm_id, use_gemma_norm);
                if (!q_norm_result) {
                    return q_norm_result.error();
                }
                q_heads_input = *q_norm_result;
            }

            if (attn_k_norm_id) {
                marmot_graph_tensor_desc_t k_norm_desc{};
                k_norm_desc.dtype = k_desc->dtype;
                k_norm_desc.ndim = 2;
                k_norm_desc.shape[0] = seq_len * meta.n_head_kv;
                k_norm_desc.shape[1] = head_dim;
                if (!marmot::graph::ensure_strides(k_norm_desc)) {
                    return MARMOT_ERROR_INVALID_ARGUMENT;
                }
                auto k_norm_view = add_reshape(graph.get(), *k_id, k_norm_desc);
                if (!k_norm_view) {
                    return k_norm_view.error();
                }
                auto k_norm_result = add_rms_norm(graph.get(), *k_norm_view, *attn_k_norm_id, use_gemma_norm);
                if (!k_norm_result) {
                    return k_norm_result.error();
                }
                k_heads_input = *k_norm_result;
            }

            marmot_graph_tensor_desc_t q_heads_desc{};
            q_heads_desc.dtype = q_desc->dtype;
            q_heads_desc.ndim = 3;
            q_heads_desc.shape[0] = meta.n_head;
            q_heads_desc.shape[1] = seq_len;
            q_heads_desc.shape[2] = head_dim;
            q_heads_desc.strides[2] = 1;
            q_heads_desc.strides[1] = q_desc->strides[0];
            q_heads_desc.strides[0] = head_dim;

            marmot_graph_tensor_desc_t kv_heads_desc{};
            kv_heads_desc.dtype = k_desc->dtype;
            kv_heads_desc.ndim = 3;
            kv_heads_desc.shape[0] = meta.n_head_kv;
            kv_heads_desc.shape[1] = seq_len;
            kv_heads_desc.shape[2] = head_dim;
            kv_heads_desc.strides[2] = 1;
            kv_heads_desc.strides[1] = k_desc->strides[0];
            kv_heads_desc.strides[0] = head_dim;

            auto q_heads_id = add_reshape(graph.get(), q_heads_input, q_heads_desc);
            if (!q_heads_id) {
                return q_heads_id.error();
            }
            (void)graph->inner.set_name(*q_heads_id, std::format("layer.{}.q_heads", layer));
            auto k_heads_id = add_reshape(graph.get(), k_heads_input, kv_heads_desc);
            if (!k_heads_id) {
                return k_heads_id.error();
            }
            (void)graph->inner.set_name(*k_heads_id, std::format("layer.{}.k_heads", layer));
            auto v_heads_id = add_reshape(graph.get(), *v_id, kv_heads_desc);
            if (!v_heads_id) {
                return v_heads_id.error();
            }
            (void)graph->inner.set_name(*v_heads_id, std::format("layer.{}.v_heads", layer));

            marmot_value_id_t q_rope_input = *q_heads_id;
            marmot_value_id_t k_rope_input = *k_heads_id;
            if (!qkv_rope_used) {
                auto q_rope_id = add_rope(graph.get(), *q_heads_id, positions_id);
                if (!q_rope_id) {
                    return q_rope_id.error();
                }
                (void)graph->inner.set_name(*q_rope_id, std::format("layer.{}.q_rope", layer));
                auto k_rope_id = add_rope(graph.get(), *k_heads_id, positions_id);
                if (!k_rope_id) {
                    return k_rope_id.error();
                }
                (void)graph->inner.set_name(*k_rope_id, std::format("layer.{}.k_rope", layer));
                q_rope_input = *q_rope_id;
                k_rope_input = *k_rope_id;
            }

            marmot_graph_tensor_desc_t attn_desc{};
            marmot_value_id_t attn_id = MARMOT_VALUE_ID_INVALID;

            marmot_graph_tensor_desc_t q_tokens_desc{};
            q_tokens_desc.dtype = q_desc->dtype;
            q_tokens_desc.ndim = 3;
            q_tokens_desc.shape[0] = seq_len;
            q_tokens_desc.shape[1] = meta.n_head;
            q_tokens_desc.shape[2] = head_dim;
            q_tokens_desc.strides[2] = 1;
            q_tokens_desc.strides[1] = head_dim;
            q_tokens_desc.strides[0] = q_desc->strides[0];

            marmot_graph_tensor_desc_t kv_tokens_desc{};
            kv_tokens_desc.dtype = k_desc->dtype;
            kv_tokens_desc.ndim = 3;
            kv_tokens_desc.shape[0] = seq_len;
            kv_tokens_desc.shape[1] = meta.n_head_kv;
            kv_tokens_desc.shape[2] = head_dim;
            kv_tokens_desc.strides[2] = 1;
            kv_tokens_desc.strides[1] = head_dim;
            kv_tokens_desc.strides[0] = k_desc->strides[0];

            auto q_tokens_id = add_reshape(graph.get(), q_rope_input, q_tokens_desc);
            if (!q_tokens_id) {
                return q_tokens_id.error();
            }
            auto k_tokens_id = add_reshape(graph.get(), k_rope_input, kv_tokens_desc);
            if (!k_tokens_id) {
                return k_tokens_id.error();
            }
            auto v_tokens_id = add_reshape(graph.get(), *v_heads_id, kv_tokens_desc);
            if (!v_tokens_id) {
                return v_tokens_id.error();
            }

            auto attn_heads_id = add_paged_attention(
                graph.get(), token_meta_id, *q_tokens_id, *k_tokens_id, *v_tokens_id, kv_k_id, kv_v_id, block_table_id,
                kv_k_scale_id, kv_v_scale_id, q_tokens_desc, static_cast<uint32_t>(layer)
            );
            if (!attn_heads_id) {
                return attn_heads_id.error();
            }

            attn_desc.dtype = q_desc->dtype;
            attn_desc.ndim = 2;
            attn_desc.shape[0] = seq_len;
            attn_desc.shape[1] = q_embd;
            attn_desc.strides[0] = q_desc->strides[0];
            attn_desc.strides[1] = 1;
            auto attn_reshape_id = add_reshape(graph.get(), *attn_heads_id, attn_desc);
            if (!attn_reshape_id) {
                return attn_reshape_id.error();
            }
            (void)graph->inner.set_name(*attn_reshape_id, std::format("layer.{}.attn_in", layer));
            attn_id = *attn_reshape_id;

            auto attn_out_desc = matmul_output_desc(attn_desc, attn_out->tensor);
            if (!attn_out_desc) {
                return attn_out_desc.error();
            }
            auto attn_proj_id = add_matmul(graph.get(), attn_id, *attn_out_id, attn_out->qscheme_id, *attn_out_desc);
            if (!attn_proj_id) {
                return attn_proj_id.error();
            }
            (void)graph->inner.set_name(*attn_proj_id, std::format("layer.{}.attn_proj", layer));

            auto residual1_id = add_add(graph.get(), current, *attn_proj_id);
            if (!residual1_id) {
                return residual1_id.error();
            }

            auto norm2_out_id = add_rms_norm(graph.get(), *residual1_id, *ffn_norm_id, use_gemma_norm);
            if (!norm2_out_id) {
                return norm2_out_id.error();
            }

            auto norm2_desc = graph->inner.get_value_desc(*norm2_out_id);
            if (!norm2_desc) {
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }

            marmot_value_id_t ffn_hidden_id = MARMOT_VALUE_ID_INVALID;
            const bool is_gated_ffn =
                arch_traits->ffn_type == MARMOT_FFN_SWIGLU || arch_traits->ffn_type == MARMOT_FFN_GEGLU;

            if (is_gated_ffn) {
                if (use_fused_gate_up) {
                    // Fused gate+up: single matmul then split
                    auto gate_up_desc = matmul_output_desc(*norm2_desc, ffn_up->tensor);
                    if (!gate_up_desc) {
                        return gate_up_desc.error();
                    }
                    auto gate_up_id =
                        add_matmul(graph.get(), *norm2_out_id, *ffn_up_id, ffn_up->qscheme_id, *gate_up_desc);
                    if (!gate_up_id) {
                        return gate_up_id.error();
                    }

                    // Split into gate and up parts
                    auto split_result = add_split_gate_up(graph.get(), *gate_up_id, use_gate_up_views);
                    if (!split_result) {
                        return split_result.error();
                    }

                    auto gated_id =
                        add_glu(graph.get(), split_result->gate_id, split_result->up_id, arch_traits->ffn_type);
                    if (!gated_id) {
                        return gated_id.error();
                    }
                    ffn_hidden_id = *gated_id;
                } else {
                    // Separate gate and up projections
                    auto gate_desc = matmul_output_desc(*norm2_desc, ffn_gate->tensor);
                    if (!gate_desc) {
                        return gate_desc.error();
                    }
                    auto gate_id =
                        add_matmul(graph.get(), *norm2_out_id, *ffn_gate_id, ffn_gate->qscheme_id, *gate_desc);
                    if (!gate_id) {
                        return gate_id.error();
                    }

                    auto up_desc = matmul_output_desc(*norm2_desc, ffn_up->tensor);
                    if (!up_desc) {
                        return up_desc.error();
                    }
                    auto up_id = add_matmul(graph.get(), *norm2_out_id, *ffn_up_id, ffn_up->qscheme_id, *up_desc);
                    if (!up_id) {
                        return up_id.error();
                    }

                    auto gated_id = add_glu(graph.get(), *gate_id, *up_id, arch_traits->ffn_type);
                    if (!gated_id) {
                        return gated_id.error();
                    }
                    ffn_hidden_id = *gated_id;
                }
            } else {
                // Plain GELU FFN: gelu(up)
                auto up_desc = matmul_output_desc(*norm2_desc, ffn_up->tensor);
                if (!up_desc) {
                    return up_desc.error();
                }
                auto up_id = add_matmul(graph.get(), *norm2_out_id, *ffn_up_id, ffn_up->qscheme_id, *up_desc);
                if (!up_id) {
                    return up_id.error();
                }

                auto gelu_id = add_gelu(graph.get(), *up_id);
                if (!gelu_id) {
                    return gelu_id.error();
                }
                ffn_hidden_id = *gelu_id;
            }

            auto ffn_hidden_desc = graph->inner.get_value_desc(ffn_hidden_id);
            if (!ffn_hidden_desc) {
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }

            auto ffn_desc = matmul_output_desc(*ffn_hidden_desc, ffn_down->tensor);
            if (!ffn_desc) {
                return ffn_desc.error();
            }
            auto ffn_out_id = add_matmul(graph.get(), ffn_hidden_id, *ffn_down_id, ffn_down->qscheme_id, *ffn_desc);
            if (!ffn_out_id) {
                return ffn_out_id.error();
            }

            auto residual2_id = add_add(graph.get(), *residual1_id, *ffn_out_id);
            if (!residual2_id) {
                return residual2_id.error();
            }
            (void)graph->inner.set_name(*residual2_id, std::format("layer.{}.post_mlp", layer));

            current = *residual2_id;
        }

        auto final_norm_id = add_rms_norm(graph.get(), current, *output_norm_id, use_gemma_norm);
        if (!final_norm_id) {
            return final_norm_id.error();
        }

        if (emit_logits_actual) {
            auto final_norm_desc = graph->inner.get_value_desc(*final_norm_id);
            if (!final_norm_desc) {
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }

            marmot_graph_tensor_desc_t gather_desc{};
            gather_desc.dtype = final_norm_desc->dtype;
            gather_desc.ndim = 2;
            gather_desc.shape[0] = sample_count;
            gather_desc.shape[1] = final_norm_desc->shape[1];
            if (!marmot::graph::ensure_strides(gather_desc)) {
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            auto gather_id = add_gather_rows(graph.get(), *final_norm_id, sample_indices_id, gather_desc);
            if (!gather_id) {
                return gather_id.error();
            }
            marmot_value_id_t logits_input_id = *gather_id;

            auto logits_input_desc = graph->inner.get_value_desc(logits_input_id);
            if (!logits_input_desc) {
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            auto logits_desc = matmul_output_desc(*logits_input_desc, output_weight->tensor);
            if (!logits_desc) {
                return logits_desc.error();
            }
            auto logits_id =
                add_matmul(graph.get(), logits_input_id, *output_weight_id, output_weight->qscheme_id, *logits_desc);
            if (!logits_id) {
                return logits_id.error();
            }
        }

        status = marmot_graph_finalize(graph.get(), backend);
        if (status != MARMOT_SUCCESS) {
            return status;
        }

        // Model-based graph does NOT own the model - caller retains ownership.
        // The caller must ensure the model outlives the graph.
        *out_graph = graph.release();
        return MARMOT_SUCCESS;
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_graph_from_model_packed threw");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
}

[[nodiscard]] static marmot_error_t marmot_graph_from_model_packed_impl(
    const marmot_gguf_model_t *model, marmot_backend_type_t backend, bool auto_backend,
    const marmot_packed_graph_options_t *packed_opts, marmot_graph_t **out_graph
) {
    if (!auto_backend) {
        return marmot_graph_from_model_packed_single_backend_impl(model, backend, packed_opts, out_graph);
    }
    if (out_graph == nullptr || model == nullptr || packed_opts == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    *out_graph = nullptr;

    marmot_routing_policy_t policy = marmot_routing_policy_from_env();

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
#if MARMOT_ENABLE_METAL
        candidates.push_back(MARMOT_BACKEND_METAL);
#endif
        candidates.push_back(MARMOT_BACKEND_CPU);
        break;
    default:
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Invalid routing policy");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

#if MARMOT_ENABLE_METAL
    std::string metal_detail;
    bool tried_metal = false;
#endif

    marmot_error_t last_status = MARMOT_ERROR_INVALID_OPERATION;
    for (marmot_backend_type_t candidate : candidates) {
        marmot_graph_t *graph = nullptr;
        marmot_error_t status =
            marmot_graph_from_model_packed_single_backend_impl(model, candidate, packed_opts, &graph);
        if (status == MARMOT_SUCCESS) {
#if MARMOT_ENABLE_METAL
            if (policy == MARMOT_ROUTING_AUTO && candidate == MARMOT_BACKEND_CPU && tried_metal) {
                if (!metal_detail.empty()) {
                    std::fprintf(
                        stderr, "[marmot] warning: auto-backend selected CPU because Metal build failed: %s\n",
                        metal_detail.c_str()
                    );
                } else {
                    std::fprintf(stderr, "[marmot] warning: auto-backend selected CPU because Metal build failed\n");
                }
            }
#endif
            marmot_clear_error();
            *out_graph = graph;
            return MARMOT_SUCCESS;
        }

#if MARMOT_ENABLE_METAL
        if (candidate == MARMOT_BACKEND_METAL) {
            tried_metal = true;
            const char *detail = marmot_get_last_error_detail();
            if (detail != nullptr && detail[0] != '\0') {
                metal_detail = detail;
            }
        }
#endif

        last_status = status;
    }

    return last_status;
}

extern "C" {

marmot_error_t marmot_packed_graph_options_init(marmot_packed_graph_options_t *opts) {
    if (opts == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    std::memset(opts, 0, sizeof(*opts));
    opts->struct_size = sizeof(marmot_packed_graph_options_t);
    opts->struct_version = MARMOT_PACKED_GRAPH_OPTIONS_VERSION;
    opts->flags = MARMOT_PACKED_GRAPH_FLAG_KV_DTYPE_AUTO;
    opts->token_count = 0;
    opts->sample_count = 0;
    opts->max_seqs = 0;
    opts->max_seq_len = 0;
    opts->block_size = 0;
    opts->num_kv_blocks = 0;
    opts->kv_dtype = MARMOT_DTYPE_FLOAT16;
    opts->pnext = nullptr;
    return MARMOT_SUCCESS;
}

marmot_error_t marmot_graph_from_model_packed(
    const marmot_gguf_model_t *model, marmot_backend_type_t backend, const marmot_packed_graph_options_t *opts,
    marmot_graph_t **out_graph
) {
    return marmot_graph_from_model_packed_impl(model, backend, false, opts, out_graph);
}

marmot_error_t marmot_graph_from_gguf_packed(
    const char *path, marmot_backend_type_t backend, const marmot_packed_graph_options_t *opts,
    marmot_graph_t **out_graph
) {
    if (out_graph == nullptr || path == nullptr || opts == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    *out_graph = nullptr;

    auto model = load_model(path, backend);
    if (!model) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_graph_t *graph = nullptr;
    marmot_error_t status = marmot_graph_from_model_packed_impl((*model).get(), backend, false, opts, &graph);
    if (status != MARMOT_SUCCESS) {
        return status;
    }

    graph->external_state = (*model).release();
    graph->external_cleanup = destroy_model;
    *out_graph = graph;
    return MARMOT_SUCCESS;
}

} // extern "C"
