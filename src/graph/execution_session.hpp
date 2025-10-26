#pragma once

#include "marmot/error.h"
#include "marmot/ops_types.h"
#include "marmot/tensor.h"

#include <array>
#include <span>
#include <vector>

#include "binding_table.hpp"
#include "graph_impl.hpp"

namespace marmot::graph {

class ExecutionSession {
  public:
    explicit ExecutionSession(Graph::Impl &impl);
    ~ExecutionSession();

    [[nodiscard]] marmot_error_t initialize(const marmot_context_t *ctx);
    [[nodiscard]] bool compatible(const marmot_context_t *ctx) const;

    [[nodiscard]] marmot_error_t execute(
        const marmot_context_t *ctx, std::span<const marmot_tensor_t *const> inputs,
        std::span<marmot_tensor_t *const> outputs
    );

    [[nodiscard]] marmot_tensor_t *
    broadcast_rope_positions(const marmot_tensor_t *positions, size_t total_seqs, size_t seq_len);
    [[nodiscard]] const marmot_rope_params_t *rope_params() const;
    [[nodiscard]] float rope_theta() const;
    [[nodiscard]] float rms_norm_eps() const;

  private:
    struct ViewAlias {
        marmot_value_id_t input_id{MARMOT_VALUE_ID_INVALID};
        marmot_value_id_t output_id{MARMOT_VALUE_ID_INVALID};
        marmot_op_id_t op_id{MARMOT_OP_COUNT};
        size_t byte_offset{0};
        marmot_graph_tensor_desc_t desc{};
        const marmot_tensor_t *last_input{nullptr};
        marmot_tensor_t *last_output{nullptr};
        const void *last_input_data{nullptr};
        size_t last_input_capacity{0};
        marmot_dtype_t last_input_dtype{MARMOT_DTYPE_COUNT};
        marmot_quant_kind_t last_input_quant_kind{MARMOT_QUANT_KIND_GENERIC};
        marmot_quant_layout_t last_input_quant_layout{MARMOT_QUANT_LAYOUT_GENERIC};
        const marmot_quant_params_t *last_input_quant_params{nullptr};
        marmot_backend_type_t last_input_backend{MARMOT_BACKEND_CPU};
        marmot_memory_location_t last_input_memory{MARMOT_MEMORY_UNKNOWN};
        bool last_input_needs_sync{false};
        uint32_t last_input_ndim{0};
        std::array<size_t, MARMOT_MAX_DIMS> last_input_shape{};
    };

    [[nodiscard]] marmot_error_t allocate_persistent_bindings(const marmot_context_t *ctx);
    void build_view_aliases();
    [[nodiscard]] marmot_error_t apply_view_aliases(const marmot_context_t *ctx);
    void release_rope_positions();
    void bind_runtime_inputs(std::span<const marmot_tensor_t *const> inputs);
    void bind_graph_outputs(std::span<marmot_tensor_t *const> outputs);

    Graph::Impl &impl_;
    BindingTable table_;
    std::vector<marmot_value_id_t> runtime_input_ids_;
    std::vector<marmot_value_id_t> graph_output_ids_;
    std::vector<ViewAlias> view_aliases_;
    const void *device_ctx_{nullptr};
    marmot_tensor_t *rope_positions_{nullptr};
    size_t rope_positions_capacity_{0};
    size_t max_seq_len_{0};
    marmot_rope_params_t rope_params_{};
    float rms_norm_eps_{1e-5f};
};

} // namespace marmot::graph
