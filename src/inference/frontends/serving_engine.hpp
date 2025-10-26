#pragma once

#include "marmot/graph/graph.h"
#include "marmot/inference/engine.h"
#include "marmot/inference/kv_pool.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "inference/common/tensor_ptr.hpp"

namespace marmot::inference {

class Model;
class KVPool;

class ServingEngine {
  public:
    static std::unique_ptr<ServingEngine> create(
        const marmot_context_t *ctx, std::shared_ptr<const Model> model, const marmot_serving_engine_options_t &opts,
        marmot_error_t &status, std::string &error
    );

    ~ServingEngine();

    [[nodiscard]] marmot_error_t submit(
        const marmot_token_id_t *prompt_tokens, size_t prompt_len, const marmot_llm_generate_options_t &gen_opts,
        const marmot_llm_sampling_options_t &sampling_opts, marmot_request_id_t &out_request_id, std::string &error
    );

    [[nodiscard]] marmot_error_t step(size_t max_steps, size_t &out_steps_done, std::string &error);

    [[nodiscard]] marmot_error_t last_batch_view(marmot_serving_engine_batch_view_t &out_batch) const noexcept;

    [[nodiscard]] marmot_llm_request_state_t request_state(marmot_request_id_t request_id) const noexcept;

    [[nodiscard]] marmot_error_t request_cancel(marmot_request_id_t request_id, std::string &error);
    [[nodiscard]] marmot_error_t request_release(marmot_request_id_t request_id, std::string &error);

  private:
    struct Request {
        marmot_request_id_t id{0};
        marmot_llm_request_state_t state{MARMOT_LLM_REQUEST_STATE_INVALID};
        marmot_llm_generate_options_t gen_opts{};
        marmot_llm_sampling_options_t sampling_opts{};
        std::vector<marmot_token_id_t> stop_tokens{};
        std::vector<marmot_token_id_t> prompt_tokens{};
        std::vector<marmot_token_id_t> generated_tokens{};
        std::vector<marmot_token_id_t> recompute_tokens{};
        std::vector<uint64_t> prefix_hashes{};
        bool needs_recompute{false};
        bool cancel_requested{false};
        marmot_seq_slot_t seq_slot{0};
        bool has_seq_slot{false};
        size_t prompt_cursor{0};
        marmot_token_id_t pending_input_token{MARMOT_TOKEN_ID_INVALID};
        bool has_pending_token{false};
        bool awaiting_sample{false};
        bool awaiting_prefix_attach{false};
        bool swapped_out{false};
        uint64_t flags{0};
        int32_t priority{0};
        std::string cache_salt{};
        size_t retention_blocks{0};
        size_t num_samples{1};
        size_t sample_index{0};
        marmot_request_id_t parent_id{0};
        std::vector<marmot_request_id_t> clone_ids{};
        std::mt19937_64 rng{0};
        std::vector<uint8_t> allowed_special_mask{};
    };

    struct PackedBatch {
        std::vector<marmot_token_id_t> token_ids{};
        std::vector<uint32_t> token_meta{};
        std::vector<uint32_t> sample_indices{};
        std::vector<marmot_request_id_t> sample_request_ids{};
    };

    struct PrefixEntry {
        marmot_block_id_t block_id{MARMOT_BLOCK_ID_INVALID};
        uint32_t generation{0};
        uint64_t last_use{0};
    };

    struct GraphDeleter {
        void operator()(marmot_graph_t *graph) const noexcept {
            marmot_graph_destroy(graph);
        }
    };

    using GraphOwner = std::unique_ptr<marmot_graph_t, GraphDeleter>;

    struct PackedGraphKey {
        uint32_t token_count{0};
        uint32_t sample_count{0};
        bool emit_logits{false};
    };

    struct PackedGraphKeyHash {
        size_t operator()(const PackedGraphKey &key) const noexcept {
            uint64_t hash = 14695981039346656037ull;
            auto mix_u64 = [&](uint64_t value) {
                hash ^= value;
                hash *= 1099511628211ull;
            };
            mix_u64(key.emit_logits ? 1ull : 0ull);
            mix_u64(static_cast<uint64_t>(key.token_count));
            mix_u64(static_cast<uint64_t>(key.sample_count));
            return static_cast<size_t>(hash);
        }
    };

    struct PackedGraphKeyEq {
        bool operator()(const PackedGraphKey &a, const PackedGraphKey &b) const noexcept {
            return a.emit_logits == b.emit_logits && a.token_count == b.token_count && a.sample_count == b.sample_count;
        }
    };

    ServingEngine(
        const marmot_context_t *ctx, std::shared_ptr<const Model> model, const marmot_serving_engine_options_t &opts,
        std::unique_ptr<KVPool> kv_pool
    );

    [[nodiscard]] bool can_accept_request(size_t additional) const noexcept;
    [[nodiscard]] marmot_error_t
    parse_request_ext(const marmot_llm_generate_options_t &gen_opts, Request &req, std::string &error);
    [[nodiscard]] marmot_error_t ensure_scratch(size_t token_count, size_t sample_count, std::string &error);
    [[nodiscard]] marmot_error_t
    ensure_packed_graph(size_t token_count, size_t sample_count, marmot_graph_t **out_graph, std::string &error);
    [[nodiscard]] marmot_error_t
    step_pipelined_greedy_decode(size_t max_steps, size_t &out_steps_done, std::string &error);

    const marmot_context_t *ctx_{nullptr};
    std::shared_ptr<const Model> model_{};
    marmot_serving_engine_options_t opts_{};
    std::unique_ptr<KVPool> kv_pool_{};
    std::unordered_map<marmot_request_id_t, Request> requests_{};
    std::deque<marmot_request_id_t> schedule_{};
    marmot_request_id_t next_request_id_{1};
    PackedBatch last_batch_{};

    const marmot_tensor_t *token_embedding_{nullptr};
    size_t max_seq_len_{0};
    size_t n_vocab_{0};
    size_t n_embd_{0};
    float embedding_scale_{1.0f};
    marmot_dtype_t activation_dtype_{MARMOT_DTYPE_FLOAT32};
    marmot_token_id_t bos_id_{MARMOT_TOKEN_ID_INVALID};
    marmot_token_id_t eos_id_{MARMOT_TOKEN_ID_INVALID};

    std::vector<uint8_t> special_mask_{};
    std::vector<float> logits_f32_{};
    std::vector<uint8_t> seen_{};
    std::vector<marmot_token_id_t> seen_tokens_{};

    TensorPtr token_ids_{};
    TensorPtr positions_{};
    TensorPtr positions_alt_{};
    TensorPtr token_meta_{};
    TensorPtr token_meta_alt_{};
    TensorPtr block_table_snapshot_{};
    TensorPtr block_table_snapshot_alt_{};
    TensorPtr sample_indices_{};
    TensorPtr sample_indices_alt_{};
    TensorPtr hidden_{};
    TensorPtr logits_{};
    TensorPtr logits_max_{};
    TensorPtr logits_argmax_{};
    TensorPtr logits_max_alt_{};
    TensorPtr logits_argmax_alt_{};
    TensorPtr hidden_out_{};
    size_t scratch_token_capacity_{0};
    size_t scratch_sample_capacity_{0};
    bool token_ids_device_ready_{false};
    struct PipelineInFlight {
        marmot_request_id_t request_id{0};
        uint32_t argmax_buffer_index{0};
    };
    std::optional<PipelineInFlight> pipeline_in_flight_{};
    uint32_t pipeline_next_argmax_buffer_index_{0};

    std::unordered_map<PackedGraphKey, GraphOwner, PackedGraphKeyHash, PackedGraphKeyEq> packed_graphs_{};
    std::unordered_map<uint64_t, PrefixEntry> prefix_cache_{};
    std::vector<uint64_t> block_hash_by_id_{};
    std::vector<uint8_t> block_hash_valid_{};
    uint64_t prefix_tick_{0};
};

} // namespace marmot::inference
