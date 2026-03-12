#include "bench_llm.h"
#include "bench_stats.h"

#include "marmot/device.h"
#include "marmot/error.h"
#include "marmot/graph/architecture.h"
#include "marmot/graph/gguf_model.h"
#include "marmot/ops/neural.h"
#include "marmot/tensor.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <new>
#include <random>
#include <span>
#include <string>
#include <time.h>
#include <unordered_map>
#include <vector>

#include "api/inference_handles.hpp"
#include "graph/fast_executor.hpp"
#include "graph/fast_plan.hpp"
#include "inference/common/tensor_ptr.hpp"
#include "inference/common/model_prepack.hpp"
#include "inference/kv_pool/kv_pool.hpp"
#include "inference/model/model.hpp"

namespace {

using marmot::graph::FastExecProfile;
using marmot::graph::FastExecutor;
using marmot::graph::FastPlan;
using marmot::graph::FastPlanBucket;
using marmot::graph::compile_fast_plan;
using marmot::inference::KVPool;
using marmot::inference::Model;
using marmot::inference::TensorPtr;

constexpr uint32_t kTokenFlagPrefill = 1u << 0;
constexpr uint32_t kTokenFlagDecode = 1u << 1;
constexpr size_t kDefaultBlockSize = 16;
constexpr uint64_t kBenchSeed = 0x4d61726d6f74424full;

bool fast_plan_profile_enabled() {
    static bool enabled = [] {
        const char *env = std::getenv("MARMOT_FAST_PLAN_PROFILE");
        return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

bool direct_bench_profile_enabled() {
    static bool enabled = [] {
        const char *env = std::getenv("MARMOT_DIRECT_PROFILE");
        return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

struct GraphDeleter {
    void operator()(marmot_graph_t *graph) const noexcept {
        if (graph != nullptr) {
            marmot_graph_destroy(graph);
        }
    }
};

using GraphOwner = std::unique_ptr<marmot_graph_t, GraphDeleter>;

struct GraphKey {
    uint32_t token_count{0};
    uint32_t sample_count{0};
    bool emit_logits{false};
};

struct GraphKeyHash {
    size_t operator()(const GraphKey &key) const noexcept {
        uint64_t hash = 14695981039346656037ull;
        auto mix = [&](uint64_t value) {
            hash ^= value;
            hash *= 1099511628211ull;
        };
        mix(key.emit_logits ? 1ull : 0ull);
        mix(static_cast<uint64_t>(key.token_count));
        mix(static_cast<uint64_t>(key.sample_count));
        return static_cast<size_t>(hash);
    }
};

struct GraphKeyEq {
    bool operator()(const GraphKey &a, const GraphKey &b) const noexcept {
        return a.token_count == b.token_count && a.sample_count == b.sample_count && a.emit_logits == b.emit_logits;
    }
};

struct GraphEntry {
    GraphOwner graph{};
    std::unique_ptr<FastPlan> fast_plan{};
};

struct Scratch {
    TensorPtr token_ids{};
    TensorPtr positions{};
    TensorPtr token_meta{};
    TensorPtr sample_indices{};
    TensorPtr hidden{};
    TensorPtr hidden_out{};
    TensorPtr logits{};
    size_t token_capacity{0};
    size_t sample_capacity{0};
};

struct DirectStepProfile {
    uint64_t kv_prepare_ns{0};
    uint64_t input_setup_ns{0};
    uint64_t embedding_ns{0};
    uint64_t graph_ns{0};
    uint64_t kv_commit_ns{0};
    size_t steps{0};
};

[[nodiscard]] uint64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
}

void fill_random_tokens(std::span<marmot_token_id_t> tokens, size_t vocab, std::mt19937_64 &rng) {
    if (tokens.empty() || vocab == 0) {
        return;
    }
    std::uniform_int_distribution<uint32_t> dist(0u, static_cast<uint32_t>(vocab - 1));
    for (marmot_token_id_t &token : tokens) {
        token = static_cast<marmot_token_id_t>(dist(rng));
    }
}

[[nodiscard]] uint64_t mix_bench_seed(uint64_t seed, uint64_t value) {
    seed ^= value + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
    return seed;
}

[[nodiscard]] std::mt19937_64 make_bench_rng(uint64_t tag, size_t depth_tokens, size_t data_tokens) {
    uint64_t seed = kBenchSeed;
    seed = mix_bench_seed(seed, tag);
    seed = mix_bench_seed(seed, static_cast<uint64_t>(depth_tokens));
    seed = mix_bench_seed(seed, static_cast<uint64_t>(data_tokens));
    return std::mt19937_64(seed);
}

[[nodiscard]] marmot_error_t execute_packed_graph(
    marmot_graph_t *graph, const FastPlan *fast_plan, const marmot_context_t *ctx,
    std::span<const marmot_tensor_t *const> inputs, std::span<marmot_tensor_t *const> outputs, bool enable_profile
) {
    if (graph == nullptr || ctx == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (fast_plan == nullptr) {
        return marmot_graph_execute(
            graph, ctx, const_cast<const marmot_tensor_t **>(inputs.data()), inputs.size(),
            const_cast<marmot_tensor_t **>(outputs.data()), outputs.size()
        );
    }

    FastExecProfile profile{};
    FastExecProfile *profile_ptr = enable_profile && fast_plan_profile_enabled() ? &profile : nullptr;
    marmot_error_t status = FastExecutor::execute(graph, fast_plan, ctx, inputs, outputs, profile_ptr);
    if (status == MARMOT_SUCCESS && profile_ptr != nullptr) {
        FastExecutor::print_profile(profile);
    }
    return status;
}

[[nodiscard]] TensorPtr create_tensor(const marmot_context_t *ctx, std::span<const size_t> shape, marmot_dtype_t dtype) {
    marmot_tensor_t *tensor = marmot_tensor_create(ctx, shape.data(), shape.size(), dtype);
    return TensorPtr(tensor);
}

class DirectBenchRunner {
  public:
    [[nodiscard]] static std::unique_ptr<DirectBenchRunner>
    create(const marmot_bench_model_t *model, std::string &error, marmot_error_t &status) {
        if (model == nullptr || !model->loaded || model->ctx == nullptr || model->model == nullptr) {
            status = MARMOT_ERROR_INVALID_ARGUMENT;
            error = "benchmark model is not loaded";
            return nullptr;
        }

        auto *handle = reinterpret_cast<const MarmotModelHandle *>(model->model);
        if (handle == nullptr || !handle->impl) {
            status = MARMOT_ERROR_INVALID_ARGUMENT;
            error = "model handle is unavailable";
            return nullptr;
        }

        const std::shared_ptr<Model> model_impl = handle->impl;
        const marmot_gguf_model_t *gguf = model_impl->gguf();
        if (gguf == nullptr) {
            status = MARMOT_ERROR_INVALID_ARGUMENT;
            error = "GGUF model is unavailable";
            return nullptr;
        }

        const marmot_tensor_t *embed = marmot_gguf_model_tensor(gguf, "token_embd.weight");
        if (embed == nullptr) {
            status = MARMOT_ERROR_INVALID_ARGUMENT;
            error = "missing token_embd.weight tensor";
            return nullptr;
        }

        marmot_gguf_model_meta_t meta{};
        if (!marmot_gguf_model_metadata(gguf, &meta)) {
            status = MARMOT_ERROR_INVALID_ARGUMENT;
            error = "failed to load GGUF model metadata";
            return nullptr;
        }

        const marmot_model_info_t &info = model_impl->info();
        if (info.n_vocab == 0 || info.n_embd == 0 || info.n_layer == 0 || info.n_head_kv == 0) {
            status = MARMOT_ERROR_INVALID_ARGUMENT;
            error = "model is missing required benchmark dimensions";
            return nullptr;
        }

        const marmot_dtype_t activation_dtype =
            marmot_activation_dtype_for_architecture(meta.architecture, model->ctx->backend_type);
        const size_t max_seq_len =
            info.context_length > 0 ? std::min(model->config.ctx_size, info.context_length) : model->config.ctx_size;
        const size_t max_num_tokens = std::min(max_seq_len, std::max<size_t>(1, model->config.batch_size));
        const size_t block_size = kDefaultBlockSize;
        size_t blocks_per_seq = (max_seq_len + block_size - 1) / block_size;
        if (blocks_per_seq == 0) {
            blocks_per_seq = 1;
        }

        KVPool::Options kv_opts{};
        kv_opts.backend = model->ctx->backend_type;
        kv_opts.max_seqs = 1;
        kv_opts.max_seq_len = max_seq_len;
        kv_opts.block_size = block_size;
        kv_opts.num_blocks = blocks_per_seq;
        kv_opts.num_layers = info.n_layer;
        kv_opts.num_kv_heads = info.n_head_kv;
        kv_opts.head_dim = meta.head_dim;
        kv_opts.kv_dtype = activation_dtype;
        kv_opts.deterministic_alloc = true;

        auto kv_pool_result = KVPool::create(model->ctx, kv_opts);
        if (!kv_pool_result) {
            status = kv_pool_result.error();
            error = "failed to create KV pool";
            return nullptr;
        }

        auto runner = std::unique_ptr<DirectBenchRunner>(new (std::nothrow) DirectBenchRunner(
            model, std::move(model_impl), std::move(meta), embed, activation_dtype, max_seq_len, max_num_tokens,
            std::move(*kv_pool_result)
        ));
        if (!runner) {
            status = MARMOT_ERROR_OUT_OF_MEMORY;
            error = "failed to allocate direct benchmark runner";
            return nullptr;
        }

        status = runner->ensure_scratch(max_num_tokens, 1, error);
        if (status != MARMOT_SUCCESS) {
            return nullptr;
        }

        status = MARMOT_SUCCESS;
        return runner;
    }

    [[nodiscard]] marmot_error_t run_prompt_benchmark(
        size_t n_depth, size_t n_prompt, size_t warmup_runs, size_t repetitions, double *samples_us, double *ttft_ns
    ) {
        if (samples_us == nullptr) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        if (n_depth + n_prompt > max_seq_len_) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "prompt benchmark exceeds context length");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }

        std::vector<marmot_token_id_t> depth_tokens(n_depth);
        std::vector<marmot_token_id_t> prompt_tokens(n_prompt);
        std::mt19937_64 rng = make_bench_rng(0x70726f6d7074ull, n_depth, n_prompt);
        fill_random_tokens(depth_tokens, n_vocab_, rng);
        fill_random_tokens(prompt_tokens, n_vocab_, rng);

        const size_t total_runs = warmup_runs + repetitions;
        for (size_t run = 0; run < total_runs; ++run) {
            marmot_seq_slot_t seq = 0;
            marmot_error_t err = kv_pool_->acquire_seq(seq);
            if (err != MARMOT_SUCCESS) {
                return err;
            }

            if (!depth_tokens.empty()) {
                err = run_prefill(seq, depth_tokens, static_cast<uint64_t *>(nullptr), false);
                if (err != MARMOT_SUCCESS) {
                    (void)kv_pool_->release_seq(seq);
                    return err;
                }
            }

            uint64_t duration_ns = 0;
            err = run_prefill(seq, prompt_tokens, &duration_ns, run >= warmup_runs);
            (void)kv_pool_->release_seq(seq);
            if (err != MARMOT_SUCCESS) {
                return err;
            }

            if (run >= warmup_runs) {
                const size_t sample_index = run - warmup_runs;
                samples_us[sample_index] = static_cast<double>(duration_ns) / 1000.0;
                if (sample_index == 0 && ttft_ns != nullptr) {
                    *ttft_ns = static_cast<double>(duration_ns);
                }
            }
        }

        return MARMOT_SUCCESS;
    }

    [[nodiscard]] marmot_error_t run_decode_benchmark(
        size_t n_depth, size_t n_gen, size_t warmup_runs, size_t repetitions, double *samples_us, double *ttft_ns
    ) {
        if (samples_us == nullptr) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        if (n_depth + n_gen > max_seq_len_) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "decode benchmark exceeds context length");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }

        std::vector<marmot_token_id_t> depth_tokens(n_depth);
        std::vector<marmot_token_id_t> decode_tokens(n_gen);
        std::mt19937_64 rng = make_bench_rng(0x6465636f6465ull, n_depth, n_gen);
        fill_random_tokens(depth_tokens, n_vocab_, rng);
        fill_random_tokens(decode_tokens, n_vocab_, rng);
        direct_profile_ = {};

        const size_t total_runs = warmup_runs + repetitions;
        for (size_t run = 0; run < total_runs; ++run) {
            marmot_seq_slot_t seq = 0;
            marmot_error_t err = kv_pool_->acquire_seq(seq);
            if (err != MARMOT_SUCCESS) {
                return err;
            }

            if (!depth_tokens.empty()) {
                err = run_prefill(seq, depth_tokens, static_cast<uint64_t *>(nullptr), false);
                if (err != MARMOT_SUCCESS) {
                    (void)kv_pool_->release_seq(seq);
                    return err;
                }
            }

            const uint64_t start_ns = now_ns();
            uint64_t first_step_ns = 0;
            for (size_t i = 0; i < decode_tokens.size(); ++i) {
                const uint64_t step_start_ns = now_ns();
                err = run_decode_step(seq, decode_tokens[i], run >= warmup_runs);
                const uint64_t step_end_ns = now_ns();
                if (err != MARMOT_SUCCESS) {
                    (void)kv_pool_->release_seq(seq);
                    return err;
                }
                if (i == 0) {
                    first_step_ns = step_end_ns - step_start_ns;
                }
            }
            const uint64_t end_ns = now_ns();
            (void)kv_pool_->release_seq(seq);

            if (run >= warmup_runs) {
                const size_t sample_index = run - warmup_runs;
                samples_us[sample_index] = static_cast<double>(end_ns - start_ns) / 1000.0;
                if (sample_index == 0 && ttft_ns != nullptr) {
                    *ttft_ns = static_cast<double>(first_step_ns);
                }
            }
        }

        if (direct_bench_profile_enabled() && direct_profile_.steps > 0) {
            const double steps = static_cast<double>(direct_profile_.steps);
            fprintf(
                stderr,
                "[direct profile] decode steps=%zu kv_prepare=%.3fms input_setup=%.3fms embedding=%.3fms "
                "graph=%.3fms kv_commit=%.3fms total=%.3fms\n",
                direct_profile_.steps, (double)direct_profile_.kv_prepare_ns / 1e6 / steps,
                (double)direct_profile_.input_setup_ns / 1e6 / steps, (double)direct_profile_.embedding_ns / 1e6 / steps,
                (double)direct_profile_.graph_ns / 1e6 / steps, (double)direct_profile_.kv_commit_ns / 1e6 / steps,
                (double)(direct_profile_.kv_prepare_ns + direct_profile_.input_setup_ns + direct_profile_.embedding_ns +
                         direct_profile_.graph_ns + direct_profile_.kv_commit_ns) /
                    1e6 / steps
            );
        }

        return MARMOT_SUCCESS;
    }

  private:
    DirectBenchRunner(
        const marmot_bench_model_t *bench_model, std::shared_ptr<Model> model_impl, marmot_gguf_model_meta_t meta,
        const marmot_tensor_t *token_embedding, marmot_dtype_t activation_dtype, size_t max_seq_len,
        size_t max_num_tokens, std::unique_ptr<KVPool> kv_pool
    )
        : ctx_(bench_model->ctx), model_impl_(std::move(model_impl)), token_embedding_(token_embedding),
          activation_dtype_(activation_dtype), max_seq_len_(max_seq_len),
          max_num_tokens_(max_num_tokens), n_vocab_(bench_model->info.n_vocab), n_embd_(bench_model->info.n_embd),
          embedding_scale_(meta.embedding_scale), kv_pool_(std::move(kv_pool)) {}

    [[nodiscard]] marmot_error_t ensure_scratch(size_t token_capacity, size_t sample_capacity, std::string &error) {
        if (scratch_.token_capacity == token_capacity && scratch_.sample_capacity == sample_capacity &&
            scratch_.token_ids && scratch_.positions && scratch_.token_meta && scratch_.hidden && scratch_.hidden_out &&
            ((sample_capacity == 0) || (scratch_.sample_indices && scratch_.logits))) {
            return MARMOT_SUCCESS;
        }

        const std::array<size_t, 1> token_shape{token_capacity};
        const std::array<size_t, 2> token_meta_shape{token_capacity, 4};
        const std::array<size_t, 2> hidden_shape{token_capacity, n_embd_};
        scratch_.token_ids = create_tensor(ctx_, token_shape, MARMOT_DTYPE_INT32);
        scratch_.positions = create_tensor(ctx_, token_shape, MARMOT_DTYPE_FLOAT32);
        scratch_.token_meta = create_tensor(ctx_, token_meta_shape, MARMOT_DTYPE_UINT32);
        scratch_.hidden = create_tensor(ctx_, hidden_shape, activation_dtype_);
        scratch_.hidden_out = create_tensor(ctx_, hidden_shape, activation_dtype_);
        if (!scratch_.token_ids || !scratch_.positions || !scratch_.token_meta || !scratch_.hidden ||
            !scratch_.hidden_out) {
            error = "failed to allocate direct benchmark scratch tensors";
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }

        if (sample_capacity > 0) {
            const std::array<size_t, 1> sample_shape{sample_capacity};
            const std::array<size_t, 2> logits_shape{sample_capacity, n_vocab_};
            scratch_.sample_indices = create_tensor(ctx_, sample_shape, MARMOT_DTYPE_UINT32);
            scratch_.logits = create_tensor(ctx_, logits_shape, activation_dtype_);
            if (!scratch_.sample_indices || !scratch_.logits) {
                error = "failed to allocate direct benchmark logits scratch";
                return MARMOT_ERROR_OUT_OF_MEMORY;
            }
        } else {
            scratch_.sample_indices.reset();
            scratch_.logits.reset();
        }

        scratch_.token_capacity = token_capacity;
        scratch_.sample_capacity = sample_capacity;
        return MARMOT_SUCCESS;
    }

    [[nodiscard]] marmot_error_t ensure_graph(
        size_t token_count, size_t sample_count, marmot_graph_t **out_graph, const FastPlan **out_fast_plan,
        std::string &error
    ) {
        if (out_graph == nullptr) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }

        *out_graph = nullptr;
        if (out_fast_plan != nullptr) {
            *out_fast_plan = nullptr;
        }

        const bool emit_logits = sample_count > 0;
        GraphKey key{
            .token_count = static_cast<uint32_t>(token_count),
            .sample_count = static_cast<uint32_t>(emit_logits ? sample_count : 0),
            .emit_logits = emit_logits,
        };

        auto found = graphs_.find(key);
        if (found == graphs_.end()) {
            marmot_packed_graph_options_t opts;
            marmot_error_t status = marmot_packed_graph_options_init(&opts);
            if (status != MARMOT_SUCCESS) {
                return status;
            }
            opts.flags = MARMOT_PACKED_GRAPH_FLAG_KV_DTYPE_AUTO;
            opts.token_count = token_count;
            opts.sample_count = emit_logits ? sample_count : 0;
            opts.max_seqs = 1;
            opts.max_seq_len = max_seq_len_;
            opts.block_size = kDefaultBlockSize;
            opts.num_kv_blocks = kv_pool_->total_block_count();
            opts.kv_dtype = activation_dtype_;

            marmot_graph_t *graph_raw = nullptr;
            status = marmot_graph_from_model_packed(model_impl_->gguf(), ctx_->backend_type, &opts, &graph_raw);
            if (status != MARMOT_SUCCESS || graph_raw == nullptr) {
                error = "failed to build packed graph";
                return status != MARMOT_SUCCESS ? status : MARMOT_ERROR_INVALID_OPERATION;
            }

            GraphEntry entry{};
            entry.graph.reset(graph_raw);
            marmot::inference::prepack_cpu_model_weights(ctx_, model_impl_->gguf(), emit_logits);
            FastPlanBucket bucket{
                .token_count = static_cast<uint32_t>(token_count),
                .sample_count = static_cast<uint32_t>(emit_logits ? sample_count : 0),
                .emit_logits = emit_logits,
            };
            if (auto fast_plan = compile_fast_plan(graph_raw, bucket)) {
                entry.fast_plan = std::make_unique<FastPlan>(std::move(*fast_plan));
            }
            found = graphs_.emplace(key, std::move(entry)).first;
        }

        *out_graph = found->second.graph.get();
        if (out_fast_plan != nullptr) {
            *out_fast_plan = found->second.fast_plan.get();
        }
        return MARMOT_SUCCESS;
    }

    [[nodiscard]] marmot_error_t run_prefill(
        marmot_seq_slot_t seq, std::span<const marmot_token_id_t> tokens, uint64_t *duration_ns, bool enable_profile
    ) {
        if (duration_ns != nullptr) {
            *duration_ns = 0;
        }
        if (tokens.empty()) {
            return MARMOT_SUCCESS;
        }

        const uint64_t start_ns = duration_ns != nullptr ? now_ns() : 0;
        size_t offset = 0;
        while (offset < tokens.size()) {
            const size_t chunk = std::min(max_num_tokens_, tokens.size() - offset);
            marmot_error_t err = run_batch(seq, tokens.subspan(offset, chunk), kTokenFlagPrefill, false, enable_profile);
            if (err != MARMOT_SUCCESS) {
                return err;
            }
            offset += chunk;
        }

        if (duration_ns != nullptr) {
            *duration_ns = now_ns() - start_ns;
        }
        return MARMOT_SUCCESS;
    }

    [[nodiscard]] marmot_error_t run_decode_step(marmot_seq_slot_t seq, marmot_token_id_t token, bool enable_profile) {
        return run_batch(seq, std::span<const marmot_token_id_t>(&token, 1), kTokenFlagDecode, true, enable_profile);
    }

    [[nodiscard]] marmot_error_t
    run_batch(
        marmot_seq_slot_t seq, std::span<const marmot_token_id_t> tokens, uint32_t flags, bool emit_logits,
        bool enable_profile
    ) {
        if (tokens.empty() || tokens.size() > max_num_tokens_) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }

        std::vector<marmot_kv_slot_t> slots(tokens.size());
        size_t start_pos = 0;
        KVPool::AppendPlan plan{};
        const bool record_profile = enable_profile && direct_bench_profile_enabled();
        const uint64_t kv_prepare_start_ns = record_profile ? now_ns() : 0;
        marmot_error_t status = kv_pool_->prepare_append(seq, tokens.size(), slots.data(), start_pos, plan);
        if (record_profile) {
            direct_profile_.kv_prepare_ns += now_ns() - kv_prepare_start_ns;
        }
        if (status != MARMOT_SUCCESS) {
            return status;
        }

        std::string error;
        uint64_t input_setup_ns = 0;
        uint64_t embedding_ns = 0;
        status = prepare_batch_inputs(
            tokens, slots, seq, start_pos, flags, emit_logits, error, &input_setup_ns, &embedding_ns
        );
        if (record_profile) {
            direct_profile_.input_setup_ns += input_setup_ns;
            direct_profile_.embedding_ns += embedding_ns;
        }
        if (status != MARMOT_SUCCESS) {
            kv_pool_->abort_append(plan);
            if (!error.empty()) {
                marmot_set_error(status, error.c_str());
            }
            return status;
        }

        const uint64_t graph_start_ns = record_profile ? now_ns() : 0;
        status = execute_batch_graph(tokens.size(), emit_logits, enable_profile, error);
        if (record_profile) {
            direct_profile_.graph_ns += now_ns() - graph_start_ns;
        }
        if (status != MARMOT_SUCCESS) {
            kv_pool_->abort_append(plan);
            if (!error.empty()) {
                marmot_set_error(status, error.c_str());
            }
            return status;
        }

        const uint64_t kv_commit_start_ns = record_profile ? now_ns() : 0;
        status = kv_pool_->commit_append(plan);
        if (record_profile) {
            direct_profile_.kv_commit_ns += now_ns() - kv_commit_start_ns;
            direct_profile_.steps += 1;
        }
        if (status != MARMOT_SUCCESS && !error.empty()) {
            marmot_set_error(status, error.c_str());
        }
        return status;
    }

    [[nodiscard]] marmot_error_t prepare_batch_inputs(
        std::span<const marmot_token_id_t> tokens, std::span<const marmot_kv_slot_t> slots, marmot_seq_slot_t seq,
        size_t start_pos, uint32_t flags, bool emit_logits, std::string &error, uint64_t *input_setup_ns,
        uint64_t *embedding_ns
    ) {
        error.clear();
        if (input_setup_ns != nullptr) {
            *input_setup_ns = 0;
        }
        if (embedding_ns != nullptr) {
            *embedding_ns = 0;
        }
        const uint64_t input_setup_start_ns = input_setup_ns != nullptr ? now_ns() : 0;
        marmot_error_t status = ensure_scratch(max_num_tokens_, emit_logits ? 1 : 0, error);
        if (status != MARMOT_SUCCESS) {
            return status;
        }

        scratch_.token_ids->shape.ndim = 1;
        scratch_.token_ids->shape.shape[0] = tokens.size();
        scratch_.token_ids->shape.strides[0] = 1;

        scratch_.positions->shape.ndim = 1;
        scratch_.positions->shape.shape[0] = tokens.size();
        scratch_.positions->shape.strides[0] = 1;

        scratch_.token_meta->shape.ndim = 2;
        scratch_.token_meta->shape.shape[0] = tokens.size();
        scratch_.token_meta->shape.shape[1] = 4;
        scratch_.token_meta->shape.strides[1] = 1;
        scratch_.token_meta->shape.strides[0] = 4;

        scratch_.hidden->shape.ndim = 2;
        scratch_.hidden->shape.shape[0] = tokens.size();
        scratch_.hidden->shape.shape[1] = n_embd_;
        scratch_.hidden->shape.strides[1] = 1;
        scratch_.hidden->shape.strides[0] = n_embd_;

        scratch_.hidden_out->shape.ndim = 2;
        scratch_.hidden_out->shape.shape[0] = tokens.size();
        scratch_.hidden_out->shape.shape[1] = n_embd_;
        scratch_.hidden_out->shape.strides[1] = 1;
        scratch_.hidden_out->shape.strides[0] = n_embd_;

        marmot_int32_t *token_ptr = marmot_tensor_data_i32_mut(ctx_, scratch_.token_ids.get());
        marmot_uint32_t *meta_ptr = marmot_tensor_data_u32_mut(ctx_, scratch_.token_meta.get());
        float *pos_ptr = marmot_tensor_data_f32_mut(ctx_, scratch_.positions.get());
        if (token_ptr == nullptr || meta_ptr == nullptr || pos_ptr == nullptr) {
            error = "failed to access direct benchmark scratch";
            return MARMOT_ERROR_INVALID_OPERATION;
        }

        for (size_t i = 0; i < tokens.size(); ++i) {
            token_ptr[i].value = static_cast<int32_t>(tokens[i]);
            pos_ptr[i] = static_cast<float>(start_pos + i);
            meta_ptr[i * 4 + 0].value = static_cast<uint32_t>(seq);
            meta_ptr[i * 4 + 1].value = static_cast<uint32_t>(start_pos + i);
            meta_ptr[i * 4 + 2].value = static_cast<uint32_t>(slots[i]);
            meta_ptr[i * 4 + 3].value = flags;
        }

        if (emit_logits) {
            scratch_.sample_indices->shape.ndim = 1;
            scratch_.sample_indices->shape.shape[0] = 1;
            scratch_.sample_indices->shape.strides[0] = 1;
            scratch_.logits->shape.ndim = 2;
            scratch_.logits->shape.shape[0] = 1;
            scratch_.logits->shape.shape[1] = n_vocab_;
            scratch_.logits->shape.strides[1] = 1;
            scratch_.logits->shape.strides[0] = n_vocab_;

            marmot_uint32_t *sample_ptr = marmot_tensor_data_u32_mut(ctx_, scratch_.sample_indices.get());
            if (sample_ptr == nullptr) {
                error = "failed to access sample indices scratch";
                return MARMOT_ERROR_INVALID_OPERATION;
            }
            sample_ptr[0].value = static_cast<uint32_t>(tokens.size() - 1);
        }

        if (ctx_->backend_type != MARMOT_BACKEND_CPU) {
            status = marmot_tensor_to_device(ctx_, scratch_.token_ids.get());
            if (status != MARMOT_SUCCESS) {
                error = "failed to sync token ids";
                return status;
            }
            status = marmot_tensor_to_device(ctx_, scratch_.positions.get());
            if (status != MARMOT_SUCCESS) {
                error = "failed to sync positions";
                return status;
            }
            status = marmot_tensor_to_device(ctx_, scratch_.token_meta.get());
            if (status != MARMOT_SUCCESS) {
                error = "failed to sync token metadata";
                return status;
            }
            if (emit_logits) {
                status = marmot_tensor_to_device(ctx_, scratch_.sample_indices.get());
                if (status != MARMOT_SUCCESS) {
                    error = "failed to sync sample indices";
                    return status;
                }
            }
        }
        if (input_setup_start_ns != 0) {
            *input_setup_ns = now_ns() - input_setup_start_ns;
        }

        marmot_embedding_gather_desc_t gather = marmot_embedding_gather_desc_default();
        gather.weights = token_embedding_;
        gather.token_ids = scratch_.token_ids.get();
        gather.out = scratch_.hidden.get();
        gather.dtype_out = scratch_.hidden->dtype;
        gather.scale = embedding_scale_;
        gather.bounds_check = true;
        gather.padding_id = -1;
        gather.prefer_gpu_private = MARMOT_PREFERENCE_DISABLE;
        gather.allow_quant_decode_on_the_fly = MARMOT_PREFERENCE_ENABLE;

        const uint64_t embedding_start_ns = embedding_ns != nullptr ? now_ns() : 0;
        status = marmot_embedding_gather(ctx_, &gather);
        if (embedding_start_ns != 0) {
            *embedding_ns = now_ns() - embedding_start_ns;
        }
        if (status != MARMOT_SUCCESS) {
            error = "embedding_gather failed";
            return status;
        }

        return MARMOT_SUCCESS;
    }

    [[nodiscard]] marmot_error_t
    execute_batch_graph(size_t token_count, bool emit_logits, bool enable_profile, std::string &error) {
        error.clear();
        marmot_graph_t *graph = nullptr;
        const FastPlan *fast_plan = nullptr;
        marmot_error_t status = ensure_graph(token_count, emit_logits ? 1 : 0, &graph, &fast_plan, error);
        if (status != MARMOT_SUCCESS) {
            return status;
        }

        marmot_tensor_t *kv_k = nullptr;
        marmot_tensor_t *kv_v = nullptr;
        marmot_tensor_t *block_table = nullptr;
        marmot_tensor_t *kv_k_scale = nullptr;
        marmot_tensor_t *kv_v_scale = nullptr;
        kv_pool_->get_tensors(&kv_k, &kv_v, &block_table, &kv_k_scale, &kv_v_scale);
        if (kv_k == nullptr || kv_v == nullptr || block_table == nullptr) {
            error = "KV pool tensors unavailable";
            return MARMOT_ERROR_INVALID_OPERATION;
        }

        if (ctx_->backend_type != MARMOT_BACKEND_CPU) {
            status = marmot_tensor_to_device(ctx_, block_table);
            if (status != MARMOT_SUCCESS) {
                error = "failed to sync block table";
                return status;
            }
        }

        const bool has_fp8_scales = kv_k_scale != nullptr && kv_v_scale != nullptr;
        if (emit_logits) {
            if (has_fp8_scales) {
                const marmot_tensor_t *inputs[] = {
                    scratch_.hidden.get(),        scratch_.positions.get(), scratch_.token_meta.get(),
                    block_table,                  kv_k,                    kv_v,
                    kv_k_scale,                  kv_v_scale,              scratch_.sample_indices.get(),
                };
                marmot_tensor_t *outputs[] = {scratch_.logits.get()};
                status = execute_packed_graph(
                    graph, fast_plan, ctx_, std::span<const marmot_tensor_t *const>(inputs, 9),
                    std::span<marmot_tensor_t *const>(outputs, 1), enable_profile
                );
            } else {
                const marmot_tensor_t *inputs[] = {
                    scratch_.hidden.get(), scratch_.positions.get(), scratch_.token_meta.get(),
                    block_table,           kv_k,                    kv_v,
                    scratch_.sample_indices.get(),
                };
                marmot_tensor_t *outputs[] = {scratch_.logits.get()};
                status = execute_packed_graph(
                    graph, fast_plan, ctx_, std::span<const marmot_tensor_t *const>(inputs, 7),
                    std::span<marmot_tensor_t *const>(outputs, 1), enable_profile
                );
            }
        } else {
            if (has_fp8_scales) {
                const marmot_tensor_t *inputs[] = {
                    scratch_.hidden.get(), scratch_.positions.get(), scratch_.token_meta.get(), block_table,
                    kv_k,                  kv_v,                    kv_k_scale,               kv_v_scale,
                };
                marmot_tensor_t *outputs[] = {scratch_.hidden_out.get()};
                status = execute_packed_graph(
                    graph, fast_plan, ctx_, std::span<const marmot_tensor_t *const>(inputs, 8),
                    std::span<marmot_tensor_t *const>(outputs, 1), enable_profile
                );
            } else {
                const marmot_tensor_t *inputs[] = {
                    scratch_.hidden.get(), scratch_.positions.get(), scratch_.token_meta.get(), block_table, kv_k, kv_v,
                };
                marmot_tensor_t *outputs[] = {scratch_.hidden_out.get()};
                status = execute_packed_graph(
                    graph, fast_plan, ctx_, std::span<const marmot_tensor_t *const>(inputs, 6),
                    std::span<marmot_tensor_t *const>(outputs, 1), enable_profile
                );
            }
        }

        if (status != MARMOT_SUCCESS) {
            const char *detail = marmot_get_last_error_detail();
            error = (detail != nullptr && detail[0] != '\0') ? detail : "packed graph execute failed";
        }
        return status;
    }

    const marmot_context_t *ctx_{nullptr};
    std::shared_ptr<Model> model_impl_{};
    const marmot_tensor_t *token_embedding_{nullptr};
    marmot_dtype_t activation_dtype_{MARMOT_DTYPE_FLOAT32};
    size_t max_seq_len_{0};
    size_t max_num_tokens_{0};
    size_t n_vocab_{0};
    size_t n_embd_{0};
    float embedding_scale_{1.0f};
    std::unique_ptr<KVPool> kv_pool_{};
    Scratch scratch_{};
    std::unordered_map<GraphKey, GraphEntry, GraphKeyHash, GraphKeyEq> graphs_{};
    DirectStepProfile direct_profile_{};
};

} // namespace

extern "C" marmot_error_t marmot_bench_llm_run_direct(
    const marmot_bench_model_t *model, const marmot_bench_llm_params_t *params, size_t repetitions,
    marmot_bench_llm_result_t *result
) {
    if (model == nullptr || params == nullptr || result == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (params->n_seqs != 1) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "direct LLM benchmark requires --concurrency 1");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    memset(result, 0, sizeof(*result));
    result->n_prompt = params->n_prompt;
    result->n_gen = params->n_gen;
    result->n_depth = params->n_depth;
    result->n_seqs = params->n_seqs;
    result->mode = params->mode;

    std::string error;
    marmot_error_t status = MARMOT_SUCCESS;
    auto runner = DirectBenchRunner::create(model, error, status);
    if (!runner) {
        if (!error.empty()) {
            marmot_set_error(status, error.c_str());
        }
        return status;
    }

    double *pp_times_us = nullptr;
    double *tg_times_us = nullptr;

    if (params->n_prompt > 0) {
        pp_times_us = static_cast<double *>(malloc(repetitions * sizeof(double)));
        if (pp_times_us == nullptr) {
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
    }
    if (params->n_gen > 0) {
        tg_times_us = static_cast<double *>(malloc(repetitions * sizeof(double)));
        if (tg_times_us == nullptr) {
            free(pp_times_us);
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
    }

    if (params->n_prompt > 0) {
        status = runner->run_prompt_benchmark(params->n_depth, params->n_prompt, params->warmup_runs, repetitions, pp_times_us, &result->ttft_ns);
        if (status != MARMOT_SUCCESS) {
            free(pp_times_us);
            free(tg_times_us);
            return status;
        }
        marmot_bench_compute_stats(pp_times_us, repetitions, 0.95, &result->pp_stats);
        result->pp_total_ns = result->pp_stats.mean_us * 1000.0;
        if (result->pp_total_ns > 0.0) {
            result->pp_tokens_per_sec = static_cast<double>(params->n_prompt) / (result->pp_total_ns / 1e9);
        }
    }

    if (params->n_gen > 0) {
        status = runner->run_decode_benchmark(params->n_depth, params->n_gen, params->warmup_runs, repetitions, tg_times_us, &result->ttft_ns);
        if (status != MARMOT_SUCCESS) {
            free(pp_times_us);
            free(tg_times_us);
            return status;
        }
        marmot_bench_compute_stats(tg_times_us, repetitions, 0.95, &result->tg_stats);
        result->tg_total_ns = result->tg_stats.mean_us * 1000.0;
        if (result->tg_total_ns > 0.0) {
            result->tg_tokens_per_sec = static_cast<double>(params->n_gen) / (result->tg_total_ns / 1e9);
        }
    }

    free(pp_times_us);
    free(tg_times_us);
    return MARMOT_SUCCESS;
}
