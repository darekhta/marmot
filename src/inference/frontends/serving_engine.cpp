#include "serving_engine.hpp"

#include "marmot/device.h"
#include "marmot/error.h"
#include "marmot/graph/architecture.h"
#include "marmot/graph/gguf_loader.h"
#include "marmot/graph/gguf_model.h"
#include "marmot/macros.h"
#include "marmot/ops/manipulation.h"
#include "marmot/ops/neural.h"
#include "marmot/ops/reduction.h"
#include "marmot/tensor.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <new>
#include <numeric>
#include <span>

#include "core/context/context_internal.h"
#include "graph/fast_executor.hpp"
#include "graph/fast_plan.hpp"
#include "inference/common/model_prepack.hpp"
#include "inference/kv_pool/kv_pool.hpp"
#include "inference/model/model.hpp"
#include "utils/dtype_ref.h"

namespace marmot::inference {

namespace {

bool step_profile_enabled() {
    static bool enabled = [] {
        const char *env = std::getenv("MARMOT_PROFILE_STEP");
        return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

bool fast_plan_profile_enabled() {
    static bool enabled = [] {
        const char *env = std::getenv("MARMOT_FAST_PLAN_PROFILE");
        return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

size_t cpu_prefill_thread_count(const marmot_context_t *ctx) {
    if (ctx == nullptr || ctx->backend_type != MARMOT_BACKEND_CPU) {
        return 0;
    }
    const auto &topology = ctx->device_caps.topology;
    if (topology.is_hybrid && topology.p_cores > 1) {
        return topology.p_cores - 1;
    }
    if (topology.total_cores > 0) {
        return topology.total_cores;
    }
    return marmot_context_get_thread_count(ctx);
}

size_t cpu_decode_thread_count(const marmot_context_t *ctx) {
    if (ctx == nullptr || ctx->backend_type != MARMOT_BACKEND_CPU) {
        return 0;
    }
    const auto &topology = ctx->device_caps.topology;
    if (topology.is_hybrid && topology.p_cores > 1) {
        return topology.p_cores - 1;
    }
    if (topology.total_cores > 0) {
        return topology.total_cores;
    }
    return marmot_context_get_thread_count(ctx);
}

marmot_error_t apply_cpu_step_thread_policy(const marmot_context_t *ctx, bool has_prefill) {
    if (ctx == nullptr || ctx->backend_type != MARMOT_BACKEND_CPU) {
        return MARMOT_SUCCESS;
    }
    if (marmot_context_thread_count_is_explicit(ctx)) {
        return MARMOT_SUCCESS;
    }

    const size_t target_threads = has_prefill ? cpu_prefill_thread_count(ctx) : cpu_decode_thread_count(ctx);
    if (target_threads == 0 || target_threads == marmot_context_get_thread_count(ctx)) {
        return MARMOT_SUCCESS;
    }
    return marmot_context_set_thread_count_auto(const_cast<marmot_context_t *>(ctx), target_threads);
}

marmot_error_t execute_packed_graph(
    marmot_graph_t *graph, const graph::FastPlan *fast_plan, const marmot_context_t *ctx,
    std::span<const marmot_tensor_t *const> inputs, std::span<marmot_tensor_t *const> outputs
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

    graph::FastExecProfile profile{};
    graph::FastExecProfile *profile_ptr = fast_plan_profile_enabled() ? &profile : nullptr;
    marmot_error_t status = graph::FastExecutor::execute(graph, fast_plan, ctx, inputs, outputs, profile_ptr);
    if (status == MARMOT_SUCCESS && profile_ptr != nullptr) {
        graph::FastExecutor::print_profile(profile);
    }
    return status;
}

struct StepTimer {
    double batch_build_ns{0.0};
    double host_sync_ns{0.0};
    double embedding_ns{0.0};
    double graph_exec_ns{0.0};
    double sampling_ns{0.0};
    double total_ns{0.0};
    size_t token_count{0};
    size_t sample_count{0};
    bool has_prefill{false};

    static double now_ns() {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return static_cast<double>(ts.tv_sec) * 1e9 + static_cast<double>(ts.tv_nsec);
    }

    void report() const {
        if (!step_profile_enabled()) {
            return;
        }
        const double total_ms = total_ns / 1e6;
        const double batch_ms = batch_build_ns / 1e6;
        const double sync_ms = host_sync_ns / 1e6;
        const double embed_ms = embedding_ns / 1e6;
        const double graph_ms = graph_exec_ns / 1e6;
        const double sample_ms = sampling_ns / 1e6;
        std::fprintf(
            stderr,
            "[profile step] tokens=%zu samples=%zu %s total=%.3fms batch=%.3fms sync=%.3fms embed=%.3fms "
            "graph=%.3fms sample=%.3fms\n",
            token_count, sample_count, has_prefill ? "prefill" : "decode", total_ms, batch_ms, sync_ms, embed_ms,
            graph_ms, sample_ms
        );
    }
};

constexpr uint32_t kTokenFlagPrefill = 1u << 0;
constexpr uint32_t kTokenFlagDecode = 1u << 1;

constexpr uint64_t kFnvOffsetBasis = 14695981039346656037ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;

struct PendingAppend {
    marmot_request_id_t request_id{0};
    KVPool::AppendPlan plan{};
};

uint64_t fnv1a_update(uint64_t hash, const void *data, size_t len) {
    const auto *bytes = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < len; ++i) {
        hash ^= bytes[i];
        hash *= kFnvPrime;
    }
    return hash;
}

uint64_t hash_seed(const std::string &salt) {
    uint64_t hash = kFnvOffsetBasis;
    if (!salt.empty()) {
        hash = fnv1a_update(hash, salt.data(), salt.size());
    }
    return hash;
}

uint64_t hash_token(uint64_t hash, marmot_token_id_t token) {
    const uint32_t value = static_cast<uint32_t>(token);
    return fnv1a_update(hash, &value, sizeof(value));
}

bool read_i32_kv(const marmot_gguf_t *file, const char *key, int32_t &out_value) {
    if (file == nullptr || key == nullptr) {
        return false;
    }
    const marmot_gguf_kv_t *kv = marmot_gguf_find_kv(file, key);
    if (kv == nullptr) {
        return false;
    }

    switch (kv->value.type) {
    case MARMOT_GGUF_TYPE_INT32:
        out_value = kv->value.data.int32_value;
        return true;
    case MARMOT_GGUF_TYPE_UINT32:
        if (kv->value.data.uint32_value > (uint32_t)INT32_MAX) {
            return false;
        }
        out_value = (int32_t)kv->value.data.uint32_value;
        return true;
    default:
        return false;
    }
}

bool is_visible_special_token(const marmot_gguf_string_t &token) {
    if (token.data == nullptr) {
        return false;
    }
    constexpr char kThink[] = "<think>";
    constexpr char kThinkEnd[] = "</think>";
    constexpr char kEndOfTurn[] = "<end_of_turn>";
    constexpr char kImEnd[] = "<|im_end|>";
    constexpr char kEotId[] = "<|eot_id|>";
    constexpr char kEndOfText[] = "<|end_of_text|>";
    if (token.length == sizeof(kThink) - 1 && std::memcmp(token.data, kThink, sizeof(kThink) - 1) == 0) {
        return true;
    }
    if (token.length == sizeof(kThinkEnd) - 1 && std::memcmp(token.data, kThinkEnd, sizeof(kThinkEnd) - 1) == 0) {
        return true;
    }
    if (token.length == sizeof(kEndOfTurn) - 1 && std::memcmp(token.data, kEndOfTurn, sizeof(kEndOfTurn) - 1) == 0) {
        return true;
    }
    if (token.length == sizeof(kImEnd) - 1 && std::memcmp(token.data, kImEnd, sizeof(kImEnd) - 1) == 0) {
        return true;
    }
    if (token.length == sizeof(kEotId) - 1 && std::memcmp(token.data, kEotId, sizeof(kEotId) - 1) == 0) {
        return true;
    }
    if (token.length == sizeof(kEndOfText) - 1 && std::memcmp(token.data, kEndOfText, sizeof(kEndOfText) - 1) == 0) {
        return true;
    }
    return false;
}

bool is_unused_special_token(const marmot_gguf_string_t &token) {
    if (token.data == nullptr || token.length < 9) {
        return false;
    }
    constexpr char kPrefix[] = "<unused";
    constexpr size_t kPrefixLen = sizeof(kPrefix) - 1;
    if (token.length <= kPrefixLen + 1) {
        return false;
    }
    if (std::memcmp(token.data, kPrefix, kPrefixLen) != 0) {
        return false;
    }
    if (token.data[token.length - 1] != '>') {
        return false;
    }
    for (size_t i = kPrefixLen; i + 1 < token.length; ++i) {
        const char c = token.data[i];
        if (c < '0' || c > '9') {
            return false;
        }
    }
    return true;
}

bool is_hex_digit(char c) {
    if (c >= '0' && c <= '9') {
        return true;
    }
    if (c >= 'a' && c <= 'f') {
        return true;
    }
    return (c >= 'A' && c <= 'F');
}

bool token_equals(const marmot_gguf_string_t &token, const char *value) {
    if (token.data == nullptr || value == nullptr) {
        return false;
    }
    const size_t len = std::strlen(value);
    if (token.length != len) {
        return false;
    }
    return std::memcmp(token.data, value, len) == 0;
}

bool is_named_special_token(const marmot_gguf_string_t &token) {
    return token_equals(token, "<pad>") || token_equals(token, "<unk>") || token_equals(token, "<bos>") ||
        token_equals(token, "<eos>") || token_equals(token, "<s>") || token_equals(token, "</s>") ||
        token_equals(token, "<mask>");
}

bool is_byte_token(const marmot_gguf_string_t &token) {
    if (token.data == nullptr || token.length < 5) {
        return false;
    }
    if (token.data[0] != '<' || token.data[1] != '0' || token.data[2] != 'x' || token.data[token.length - 1] != '>') {
        return false;
    }
    for (size_t i = 3; i + 1 < token.length; ++i) {
        if (!is_hex_digit(token.data[i])) {
            return false;
        }
    }
    return true;
}

bool is_wrapped_token(const marmot_gguf_string_t &token) {
    if (token.data == nullptr || token.length < 2) {
        return false;
    }
    return token.data[0] == '<' && token.data[token.length - 1] == '>';
}

std::vector<uint8_t> build_special_mask(const marmot_gguf_t *file, size_t vocab) {
    if (file == nullptr || vocab == 0) {
        return {};
    }

    const marmot_gguf_kv_t *types_kv = marmot_gguf_find_kv(file, "tokenizer.ggml.token_type");
    const int32_t *types = nullptr;
    size_t types_len = 0;
    if (types_kv != nullptr && types_kv->value.type == MARMOT_GGUF_TYPE_ARRAY &&
        types_kv->value.data.array_value.type == MARMOT_GGUF_TYPE_INT32) {
        types_len = types_kv->value.data.array_value.length;
        if (types_len >= vocab) {
            types = types_kv->value.data.array_value.data.int32_values;
        }
    }

    const marmot_gguf_kv_t *tokens_kv = marmot_gguf_find_kv(file, "tokenizer.ggml.tokens");
    const marmot_gguf_string_t *tokens = nullptr;
    size_t tokens_len = 0;
    if (tokens_kv != nullptr && tokens_kv->value.type == MARMOT_GGUF_TYPE_ARRAY &&
        tokens_kv->value.data.array_value.type == MARMOT_GGUF_TYPE_STRING) {
        tokens_len = tokens_kv->value.data.array_value.length;
        tokens = tokens_kv->value.data.array_value.data.string_values;
    }

    if (types == nullptr && tokens == nullptr) {
        return {};
    }

    constexpr int32_t kTokenTypeNormal = 1;
    constexpr int32_t kTokenTypeByte = 6;

    std::vector<uint8_t> mask(vocab, 0);
    for (size_t i = 0; i < vocab; ++i) {
        bool is_special = false;
        if (types != nullptr) {
            const int32_t type = types[i];
            is_special = (type != kTokenTypeNormal && type != kTokenTypeByte);
        }
        bool allow_special = false;
        if (tokens != nullptr && i < tokens_len) {
            if (!is_special && is_unused_special_token(tokens[i])) {
                is_special = true;
            }
            if (!is_special && is_named_special_token(tokens[i])) {
                is_special = true;
            }
            if (types == nullptr && !is_special && is_wrapped_token(tokens[i]) && !is_byte_token(tokens[i])) {
                is_special = true;
            }
            allow_special = is_visible_special_token(tokens[i]);
        }
        mask[i] = (is_special && !allow_special) ? 1u : 0u;
    }

    auto mark_special_id = [&](const char *key) {
        int32_t id = -1;
        if (!read_i32_kv(file, key, id)) {
            return;
        }
        if (id < 0 || (size_t)id >= mask.size()) {
            return;
        }
        mask[(size_t)id] = 1u;
    };

    mark_special_id("tokenizer.ggml.bos_token_id");
    mark_special_id("tokenizer.ggml.eos_token_id");
    mark_special_id("tokenizer.ggml.unk_token_id");
    mark_special_id("tokenizer.ggml.pad_token_id");

    return mask;
}

void apply_repetition_penalty(
    std::vector<float> &logits, const std::vector<marmot_token_id_t> &history, float penalty,
    std::vector<uint8_t> &seen, std::vector<marmot_token_id_t> &seen_tokens
) {
    if (!(penalty > 1.0f) || history.empty() || logits.empty()) {
        return;
    }

    const size_t vocab = logits.size();
    for (marmot_token_id_t token : history) {
        if (token < 0) {
            continue;
        }
        const size_t idx = (size_t)token;
        if (idx >= vocab) {
            continue;
        }
        if (seen[idx] != 0) {
            continue;
        }
        seen[idx] = 1;
        seen_tokens.push_back(token);
        float &logit = logits[idx];
        logit = (logit >= 0.0f) ? (logit / penalty) : (logit * penalty);
    }

    for (marmot_token_id_t token : seen_tokens) {
        const size_t idx = (size_t)token;
        if (idx < vocab) {
            seen[idx] = 0;
        }
    }
    seen_tokens.clear();
}

} // namespace

std::unique_ptr<ServingEngine> ServingEngine::create(
    const marmot_context_t *ctx, std::shared_ptr<const Model> model, const marmot_serving_engine_options_t &opts,
    marmot_error_t &status, std::string &error
) {
    status = MARMOT_ERROR_INVALID_OPERATION;
    error.clear();

    if (ctx == nullptr || model == nullptr) {
        status = MARMOT_ERROR_INVALID_ARGUMENT;
        error = "serving engine requires non-null context and model";
        return nullptr;
    }
    if (opts.max_seqs == 0 || opts.max_batch_seqs == 0 || opts.max_num_tokens == 0 || opts.max_seq_len == 0) {
        status = MARMOT_ERROR_INVALID_ARGUMENT;
        error = "serving engine limits must be non-zero";
        return nullptr;
    }
    if (opts.max_batch_seqs > opts.max_num_tokens) {
        status = MARMOT_ERROR_INVALID_ARGUMENT;
        error = "serving engine max_batch_seqs must be <= max_num_tokens";
        return nullptr;
    }
    if (opts.block_size <= 1 || !std::has_single_bit(opts.block_size)) {
        status = MARMOT_ERROR_INVALID_ARGUMENT;
        error = "serving engine block_size must be power-of-two > 1";
        return nullptr;
    }
    if (opts.num_kv_blocks == 0) {
        status = MARMOT_ERROR_INVALID_ARGUMENT;
        error = "serving engine num_kv_blocks must be non-zero";
        return nullptr;
    }
    if ((opts.flags & MARMOT_SERVING_ENGINE_FLAG_ENABLE_SWAP) != 0 && opts.num_swap_blocks == 0) {
        status = MARMOT_ERROR_INVALID_ARGUMENT;
        error = "serving engine num_swap_blocks must be non-zero when swap is enabled";
        return nullptr;
    }

    const marmot_model_info_t &info = model->info();
    if (info.n_layer == 0 || info.n_head == 0 || info.n_head_kv == 0 || info.n_embd == 0) {
        status = MARMOT_ERROR_INVALID_ARGUMENT;
        error = "serving engine requires model attention dimensions";
        return nullptr;
    }
    if (info.n_embd % info.n_head != 0) {
        status = MARMOT_ERROR_INVALID_ARGUMENT;
        error = "serving engine requires n_embd divisible by n_head";
        return nullptr;
    }
    if (info.n_vocab == 0) {
        status = MARMOT_ERROR_INVALID_ARGUMENT;
        error = "serving engine requires non-zero vocab size";
        return nullptr;
    }

    marmot_serving_engine_options_t normalized_opts = opts;
    if (normalized_opts.prefill_chunk_size == 0) {
        if (normalized_opts.max_seqs == 1 && normalized_opts.max_batch_seqs == 1) {
            normalized_opts.prefill_chunk_size = normalized_opts.max_num_tokens;
        } else {
            normalized_opts.prefill_chunk_size = normalized_opts.block_size;
        }
    }
    if (!std::isfinite(normalized_opts.kv_block_watermark) || normalized_opts.kv_block_watermark < 0.0f ||
        normalized_opts.kv_block_watermark >= 1.0f) {
        status = MARMOT_ERROR_INVALID_ARGUMENT;
        error = "serving engine kv_block_watermark must be in [0, 1)";
        return nullptr;
    }
    if (info.context_length > 0) {
        normalized_opts.max_seq_len = std::min(normalized_opts.max_seq_len, info.context_length);
    }
    if (normalized_opts.max_num_tokens > normalized_opts.max_seq_len) {
        status = MARMOT_ERROR_INVALID_ARGUMENT;
        error = "serving engine max_num_tokens must be <= max_seq_len";
        return nullptr;
    }

    const marmot_tensor_t *embed = marmot_gguf_model_tensor(model->gguf(), "token_embd.weight");
    if (embed == nullptr) {
        status = MARMOT_ERROR_INVALID_ARGUMENT;
        error = "missing token_embd.weight tensor";
        return nullptr;
    }

    marmot_gguf_model_meta_t meta{};
    if (!marmot_gguf_model_metadata(model->gguf(), &meta)) {
        status = MARMOT_ERROR_INVALID_ARGUMENT;
        error = "failed to load model metadata";
        return nullptr;
    }

    const marmot_dtype_t activation_dtype =
        marmot_activation_dtype_for_architecture(meta.architecture, ctx->backend_type);
    bool kv_is_fp8 = false;
#if MARMOT_ENABLE_FP8
    kv_is_fp8 = normalized_opts.kv_dtype == MARMOT_DTYPE_FLOAT8_E4M3;
    if (normalized_opts.kv_dtype == MARMOT_DTYPE_FLOAT8_E5M2) {
        status = MARMOT_ERROR_UNSUPPORTED_DTYPE;
        error = "serving engine FP8 E5M2 KV not supported";
        return nullptr;
    }
#endif
    if (kv_is_fp8) {
        if (ctx->backend_type != MARMOT_BACKEND_CPU) {
            status = MARMOT_ERROR_UNSUPPORTED_DTYPE;
            error = "serving engine FP8 KV only supported on CPU backend";
            return nullptr;
        }
    } else if (normalized_opts.kv_dtype != activation_dtype) {
        status = MARMOT_ERROR_UNSUPPORTED_DTYPE;
        error = "serving engine kv_dtype must match activation dtype";
        return nullptr;
    }

    KVPool::Options kv_opts{};
    kv_opts.backend = ctx->backend_type;
    kv_opts.max_seqs = normalized_opts.max_seqs;
    kv_opts.max_seq_len = normalized_opts.max_seq_len;
    kv_opts.block_size = normalized_opts.block_size;
    kv_opts.num_blocks = normalized_opts.num_kv_blocks;
    kv_opts.num_layers = info.n_layer;
    kv_opts.num_kv_heads = info.n_head_kv;
    kv_opts.head_dim = meta.head_dim;
    kv_opts.kv_dtype = normalized_opts.kv_dtype;
    kv_opts.num_swap_blocks =
        (normalized_opts.flags & MARMOT_SERVING_ENGINE_FLAG_ENABLE_SWAP) != 0 ? normalized_opts.num_swap_blocks : 0;
    kv_opts.deterministic_alloc = (normalized_opts.flags & MARMOT_SERVING_ENGINE_FLAG_DETERMINISTIC_KV) != 0;

    auto pool_result = KVPool::create(ctx, kv_opts);
    if (!pool_result) {
        status = pool_result.error();
        error = "failed to create KV pool";
        return nullptr;
    }

    auto engine = std::unique_ptr<ServingEngine>(
        new (std::nothrow) ServingEngine(ctx, std::move(model), normalized_opts, std::move(*pool_result))
    );
    if (!engine) {
        status = MARMOT_ERROR_OUT_OF_MEMORY;
        error = "failed to allocate serving engine";
        return nullptr;
    }

    engine->block_hash_by_id_.assign(normalized_opts.num_kv_blocks, 0);
    engine->block_hash_valid_.assign(normalized_opts.num_kv_blocks, 0);

    const marmot_gguf_t *gguf_file = marmot_gguf_model_file(engine->model_->gguf());
    int32_t bos_id = (int32_t)MARMOT_TOKEN_ID_INVALID;
    if (read_i32_kv(gguf_file, "tokenizer.ggml.bos_token_id", bos_id)) {
        engine->bos_id_ = (marmot_token_id_t)bos_id;
    }
    int32_t eos_id = (int32_t)MARMOT_TOKEN_ID_INVALID;
    if (read_i32_kv(gguf_file, "tokenizer.ggml.eos_token_id", eos_id)) {
        engine->eos_id_ = (marmot_token_id_t)eos_id;
    }

    engine->token_embedding_ = embed;
    engine->embedding_scale_ = meta.embedding_scale;
    engine->activation_dtype_ = activation_dtype;
    engine->max_seq_len_ = normalized_opts.max_seq_len;
    engine->n_vocab_ = info.n_vocab;
    engine->n_embd_ = info.n_embd;
    engine->special_mask_ = build_special_mask(gguf_file, info.n_vocab);

    status = MARMOT_SUCCESS;
    return engine;
}

ServingEngine::~ServingEngine() = default;

ServingEngine::ServingEngine(
    const marmot_context_t *ctx, std::shared_ptr<const Model> model, const marmot_serving_engine_options_t &opts,
    std::unique_ptr<KVPool> kv_pool
)
    : ctx_(ctx), model_(std::move(model)), opts_(opts), kv_pool_(std::move(kv_pool)) {}

bool ServingEngine::can_accept_request(size_t additional) const noexcept {
    if (opts_.max_seqs == 0) {
        return true;
    }

    size_t active = 0;
    for (const auto &kv : requests_) {
        const Request &req = kv.second;
        const marmot_llm_request_state_t state = req.state;
        if (state == MARMOT_LLM_REQUEST_STATE_PENDING || state == MARMOT_LLM_REQUEST_STATE_PREFILL ||
            state == MARMOT_LLM_REQUEST_STATE_DECODING) {
            if (!req.has_seq_slot) {
                continue;
            }
            active++;
        }
    }

    if (active >= opts_.max_seqs) {
        return false;
    }
    return (opts_.max_seqs - active) >= additional;
}

marmot_error_t
ServingEngine::parse_request_ext(const marmot_llm_generate_options_t &gen_opts, Request &req, std::string &error) {
    error.clear();
    if (gen_opts.pnext == nullptr) {
        return MARMOT_SUCCESS;
    }

    const auto *ext = static_cast<const marmot_serving_request_ext_t *>(gen_opts.pnext);
    if (ext->struct_version != MARMOT_SERVING_REQUEST_EXT_VERSION) {
        error = "invalid serving request extension version";
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (ext->struct_size < sizeof(marmot_serving_request_ext_t)) {
        error = "invalid serving request extension size";
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    req.flags = ext->flags;
    req.priority = ext->priority;
    req.retention_blocks = ext->retention_blocks;
    req.num_samples = (ext->num_samples == 0) ? 1 : ext->num_samples;
    if (ext->cache_salt != nullptr && ext->cache_salt_len > 0) {
        req.cache_salt.assign(ext->cache_salt, ext->cache_salt_len);
    }

    return MARMOT_SUCCESS;
}

marmot_error_t ServingEngine::ensure_scratch(size_t token_count, size_t sample_count, std::string &error) {
    error.clear();
    if (ctx_ == nullptr || token_embedding_ == nullptr) {
        return MARMOT_ERROR_INVALID_OPERATION;
    }
    if (token_count == 0 || n_embd_ == 0 || n_vocab_ == 0 || opts_.max_num_tokens == 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (token_count > opts_.max_num_tokens) {
        error = "token_count exceeds serving engine capacity";
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (sample_count > opts_.max_batch_seqs) {
        error = "sample_count exceeds serving engine capacity";
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const size_t token_capacity = opts_.max_num_tokens;

    if (scratch_token_capacity_ != token_capacity || !token_ids_ || token_ids_->dtype != MARMOT_DTYPE_INT32) {
        size_t shape[1] = {token_capacity};
        token_ids_.reset(marmot_tensor_create(ctx_, shape, 1, MARMOT_DTYPE_INT32));
        if (!token_ids_) {
            error = "failed to allocate token_ids";
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        token_ids_device_ready_ = false;
    }

    if (scratch_token_capacity_ != token_capacity || !positions_ || positions_->dtype != MARMOT_DTYPE_FLOAT32) {
        size_t shape[1] = {token_capacity};
        positions_.reset(marmot_tensor_create(ctx_, shape, 1, MARMOT_DTYPE_FLOAT32));
        if (!positions_) {
            error = "failed to allocate positions";
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
    }

    if (scratch_token_capacity_ != token_capacity || !positions_alt_ || positions_alt_->dtype != MARMOT_DTYPE_FLOAT32) {
        size_t shape[1] = {token_capacity};
        positions_alt_.reset(marmot_tensor_create(ctx_, shape, 1, MARMOT_DTYPE_FLOAT32));
        if (!positions_alt_) {
            error = "failed to allocate positions_alt";
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
    }

    if (scratch_token_capacity_ != token_capacity || !token_meta_ || token_meta_->dtype != MARMOT_DTYPE_UINT32) {
        size_t shape[2] = {token_capacity, 4};
        token_meta_.reset(marmot_tensor_create(ctx_, shape, 2, MARMOT_DTYPE_UINT32));
        if (!token_meta_) {
            error = "failed to allocate token_meta";
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
    }

    if (scratch_token_capacity_ != token_capacity || !token_meta_alt_ ||
        token_meta_alt_->dtype != MARMOT_DTYPE_UINT32) {
        size_t shape[2] = {token_capacity, 4};
        token_meta_alt_.reset(marmot_tensor_create(ctx_, shape, 2, MARMOT_DTYPE_UINT32));
        if (!token_meta_alt_) {
            error = "failed to allocate token_meta_alt";
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
    }

    if (scratch_token_capacity_ != token_capacity || !hidden_ || hidden_->dtype != activation_dtype_) {
        size_t shape[2] = {token_capacity, n_embd_};
        hidden_.reset(marmot_tensor_create(ctx_, shape, 2, activation_dtype_));
        if (!hidden_) {
            error = "failed to allocate hidden";
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
    }

    if (scratch_token_capacity_ != token_capacity || !hidden_out_ || hidden_out_->dtype != activation_dtype_) {
        size_t shape[2] = {token_capacity, n_embd_};
        hidden_out_.reset(marmot_tensor_create(ctx_, shape, 2, activation_dtype_));
        if (!hidden_out_) {
            error = "failed to allocate hidden_out";
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
    }

    scratch_token_capacity_ = token_capacity;

    if (sample_count > 0) {
        if (opts_.max_batch_seqs == 0) {
            error = "sample_count requires max_batch_seqs capacity";
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        const size_t sample_capacity = opts_.max_batch_seqs;

        if (scratch_sample_capacity_ != sample_capacity || !sample_indices_ ||
            sample_indices_->dtype != MARMOT_DTYPE_UINT32) {
            size_t shape[1] = {sample_capacity};
            sample_indices_.reset(marmot_tensor_create(ctx_, shape, 1, MARMOT_DTYPE_UINT32));
            if (!sample_indices_) {
                error = "failed to allocate sample_indices";
                return MARMOT_ERROR_OUT_OF_MEMORY;
            }
        }

        if (scratch_sample_capacity_ != sample_capacity || !sample_indices_alt_ ||
            sample_indices_alt_->dtype != MARMOT_DTYPE_UINT32) {
            size_t shape[1] = {sample_capacity};
            sample_indices_alt_.reset(marmot_tensor_create(ctx_, shape, 1, MARMOT_DTYPE_UINT32));
            if (!sample_indices_alt_) {
                error = "failed to allocate sample_indices_alt";
                return MARMOT_ERROR_OUT_OF_MEMORY;
            }
        }

        if (scratch_sample_capacity_ != sample_capacity || !logits_ || logits_->dtype != activation_dtype_) {
            size_t shape[2] = {sample_capacity, n_vocab_};
            logits_.reset(marmot_tensor_create(ctx_, shape, 2, activation_dtype_));
            if (!logits_) {
                error = "failed to allocate logits";
                return MARMOT_ERROR_OUT_OF_MEMORY;
            }
        }

        if (scratch_sample_capacity_ != sample_capacity || !logits_max_ || logits_max_->dtype != activation_dtype_) {
            size_t shape[1] = {sample_capacity};
            logits_max_.reset(marmot_tensor_create(ctx_, shape, 1, activation_dtype_));
            if (!logits_max_) {
                error = "failed to allocate logits_max";
                return MARMOT_ERROR_OUT_OF_MEMORY;
            }
        }

        if (scratch_sample_capacity_ != sample_capacity || !logits_max_alt_ ||
            logits_max_alt_->dtype != activation_dtype_) {
            size_t shape[1] = {sample_capacity};
            logits_max_alt_.reset(marmot_tensor_create(ctx_, shape, 1, activation_dtype_));
            if (!logits_max_alt_) {
                error = "failed to allocate logits_max_alt";
                return MARMOT_ERROR_OUT_OF_MEMORY;
            }
        }

        if (scratch_sample_capacity_ != sample_capacity || !logits_argmax_ ||
            logits_argmax_->dtype != MARMOT_DTYPE_UINT64) {
            size_t shape[1] = {sample_capacity};
            logits_argmax_.reset(marmot_tensor_create(ctx_, shape, 1, MARMOT_DTYPE_UINT64));
            if (!logits_argmax_) {
                error = "failed to allocate logits_argmax";
                return MARMOT_ERROR_OUT_OF_MEMORY;
            }
        }

        if (scratch_sample_capacity_ != sample_capacity || !logits_argmax_alt_ ||
            logits_argmax_alt_->dtype != MARMOT_DTYPE_UINT64) {
            size_t shape[1] = {sample_capacity};
            logits_argmax_alt_.reset(marmot_tensor_create(ctx_, shape, 1, MARMOT_DTYPE_UINT64));
            if (!logits_argmax_alt_) {
                error = "failed to allocate logits_argmax_alt";
                return MARMOT_ERROR_OUT_OF_MEMORY;
            }
        }

        scratch_sample_capacity_ = sample_capacity;
    }

    if (logits_f32_.size() != n_vocab_) {
        logits_f32_.assign(n_vocab_, 0.0f);
        seen_.assign(n_vocab_, 0);
        seen_tokens_.clear();
    }

    return MARMOT_SUCCESS;
}

marmot_error_t ServingEngine::ensure_packed_graph(
    size_t token_count, size_t sample_count, marmot_graph_t **out_graph, const graph::FastPlan **out_fast_plan,
    std::string &error
) {
    error.clear();
    if (ctx_ == nullptr || model_ == nullptr || out_graph == nullptr) {
        return MARMOT_ERROR_INVALID_OPERATION;
    }
    *out_graph = nullptr;
    if (out_fast_plan != nullptr) {
        *out_fast_plan = nullptr;
    }

    if (token_count == 0 || token_count > opts_.max_num_tokens || token_count > UINT32_MAX ||
        sample_count > UINT32_MAX) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const bool emit_logits = sample_count > 0;
    if (emit_logits && sample_count > token_count) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    PackedGraphKey key{
        .token_count = static_cast<uint32_t>(token_count),
        .sample_count = static_cast<uint32_t>(emit_logits ? sample_count : 0),
        .emit_logits = emit_logits,
    };

    auto found = packed_graphs_.find(key);
    if (found != packed_graphs_.end() && found->second.graph) {
        *out_graph = found->second.graph.get();
        if (out_fast_plan != nullptr) {
            *out_fast_plan = found->second.fast_plan.get();
        }
        return MARMOT_SUCCESS;
    }
    if (emit_logits && opts_.max_batch_seqs == 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_packed_graph_options_t opts{};
    marmot_error_t init_status = marmot_packed_graph_options_init(&opts);
    if (init_status != MARMOT_SUCCESS) {
        return init_status;
    }
    opts.token_count = token_count;
    opts.sample_count = emit_logits ? sample_count : 0;
    opts.max_seqs = opts_.max_seqs;
    opts.max_seq_len = max_seq_len_;
    opts.block_size = opts_.block_size;
    opts.num_kv_blocks = opts_.num_kv_blocks;
    opts.kv_dtype = opts_.kv_dtype;

    marmot_graph_t *graph_raw = nullptr;
    marmot_error_t status = marmot_graph_from_model_packed(model_->gguf(), ctx_->backend_type, &opts, &graph_raw);
    if (status != MARMOT_SUCCESS || graph_raw == nullptr) {
        const char *detail = marmot_get_last_error_detail();
        if (detail != nullptr && detail[0] != '\0') {
            error = detail;
        } else {
            error = "failed to build packed graph";
        }
        return status != MARMOT_SUCCESS ? status : MARMOT_ERROR_INVALID_OPERATION;
    }

    PackedGraphEntry entry{};
    entry.graph.reset(graph_raw);
    prepack_cpu_model_weights(ctx_, model_->gguf(), emit_logits);

    graph::FastPlanBucket bucket{
        .token_count = key.token_count,
        .sample_count = key.sample_count,
        .emit_logits = key.emit_logits,
    };
    if (auto fast_plan = graph::compile_fast_plan(graph_raw, bucket)) {
        entry.fast_plan = std::make_unique<graph::FastPlan>(std::move(*fast_plan));
    } else {
        marmot_clear_error();
    }

    packed_graphs_.insert_or_assign(key, std::move(entry));
    auto inserted = packed_graphs_.find(key);
    if (inserted == packed_graphs_.end() || !inserted->second.graph) {
        error = "failed to cache packed graph";
        return MARMOT_ERROR_INVALID_OPERATION;
    }
    *out_graph = inserted->second.graph.get();
    if (out_fast_plan != nullptr) {
        *out_fast_plan = inserted->second.fast_plan.get();
    }
    return MARMOT_SUCCESS;
}

marmot_error_t ServingEngine::submit(
    const marmot_token_id_t *prompt_tokens, size_t prompt_len, const marmot_llm_generate_options_t &gen_opts,
    const marmot_llm_sampling_options_t &sampling_opts, marmot_request_id_t &out_request_id, std::string &error
) {
    error.clear();
    if (ctx_ == nullptr || model_ == nullptr || kv_pool_ == nullptr) {
        return MARMOT_ERROR_INVALID_OPERATION;
    }
    if (prompt_tokens == nullptr && prompt_len != 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const auto *ext =
        (gen_opts.pnext != nullptr) ? static_cast<const marmot_serving_request_ext_t *>(gen_opts.pnext) : nullptr;

    Request base{};
    base.state = MARMOT_LLM_REQUEST_STATE_PENDING;
    base.gen_opts = gen_opts;
    base.sampling_opts = sampling_opts;

    if (prompt_len > 0) {
        base.prompt_tokens.assign(prompt_tokens, prompt_tokens + prompt_len);
    } else {
        marmot_token_id_t bos = bos_id_ != MARMOT_TOKEN_ID_INVALID ? bos_id_ : (marmot_token_id_t)0;
        base.prompt_tokens.push_back(bos);
        prompt_len = 1;
    }

    if (max_seq_len_ > 0 && prompt_len > max_seq_len_) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (gen_opts.stop_tokens != nullptr && gen_opts.stop_tokens_len > 0) {
        base.stop_tokens.assign(gen_opts.stop_tokens, gen_opts.stop_tokens + gen_opts.stop_tokens_len);
        base.gen_opts.stop_tokens = base.stop_tokens.data();
        base.gen_opts.stop_tokens_len = base.stop_tokens.size();
    } else {
        base.gen_opts.stop_tokens = nullptr;
        base.gen_opts.stop_tokens_len = 0;
    }

    marmot_error_t ext_status = parse_request_ext(gen_opts, base, error);
    if (ext_status != MARMOT_SUCCESS) {
        return ext_status;
    }

    size_t num_samples = base.num_samples;
    if (num_samples == 0) {
        num_samples = 1;
    }
    if (num_samples > 1 && base.gen_opts.max_new_tokens == 0) {
        error = "num_samples requires max_new_tokens";
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (num_samples > 1) {
        if (ext == nullptr || ext->out_request_ids == nullptr || ext->out_request_ids_capacity < num_samples) {
            error = "num_samples requires out_request_ids";
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        if (ext->sample_user_data != nullptr && ext->sample_user_data_len < num_samples) {
            error = "sample_user_data_len is too small";
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
    }

    if (!can_accept_request(num_samples)) {
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    std::vector<marmot_seq_slot_t> seq_slots;
    seq_slots.reserve(num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
        marmot_seq_slot_t seq_slot = 0;
        marmot_error_t seq_status = kv_pool_->acquire_seq(seq_slot);
        if (seq_status != MARMOT_SUCCESS) {
            for (marmot_seq_slot_t slot : seq_slots) {
                (void)kv_pool_->release_seq(slot);
            }
            return seq_status;
        }
        seq_slots.push_back(seq_slot);
    }

    base.allowed_special_mask.clear();
    if (base.gen_opts.stop_tokens != nullptr && base.gen_opts.stop_tokens_len > 0 && n_vocab_ > 0) {
        base.allowed_special_mask.assign(n_vocab_, 0);
        for (size_t i = 0; i < base.gen_opts.stop_tokens_len; ++i) {
            const marmot_token_id_t token = base.gen_opts.stop_tokens[i];
            if (token < 0) {
                continue;
            }
            const size_t idx = (size_t)token;
            if (idx < n_vocab_) {
                base.allowed_special_mask[idx] = 1u;
            }
        }
    }

    auto seed_for_sample = [](uint64_t seed, size_t sample_index) -> uint64_t {
        uint64_t value = seed + static_cast<uint64_t>(sample_index);
        value += 0x9e3779b97f4a7c15ull;
        value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
        value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
        return value ^ (value >> 31);
    };

    auto init_request = [&](Request &req, size_t sample_index, marmot_request_id_t parent_id,
                            marmot_seq_slot_t seq_slot) {
        req.state = MARMOT_LLM_REQUEST_STATE_PENDING;
        req.gen_opts = base.gen_opts;
        req.sampling_opts = base.sampling_opts;
        req.stop_tokens = base.stop_tokens;
        req.gen_opts.stop_tokens = req.stop_tokens.empty() ? nullptr : req.stop_tokens.data();
        req.gen_opts.stop_tokens_len = req.stop_tokens.size();
        req.prompt_tokens = base.prompt_tokens;
        req.allowed_special_mask = base.allowed_special_mask;
        req.flags = base.flags;
        req.priority = base.priority;
        req.cache_salt = base.cache_salt;
        req.retention_blocks = base.retention_blocks;
        req.num_samples = (sample_index == 0) ? num_samples : 1;
        req.sample_index = sample_index;
        req.parent_id = parent_id;
        req.seq_slot = seq_slot;
        req.has_seq_slot = true;
        req.prompt_cursor = 0;
        req.pending_input_token = MARMOT_TOKEN_ID_INVALID;
        req.has_pending_token = false;
        req.awaiting_sample = false;
        req.awaiting_prefix_attach = (sample_index != 0);
        req.swapped_out = false;
        req.needs_recompute = false;
        req.prefix_hashes.clear();
        req.generated_tokens.clear();
        req.recompute_tokens.clear();
        req.clone_ids.clear();

        uint64_t seed = base.sampling_opts.seed;
        if (sample_index != 0) {
            seed = seed_for_sample(seed, sample_index);
        }
        req.sampling_opts.seed = seed;
        req.rng.seed(seed);
        if (ext != nullptr && ext->sample_user_data != nullptr && sample_index < ext->sample_user_data_len) {
            req.gen_opts.user_data = ext->sample_user_data[sample_index];
        }
    };

    std::vector<marmot_request_id_t> request_ids;
    request_ids.reserve(num_samples);

    Request parent{};
    parent.id = next_request_id_++;
    init_request(parent, 0, 0, seq_slots[0]);

    if ((opts_.flags & MARMOT_SERVING_ENGINE_FLAG_ENABLE_PREFIX_CACHE) != 0 &&
        (parent.flags & MARMOT_SERVING_REQUEST_FLAG_DISABLE_PREFIX_CACHE) == 0 && opts_.block_size > 0) {
        const size_t block_size = opts_.block_size;
        size_t full_blocks = prompt_len / block_size;
        if (parent.gen_opts.max_new_tokens != 0 && prompt_len % block_size == 0 && full_blocks > 0) {
            full_blocks -= 1;
        }

        std::vector<marmot_block_id_t> block_ids;
        if (full_blocks > 0) {
            block_ids.reserve(full_blocks);
            parent.prefix_hashes.clear();
            parent.prefix_hashes.reserve(full_blocks);

            uint64_t hash = hash_seed(parent.cache_salt);
            for (size_t block = 0; block < full_blocks; ++block) {
                const size_t base_idx = block * block_size;
                for (size_t i = 0; i < block_size; ++i) {
                    hash = hash_token(hash, parent.prompt_tokens[base_idx + i]);
                }

                auto it = prefix_cache_.find(hash);
                if (it == prefix_cache_.end()) {
                    break;
                }

                const PrefixEntry entry = it->second;
                const uint32_t generation = kv_pool_->block_generation(entry.block_id);
                if (generation == 0 || generation != entry.generation || entry.block_id == MARMOT_BLOCK_ID_INVALID) {
                    prefix_cache_.erase(it);
                    if (entry.block_id < block_hash_valid_.size() && block_hash_valid_[entry.block_id] != 0 &&
                        block_hash_by_id_[entry.block_id] == hash) {
                        block_hash_valid_[entry.block_id] = 0;
                    }
                    break;
                }

                it->second.last_use = ++prefix_tick_;
                block_ids.push_back(entry.block_id);
                parent.prefix_hashes.push_back(hash);
            }
        }

        if (!block_ids.empty()) {
            KVPool::PrefixPlan plan{};
            const size_t prefix_len = block_ids.size() * block_size;
            marmot_error_t attach_status =
                kv_pool_->prepare_prefix_attach(parent.seq_slot, block_ids.data(), block_ids.size(), prefix_len, plan);
            if (attach_status == MARMOT_SUCCESS) {
                attach_status = kv_pool_->commit_prefix_attach(plan);
            }
            if (attach_status != MARMOT_SUCCESS) {
                kv_pool_->abort_prefix_attach(plan);
                parent.prefix_hashes.clear();
            } else {
                parent.prompt_cursor = block_ids.size() * block_size;
                if (parent.retention_blocks > 0) {
                    const size_t retain_blocks = std::min(parent.retention_blocks, block_ids.size());
                    for (size_t i = 0; i < retain_blocks; ++i) {
                        kv_pool_->set_block_retained(block_ids[i], true);
                    }
                }
            }
        }
    }

    if (parent.prompt_cursor >= prompt_len && parent.gen_opts.max_new_tokens == 0) {
        if (parent.has_seq_slot && kv_pool_ != nullptr) {
            (void)kv_pool_->release_seq(parent.seq_slot);
            parent.has_seq_slot = false;
        }
        parent.state = MARMOT_LLM_REQUEST_STATE_DONE;
    }

    const marmot_request_id_t parent_id = parent.id;
    requests_.insert_or_assign(parent_id, std::move(parent));
    schedule_.push_back(parent_id);
    request_ids.push_back(parent_id);

    for (size_t i = 1; i < num_samples; ++i) {
        Request clone{};
        clone.id = next_request_id_++;
        init_request(clone, i, request_ids[0], seq_slots[i]);
        requests_.insert_or_assign(clone.id, std::move(clone));
        schedule_.push_back(clone.id);
        request_ids.push_back(clone.id);
    }

    if (num_samples > 1) {
        auto it = requests_.find(request_ids[0]);
        if (it != requests_.end()) {
            it->second.clone_ids.assign(request_ids.begin() + 1, request_ids.end());
        }
    }

    if (ext != nullptr && ext->out_request_ids != nullptr && ext->out_request_ids_capacity > 0) {
        const size_t count = std::min(ext->out_request_ids_capacity, request_ids.size());
        for (size_t i = 0; i < count; ++i) {
            ext->out_request_ids[i] = request_ids[i];
        }
    }

    out_request_id = request_ids.empty() ? 0 : request_ids[0];
    return MARMOT_SUCCESS;
}

marmot_error_t
ServingEngine::step_pipelined_greedy_decode(size_t max_steps, size_t &out_steps_done, std::string &error) {
    out_steps_done = 0;
    error.clear();

    if (ctx_ == nullptr || kv_pool_ == nullptr) {
        return MARMOT_ERROR_INVALID_OPERATION;
    }
    static bool disabled = [] {
        const char *env = std::getenv("MARMOT_DISABLE_PIPELINED_GREEDY");
        return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    if (disabled) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (max_steps == 0) {
        return MARMOT_SUCCESS;
    }
    if (ctx_->backend_type != MARMOT_BACKEND_METAL) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    auto greedy_device_argmax_supported = [&](const Request &req) {
        const marmot_llm_sampling_options_t &sampling = req.sampling_opts;
        if (sampling.temperature > 0.0f) {
            return false;
        }
        if (sampling.top_k > 1) {
            return false;
        }
        if (sampling.repetition_penalty != 1.0f) {
            return false;
        }
        if ((sampling.flags & MARMOT_LLM_SAMPLING_FLAG_SUPPRESS_SPECIAL_TOKENS) != 0) {
            return false;
        }
        return true;
    };

    auto argmax_buffers_for = [&](uint32_t buffer_index, marmot_tensor_t **out_values, marmot_tensor_t **out_indices) {
        if (out_values == nullptr || out_indices == nullptr) {
            return false;
        }
        if (buffer_index == 0) {
            *out_values = logits_max_.get();
            *out_indices = logits_argmax_.get();
            return *out_values != nullptr && *out_indices != nullptr;
        }
        if (buffer_index == 1) {
            *out_values = logits_max_alt_.get();
            *out_indices = logits_argmax_alt_.get();
            return *out_values != nullptr && *out_indices != nullptr;
        }
        return false;
    };

    auto finished_for_token = [&](const Request &req, marmot_token_id_t token) {
        if (req.gen_opts.stop_on_eos && eos_id_ != MARMOT_TOKEN_ID_INVALID && token == eos_id_) {
            return true;
        }
        if (req.gen_opts.stop_tokens != nullptr && req.gen_opts.stop_tokens_len > 0) {
            for (size_t j = 0; j < req.gen_opts.stop_tokens_len; ++j) {
                if (req.gen_opts.stop_tokens[j] == token) {
                    return true;
                }
            }
        }
        if (req.gen_opts.max_new_tokens > 0 && req.generated_tokens.size() + 1 >= req.gen_opts.max_new_tokens) {
            return true;
        }
        return false;
    };

    auto submit_decode_token = [&](Request &req, uint32_t argmax_buffer_index) -> marmot_error_t {
        error.clear();

        if (req.state != MARMOT_LLM_REQUEST_STATE_DECODING || req.cancel_requested || !req.has_pending_token ||
            !req.has_seq_slot) {
            return MARMOT_ERROR_NOT_IMPLEMENTED;
        }
        if (!greedy_device_argmax_supported(req)) {
            return MARMOT_ERROR_NOT_IMPLEMENTED;
        }

        marmot_kv_slot_t slot = 0;
        size_t start_pos = 0;
        KVPool::AppendPlan plan{};
        marmot_error_t status = kv_pool_->prepare_append(req.seq_slot, 1, &slot, start_pos, plan);
        if (status != MARMOT_SUCCESS) {
            return status;
        }

        PackedBatch batch{};
        batch.token_ids.push_back(req.pending_input_token);
        batch.sample_indices.push_back(0u);
        batch.sample_request_ids.push_back(req.id);
        batch.token_meta.push_back(static_cast<uint32_t>(req.seq_slot));
        batch.token_meta.push_back(static_cast<uint32_t>(start_pos));
        batch.token_meta.push_back(static_cast<uint32_t>(slot));
        batch.token_meta.push_back(kTokenFlagDecode);

        marmot_error_t scratch_status = ensure_scratch(1, 1, error);
        if (scratch_status != MARMOT_SUCCESS) {
            kv_pool_->abort_append(plan);
            return scratch_status;
        }

        marmot_tensor_t *argmax_values = nullptr;
        marmot_tensor_t *argmax_indices = nullptr;
        if (!argmax_buffers_for(argmax_buffer_index, &argmax_values, &argmax_indices)) {
            kv_pool_->abort_append(plan);
            error = "argmax buffers unavailable";
            return MARMOT_ERROR_INVALID_OPERATION;
        }

        const size_t token_count = 1;
        const size_t sample_count = 1;
        const bool use_device_token_ids =
            token_ids_device_ready_ && ctx_->backend_type != MARMOT_BACKEND_CPU && sample_count == token_count;
        token_ids_device_ready_ = false;

        const size_t graph_token_count = 1;
        const size_t graph_sample_count = 1;

        token_ids_->shape.ndim = 1;
        token_ids_->shape.shape[0] = graph_token_count;
        token_ids_->shape.strides[0] = 1;

        marmot_tensor_t *positions = (argmax_buffer_index == 0) ? positions_.get() : positions_alt_.get();
        marmot_tensor_t *token_meta = (argmax_buffer_index == 0) ? token_meta_.get() : token_meta_alt_.get();
        marmot_tensor_t *sample_indices =
            (argmax_buffer_index == 0) ? sample_indices_.get() : sample_indices_alt_.get();
        if (positions == nullptr || token_meta == nullptr || sample_indices == nullptr) {
            kv_pool_->abort_append(plan);
            error = "pipelined scratch tensors unavailable";
            return MARMOT_ERROR_INVALID_OPERATION;
        }

        positions->shape.ndim = 1;
        positions->shape.shape[0] = graph_token_count;
        positions->shape.strides[0] = 1;

        token_meta->shape.ndim = 2;
        token_meta->shape.shape[0] = token_count;
        token_meta->shape.shape[1] = 4;
        token_meta->shape.strides[1] = 1;
        token_meta->shape.strides[0] = 4;

        hidden_->shape.ndim = 2;
        hidden_->shape.shape[0] = graph_token_count;
        hidden_->shape.shape[1] = n_embd_;
        hidden_->shape.strides[1] = 1;
        hidden_->shape.strides[0] = n_embd_;

        hidden_out_->shape.ndim = 2;
        hidden_out_->shape.shape[0] = graph_token_count;
        hidden_out_->shape.shape[1] = n_embd_;
        hidden_out_->shape.strides[1] = 1;
        hidden_out_->shape.strides[0] = n_embd_;

        sample_indices->shape.ndim = 1;
        sample_indices->shape.shape[0] = graph_sample_count;
        sample_indices->shape.strides[0] = 1;

        logits_->shape.ndim = 2;
        logits_->shape.shape[0] = graph_sample_count;
        logits_->shape.shape[1] = n_vocab_;
        logits_->shape.strides[1] = 1;
        logits_->shape.strides[0] = n_vocab_;

        argmax_values->shape.ndim = 1;
        argmax_values->shape.shape[0] = graph_sample_count;
        argmax_values->shape.strides[0] = 1;

        argmax_indices->shape.ndim = 1;
        argmax_indices->shape.shape[0] = graph_sample_count;
        argmax_indices->shape.strides[0] = 1;

        if (batch.token_meta.size() != token_count * 4) {
            kv_pool_->abort_append(plan);
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }

        int32_t padding_token_id = -1;
        bool bounds_check = true;
        if (n_vocab_ <= (size_t)INT32_MAX) {
            padding_token_id = (int32_t)n_vocab_;
        } else {
            bounds_check = false;
        }

        marmot_int32_t *token_ptr = use_device_token_ids ? nullptr : marmot_tensor_data_i32_mut(ctx_, token_ids_.get());
        marmot_uint32_t *meta_ptr = marmot_tensor_data_u32_mut(ctx_, token_meta);
        float *pos_ptr = marmot_tensor_data_f32_mut(ctx_, positions);
        marmot_uint32_t *sample_ptr = marmot_tensor_data_u32_mut(ctx_, sample_indices);
        if ((!use_device_token_ids && token_ptr == nullptr) || meta_ptr == nullptr || pos_ptr == nullptr ||
            sample_ptr == nullptr) {
            kv_pool_->abort_append(plan);
            const marmot_error_t data_err = marmot_get_last_error();
            const char *detail = marmot_get_last_error_detail();
            error = (detail != nullptr && detail[0] != '\0') ? detail : "failed to access scratch buffers";
            return data_err != MARMOT_SUCCESS ? data_err : MARMOT_ERROR_INVALID_OPERATION;
        }

        if (!use_device_token_ids) {
            token_ptr[0].value = (int32_t)batch.token_ids[0];
        }
        pos_ptr[0] = (float)batch.token_meta[1];
        meta_ptr[0].value = batch.token_meta[0];
        meta_ptr[1].value = batch.token_meta[1];
        meta_ptr[2].value = batch.token_meta[2];
        meta_ptr[3].value = batch.token_meta[3];
        sample_ptr[0].value = 0u;

        if (ctx_->backend_type != MARMOT_BACKEND_CPU) {
            marmot_error_t sync_status = MARMOT_SUCCESS;
            if (!use_device_token_ids) {
                sync_status = marmot_tensor_to_device(ctx_, token_ids_.get());
                if (sync_status != MARMOT_SUCCESS) {
                    kv_pool_->abort_append(plan);
                    error = "failed to sync token ids";
                    return sync_status;
                }
            }
            sync_status = marmot_tensor_to_device(ctx_, positions);
            if (sync_status != MARMOT_SUCCESS) {
                kv_pool_->abort_append(plan);
                error = "failed to sync positions";
                return sync_status;
            }
            sync_status = marmot_tensor_to_device(ctx_, token_meta);
            if (sync_status != MARMOT_SUCCESS) {
                kv_pool_->abort_append(plan);
                error = "failed to sync token metadata";
                return sync_status;
            }
            sync_status = marmot_tensor_to_device(ctx_, sample_indices);
            if (sync_status != MARMOT_SUCCESS) {
                kv_pool_->abort_append(plan);
                error = "failed to sync sample indices";
                return sync_status;
            }
        }

        marmot_embedding_gather_desc_t gather = marmot_embedding_gather_desc_default();
        gather.weights = token_embedding_;
        gather.token_ids = token_ids_.get();
        gather.out = hidden_.get();
        gather.dtype_out = hidden_->dtype;
        gather.scale = embedding_scale_;
        gather.bounds_check = bounds_check;
        gather.padding_id = padding_token_id;
        gather.prefer_gpu_private = MARMOT_PREFERENCE_DISABLE;
        gather.allow_quant_decode_on_the_fly = MARMOT_PREFERENCE_ENABLE;

        marmot_error_t gather_status = marmot_embedding_gather(ctx_, &gather);
        if (gather_status != MARMOT_SUCCESS) {
            kv_pool_->abort_append(plan);
            const char *detail = marmot_get_last_error_detail();
            error = (detail != nullptr && detail[0] != '\0') ? detail : "embedding_gather failed";
            return gather_status;
        }

        marmot_graph_t *graph = nullptr;
        const graph::FastPlan *fast_plan = nullptr;
        marmot_error_t graph_status =
            ensure_packed_graph(graph_token_count, graph_sample_count, &graph, &fast_plan, error);
        if (graph_status != MARMOT_SUCCESS) {
            kv_pool_->abort_append(plan);
            return graph_status;
        }

        marmot_tensor_t *kv_k = nullptr;
        marmot_tensor_t *kv_v = nullptr;
        marmot_tensor_t *block_table = nullptr;
        marmot_tensor_t *kv_k_scale = nullptr;
        marmot_tensor_t *kv_v_scale = nullptr;
        kv_pool_->get_tensors(&kv_k, &kv_v, &block_table, &kv_k_scale, &kv_v_scale);
        if (kv_k == nullptr || kv_v == nullptr || block_table == nullptr) {
            kv_pool_->abort_append(plan);
            error = "KV pool tensors unavailable";
            return MARMOT_ERROR_INVALID_OPERATION;
        }
        marmot_tensor_t *graph_block_table = block_table;
        if (ctx_->backend_type == MARMOT_BACKEND_METAL) {
            TensorPtr &block_table_snapshot =
                (argmax_buffer_index == 0) ? block_table_snapshot_ : block_table_snapshot_alt_;
            if (!block_table_snapshot || block_table_snapshot->dtype != block_table->dtype ||
                block_table_snapshot->shape.ndim != 2 || block_table->shape.ndim != 2 ||
                block_table_snapshot->shape.shape[0] != block_table->shape.shape[0] ||
                block_table_snapshot->shape.shape[1] != block_table->shape.shape[1]) {
                size_t shape[2] = {block_table->shape.shape[0], block_table->shape.shape[1]};
                block_table_snapshot.reset(marmot_tensor_create(ctx_, shape, 2, block_table->dtype));
                if (!block_table_snapshot) {
                    kv_pool_->abort_append(plan);
                    error = "failed to allocate block table snapshot";
                    return MARMOT_ERROR_OUT_OF_MEMORY;
                }
            }

            const size_t bytes = marmot_tensor_size_bytes(block_table);
            if (bytes != marmot_tensor_size_bytes(block_table_snapshot.get())) {
                kv_pool_->abort_append(plan);
                error = "block table snapshot size mismatch";
                return MARMOT_ERROR_INVALID_OPERATION;
            }
            memcpy(block_table_snapshot->data, block_table->data, bytes);
            graph_block_table = block_table_snapshot.get();
        }

        if (ctx_->backend_type != MARMOT_BACKEND_CPU) {
            marmot_error_t sync_status = marmot_tensor_to_device(ctx_, graph_block_table);
            if (sync_status != MARMOT_SUCCESS) {
                kv_pool_->abort_append(plan);
                error = "failed to sync block table";
                return sync_status;
            }
        }

        const marmot_tensor_t *inputs[] = {
            hidden_.get(), positions, token_meta, graph_block_table, kv_k, kv_v, sample_indices,
        };
        marmot_tensor_t *outputs[] = {logits_.get()};
        marmot_error_t run_status = execute_packed_graph(
            graph, fast_plan, ctx_, std::span<const marmot_tensor_t *const>(inputs, ARRAY_LENGTH(inputs)),
            std::span<marmot_tensor_t *const>(outputs, ARRAY_LENGTH(outputs))
        );
        if (run_status != MARMOT_SUCCESS) {
            kv_pool_->abort_append(plan);
            const char *detail = marmot_get_last_error_detail();
            error = (detail != nullptr && detail[0] != '\0') ? detail : "packed graph execute failed";
            return run_status;
        }

        {
            int32_t axis = 1;
            marmot_reduction_desc_t argmax = marmot_reduction_desc_default();
            argmax.input = logits_.get();
            argmax.out = argmax_values;
            argmax.indices_out = argmax_indices;
            argmax.axes = &axis;
            argmax.num_axes = 1;
            argmax.keepdims = false;

            marmot_error_t argmax_status = marmot_reduce_argmax(ctx_, &argmax);
            if (argmax_status != MARMOT_SUCCESS) {
                kv_pool_->abort_append(plan);
                const char *detail = marmot_get_last_error_detail();
                error = (detail != nullptr && detail[0] != '\0') ? detail : "argmax failed";
                return argmax_status;
            }
        }

        marmot_error_t scatter_status =
            marmot_scatter_u64_to_i32(ctx_, argmax_indices, sample_indices, token_ids_.get());
        if (scatter_status != MARMOT_SUCCESS) {
            kv_pool_->abort_append(plan);
            const char *detail = marmot_get_last_error_detail();
            error = (detail != nullptr && detail[0] != '\0') ? detail : "scatter_u64_to_i32 failed";
            return scatter_status;
        }
        token_ids_device_ready_ = true;

        marmot_error_t commit_status = kv_pool_->commit_append(plan);
        if (commit_status != MARMOT_SUCCESS) {
            error = "KV pool commit failed";
            return commit_status;
        }

        last_batch_ = batch;
        return MARMOT_SUCCESS;
    };

    auto sample_decode_token = [&](Request &req, uint32_t argmax_buffer_index, bool &out_finished) -> marmot_error_t {
        out_finished = false;
        error.clear();

        marmot_tensor_t *argmax_values = nullptr;
        marmot_tensor_t *argmax_indices = nullptr;
        if (!argmax_buffers_for(argmax_buffer_index, &argmax_values, &argmax_indices)) {
            error = "argmax buffers unavailable";
            return MARMOT_ERROR_INVALID_OPERATION;
        }

        const marmot_uint64_t *device_argmax = marmot_tensor_data_u64(ctx_, argmax_indices);
        if (device_argmax == nullptr) {
            const marmot_error_t data_err = marmot_get_last_error();
            const char *detail = marmot_get_last_error_detail();
            error = (detail != nullptr && detail[0] != '\0') ? detail : "argmax indices unavailable";
            return data_err != MARMOT_SUCCESS ? data_err : MARMOT_ERROR_INVALID_OPERATION;
        }

        const uint64_t index = device_argmax[0].value;
        if (index >= n_vocab_) {
            error = "argmax index out of range";
            return MARMOT_ERROR_INVALID_OPERATION;
        }
        const marmot_token_id_t token = static_cast<marmot_token_id_t>(index);

        const bool finished = req.cancel_requested || finished_for_token(req, token);

        if (!req.cancel_requested) {
            req.generated_tokens.push_back(token);
            if (req.gen_opts.on_token != nullptr) {
                req.gen_opts.on_token(req.gen_opts.user_data, token);
            }
        }
        out_finished = finished;

        if (finished) {
            req.state = req.cancel_requested ? MARMOT_LLM_REQUEST_STATE_CANCELED : MARMOT_LLM_REQUEST_STATE_DONE;
            req.has_pending_token = false;
            req.pending_input_token = MARMOT_TOKEN_ID_INVALID;
        } else {
            req.state = MARMOT_LLM_REQUEST_STATE_DECODING;
            req.pending_input_token = token;
            req.has_pending_token = true;
        }

        return MARMOT_SUCCESS;
    };

    // Drain: force device-to-host sync for argmax results, detect transfer errors
    auto drain_decode_token = [&](uint32_t argmax_buffer_index) -> marmot_error_t {
        error.clear();
        marmot_tensor_t *argmax_values = nullptr;
        marmot_tensor_t *argmax_indices = nullptr;
        if (!argmax_buffers_for(argmax_buffer_index, &argmax_values, &argmax_indices)) {
            error = "argmax buffers unavailable";
            return MARMOT_ERROR_INVALID_OPERATION;
        }
        if (marmot_tensor_data_u64(ctx_, argmax_indices) == nullptr) {
            const marmot_error_t data_err = marmot_get_last_error();
            const char *detail = marmot_get_last_error_detail();
            error = (detail != nullptr && detail[0] != '\0') ? detail : "argmax indices unavailable";
            return data_err != MARMOT_SUCCESS ? data_err : MARMOT_ERROR_INVALID_OPERATION;
        }
        return MARMOT_SUCCESS;
    };

    while (out_steps_done < max_steps) {
        PipelineInFlight inflight{};
        if (pipeline_in_flight_.has_value()) {
            inflight = *pipeline_in_flight_;
        } else {
            if (requests_.size() != 1) {
                return out_steps_done == 0 ? MARMOT_ERROR_NOT_IMPLEMENTED : MARMOT_SUCCESS;
            }

            Request &req = requests_.begin()->second;
            if (req.state != MARMOT_LLM_REQUEST_STATE_DECODING || req.cancel_requested || !req.has_pending_token ||
                !req.has_seq_slot) {
                return out_steps_done == 0 ? MARMOT_ERROR_NOT_IMPLEMENTED : MARMOT_SUCCESS;
            }
            if (!greedy_device_argmax_supported(req)) {
                return out_steps_done == 0 ? MARMOT_ERROR_NOT_IMPLEMENTED : MARMOT_SUCCESS;
            }

            const uint32_t argmax_idx = pipeline_next_argmax_buffer_index_ & 1u;
            marmot_error_t submit_status = submit_decode_token(req, argmax_idx);
            if (submit_status != MARMOT_SUCCESS) {
                return submit_status;
            }
            inflight = PipelineInFlight{req.id, argmax_idx};
            pipeline_next_argmax_buffer_index_ = argmax_idx ^ 1u;
            pipeline_in_flight_ = inflight;
        }

        auto it = requests_.find(inflight.request_id);
        if (it == requests_.end()) {
            pipeline_in_flight_.reset();
            token_ids_device_ready_ = false;
            return MARMOT_ERROR_INVALID_OPERATION;
        }
        Request &req = it->second;

        bool prefetched = false;
        PipelineInFlight next_inflight{};
        if (requests_.size() == 1 && req.state == MARMOT_LLM_REQUEST_STATE_DECODING && !req.cancel_requested &&
            req.has_pending_token && req.has_seq_slot && token_ids_device_ready_) {
            const uint32_t prefetch_idx = pipeline_next_argmax_buffer_index_ & 1u;
            marmot_error_t prefetch_status = submit_decode_token(req, prefetch_idx);
            if (prefetch_status == MARMOT_SUCCESS) {
                next_inflight = PipelineInFlight{req.id, prefetch_idx};
                pipeline_next_argmax_buffer_index_ = prefetch_idx ^ 1u;
                prefetched = true;
            } else if (prefetch_status != MARMOT_ERROR_OUT_OF_MEMORY &&
                       prefetch_status != MARMOT_ERROR_NOT_IMPLEMENTED) {
                return prefetch_status;
            }
        }

        bool finished = false;
        marmot_error_t sample_status = sample_decode_token(req, inflight.argmax_buffer_index, finished);
        if (sample_status != MARMOT_SUCCESS) {
            return sample_status;
        }
        out_steps_done += 1;

        if (finished) {
            if (prefetched) {
                marmot_error_t drain_status = drain_decode_token(next_inflight.argmax_buffer_index);
                if (drain_status != MARMOT_SUCCESS) {
                    return drain_status;
                }
            }

            pipeline_in_flight_.reset();
            token_ids_device_ready_ = false;
            if (req.has_seq_slot) {
                (void)kv_pool_->release_seq(req.seq_slot);
            }
            requests_.clear();
            schedule_.clear();
            break;
        }

        if (prefetched) {
            pipeline_in_flight_ = next_inflight;
        } else {
            pipeline_in_flight_.reset();
        }
    }

    return MARMOT_SUCCESS;
}

marmot_error_t ServingEngine::step(size_t max_steps, size_t &out_steps_done, std::string &error) {
    out_steps_done = 0;
    error.clear();

    if (ctx_ == nullptr || kv_pool_ == nullptr) {
        return MARMOT_ERROR_INVALID_OPERATION;
    }
    if (max_steps == 0) {
        return MARMOT_SUCCESS;
    }

    if (ctx_->backend_type == MARMOT_BACKEND_METAL) {
        marmot_error_t pipelined_status = step_pipelined_greedy_decode(max_steps, out_steps_done, error);
        if (pipelined_status != MARMOT_ERROR_NOT_IMPLEMENTED) {
            return pipelined_status;
        }
    }

    auto build_recompute_tokens = [](Request &req) {
        req.recompute_tokens.clear();
        req.recompute_tokens.reserve(req.prompt_tokens.size() + req.generated_tokens.size());
        req.recompute_tokens.insert(req.recompute_tokens.end(), req.prompt_tokens.begin(), req.prompt_tokens.end());
        req.recompute_tokens.insert(
            req.recompute_tokens.end(), req.generated_tokens.begin(), req.generated_tokens.end()
        );
    };

    auto abort_append_plans = [&](std::vector<PendingAppend> &plans) {
        for (const auto &pending : plans) {
            kv_pool_->abort_append(pending.plan);
        }
    };

    size_t watermark_blocks = 0;
    const size_t total_blocks = kv_pool_->total_block_count();
    if (opts_.kv_block_watermark > 0.0f && total_blocks > 0) {
        const double raw_blocks = static_cast<double>(total_blocks) * opts_.kv_block_watermark;
        watermark_blocks = static_cast<size_t>(std::ceil(raw_blocks));
        if (watermark_blocks > total_blocks) {
            watermark_blocks = total_blocks;
        }
    }

    for (; out_steps_done < max_steps; ++out_steps_done) {
        const bool profiling = step_profile_enabled();
        StepTimer timer;
        const double step_start = profiling ? StepTimer::now_ns() : 0.0;
        double batch_end = 0.0;
        double sync_end = 0.0;
        double embed_end = 0.0;
        double graph_end = 0.0;

        PackedBatch batch{};
        std::vector<PendingAppend> append_plans;
        std::vector<marmot_request_id_t> protected_ids;
        std::vector<marmot_request_id_t> preempted_ids;
        struct PrefillWork {
            Request *req{nullptr};
            size_t chunk{0};
            size_t prompt_len{0};
            bool finish_without_sample{false};
        };
        std::vector<PrefillWork> prefill_work;
        size_t token_budget = opts_.max_num_tokens;
        size_t seq_budget = opts_.max_batch_seqs;
        bool batch_has_prefill = false;

        auto is_protected = [&](marmot_request_id_t request_id, marmot_request_id_t current_id) {
            if (request_id == current_id) {
                return true;
            }
            return std::find(protected_ids.begin(), protected_ids.end(), request_id) != protected_ids.end();
        };

        auto was_preempted = [&](marmot_request_id_t request_id) {
            return std::find(preempted_ids.begin(), preempted_ids.end(), request_id) != preempted_ids.end();
        };

        auto preempt_one = [&](marmot_request_id_t current_id, std::string &preempt_error) -> bool {
            preempt_error.clear();
            Request *victim = nullptr;
            size_t victim_len = 0;
            for (auto &kv : requests_) {
                Request &candidate = kv.second;
                if (!candidate.has_seq_slot) {
                    continue;
                }
                if (candidate.cancel_requested) {
                    continue;
                }
                if (candidate.state != MARMOT_LLM_REQUEST_STATE_PENDING &&
                    candidate.state != MARMOT_LLM_REQUEST_STATE_PREFILL &&
                    candidate.state != MARMOT_LLM_REQUEST_STATE_DECODING) {
                    continue;
                }
                if (candidate.awaiting_prefix_attach) {
                    continue;
                }
                if (is_protected(candidate.id, current_id)) {
                    continue;
                }

                const size_t candidate_len = kv_pool_->seq_len(candidate.seq_slot);
                if (victim == nullptr || candidate.priority < victim->priority ||
                    (candidate.priority == victim->priority && candidate_len > victim_len)) {
                    victim = &candidate;
                    victim_len = candidate_len;
                }
            }

            if (victim == nullptr) {
                return false;
            }

            marmot_error_t release_status = kv_pool_->release_seq(victim->seq_slot);
            if (release_status != MARMOT_SUCCESS) {
                const char *detail = marmot_get_last_error_detail();
                preempt_error = (detail != nullptr && detail[0] != '\0') ? detail : "KV pool preemption failed";
                return false;
            }

            victim->has_seq_slot = false;
            victim->seq_slot = 0;
            victim->prompt_cursor = 0;
            victim->pending_input_token = MARMOT_TOKEN_ID_INVALID;
            victim->has_pending_token = false;
            victim->awaiting_sample = false;
            victim->swapped_out = false;
            victim->state = MARMOT_LLM_REQUEST_STATE_PENDING;
            victim->needs_recompute = true;
            victim->prefix_hashes.clear();
            build_recompute_tokens(*victim);
            if (!victim->clone_ids.empty()) {
                for (marmot_request_id_t clone_id : victim->clone_ids) {
                    auto clone_it = requests_.find(clone_id);
                    if (clone_it == requests_.end()) {
                        continue;
                    }
                    Request &clone = clone_it->second;
                    if (!clone.awaiting_prefix_attach) {
                        continue;
                    }
                    clone.cancel_requested = true;
                    if (clone.state == MARMOT_LLM_REQUEST_STATE_PENDING) {
                        clone.state = MARMOT_LLM_REQUEST_STATE_CANCELED;
                    }
                }
                victim->clone_ids.clear();
                victim->num_samples = 1;
            }
            preempted_ids.push_back(victim->id);
            return true;
        };

        auto swap_out_one = [&](marmot_request_id_t current_id, std::string &swap_error) -> bool {
            swap_error.clear();
            if ((opts_.flags & MARMOT_SERVING_ENGINE_FLAG_ENABLE_SWAP) == 0) {
                return false;
            }
            Request *victim = nullptr;
            size_t victim_len = 0;
            for (auto &kv : requests_) {
                Request &candidate = kv.second;
                if (!candidate.has_seq_slot || candidate.swapped_out) {
                    continue;
                }
                if (candidate.cancel_requested) {
                    continue;
                }
                if (candidate.state != MARMOT_LLM_REQUEST_STATE_PENDING &&
                    candidate.state != MARMOT_LLM_REQUEST_STATE_PREFILL &&
                    candidate.state != MARMOT_LLM_REQUEST_STATE_DECODING) {
                    continue;
                }
                if (candidate.awaiting_prefix_attach) {
                    continue;
                }
                if (is_protected(candidate.id, current_id)) {
                    continue;
                }

                const size_t candidate_len = kv_pool_->seq_len(candidate.seq_slot);
                if (candidate_len == 0) {
                    continue;
                }
                if (victim == nullptr || candidate.priority < victim->priority ||
                    (candidate.priority == victim->priority && candidate_len > victim_len)) {
                    victim = &candidate;
                    victim_len = candidate_len;
                }
            }

            if (victim == nullptr) {
                return false;
            }

            marmot_error_t swap_status = kv_pool_->swap_out_seq(victim->seq_slot);
            if (swap_status == MARMOT_ERROR_OUT_OF_MEMORY) {
                return false;
            }
            if (swap_status != MARMOT_SUCCESS) {
                const char *detail = marmot_get_last_error_detail();
                swap_error = (detail != nullptr && detail[0] != '\0') ? detail : "KV pool swap failed";
                return false;
            }

            victim->swapped_out = true;
            return true;
        };

        auto ensure_seq_resident = [&](Request &req) -> marmot_error_t {
            if (!req.swapped_out) {
                return MARMOT_SUCCESS;
            }
            if ((opts_.flags & MARMOT_SERVING_ENGINE_FLAG_ENABLE_SWAP) == 0) {
                error = "sequence is swapped out but swap is disabled";
                return MARMOT_ERROR_INVALID_OPERATION;
            }
            while (true) {
                marmot_error_t status = kv_pool_->swap_in_seq(req.seq_slot);
                if (status == MARMOT_SUCCESS) {
                    req.swapped_out = false;
                    return MARMOT_SUCCESS;
                }
                if (status != MARMOT_ERROR_OUT_OF_MEMORY) {
                    const char *detail = marmot_get_last_error_detail();
                    error = (detail != nullptr && detail[0] != '\0') ? detail : "KV pool swap-in failed";
                    return status;
                }

                std::string swap_error;
                if (!swap_out_one(req.id, swap_error)) {
                    if (!swap_error.empty()) {
                        error = swap_error;
                        return MARMOT_ERROR_INVALID_OPERATION;
                    }
                    return MARMOT_ERROR_OUT_OF_MEMORY;
                }
            }
        };

        auto enforce_kv_watermark = [&](marmot_request_id_t current_id) -> marmot_error_t {
            if (watermark_blocks == 0) {
                return MARMOT_SUCCESS;
            }
            size_t free_blocks = kv_pool_->free_block_count();
            if (free_blocks > watermark_blocks) {
                return MARMOT_SUCCESS;
            }
            while (free_blocks <= watermark_blocks) {
                std::string swap_error;
                if (swap_out_one(current_id, swap_error)) {
                    free_blocks = kv_pool_->free_block_count();
                    continue;
                }
                if (!swap_error.empty()) {
                    error = swap_error;
                    return MARMOT_ERROR_INVALID_OPERATION;
                }
                std::string preempt_error;
                if (preempt_one(current_id, preempt_error)) {
                    free_blocks = kv_pool_->free_block_count();
                    continue;
                }
                if (!preempt_error.empty()) {
                    error = preempt_error;
                    return MARMOT_ERROR_INVALID_OPERATION;
                }
                break;
            }
            return MARMOT_SUCCESS;
        };

        auto push_token_meta = [&](marmot_seq_slot_t seq_slot, size_t pos, marmot_kv_slot_t slot, uint32_t flags) {
            batch.token_meta.push_back(static_cast<uint32_t>(seq_slot));
            batch.token_meta.push_back(static_cast<uint32_t>(pos));
            batch.token_meta.push_back(static_cast<uint32_t>(slot));
            batch.token_meta.push_back(flags);
        };

        std::vector<Request *> decode_reqs;
        decode_reqs.reserve(requests_.size());
        for (auto &kv : requests_) {
            Request &req = kv.second;
            if (req.state != MARMOT_LLM_REQUEST_STATE_DECODING || req.cancel_requested || !req.has_pending_token ||
                req.awaiting_sample || !req.has_seq_slot) {
                continue;
            }
            decode_reqs.push_back(&req);
        }

        std::sort(decode_reqs.begin(), decode_reqs.end(), [](const Request *a, const Request *b) {
            if (a->priority != b->priority) {
                return a->priority > b->priority;
            }
            return a->id < b->id;
        });

        for (Request *req : decode_reqs) {
            if (token_budget == 0 || seq_budget == 0) {
                break;
            }
            if (req->state != MARMOT_LLM_REQUEST_STATE_DECODING || req->cancel_requested || !req->has_pending_token ||
                req->awaiting_sample || !req->has_seq_slot) {
                continue;
            }
            if (req->swapped_out) {
                marmot_error_t swap_status = ensure_seq_resident(*req);
                if (swap_status == MARMOT_ERROR_OUT_OF_MEMORY) {
                    continue;
                }
                if (swap_status != MARMOT_SUCCESS) {
                    abort_append_plans(append_plans);
                    return swap_status;
                }
            }

            marmot_kv_slot_t slot = 0;
            size_t start_pos = 0;
            KVPool::AppendPlan plan{};
            marmot_error_t status = MARMOT_SUCCESS;
            while (true) {
                status = kv_pool_->prepare_append(req->seq_slot, 1, &slot, start_pos, plan);
                if (status != MARMOT_ERROR_OUT_OF_MEMORY) {
                    break;
                }
                std::string swap_error;
                if (swap_out_one(req->id, swap_error)) {
                    continue;
                }
                if (!swap_error.empty()) {
                    abort_append_plans(append_plans);
                    error = swap_error;
                    return MARMOT_ERROR_INVALID_OPERATION;
                }
                std::string preempt_error;
                if (!preempt_one(req->id, preempt_error)) {
                    if (!preempt_error.empty()) {
                        abort_append_plans(append_plans);
                        error = preempt_error;
                        return MARMOT_ERROR_INVALID_OPERATION;
                    }
                    break;
                }
            }
            if (status == MARMOT_ERROR_OUT_OF_MEMORY) {
                break;
            }
            if (status != MARMOT_SUCCESS) {
                abort_append_plans(append_plans);
                return status;
            }

            if (start_pos > UINT32_MAX) {
                abort_append_plans(append_plans);
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }

            const size_t token_index = batch.token_ids.size();
            batch.token_ids.push_back(req->pending_input_token);
            push_token_meta(req->seq_slot, start_pos, slot, kTokenFlagDecode);
            batch.sample_indices.push_back(static_cast<uint32_t>(token_index));
            batch.sample_request_ids.push_back(req->id);
            append_plans.push_back(PendingAppend{req->id, std::move(plan)});
            protected_ids.push_back(req->id);

            token_budget -= 1;
            seq_budget -= 1;
        }

        if (watermark_blocks > 0) {
            marmot_error_t watermark_status = enforce_kv_watermark(0);
            if (watermark_status != MARMOT_SUCCESS) {
                abort_append_plans(append_plans);
                return watermark_status;
            }
        }

        for (marmot_request_id_t request_id : schedule_) {
            if (token_budget == 0 || seq_budget == 0) {
                break;
            }
            if (watermark_blocks > 0 && kv_pool_->free_block_count() <= watermark_blocks) {
                break;
            }

            auto it = requests_.find(request_id);
            if (it == requests_.end()) {
                continue;
            }

            Request &req = it->second;
            if (req.cancel_requested) {
                req.state = MARMOT_LLM_REQUEST_STATE_CANCELED;
                continue;
            }
            if (req.state != MARMOT_LLM_REQUEST_STATE_PENDING && req.state != MARMOT_LLM_REQUEST_STATE_PREFILL) {
                continue;
            }
            if (req.awaiting_prefix_attach) {
                continue;
            }
            if (req.swapped_out) {
                marmot_error_t swap_status = ensure_seq_resident(req);
                if (swap_status == MARMOT_ERROR_OUT_OF_MEMORY) {
                    continue;
                }
                if (swap_status != MARMOT_SUCCESS) {
                    abort_append_plans(append_plans);
                    return swap_status;
                }
            }

            if (!req.has_seq_slot) {
                if (was_preempted(request_id)) {
                    continue;
                }
                marmot_seq_slot_t seq_slot = 0;
                marmot_error_t acquire_status = kv_pool_->acquire_seq(seq_slot);
                if (acquire_status == MARMOT_ERROR_OUT_OF_MEMORY) {
                    continue;
                }
                if (acquire_status != MARMOT_SUCCESS) {
                    abort_append_plans(append_plans);
                    return acquire_status;
                }
                req.seq_slot = seq_slot;
                req.has_seq_slot = true;
                req.prompt_cursor = 0;
            }

            if (req.needs_recompute && req.recompute_tokens.empty()) {
                build_recompute_tokens(req);
            }

            const std::vector<marmot_token_id_t> &prefill_tokens =
                req.needs_recompute ? req.recompute_tokens : req.prompt_tokens;
            const size_t prompt_len = prefill_tokens.size();
            if (req.prompt_cursor >= prompt_len) {
                continue;
            }

            size_t remaining = prompt_len - req.prompt_cursor;
            size_t chunk = std::min(remaining, token_budget);
            if (opts_.prefill_chunk_size > 0) {
                chunk = std::min(chunk, opts_.prefill_chunk_size);
            }
            if (remaining > chunk) {
                const size_t aligned = chunk - (chunk % opts_.block_size);
                if (aligned == 0) {
                    continue;
                }
                chunk = aligned;
            }

            std::vector<marmot_kv_slot_t> slots(chunk);
            size_t start_pos = 0;
            KVPool::AppendPlan plan{};
            marmot_error_t status = MARMOT_SUCCESS;
            while (true) {
                status = kv_pool_->prepare_append(req.seq_slot, chunk, slots.data(), start_pos, plan);
                if (status != MARMOT_ERROR_OUT_OF_MEMORY) {
                    break;
                }
                std::string swap_error;
                if (swap_out_one(req.id, swap_error)) {
                    continue;
                }
                if (!swap_error.empty()) {
                    abort_append_plans(append_plans);
                    error = swap_error;
                    return MARMOT_ERROR_INVALID_OPERATION;
                }
                std::string preempt_error;
                if (!preempt_one(req.id, preempt_error)) {
                    if (!preempt_error.empty()) {
                        abort_append_plans(append_plans);
                        error = preempt_error;
                        return MARMOT_ERROR_INVALID_OPERATION;
                    }
                    break;
                }
            }
            if (status == MARMOT_ERROR_OUT_OF_MEMORY) {
                break;
            }
            if (status != MARMOT_SUCCESS) {
                abort_append_plans(append_plans);
                return status;
            }

            if (start_pos > UINT32_MAX || start_pos + chunk > UINT32_MAX) {
                abort_append_plans(append_plans);
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }

            for (size_t i = 0; i < chunk; ++i) {
                batch.token_ids.push_back(prefill_tokens[req.prompt_cursor + i]);
                push_token_meta(req.seq_slot, start_pos + i, slots[i], kTokenFlagPrefill);
            }
            batch_has_prefill = true;

            append_plans.push_back(PendingAppend{req.id, std::move(plan)});
            protected_ids.push_back(req.id);
            prefill_work.push_back(
                PrefillWork{
                    .req = &req,
                    .chunk = chunk,
                    .prompt_len = prompt_len,
                    .finish_without_sample =
                        (req.gen_opts.max_new_tokens == 0 && req.prompt_cursor + chunk == prompt_len),
                }
            );

            token_budget -= chunk;
            seq_budget -= 1;

            if (req.prompt_cursor + chunk == prompt_len && req.gen_opts.max_new_tokens != 0) {
                const size_t token_index = batch.token_ids.size() - 1;
                batch.sample_indices.push_back(static_cast<uint32_t>(token_index));
                batch.sample_request_ids.push_back(req.id);
            }
        }

        if (batch.token_ids.empty()) {
            last_batch_ = {};
            break;
        }

        if (profiling) {
            batch_end = StepTimer::now_ns();
        }

        const size_t token_count = batch.token_ids.size();
        const size_t sample_count = batch.sample_indices.size();
        marmot_error_t thread_status = apply_cpu_step_thread_policy(ctx_, batch_has_prefill);
        if (thread_status != MARMOT_SUCCESS) {
            abort_append_plans(append_plans);
            return thread_status;
        }
        marmot_error_t scratch_status = ensure_scratch(token_count, sample_count, error);
        if (scratch_status != MARMOT_SUCCESS) {
            abort_append_plans(append_plans);
            return scratch_status;
        }

        const size_t token_capacity = scratch_token_capacity_;
        const size_t sample_capacity = scratch_sample_capacity_;
        const bool use_compact_graph = !batch_has_prefill && token_count > 0 && token_count < token_capacity;
        const size_t graph_token_count = use_compact_graph ? token_count : token_capacity;
        const size_t graph_sample_count =
            sample_count > 0 ? (use_compact_graph ? sample_count : std::min(sample_capacity, graph_token_count)) : 0;

        token_ids_->shape.ndim = 1;
        token_ids_->shape.shape[0] = graph_token_count;
        token_ids_->shape.strides[0] = 1;

        positions_->shape.ndim = 1;
        positions_->shape.shape[0] = graph_token_count;
        positions_->shape.strides[0] = 1;

        token_meta_->shape.ndim = 2;
        token_meta_->shape.shape[0] = token_count;
        token_meta_->shape.shape[1] = 4;
        token_meta_->shape.strides[1] = 1;
        token_meta_->shape.strides[0] = 4;

        hidden_->shape.ndim = 2;
        hidden_->shape.shape[0] = graph_token_count;
        hidden_->shape.shape[1] = n_embd_;
        hidden_->shape.strides[1] = 1;
        hidden_->shape.strides[0] = n_embd_;

        hidden_out_->shape.ndim = 2;
        hidden_out_->shape.shape[0] = graph_token_count;
        hidden_out_->shape.shape[1] = n_embd_;
        hidden_out_->shape.strides[1] = 1;
        hidden_out_->shape.strides[0] = n_embd_;

        if (sample_count > 0) {
            sample_indices_->shape.ndim = 1;
            sample_indices_->shape.shape[0] = graph_sample_count;
            sample_indices_->shape.strides[0] = 1;

            logits_->shape.ndim = 2;
            logits_->shape.shape[0] = graph_sample_count;
            logits_->shape.shape[1] = n_vocab_;
            logits_->shape.strides[1] = 1;
            logits_->shape.strides[0] = n_vocab_;

            logits_max_->shape.ndim = 1;
            logits_max_->shape.shape[0] = graph_sample_count;
            logits_max_->shape.strides[0] = 1;

            logits_argmax_->shape.ndim = 1;
            logits_argmax_->shape.shape[0] = graph_sample_count;
            logits_argmax_->shape.strides[0] = 1;
        }

        if (batch.token_meta.size() != token_count * 4) {
            abort_append_plans(append_plans);
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }

        const bool use_device_token_ids = token_ids_device_ready_ && !batch_has_prefill &&
            sample_count == token_count && ctx_->backend_type != MARMOT_BACKEND_CPU;
        token_ids_device_ready_ = false;

        marmot_int32_t *token_ptr = use_device_token_ids ? nullptr : marmot_tensor_data_i32_mut(ctx_, token_ids_.get());
        marmot_uint32_t *meta_ptr = marmot_tensor_data_u32_mut(ctx_, token_meta_.get());
        float *pos_ptr = marmot_tensor_data_f32_mut(ctx_, positions_.get());
        if ((!use_device_token_ids && token_ptr == nullptr) || meta_ptr == nullptr || pos_ptr == nullptr) {
            const marmot_error_t data_err = marmot_get_last_error();
            const char *detail = marmot_get_last_error_detail();
            error = (detail != nullptr && detail[0] != '\0') ? detail : "failed to access scratch buffers";
            abort_append_plans(append_plans);
            return data_err != MARMOT_SUCCESS ? data_err : MARMOT_ERROR_INVALID_OPERATION;
        }

        int32_t padding_token_id = -1;
        bool bounds_check = true;
        if (n_vocab_ <= (size_t)INT32_MAX) {
            padding_token_id = (int32_t)n_vocab_;
        } else {
            bounds_check = false;
        }

        for (size_t i = 0; i < graph_token_count; ++i) {
            if (!use_device_token_ids) {
                token_ptr[i].value = padding_token_id;
            }
            pos_ptr[i] = 0.0f;
        }
        for (size_t i = 0; i < token_count; ++i) {
            if (!use_device_token_ids) {
                token_ptr[i].value = (int32_t)batch.token_ids[i];
            }
            pos_ptr[i] = (float)batch.token_meta[i * 4 + 1];
        }

        for (size_t i = 0; i < token_count * 4; ++i) {
            meta_ptr[i].value = batch.token_meta[i];
        }

        if (sample_count > 0) {
            marmot_uint32_t *sample_ptr = marmot_tensor_data_u32_mut(ctx_, sample_indices_.get());
            if (sample_ptr == nullptr) {
                const marmot_error_t data_err = marmot_get_last_error();
                const char *detail = marmot_get_last_error_detail();
                error = (detail != nullptr && detail[0] != '\0') ? detail : "failed to access sample indices";
                abort_append_plans(append_plans);
                return data_err != MARMOT_SUCCESS ? data_err : MARMOT_ERROR_INVALID_OPERATION;
            }
            for (size_t i = 0; i < graph_sample_count; ++i) {
                sample_ptr[i].value = (i < sample_count) ? batch.sample_indices[i] : 0u;
            }
        }

        if (ctx_->backend_type != MARMOT_BACKEND_CPU) {
            marmot_error_t sync_status = MARMOT_SUCCESS;
            if (!use_device_token_ids) {
                sync_status = marmot_tensor_to_device(ctx_, token_ids_.get());
                if (sync_status != MARMOT_SUCCESS) {
                    error = "failed to sync token ids";
                    abort_append_plans(append_plans);
                    return sync_status;
                }
            }
            sync_status = marmot_tensor_to_device(ctx_, positions_.get());
            if (sync_status != MARMOT_SUCCESS) {
                error = "failed to sync positions";
                abort_append_plans(append_plans);
                return sync_status;
            }
            sync_status = marmot_tensor_to_device(ctx_, token_meta_.get());
            if (sync_status != MARMOT_SUCCESS) {
                error = "failed to sync token metadata";
                abort_append_plans(append_plans);
                return sync_status;
            }
            if (sample_count > 0) {
                sync_status = marmot_tensor_to_device(ctx_, sample_indices_.get());
                if (sync_status != MARMOT_SUCCESS) {
                    error = "failed to sync sample indices";
                    abort_append_plans(append_plans);
                    return sync_status;
                }
            }
        }

        if (profiling) {
            sync_end = StepTimer::now_ns();
        }

        marmot_embedding_gather_desc_t gather = marmot_embedding_gather_desc_default();
        gather.weights = token_embedding_;
        gather.token_ids = token_ids_.get();
        gather.out = hidden_.get();
        gather.dtype_out = hidden_->dtype;
        gather.scale = embedding_scale_;
        gather.bounds_check = bounds_check;
        gather.padding_id = padding_token_id;
        gather.prefer_gpu_private = MARMOT_PREFERENCE_DISABLE;
        gather.allow_quant_decode_on_the_fly = MARMOT_PREFERENCE_ENABLE;

        marmot_error_t gather_status = marmot_embedding_gather(ctx_, &gather);
        if (gather_status != MARMOT_SUCCESS) {
            const char *detail = marmot_get_last_error_detail();
            error = (detail != nullptr && detail[0] != '\0') ? detail : "embedding_gather failed";
            abort_append_plans(append_plans);
            return gather_status;
        }

        if (profiling) {
            embed_end = StepTimer::now_ns();
        }

        marmot_graph_t *graph = nullptr;
        const graph::FastPlan *fast_plan = nullptr;
        marmot_error_t graph_status =
            ensure_packed_graph(graph_token_count, graph_sample_count, &graph, &fast_plan, error);
        if (graph_status != MARMOT_SUCCESS) {
            abort_append_plans(append_plans);
            return graph_status;
        }

        marmot_tensor_t *kv_k = nullptr;
        marmot_tensor_t *kv_v = nullptr;
        marmot_tensor_t *block_table = nullptr;
        marmot_tensor_t *kv_k_scale = nullptr;
        marmot_tensor_t *kv_v_scale = nullptr;
        kv_pool_->get_tensors(&kv_k, &kv_v, &block_table, &kv_k_scale, &kv_v_scale);
        if (kv_k == nullptr || kv_v == nullptr || block_table == nullptr) {
            error = "KV pool tensors unavailable";
            abort_append_plans(append_plans);
            return MARMOT_ERROR_INVALID_OPERATION;
        }
        if (ctx_->backend_type != MARMOT_BACKEND_CPU) {
            marmot_error_t sync_status = marmot_tensor_to_device(ctx_, block_table);
            if (sync_status != MARMOT_SUCCESS) {
                error = "failed to sync block table";
                abort_append_plans(append_plans);
                return sync_status;
            }
        }
        bool use_fp8_kv = false;
#if MARMOT_ENABLE_FP8
        use_fp8_kv = opts_.kv_dtype == MARMOT_DTYPE_FLOAT8_E4M3;
#endif
        if (use_fp8_kv && (kv_k_scale == nullptr || kv_v_scale == nullptr)) {
            error = "KV scale tensors unavailable";
            abort_append_plans(append_plans);
            return MARMOT_ERROR_INVALID_OPERATION;
        }

        if (sample_count > 0) {
            if (use_fp8_kv) {
                const marmot_tensor_t *inputs[] = {
                    hidden_.get(), positions_.get(), token_meta_.get(), block_table,           kv_k,
                    kv_v,          kv_k_scale,       kv_v_scale,        sample_indices_.get(),
                };
                marmot_tensor_t *outputs[] = {logits_.get()};
                marmot_error_t run_status = execute_packed_graph(
                    graph, fast_plan, ctx_, std::span<const marmot_tensor_t *const>(inputs, ARRAY_LENGTH(inputs)),
                    std::span<marmot_tensor_t *const>(outputs, ARRAY_LENGTH(outputs))
                );
                if (run_status != MARMOT_SUCCESS) {
                    const char *detail = marmot_get_last_error_detail();
                    error = (detail != nullptr && detail[0] != '\0') ? detail : "packed graph execute failed";
                    abort_append_plans(append_plans);
                    return run_status;
                }
            } else {
                const marmot_tensor_t *inputs[] = {
                    hidden_.get(), positions_.get(), token_meta_.get(), block_table, kv_k, kv_v, sample_indices_.get(),
                };
                marmot_tensor_t *outputs[] = {logits_.get()};
                marmot_error_t run_status = execute_packed_graph(
                    graph, fast_plan, ctx_, std::span<const marmot_tensor_t *const>(inputs, ARRAY_LENGTH(inputs)),
                    std::span<marmot_tensor_t *const>(outputs, ARRAY_LENGTH(outputs))
                );
                if (run_status != MARMOT_SUCCESS) {
                    const char *detail = marmot_get_last_error_detail();
                    error = (detail != nullptr && detail[0] != '\0') ? detail : "packed graph execute failed";
                    abort_append_plans(append_plans);
                    return run_status;
                }
            }
        } else {
            if (use_fp8_kv) {
                const marmot_tensor_t *inputs[] = {
                    hidden_.get(), positions_.get(), token_meta_.get(), block_table, kv_k, kv_v, kv_k_scale, kv_v_scale,
                };
                marmot_tensor_t *outputs[] = {hidden_out_.get()};
                marmot_error_t run_status = execute_packed_graph(
                    graph, fast_plan, ctx_, std::span<const marmot_tensor_t *const>(inputs, ARRAY_LENGTH(inputs)),
                    std::span<marmot_tensor_t *const>(outputs, ARRAY_LENGTH(outputs))
                );
                if (run_status != MARMOT_SUCCESS) {
                    const char *detail = marmot_get_last_error_detail();
                    error = (detail != nullptr && detail[0] != '\0') ? detail : "packed graph execute failed";
                    abort_append_plans(append_plans);
                    return run_status;
                }
            } else {
                const marmot_tensor_t *inputs[] = {
                    hidden_.get(), positions_.get(), token_meta_.get(), block_table, kv_k, kv_v,
                };
                marmot_tensor_t *outputs[] = {hidden_out_.get()};
                marmot_error_t run_status = execute_packed_graph(
                    graph, fast_plan, ctx_, std::span<const marmot_tensor_t *const>(inputs, ARRAY_LENGTH(inputs)),
                    std::span<marmot_tensor_t *const>(outputs, ARRAY_LENGTH(outputs))
                );
                if (run_status != MARMOT_SUCCESS) {
                    const char *detail = marmot_get_last_error_detail();
                    error = (detail != nullptr && detail[0] != '\0') ? detail : "packed graph execute failed";
                    abort_append_plans(append_plans);
                    return run_status;
                }
            }
        }

        if (profiling) {
            graph_end = StepTimer::now_ns();
        }

        struct SampleResult {
            Request *req{nullptr};
            marmot_token_id_t token{MARMOT_TOKEN_ID_INVALID};
            bool finished{false};
        };
        std::vector<SampleResult> sample_results;

        if (sample_count > 0) {
            const marmot_tensor_t *logits_tensor = logits_.get();
            const size_t vocab = n_vocab_;
            const size_t total_elems = marmot_tensor_num_elements(logits_tensor);

            bool use_device_argmax = false;
            if (ctx_->backend_type != MARMOT_BACKEND_CPU && logits_argmax_ && logits_max_) {
                use_device_argmax = true;
                for (marmot_request_id_t req_id : batch.sample_request_ids) {
                    auto it = requests_.find(req_id);
                    if (it == requests_.end()) {
                        use_device_argmax = false;
                        break;
                    }
                    const Request &req = it->second;
                    if (req.cancel_requested) {
                        continue;
                    }
                    const marmot_llm_sampling_options_t &sampling = req.sampling_opts;
                    if (sampling.temperature > 0.0f) {
                        use_device_argmax = false;
                        break;
                    }
                    if (sampling.top_k > 1) {
                        use_device_argmax = false;
                        break;
                    }
                    if (sampling.repetition_penalty != 1.0f) {
                        use_device_argmax = false;
                        break;
                    }
                    if ((sampling.flags & MARMOT_LLM_SAMPLING_FLAG_SUPPRESS_SPECIAL_TOKENS) != 0) {
                        use_device_argmax = false;
                        break;
                    }
                }
            }

            const marmot_uint64_t *device_argmax = nullptr;
            if (use_device_argmax) {
                int32_t axis = 1;
                marmot_reduction_desc_t argmax = marmot_reduction_desc_default();
                argmax.input = logits_tensor;
                argmax.out = logits_max_.get();
                argmax.indices_out = logits_argmax_.get();
                argmax.axes = &axis;
                argmax.num_axes = 1;
                argmax.keepdims = false;

                marmot_error_t argmax_status = marmot_reduce_argmax(ctx_, &argmax);
                if (argmax_status != MARMOT_SUCCESS) {
                    const char *detail = marmot_get_last_error_detail();
                    error = (detail != nullptr && detail[0] != '\0') ? detail : "argmax failed";
                    abort_append_plans(append_plans);
                    return argmax_status;
                }

                if (!batch_has_prefill && sample_count == token_count && ctx_->backend_type != MARMOT_BACKEND_CPU) {
                    marmot_error_t scatter_status =
                        marmot_scatter_u64_to_i32(ctx_, logits_argmax_.get(), sample_indices_.get(), token_ids_.get());
                    if (scatter_status != MARMOT_SUCCESS) {
                        const char *detail = marmot_get_last_error_detail();
                        error = (detail != nullptr && detail[0] != '\0') ? detail : "scatter_u64_to_i32 failed";
                        abort_append_plans(append_plans);
                        return scatter_status;
                    }
                    token_ids_device_ready_ = true;
                }

                device_argmax = marmot_tensor_data_u64(ctx_, logits_argmax_.get());
                if (device_argmax == nullptr) {
                    const marmot_error_t data_err = marmot_get_last_error();
                    const char *detail = marmot_get_last_error_detail();
                    error = (detail != nullptr && detail[0] != '\0') ? detail : "argmax indices unavailable";
                    abort_append_plans(append_plans);
                    return data_err != MARMOT_SUCCESS ? data_err : MARMOT_ERROR_INVALID_OPERATION;
                }
            }

            auto sample_from_device_argmax = [&](size_t row_index, marmot_token_id_t &out_token) {
                if (device_argmax == nullptr || row_index >= graph_sample_count) {
                    error = "argmax row out of bounds";
                    return MARMOT_ERROR_INVALID_OPERATION;
                }
                const uint64_t index = device_argmax[row_index].value;
                if (index >= vocab) {
                    error = "argmax index out of range";
                    return MARMOT_ERROR_INVALID_OPERATION;
                }
                out_token = static_cast<marmot_token_id_t>(index);
                return MARMOT_SUCCESS;
            };

            auto sample_from_logits = [&](Request &req, size_t row_index, marmot_token_id_t &out_token) {
                if (row_index > SIZE_MAX / vocab) {
                    error = "logits offset overflow";
                    return MARMOT_ERROR_INVALID_OPERATION;
                }
                const size_t offset = row_index * vocab;
                if (offset + vocab > total_elems) {
                    error = "logits row out of bounds";
                    return MARMOT_ERROR_INVALID_OPERATION;
                }

                if (logits_tensor->dtype == MARMOT_DTYPE_FLOAT16) {
                    const marmot_float16_t *logits_f16 = marmot_tensor_data_f16(ctx_, logits_.get());
                    if (logits_f16 == nullptr) {
                        error = "logits tensor missing data";
                        return MARMOT_ERROR_INVALID_OPERATION;
                    }
                    const marmot_float16_t *row = logits_f16 + offset;
                    for (size_t i = 0; i < vocab; ++i) {
                        logits_f32_[i] = marmot_f16_to_f32_ref(row[i]);
                    }
                } else if (logits_tensor->dtype == MARMOT_DTYPE_BFLOAT16) {
                    const marmot_bfloat16_t *logits_bf16 = marmot_tensor_data_bf16(ctx_, logits_.get());
                    if (logits_bf16 == nullptr) {
                        error = "logits tensor missing data";
                        return MARMOT_ERROR_INVALID_OPERATION;
                    }
                    const marmot_bfloat16_t *row = logits_bf16 + offset;
                    for (size_t i = 0; i < vocab; ++i) {
                        logits_f32_[i] = marmot_bf16_to_f32_ref(row[i]);
                    }
                } else if (logits_tensor->dtype == MARMOT_DTYPE_FLOAT32) {
                    const float *logits_f32 = marmot_tensor_data_f32(ctx_, logits_.get());
                    if (logits_f32 == nullptr) {
                        error = "logits tensor missing data";
                        return MARMOT_ERROR_INVALID_OPERATION;
                    }
                    const float *row = logits_f32 + offset;
                    std::copy(row, row + vocab, logits_f32_.begin());
                } else {
                    error = "unsupported logits dtype";
                    return MARMOT_ERROR_UNSUPPORTED_DTYPE;
                }

                std::vector<marmot_token_id_t> history;
                history.reserve(req.prompt_tokens.size() + req.generated_tokens.size());
                history.insert(history.end(), req.prompt_tokens.begin(), req.prompt_tokens.end());
                history.insert(history.end(), req.generated_tokens.begin(), req.generated_tokens.end());

                apply_repetition_penalty(
                    logits_f32_, history, req.sampling_opts.repetition_penalty, seen_, seen_tokens_
                );

                if ((req.sampling_opts.flags & MARMOT_LLM_SAMPLING_FLAG_SUPPRESS_SPECIAL_TOKENS) != 0 &&
                    !special_mask_.empty()) {
                    const size_t limit = std::min(vocab, special_mask_.size());
                    const size_t allow_limit =
                        req.allowed_special_mask.empty() ? 0 : std::min(vocab, req.allowed_special_mask.size());
                    for (size_t i = 0; i < limit; ++i) {
                        if (special_mask_[i] == 0u) {
                            continue;
                        }
                        if (eos_id_ != MARMOT_TOKEN_ID_INVALID && (marmot_token_id_t)i == eos_id_) {
                            continue;
                        }
                        if (allow_limit > 0 && i < allow_limit && req.allowed_special_mask[i] != 0u) {
                            continue;
                        }
                        logits_f32_[i] = -INFINITY;
                    }
                }

                const float temperature = req.sampling_opts.temperature;
                if (!(temperature > 0.0f)) {
                    size_t best = 0;
                    float best_logit = logits_f32_[0];
                    for (size_t i = 1; i < vocab; ++i) {
                        if (logits_f32_[i] > best_logit) {
                            best_logit = logits_f32_[i];
                            best = i;
                        }
                    }
                    out_token = (marmot_token_id_t)best;
                    return MARMOT_SUCCESS;
                }

                std::vector<size_t> indices(vocab);
                std::iota(indices.begin(), indices.end(), 0);

                if (req.sampling_opts.top_k > 0 && req.sampling_opts.top_k < vocab) {
                    const size_t k = req.sampling_opts.top_k;
                    std::nth_element(indices.begin(), indices.begin() + k, indices.end(), [&](size_t a, size_t b) {
                        return logits_f32_[a] > logits_f32_[b];
                    });
                    indices.resize(k);
                }

                for (size_t idx : indices) {
                    logits_f32_[idx] /= temperature;
                }

                std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
                    return logits_f32_[a] > logits_f32_[b];
                });
                const float max_logit = logits_f32_[indices.front()];

                std::vector<float> probs(indices.size(), 0.0f);
                double sum = 0.0;
                for (size_t i = 0; i < indices.size(); ++i) {
                    const float x = logits_f32_[indices[i]] - max_logit;
                    const float p = std::exp(x);
                    probs[i] = p;
                    sum += (double)p;
                }
                if (sum <= 0.0) {
                    out_token = (marmot_token_id_t)indices.front();
                    return MARMOT_SUCCESS;
                }

                for (float &p : probs) {
                    p = (float)((double)p / sum);
                }

                if (req.sampling_opts.top_p > 0.0f && req.sampling_opts.top_p < 1.0f) {
                    float cumulative = 0.0f;
                    size_t keep = 0;
                    for (; keep < probs.size(); ++keep) {
                        cumulative += probs[keep];
                        if (cumulative >= req.sampling_opts.top_p) {
                            keep += 1;
                            break;
                        }
                    }
                    keep = std::max<size_t>(1, std::min(keep, probs.size()));
                    indices.resize(keep);
                    probs.resize(keep);

                    const float renorm = std::accumulate(probs.begin(), probs.end(), 0.0f);
                    if (renorm > 0.0f) {
                        for (float &p : probs) {
                            p /= renorm;
                        }
                    }
                }

                if (req.sampling_opts.min_p > 0.0f) {
                    const float max_prob = probs.empty() ? 0.0f : probs.front();
                    const float threshold = max_prob * req.sampling_opts.min_p;
                    size_t keep = 0;
                    for (; keep < probs.size(); ++keep) {
                        if (probs[keep] < threshold) {
                            break;
                        }
                    }
                    keep = std::max<size_t>(1, std::min(keep, probs.size()));
                    indices.resize(keep);
                    probs.resize(keep);

                    const float renorm = std::accumulate(probs.begin(), probs.end(), 0.0f);
                    if (renorm > 0.0f) {
                        for (float &p : probs) {
                            p /= renorm;
                        }
                    }
                }

                std::discrete_distribution<size_t> dist(probs.begin(), probs.end());
                const size_t pick = dist(req.rng);
                out_token = (marmot_token_id_t)indices[pick];
                return MARMOT_SUCCESS;
            };

            auto finished_for_token = [&](const Request &req, marmot_token_id_t token) {
                if (req.gen_opts.stop_on_eos && eos_id_ != MARMOT_TOKEN_ID_INVALID && token == eos_id_) {
                    return true;
                }
                if (req.gen_opts.stop_tokens != nullptr && req.gen_opts.stop_tokens_len > 0) {
                    for (size_t j = 0; j < req.gen_opts.stop_tokens_len; ++j) {
                        if (req.gen_opts.stop_tokens[j] == token) {
                            return true;
                        }
                    }
                }
                if (req.gen_opts.max_new_tokens > 0 && req.generated_tokens.size() + 1 >= req.gen_opts.max_new_tokens) {
                    return true;
                }
                return false;
            };

            auto attach_clone_prefix = [&](Request &parent) -> marmot_error_t {
                if (parent.clone_ids.empty()) {
                    return MARMOT_SUCCESS;
                }
                if (opts_.block_size == 0) {
                    error = "block_size is zero";
                    return MARMOT_ERROR_INVALID_OPERATION;
                }
                const size_t prompt_len = parent.prompt_tokens.size();
                const size_t num_blocks = (prompt_len + opts_.block_size - 1) / opts_.block_size;
                if (num_blocks == 0) {
                    return MARMOT_SUCCESS;
                }

                std::vector<marmot_block_id_t> block_ids;
                block_ids.reserve(num_blocks);
                for (size_t block = 0; block < num_blocks; ++block) {
                    const marmot_block_id_t block_id = kv_pool_->block_id(parent.seq_slot, block);
                    if (block_id == MARMOT_BLOCK_ID_INVALID) {
                        error = "clone prefix attach missing block";
                        return MARMOT_ERROR_INVALID_OPERATION;
                    }
                    block_ids.push_back(block_id);
                }

                for (marmot_request_id_t clone_id : parent.clone_ids) {
                    auto it = requests_.find(clone_id);
                    if (it == requests_.end()) {
                        continue;
                    }
                    Request &clone = it->second;
                    if (clone.cancel_requested) {
                        continue;
                    }
                    if (!clone.awaiting_prefix_attach) {
                        continue;
                    }
                    KVPool::PrefixPlan plan{};
                    marmot_error_t attach_status = kv_pool_->prepare_prefix_attach(
                        clone.seq_slot, block_ids.data(), block_ids.size(), prompt_len, plan
                    );
                    if (attach_status == MARMOT_SUCCESS) {
                        attach_status = kv_pool_->commit_prefix_attach(plan);
                    }
                    if (attach_status != MARMOT_SUCCESS) {
                        kv_pool_->abort_prefix_attach(plan);
                        error = "clone prefix attach failed";
                        return attach_status;
                    }
                    clone.prompt_cursor = prompt_len;
                    clone.awaiting_prefix_attach = false;
                    if (clone.retention_blocks > 0) {
                        const size_t retain_blocks = std::min(clone.retention_blocks, block_ids.size());
                        for (size_t block = 0; block < retain_blocks; ++block) {
                            kv_pool_->set_block_retained(block_ids[block], true);
                        }
                    }
                }

                return MARMOT_SUCCESS;
            };

            sample_results.reserve(sample_count);
            for (size_t i = 0; i < sample_count; ++i) {
                const marmot_request_id_t request_id = batch.sample_request_ids[i];
                auto it = requests_.find(request_id);
                if (it == requests_.end()) {
                    error = "sample request not found";
                    abort_append_plans(append_plans);
                    return MARMOT_ERROR_INVALID_OPERATION;
                }
                Request &req = it->second;
                if (req.cancel_requested) {
                    sample_results.push_back(SampleResult{&req, MARMOT_TOKEN_ID_INVALID, true});
                    continue;
                }

                marmot_token_id_t token = MARMOT_TOKEN_ID_INVALID;
                marmot_error_t sample_status =
                    use_device_argmax ? sample_from_device_argmax(i, token) : sample_from_logits(req, i, token);
                if (sample_status != MARMOT_SUCCESS) {
                    abort_append_plans(append_plans);
                    return sample_status;
                }
                sample_results.push_back(SampleResult{&req, token, finished_for_token(req, token)});

                const uint32_t token_index = batch.sample_indices[i];
                const uint32_t flags = batch.token_meta[token_index * 4 + 3];
                if ((flags & kTokenFlagPrefill) == 0) {
                    continue;
                }
                if (req.sample_index != 0 || req.num_samples <= 1 || req.needs_recompute) {
                    continue;
                }

                bool has_pending_clones = false;
                for (marmot_request_id_t clone_id : req.clone_ids) {
                    auto clone_it = requests_.find(clone_id);
                    if (clone_it != requests_.end() && clone_it->second.awaiting_prefix_attach &&
                        !clone_it->second.cancel_requested) {
                        has_pending_clones = true;
                        break;
                    }
                }
                if (!has_pending_clones) {
                    continue;
                }

                marmot_error_t attach_status = attach_clone_prefix(req);
                if (attach_status != MARMOT_SUCCESS) {
                    abort_append_plans(append_plans);
                    return attach_status;
                }

                for (marmot_request_id_t clone_id : req.clone_ids) {
                    auto clone_it = requests_.find(clone_id);
                    if (clone_it == requests_.end()) {
                        continue;
                    }
                    Request &clone = clone_it->second;
                    if (clone.awaiting_prefix_attach) {
                        continue;
                    }
                    if (clone.cancel_requested) {
                        sample_results.push_back(SampleResult{&clone, MARMOT_TOKEN_ID_INVALID, true});
                        continue;
                    }
                    marmot_token_id_t clone_token = MARMOT_TOKEN_ID_INVALID;
                    marmot_error_t clone_status = use_device_argmax ? sample_from_device_argmax(i, clone_token)
                                                                    : sample_from_logits(clone, i, clone_token);
                    if (clone_status != MARMOT_SUCCESS) {
                        abort_append_plans(append_plans);
                        return clone_status;
                    }
                    sample_results.push_back(SampleResult{&clone, clone_token, finished_for_token(clone, clone_token)});
                }
            }
        }

        for (const auto &pending : append_plans) {
            marmot_error_t status = kv_pool_->commit_append(pending.plan);
            if (status != MARMOT_SUCCESS) {
                error = "KV pool commit failed";
                abort_append_plans(append_plans);
                return status;
            }
        }

        if ((opts_.flags & MARMOT_SERVING_ENGINE_FLAG_ENABLE_PREFIX_CACHE) != 0 && opts_.block_size > 0) {
            const size_t block_size = opts_.block_size;
            for (const auto &pending : append_plans) {
                auto it = requests_.find(pending.request_id);
                if (it == requests_.end()) {
                    continue;
                }
                Request &req = it->second;
                if ((req.flags & MARMOT_SERVING_REQUEST_FLAG_DISABLE_PREFIX_CACHE) != 0) {
                    continue;
                }
                if (!req.has_seq_slot) {
                    continue;
                }

                const size_t start_pos = pending.plan.start_pos;
                const size_t new_seq_len = pending.plan.start_pos + pending.plan.token_count;
                const size_t old_full_blocks = start_pos / block_size;
                const size_t new_full_blocks = new_seq_len / block_size;
                if (new_full_blocks <= old_full_blocks) {
                    continue;
                }

                const size_t prompt_len = req.prompt_tokens.size();
                size_t appended_gen = 0;
                if (new_seq_len > prompt_len) {
                    appended_gen = new_seq_len - prompt_len;
                    if (appended_gen > req.generated_tokens.size()) {
                        appended_gen = req.generated_tokens.size();
                    }
                }

                auto token_at = [&](size_t index) -> marmot_token_id_t {
                    if (index < prompt_len) {
                        return req.prompt_tokens[index];
                    }
                    const size_t gen_index = index - prompt_len;
                    if (gen_index < appended_gen && gen_index < req.generated_tokens.size()) {
                        return req.generated_tokens[gen_index];
                    }
                    return MARMOT_TOKEN_ID_INVALID;
                };

                if (req.prefix_hashes.size() > old_full_blocks) {
                    req.prefix_hashes.resize(old_full_blocks);
                }

                uint64_t hash = hash_seed(req.cache_salt);
                if (old_full_blocks > 0) {
                    if (req.prefix_hashes.size() == old_full_blocks) {
                        hash = req.prefix_hashes.back();
                    } else {
                        req.prefix_hashes.clear();
                        bool valid = true;
                        for (size_t block = 0; block < old_full_blocks; ++block) {
                            const size_t base = block * block_size;
                            for (size_t i = 0; i < block_size; ++i) {
                                const marmot_token_id_t token = token_at(base + i);
                                if (token == MARMOT_TOKEN_ID_INVALID) {
                                    valid = false;
                                    break;
                                }
                                hash = hash_token(hash, token);
                            }
                            if (!valid) {
                                break;
                            }
                            req.prefix_hashes.push_back(hash);
                        }
                        if (!valid || req.prefix_hashes.size() != old_full_blocks) {
                            continue;
                        }
                    }
                }

                for (size_t block = old_full_blocks; block < new_full_blocks; ++block) {
                    const size_t base = block * block_size;
                    bool valid = true;
                    for (size_t i = 0; i < block_size; ++i) {
                        const marmot_token_id_t token = token_at(base + i);
                        if (token == MARMOT_TOKEN_ID_INVALID) {
                            valid = false;
                            break;
                        }
                        hash = hash_token(hash, token);
                    }
                    if (!valid) {
                        break;
                    }

                    req.prefix_hashes.push_back(hash);

                    const marmot_block_id_t block_id = kv_pool_->block_id(req.seq_slot, block);
                    const uint32_t generation = kv_pool_->block_generation(block_id);
                    if (block_id == MARMOT_BLOCK_ID_INVALID || generation == 0) {
                        continue;
                    }

                    if (block_id < block_hash_valid_.size() && block_hash_valid_[block_id] != 0) {
                        const uint64_t old_hash = block_hash_by_id_[block_id];
                        auto entry_it = prefix_cache_.find(old_hash);
                        if (entry_it != prefix_cache_.end() && entry_it->second.block_id == block_id) {
                            prefix_cache_.erase(entry_it);
                        }
                    }

                    if (block_id < block_hash_by_id_.size() && block_id < block_hash_valid_.size()) {
                        block_hash_by_id_[block_id] = hash;
                        block_hash_valid_[block_id] = 1;
                    }
                    prefix_cache_[hash] = PrefixEntry{block_id, generation, ++prefix_tick_};
                }

                if (req.retention_blocks > 0) {
                    size_t prompt_blocks = prompt_len / block_size;
                    if (req.gen_opts.max_new_tokens != 0 && prompt_len % block_size == 0 && prompt_blocks > 0) {
                        prompt_blocks -= 1;
                    }
                    const size_t retain_blocks = std::min(req.retention_blocks, prompt_blocks);
                    for (size_t block = 0; block < retain_blocks; ++block) {
                        const marmot_block_id_t block_id = kv_pool_->block_id(req.seq_slot, block);
                        if (block_id != MARMOT_BLOCK_ID_INVALID) {
                            kv_pool_->set_block_retained(block_id, true);
                        }
                    }
                }
            }
        }

        for (const auto &work : prefill_work) {
            if (work.req == nullptr) {
                continue;
            }
            if (work.req->cancel_requested) {
                work.req->state = MARMOT_LLM_REQUEST_STATE_CANCELED;
                work.req->has_pending_token = false;
                work.req->pending_input_token = MARMOT_TOKEN_ID_INVALID;
                continue;
            }
            work.req->prompt_cursor += work.chunk;
            if (work.req->prompt_cursor < work.prompt_len) {
                work.req->state = MARMOT_LLM_REQUEST_STATE_PREFILL;
            } else if (work.finish_without_sample) {
                work.req->state = MARMOT_LLM_REQUEST_STATE_DONE;
                work.req->has_pending_token = false;
                work.req->pending_input_token = MARMOT_TOKEN_ID_INVALID;
            }
            if (work.req->needs_recompute && work.req->prompt_cursor >= work.prompt_len) {
                work.req->needs_recompute = false;
                work.req->recompute_tokens.clear();
                work.req->prompt_cursor = work.req->prompt_tokens.size();
            }
        }

        for (const auto &result : sample_results) {
            if (result.req == nullptr) {
                continue;
            }
            Request &req = *result.req;
            if (req.cancel_requested) {
                req.state = MARMOT_LLM_REQUEST_STATE_CANCELED;
                req.has_pending_token = false;
                req.pending_input_token = MARMOT_TOKEN_ID_INVALID;
                continue;
            }
            if (result.token != MARMOT_TOKEN_ID_INVALID) {
                req.generated_tokens.push_back(result.token);
                if (req.gen_opts.on_token != nullptr) {
                    req.gen_opts.on_token(req.gen_opts.user_data, result.token);
                }
            }
            if (result.finished) {
                req.state = MARMOT_LLM_REQUEST_STATE_DONE;
                req.has_pending_token = false;
                req.pending_input_token = MARMOT_TOKEN_ID_INVALID;
            } else {
                req.state = MARMOT_LLM_REQUEST_STATE_DECODING;
                req.pending_input_token = result.token;
                req.has_pending_token = true;
            }
        }

        last_batch_ = std::move(batch);

        for (auto it = requests_.begin(); it != requests_.end();) {
            marmot_llm_request_state_t state = it->second.state;
            if (state == MARMOT_LLM_REQUEST_STATE_CANCELED || state == MARMOT_LLM_REQUEST_STATE_DONE ||
                state == MARMOT_LLM_REQUEST_STATE_FAILED) {
                if (it->second.has_seq_slot && kv_pool_ != nullptr) {
                    (void)kv_pool_->release_seq(it->second.seq_slot);
                }
                it = requests_.erase(it);
                continue;
            }
            ++it;
        }

        if (!schedule_.empty()) {
            schedule_.erase(
                std::remove_if(
                    schedule_.begin(), schedule_.end(),
                    [&](marmot_request_id_t request_id) { return requests_.find(request_id) == requests_.end(); }
                ),
                schedule_.end()
            );
        }

        if (profiling) {
            const double step_end = StepTimer::now_ns();
            timer.batch_build_ns = batch_end - step_start;
            timer.host_sync_ns = sync_end - batch_end;
            timer.embedding_ns = embed_end - sync_end;
            timer.graph_exec_ns = graph_end - embed_end;
            timer.sampling_ns = step_end - graph_end;
            timer.total_ns = step_end - step_start;
            timer.token_count = token_count;
            timer.sample_count = sample_count;
            timer.has_prefill = batch_has_prefill;
            timer.report();
        }
    }

    return MARMOT_SUCCESS;
}

marmot_error_t ServingEngine::last_batch_view(marmot_serving_engine_batch_view_t &out_batch) const noexcept {
    out_batch.token_count = last_batch_.token_ids.size();
    out_batch.sample_count = last_batch_.sample_indices.size();
    out_batch.token_ids = last_batch_.token_ids.empty() ? nullptr : last_batch_.token_ids.data();
    out_batch.token_meta = last_batch_.token_meta.empty() ? nullptr : last_batch_.token_meta.data();
    out_batch.sample_indices = last_batch_.sample_indices.empty() ? nullptr : last_batch_.sample_indices.data();
    out_batch.sample_request_ids =
        last_batch_.sample_request_ids.empty() ? nullptr : last_batch_.sample_request_ids.data();
    return MARMOT_SUCCESS;
}

marmot_llm_request_state_t ServingEngine::request_state(marmot_request_id_t request_id) const noexcept {
    auto it = requests_.find(request_id);
    if (it == requests_.end()) {
        return MARMOT_LLM_REQUEST_STATE_INVALID;
    }
    return it->second.state;
}

marmot_error_t ServingEngine::request_cancel(marmot_request_id_t request_id, std::string &error) {
    error.clear();
    auto it = requests_.find(request_id);
    if (it == requests_.end()) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    it->second.cancel_requested = true;
    if (it->second.state == MARMOT_LLM_REQUEST_STATE_PENDING) {
        it->second.state = MARMOT_LLM_REQUEST_STATE_CANCELED;
    }
    if (!it->second.clone_ids.empty()) {
        for (marmot_request_id_t clone_id : it->second.clone_ids) {
            auto clone_it = requests_.find(clone_id);
            if (clone_it == requests_.end()) {
                continue;
            }
            Request &clone = clone_it->second;
            if (!clone.awaiting_prefix_attach) {
                continue;
            }
            clone.cancel_requested = true;
            if (clone.state == MARMOT_LLM_REQUEST_STATE_PENDING) {
                clone.state = MARMOT_LLM_REQUEST_STATE_CANCELED;
            }
        }
    }
    return MARMOT_SUCCESS;
}

marmot_error_t ServingEngine::request_release(marmot_request_id_t request_id, std::string &error) {
    error.clear();
    auto it = requests_.find(request_id);
    if (it == requests_.end()) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (it->second.has_seq_slot && kv_pool_ != nullptr) {
        marmot_error_t status = kv_pool_->release_seq(it->second.seq_slot);
        if (status != MARMOT_SUCCESS) {
            return status;
        }
    }

    requests_.erase(it);
    schedule_.erase(std::remove(schedule_.begin(), schedule_.end(), request_id), schedule_.end());
    return MARMOT_SUCCESS;
}

} // namespace marmot::inference
