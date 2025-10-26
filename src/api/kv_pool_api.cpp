#include "marmot/inference/kv_pool.h"

#include <cstring>
#include <memory>
#include <new>

#include "inference/kv_pool/kv_pool.hpp"

struct marmot_kv_pool {
    std::unique_ptr<marmot::inference::KVPool> inner;
};

namespace {

[[nodiscard]] marmot_error_t validate_options(const marmot_kv_pool_options_t *opts) {
    if (opts == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (opts->struct_version != MARMOT_KV_POOL_OPTIONS_VERSION) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (opts->struct_size < sizeof(marmot_kv_pool_options_t)) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    return MARMOT_SUCCESS;
}

} // namespace

extern "C" {

marmot_error_t marmot_kv_pool_options_init(marmot_kv_pool_options_t *opts) {
    try {
        if (opts == nullptr) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        std::memset(opts, 0, sizeof(*opts));
        opts->struct_size = sizeof(marmot_kv_pool_options_t);
        opts->struct_version = MARMOT_KV_POOL_OPTIONS_VERSION;
        opts->flags = 0;
        opts->backend = MARMOT_BACKEND_CPU;
        opts->max_seqs = 1;
        opts->max_seq_len = 2048;
        opts->block_size = 16;
        opts->num_blocks = 256;
        opts->num_layers = 1;
        opts->num_kv_heads = 1;
        opts->head_dim = 64;
        opts->num_swap_blocks = 0;
        opts->kv_dtype = MARMOT_DTYPE_FLOAT16;
        opts->pnext = nullptr;
        return MARMOT_SUCCESS;
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_kv_pool_options_init threw");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
}

marmot_error_t
marmot_kv_pool_create(const marmot_context_t *ctx, const marmot_kv_pool_options_t *opts, marmot_kv_pool_t **out_pool) {
    try {
        if (out_pool == nullptr) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        *out_pool = nullptr;

        marmot_error_t status = validate_options(opts);
        if (status != MARMOT_SUCCESS) {
            return status;
        }

        if (ctx == nullptr) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "KV pool create requires valid context");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }

        marmot::inference::KVPool::Options cpp_opts{};
        cpp_opts.backend = opts->backend;
        cpp_opts.max_seqs = opts->max_seqs;
        cpp_opts.max_seq_len = opts->max_seq_len;
        cpp_opts.block_size = opts->block_size;
        cpp_opts.num_blocks = opts->num_blocks;
        cpp_opts.num_layers = opts->num_layers;
        cpp_opts.num_kv_heads = opts->num_kv_heads;
        cpp_opts.head_dim = opts->head_dim;
        cpp_opts.kv_dtype = opts->kv_dtype;
        cpp_opts.num_swap_blocks = opts->num_swap_blocks;
        cpp_opts.deterministic_alloc = (opts->flags & MARMOT_KV_POOL_FLAG_DETERMINISTIC_ALLOC) != 0;

        auto result = marmot::inference::KVPool::create(ctx, cpp_opts);
        if (!result) {
            return result.error();
        }

        auto *pool = new (std::nothrow) marmot_kv_pool();
        if (pool == nullptr) {
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to allocate KV pool handle");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }

        pool->inner = std::move(*result);
        *out_pool = pool;
        return MARMOT_SUCCESS;
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_kv_pool_create threw");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
}

void marmot_kv_pool_destroy(marmot_kv_pool_t *pool) {
    try {
        delete pool;
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_kv_pool_destroy threw");
    }
}

marmot_error_t marmot_kv_pool_acquire_seq(marmot_kv_pool_t *pool, marmot_seq_slot_t *out_seq) {
    try {
        if (pool == nullptr || pool->inner == nullptr || out_seq == nullptr) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        return pool->inner->acquire_seq(*out_seq);
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_kv_pool_acquire_seq threw");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
}

marmot_error_t marmot_kv_pool_release_seq(marmot_kv_pool_t *pool, marmot_seq_slot_t seq) {
    try {
        if (pool == nullptr || pool->inner == nullptr) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        return pool->inner->release_seq(seq);
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_kv_pool_release_seq threw");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
}

marmot_error_t marmot_kv_pool_prepare_append(
    marmot_kv_pool_t *pool, marmot_seq_slot_t seq, size_t token_count, marmot_kv_append_plan_t *out_plan,
    marmot_kv_slot_t *out_slots, size_t *out_start_pos
) {
    try {
        if (pool == nullptr || pool->inner == nullptr || out_plan == nullptr || out_start_pos == nullptr) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        if (token_count != 0 && out_slots == nullptr) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }

        out_plan->seq = seq;
        out_plan->token_count = token_count;
        out_plan->cookie = 0;

        auto *plan = new (std::nothrow) marmot::inference::KVPool::AppendPlan();
        if (plan == nullptr) {
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }

        size_t start_pos = 0;
        marmot_error_t status = pool->inner->prepare_append(seq, token_count, out_slots, start_pos, *plan);
        if (status != MARMOT_SUCCESS) {
            delete plan;
            return status;
        }

        out_plan->cookie = reinterpret_cast<uint64_t>(plan);
        *out_start_pos = start_pos;
        return MARMOT_SUCCESS;
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_kv_pool_prepare_append threw");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
}

marmot_error_t marmot_kv_pool_commit_append(marmot_kv_pool_t *pool, const marmot_kv_append_plan_t *plan) {
    try {
        if (pool == nullptr || pool->inner == nullptr || plan == nullptr) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        if (plan->cookie == 0) {
            return MARMOT_SUCCESS;
        }

        auto *cpp_plan = reinterpret_cast<marmot::inference::KVPool::AppendPlan *>(plan->cookie);
        marmot_error_t status = pool->inner->commit_append(*cpp_plan);
        if (status == MARMOT_SUCCESS) {
            delete cpp_plan;
        }
        return status;
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_kv_pool_commit_append threw");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
}

void marmot_kv_pool_abort_append(marmot_kv_pool_t *pool, const marmot_kv_append_plan_t *plan) {
    try {
        if (pool == nullptr || pool->inner == nullptr || plan == nullptr) {
            return;
        }
        if (plan->cookie == 0) {
            return;
        }

        auto *cpp_plan = reinterpret_cast<marmot::inference::KVPool::AppendPlan *>(plan->cookie);
        pool->inner->abort_append(*cpp_plan);
        delete cpp_plan;
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_kv_pool_abort_append threw");
    }
}

marmot_error_t marmot_kv_pool_prepare_prefix_attach(
    marmot_kv_pool_t *pool, marmot_seq_slot_t seq, const marmot_block_id_t *block_ids, size_t num_blocks,
    size_t prefix_len, marmot_kv_prefix_plan_t *out_plan
) {
    try {
        if (pool == nullptr || pool->inner == nullptr || out_plan == nullptr) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }

        out_plan->seq = seq;
        out_plan->num_blocks = num_blocks;
        out_plan->prefix_len = prefix_len;
        out_plan->cookie = 0;

        auto *plan = new (std::nothrow) marmot::inference::KVPool::PrefixPlan();
        if (plan == nullptr) {
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }

        marmot_error_t status = pool->inner->prepare_prefix_attach(seq, block_ids, num_blocks, prefix_len, *plan);
        if (status != MARMOT_SUCCESS) {
            delete plan;
            return status;
        }

        out_plan->cookie = reinterpret_cast<uint64_t>(plan);
        return MARMOT_SUCCESS;
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_kv_pool_prepare_prefix_attach threw");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
}

marmot_error_t marmot_kv_pool_commit_prefix_attach(marmot_kv_pool_t *pool, const marmot_kv_prefix_plan_t *plan) {
    try {
        if (pool == nullptr || pool->inner == nullptr || plan == nullptr) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        if (plan->cookie == 0) {
            return MARMOT_SUCCESS;
        }

        auto *cpp_plan = reinterpret_cast<marmot::inference::KVPool::PrefixPlan *>(plan->cookie);
        marmot_error_t status = pool->inner->commit_prefix_attach(*cpp_plan);
        if (status == MARMOT_SUCCESS) {
            delete cpp_plan;
        }
        return status;
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_kv_pool_commit_prefix_attach threw");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
}

void marmot_kv_pool_abort_prefix_attach(marmot_kv_pool_t *pool, const marmot_kv_prefix_plan_t *plan) {
    try {
        if (pool == nullptr || pool->inner == nullptr || plan == nullptr) {
            return;
        }
        if (plan->cookie == 0) {
            return;
        }

        auto *cpp_plan = reinterpret_cast<marmot::inference::KVPool::PrefixPlan *>(plan->cookie);
        pool->inner->abort_prefix_attach(*cpp_plan);
        delete cpp_plan;
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_kv_pool_abort_prefix_attach threw");
    }
}

marmot_error_t marmot_kv_pool_get_tensors(
    marmot_kv_pool_t *pool, marmot_tensor_t **out_k, marmot_tensor_t **out_v, marmot_tensor_t **out_block_table
) {
    try {
        if (pool == nullptr || pool->inner == nullptr) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        if (out_k == nullptr && out_v == nullptr && out_block_table == nullptr) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        pool->inner->get_tensors(out_k, out_v, out_block_table);
        return MARMOT_SUCCESS;
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_kv_pool_get_tensors threw");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
}

marmot_error_t
marmot_kv_pool_get_scale_tensors(marmot_kv_pool_t *pool, marmot_tensor_t **out_k_scale, marmot_tensor_t **out_v_scale) {
    try {
        if (pool == nullptr || pool->inner == nullptr) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        if (out_k_scale == nullptr && out_v_scale == nullptr) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        pool->inner->get_tensors(nullptr, nullptr, nullptr, out_k_scale, out_v_scale);
        return MARMOT_SUCCESS;
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_kv_pool_get_scale_tensors threw");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
}

size_t marmot_kv_pool_free_block_count(const marmot_kv_pool_t *pool) {
    try {
        if (pool == nullptr || pool->inner == nullptr) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "marmot_kv_pool_free_block_count requires valid pool");
            return 0;
        }
        return pool->inner->free_block_count();
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_kv_pool_free_block_count threw");
        return 0;
    }
}

size_t marmot_kv_pool_total_block_count(const marmot_kv_pool_t *pool) {
    try {
        if (pool == nullptr || pool->inner == nullptr) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "marmot_kv_pool_total_block_count requires valid pool");
            return 0;
        }
        return pool->inner->total_block_count();
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_kv_pool_total_block_count threw");
        return 0;
    }
}

marmot_error_t marmot_kv_pool_swap_out_seq(marmot_kv_pool_t *pool, marmot_seq_slot_t seq) {
    try {
        if (pool == nullptr || pool->inner == nullptr) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        return pool->inner->swap_out_seq(seq);
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_kv_pool_swap_out_seq threw");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
}

marmot_error_t marmot_kv_pool_swap_in_seq(marmot_kv_pool_t *pool, marmot_seq_slot_t seq) {
    try {
        if (pool == nullptr || pool->inner == nullptr) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        return pool->inner->swap_in_seq(seq);
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_kv_pool_swap_in_seq threw");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
}

marmot_error_t marmot_kv_pool_is_seq_swapped(const marmot_kv_pool_t *pool, marmot_seq_slot_t seq, bool *out_swapped) {
    try {
        if (pool == nullptr || pool->inner == nullptr || out_swapped == nullptr) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        *out_swapped = pool->inner->seq_swapped(seq);
        return MARMOT_SUCCESS;
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_kv_pool_is_seq_swapped threw");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
}

} // extern "C"
