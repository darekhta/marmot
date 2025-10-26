#include "kv_pool.hpp"

#include "marmot/config.h"
#include "marmot/device.h"
#include "marmot/error.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <new>
#include <span>

namespace marmot::inference {

namespace {

constexpr marmot_block_id_t kInvalidBlock = MARMOT_BLOCK_ID_INVALID;

size_t ceil_div(size_t value, size_t divisor) {
    return (value + divisor - 1) / divisor;
}

} // namespace

void KVPool::free_block_set_init(FreeBlockSet &set, size_t capacity, bool deterministic) {
    set.deterministic = deterministic;
    set.blocks.clear();
    set.blocks.reserve(capacity);
    set.index.clear();
    set.mask.clear();
    set.count = 0;
    if (deterministic) {
        set.mask.assign(capacity, 0);
    } else {
        set.index.assign(capacity, -1);
    }
}

void KVPool::free_block_set_push(FreeBlockSet &set, marmot_block_id_t block_id) {
    if (set.deterministic) {
        if (block_id >= set.mask.size()) {
            return;
        }
        if (set.mask[block_id] != 0) {
            return;
        }
        set.mask[block_id] = 1;
        set.blocks.push_back(block_id);
        std::push_heap(set.blocks.begin(), set.blocks.end(), std::greater<marmot_block_id_t>());
        set.count += 1;
        return;
    }
    if (block_id >= set.index.size()) {
        return;
    }
    if (set.index[block_id] >= 0) {
        return;
    }
    set.index[block_id] = static_cast<int32_t>(set.blocks.size());
    set.blocks.push_back(block_id);
    set.count += 1;
}

bool KVPool::free_block_set_remove(FreeBlockSet &set, marmot_block_id_t block_id) {
    if (set.deterministic) {
        if (block_id >= set.mask.size()) {
            return false;
        }
        if (set.mask[block_id] == 0) {
            return false;
        }
        set.mask[block_id] = 0;
        if (set.count > 0) {
            set.count -= 1;
        }
        return true;
    }
    if (set.blocks.empty()) {
        return false;
    }
    if (block_id >= set.index.size()) {
        return false;
    }
    const int32_t index = set.index[block_id];
    if (index < 0) {
        return false;
    }
    const size_t idx = static_cast<size_t>(index);
    const size_t last_index = set.blocks.size() - 1;
    const marmot_block_id_t last_block = set.blocks[last_index];
    set.blocks[idx] = last_block;
    set.index[last_block] = static_cast<int32_t>(idx);
    set.blocks.pop_back();
    set.index[block_id] = -1;
    if (set.count > 0) {
        set.count -= 1;
    }
    return true;
}

bool KVPool::free_block_set_pop(FreeBlockSet &set, marmot_block_id_t &out_block) {
    if (set.deterministic) {
        while (!set.blocks.empty()) {
            std::pop_heap(set.blocks.begin(), set.blocks.end(), std::greater<marmot_block_id_t>());
            out_block = set.blocks.back();
            set.blocks.pop_back();
            if (out_block < set.mask.size() && set.mask[out_block] != 0) {
                set.mask[out_block] = 0;
                if (set.count > 0) {
                    set.count -= 1;
                }
                return true;
            }
        }
        return false;
    }
    if (set.blocks.empty()) {
        return false;
    }
    out_block = set.blocks.back();
    set.blocks.pop_back();
    if (out_block < set.index.size()) {
        set.index[out_block] = -1;
    }
    if (set.count > 0) {
        set.count -= 1;
    }
    return true;
}

std::expected<std::unique_ptr<KVPool>, marmot_error_t>
KVPool::create(const marmot_context_t *ctx, const Options &opts) {
    if (ctx == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "KV pool requires valid context");
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }
    if (opts.max_seqs == 0 || opts.max_seq_len == 0 || opts.block_size == 0 || opts.num_blocks == 0 ||
        opts.num_layers == 0 || opts.num_kv_heads == 0 || opts.head_dim == 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "KV pool dimensions must be non-zero");
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }
    if (opts.backend != ctx->backend_type) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "KV pool backend mismatch with context");
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }
    if (opts.block_size <= 1 || !std::has_single_bit(opts.block_size)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "KV pool block_size must be power-of-two > 1");
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }
    if (opts.block_size > UINT32_MAX) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "KV pool block_size exceeds uint32");
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }
    if (opts.num_blocks > UINT32_MAX) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "KV pool num_blocks exceeds uint32");
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }
    if (opts.num_swap_blocks > UINT32_MAX) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "KV pool num_swap_blocks exceeds uint32");
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }

    const uint32_t block_shift = static_cast<uint32_t>(std::countr_zero(opts.block_size));
    if (block_shift >= 32) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "KV pool block_size too large for kv_slot encoding");
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }

    const uint64_t max_blocks_for_slot = 1ULL << (32 - block_shift);
    if (opts.num_blocks > max_blocks_for_slot) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "KV pool num_blocks exceeds kv_slot capacity");
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }

    const size_t max_blocks_per_seq = ceil_div(opts.max_seq_len, opts.block_size);
    if (max_blocks_per_seq == 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "KV pool max_blocks_per_seq is zero");
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }

    const size_t kv_shape[5] = {opts.num_blocks, opts.num_layers, opts.num_kv_heads, opts.block_size, opts.head_dim};
    auto kv_k = StorageBlock::create(opts.backend, std::span(kv_shape, 5), opts.kv_dtype, ctx);
    if (!kv_k) {
        return std::unexpected(kv_k.error());
    }

    auto kv_v = StorageBlock::create(opts.backend, std::span(kv_shape, 5), opts.kv_dtype, ctx);
    if (!kv_v) {
        return std::unexpected(kv_v.error());
    }

    const size_t k_bytes = kv_k->size_bytes();
    const size_t v_bytes = kv_v->size_bytes();
    if (k_bytes == 0 || v_bytes == 0 || k_bytes != v_bytes) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "KV pool storage bytes invalid");
        return std::unexpected(MARMOT_ERROR_INVALID_OPERATION);
    }
    if (opts.num_blocks == 0 || (k_bytes % opts.num_blocks) != 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "KV pool block bytes misaligned");
        return std::unexpected(MARMOT_ERROR_INVALID_OPERATION);
    }
    const size_t block_bytes = k_bytes / opts.num_blocks;

    bool use_fp8_scales = false;
#if MARMOT_ENABLE_FP8
    if (opts.kv_dtype == MARMOT_DTYPE_FLOAT8_E4M3) {
        use_fp8_scales = true;
    } else if (opts.kv_dtype == MARMOT_DTYPE_FLOAT8_E5M2) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "KV pool FP8 E5M2 scales not supported");
        return std::unexpected(MARMOT_ERROR_UNSUPPORTED_DTYPE);
    }
#endif

    StorageBlock kv_k_scale;
    StorageBlock kv_v_scale;
    size_t scale_block_bytes = 0;
    if (use_fp8_scales) {
        const size_t scale_shape[3] = {opts.num_blocks, opts.num_layers, opts.num_kv_heads};
        auto kv_k_scale_res = StorageBlock::create(opts.backend, std::span(scale_shape, 3), MARMOT_DTYPE_FLOAT32, ctx);
        if (!kv_k_scale_res) {
            return std::unexpected(kv_k_scale_res.error());
        }
        auto kv_v_scale_res = StorageBlock::create(opts.backend, std::span(scale_shape, 3), MARMOT_DTYPE_FLOAT32, ctx);
        if (!kv_v_scale_res) {
            return std::unexpected(kv_v_scale_res.error());
        }
        const size_t scale_bytes = kv_k_scale_res->size_bytes();
        if (scale_bytes == 0 || opts.num_blocks == 0 || (scale_bytes % opts.num_blocks) != 0) {
            marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "KV pool scale bytes misaligned");
            return std::unexpected(MARMOT_ERROR_INVALID_OPERATION);
        }
        scale_block_bytes = scale_bytes / opts.num_blocks;
        kv_k_scale = std::move(*kv_k_scale_res);
        kv_v_scale = std::move(*kv_v_scale_res);
    }

    const size_t block_table_shape[2] = {opts.max_seqs, max_blocks_per_seq};
    auto block_table = StorageBlock::create(opts.backend, std::span(block_table_shape, 2), MARMOT_DTYPE_UINT32, ctx);
    if (!block_table) {
        return std::unexpected(block_table.error());
    }

    StorageBlock swap_k;
    StorageBlock swap_v;
    StorageBlock swap_k_scale;
    StorageBlock swap_v_scale;
    if (opts.num_swap_blocks > 0) {
        const size_t swap_shape[5] = {
            opts.num_swap_blocks, opts.num_layers, opts.num_kv_heads, opts.block_size, opts.head_dim
        };
        auto swap_k_res = StorageBlock::create(MARMOT_BACKEND_CPU, std::span(swap_shape, 5), opts.kv_dtype, ctx);
        if (!swap_k_res) {
            return std::unexpected(swap_k_res.error());
        }
        auto swap_v_res = StorageBlock::create(MARMOT_BACKEND_CPU, std::span(swap_shape, 5), opts.kv_dtype, ctx);
        if (!swap_v_res) {
            return std::unexpected(swap_v_res.error());
        }
        swap_k = std::move(*swap_k_res);
        swap_v = std::move(*swap_v_res);
        if (use_fp8_scales) {
            const size_t swap_scale_shape[3] = {opts.num_swap_blocks, opts.num_layers, opts.num_kv_heads};
            auto swap_k_scale_res =
                StorageBlock::create(MARMOT_BACKEND_CPU, std::span(swap_scale_shape, 3), MARMOT_DTYPE_FLOAT32, ctx);
            if (!swap_k_scale_res) {
                return std::unexpected(swap_k_scale_res.error());
            }
            auto swap_v_scale_res =
                StorageBlock::create(MARMOT_BACKEND_CPU, std::span(swap_scale_shape, 3), MARMOT_DTYPE_FLOAT32, ctx);
            if (!swap_v_scale_res) {
                return std::unexpected(swap_v_scale_res.error());
            }
            swap_k_scale = std::move(*swap_k_scale_res);
            swap_v_scale = std::move(*swap_v_scale_res);
        }
    }

    auto pool = std::unique_ptr<KVPool>(new (std::nothrow) KVPool(
        ctx, opts, max_blocks_per_seq, block_shift, block_bytes, scale_block_bytes, std::move(*kv_k), std::move(*kv_v),
        std::move(kv_k_scale), std::move(kv_v_scale), std::move(*block_table), std::move(swap_k), std::move(swap_v),
        std::move(swap_k_scale), std::move(swap_v_scale)
    ));
    if (!pool) {
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to allocate KV pool");
        return std::unexpected(MARMOT_ERROR_OUT_OF_MEMORY);
    }

    return pool;
}

KVPool::KVPool(
    const marmot_context_t *ctx, const Options &opts, size_t max_blocks_per_seq, uint32_t block_shift,
    size_t block_bytes, size_t scale_block_bytes, StorageBlock kv_k, StorageBlock kv_v, StorageBlock kv_k_scale,
    StorageBlock kv_v_scale, StorageBlock block_table, StorageBlock swap_k, StorageBlock swap_v,
    StorageBlock swap_k_scale, StorageBlock swap_v_scale
)
    : ctx_(ctx), opts_(opts), max_blocks_per_seq_(max_blocks_per_seq), block_shift_(block_shift),
      block_mask_((1u << block_shift) - 1u), block_bytes_(block_bytes), scale_block_bytes_(scale_block_bytes),
      kv_k_(std::move(kv_k)), kv_v_(std::move(kv_v)), kv_k_scale_(std::move(kv_k_scale)),
      kv_v_scale_(std::move(kv_v_scale)), block_table_(std::move(block_table)), swap_k_(std::move(swap_k)),
      swap_v_(std::move(swap_v)), swap_k_scale_(std::move(swap_k_scale)), swap_v_scale_(std::move(swap_v_scale)),
      num_swap_blocks_(opts.num_swap_blocks) {
    block_table_data_ = static_cast<marmot_block_id_t *>(block_table_.data());
    swap_table_.assign(opts_.max_seqs * max_blocks_per_seq_, kInvalidBlock);
    swap_table_data_ = swap_table_.data();

    seq_len_.assign(opts_.max_seqs, 0);
    seq_active_.assign(opts_.max_seqs, 0);
    seq_pending_.assign(opts_.max_seqs, 0);
    seq_swapped_.assign(opts_.max_seqs, 0);

    free_seqs_.reserve(opts_.max_seqs);
    for (size_t i = 0; i < opts_.max_seqs; ++i) {
        free_seqs_.push_back(static_cast<marmot_seq_slot_t>(opts_.max_seqs - 1 - i));
    }

    free_block_set_init(free_blocks_, opts_.num_blocks, opts_.deterministic_alloc);
    free_block_set_init(retained_free_blocks_, opts_.num_blocks, opts_.deterministic_alloc);
    for (size_t i = 0; i < opts_.num_blocks; ++i) {
        const marmot_block_id_t block_id = static_cast<marmot_block_id_t>(opts_.num_blocks - 1 - i);
        free_block_set_push(free_blocks_, block_id);
    }

    block_refcounts_.assign(opts_.num_blocks, 0);
    block_retained_.assign(opts_.num_blocks, 0);
    block_generation_.assign(opts_.num_blocks, 0);

    if (num_swap_blocks_ > 0) {
        free_block_set_init(swap_free_blocks_, num_swap_blocks_, opts_.deterministic_alloc);
        for (size_t i = 0; i < num_swap_blocks_; ++i) {
            const marmot_block_id_t block_id = static_cast<marmot_block_id_t>(num_swap_blocks_ - 1 - i);
            free_block_set_push(swap_free_blocks_, block_id);
        }
    }

    if (block_table_data_ != nullptr) {
        std::fill(block_table_data_, block_table_data_ + opts_.max_seqs * max_blocks_per_seq_, kInvalidBlock);
    }
}

marmot_error_t KVPool::acquire_seq(marmot_seq_slot_t &out_seq) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (free_seqs_.empty()) {
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "KV pool has no free sequence slots");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    marmot_seq_slot_t seq = free_seqs_.back();
    free_seqs_.pop_back();
    seq_active_[seq] = 1;
    seq_pending_[seq] = 0;
    seq_swapped_[seq] = 0;
    seq_len_[seq] = 0;

    marmot_block_id_t *row = block_table_row(seq);
    std::fill(row, row + max_blocks_per_seq_, kInvalidBlock);
    marmot_block_id_t *swap_row = swap_table_row(seq);
    std::fill(swap_row, swap_row + max_blocks_per_seq_, kInvalidBlock);

    out_seq = seq;
    return MARMOT_SUCCESS;
}

marmot_error_t KVPool::release_seq(marmot_seq_slot_t seq) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!seq_in_range(seq) || seq_active_[seq] == 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "KV pool release requires active sequence slot");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (seq_pending_[seq] != 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "KV pool release requires no pending plan");
        return MARMOT_ERROR_INVALID_OPERATION;
    }

    marmot_block_id_t *row = block_table_row(seq);
    if (seq_swapped_[seq] != 0) {
        marmot_block_id_t *swap_row = swap_table_row(seq);
        for (size_t i = 0; i < max_blocks_per_seq_; ++i) {
            const marmot_block_id_t swap_block = swap_row[i];
            if (swap_block == kInvalidBlock) {
                continue;
            }
            release_swap_block(swap_block);
            swap_row[i] = kInvalidBlock;
        }
        seq_swapped_[seq] = 0;
    }
    for (size_t i = 0; i < max_blocks_per_seq_; ++i) {
        const marmot_block_id_t block_id = row[i];
        if (block_id == kInvalidBlock) {
            continue;
        }
        if (block_id >= block_refcounts_.size()) {
            continue;
        }
        if (block_refcounts_[block_id] > 0) {
            block_refcounts_[block_id] -= 1;
            if (block_refcounts_[block_id] == 0) {
                release_block(block_id);
            }
        }
        row[i] = kInvalidBlock;
    }

    seq_len_[seq] = 0;
    seq_active_[seq] = 0;
    seq_pending_[seq] = 0;
    free_seqs_.push_back(seq);
    return MARMOT_SUCCESS;
}

marmot_error_t KVPool::prepare_append(
    marmot_seq_slot_t seq, size_t token_count, marmot_kv_slot_t *out_slots, size_t &out_start_pos, AppendPlan &out_plan
) {
    std::lock_guard<std::mutex> lock(mutex_);

    out_plan = {};
    out_plan.seq = seq;
    out_plan.token_count = token_count;
    out_start_pos = 0;

    if (!seq_in_range(seq) || seq_active_[seq] == 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "KV pool append requires active sequence slot");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (seq_pending_[seq] != 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "KV pool append requires no pending plan");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
    if (seq_swapped_[seq] != 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "KV pool append requires resident sequence");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
    if (token_count == 0) {
        out_start_pos = seq_len_[seq];
        out_plan.start_pos = out_start_pos;
        return MARMOT_SUCCESS;
    }
    if (out_slots == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "KV pool append requires output slots");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const size_t start_pos = seq_len_[seq];
    if (start_pos > opts_.max_seq_len) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "KV pool sequence length exceeds max_seq_len");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
    if (token_count > opts_.max_seq_len - start_pos) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "KV pool append exceeds max_seq_len");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const size_t end_pos = start_pos + token_count;
    const size_t start_block = start_pos / opts_.block_size;
    const size_t end_block = (end_pos - 1) / opts_.block_size;

    if (end_block >= max_blocks_per_seq_) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "KV pool append exceeds block table");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_block_id_t *row = block_table_row(seq);

    auto restore_block = [&](marmot_block_id_t block_id) {
        if (block_id >= block_refcounts_.size()) {
            return;
        }
        if (block_refcounts_[block_id] == 0) {
            (void)free_block_set_remove(free_blocks_, block_id);
            (void)free_block_set_remove(retained_free_blocks_, block_id);
        }
        block_refcounts_[block_id] += 1;
    };

    auto reset_block_scale = [&](marmot_block_id_t block_id) {
        if (scale_block_bytes_ == 0) {
            return;
        }
        uint8_t *k_scale_base = static_cast<uint8_t *>(kv_k_scale_.data());
        uint8_t *v_scale_base = static_cast<uint8_t *>(kv_v_scale_.data());
        if (k_scale_base == nullptr || v_scale_base == nullptr) {
            return;
        }
        const size_t scale_offset = static_cast<size_t>(block_id) * scale_block_bytes_;
        std::memset(k_scale_base + scale_offset, 0, scale_block_bytes_);
        std::memset(v_scale_base + scale_offset, 0, scale_block_bytes_);
    };

    auto rollback_plan = [&]() {
        for (const auto &clone : out_plan.cloned_blocks) {
            row[clone.logical_block] = clone.old_block;
            if (clone.new_block < block_refcounts_.size() && block_refcounts_[clone.new_block] > 0) {
                block_refcounts_[clone.new_block] -= 1;
                if (block_refcounts_[clone.new_block] == 0) {
                    release_block(clone.new_block);
                }
            }
            restore_block(clone.old_block);
        }
        for (const auto &alloc : out_plan.new_blocks) {
            row[alloc.logical_block] = kInvalidBlock;
            if (alloc.block_id < block_refcounts_.size() && block_refcounts_[alloc.block_id] > 0) {
                block_refcounts_[alloc.block_id] -= 1;
                if (block_refcounts_[alloc.block_id] == 0) {
                    release_block(alloc.block_id);
                }
            }
        }
    };

    auto copy_block = [&](marmot_block_id_t dst, marmot_block_id_t src) -> marmot_error_t {
        if (ctx_ == nullptr || ctx_->ops == nullptr) {
            return MARMOT_ERROR_INVALID_OPERATION;
        }
        if (dst >= opts_.num_blocks || src >= opts_.num_blocks || block_bytes_ == 0) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        uint8_t *k_base = static_cast<uint8_t *>(kv_k_.data());
        uint8_t *v_base = static_cast<uint8_t *>(kv_v_.data());
        if (k_base == nullptr || v_base == nullptr) {
            return MARMOT_ERROR_INVALID_OPERATION;
        }
        uint8_t *k_scale_base = static_cast<uint8_t *>(kv_k_scale_.data());
        uint8_t *v_scale_base = static_cast<uint8_t *>(kv_v_scale_.data());
        if (scale_block_bytes_ != 0 && (k_scale_base == nullptr || v_scale_base == nullptr)) {
            return MARMOT_ERROR_INVALID_OPERATION;
        }
        const size_t src_offset = static_cast<size_t>(src) * block_bytes_;
        const size_t dst_offset = static_cast<size_t>(dst) * block_bytes_;
        marmot_error_t status =
            ctx_->ops->memcpy_to_device(ctx_->device_ctx, k_base + dst_offset, k_base + src_offset, block_bytes_);
        if (status != MARMOT_SUCCESS) {
            return status;
        }
        status = ctx_->ops->memcpy_to_device(ctx_->device_ctx, v_base + dst_offset, v_base + src_offset, block_bytes_);
        if (status != MARMOT_SUCCESS) {
            return status;
        }
        if (scale_block_bytes_ != 0) {
            const size_t scale_src_offset = static_cast<size_t>(src) * scale_block_bytes_;
            const size_t scale_dst_offset = static_cast<size_t>(dst) * scale_block_bytes_;
            status = ctx_->ops->memcpy_to_device(
                ctx_->device_ctx, k_scale_base + scale_dst_offset, k_scale_base + scale_src_offset, scale_block_bytes_
            );
            if (status != MARMOT_SUCCESS) {
                return status;
            }
            status = ctx_->ops->memcpy_to_device(
                ctx_->device_ctx, v_scale_base + scale_dst_offset, v_scale_base + scale_src_offset, scale_block_bytes_
            );
        }
        return status;
    };

    if ((start_pos & block_mask_) != 0) {
        const size_t logical_block = start_block;
        const marmot_block_id_t existing = row[logical_block];
        if (existing != kInvalidBlock && existing < block_refcounts_.size() && block_refcounts_[existing] > 1) {
            marmot_block_id_t block_id = kInvalidBlock;
            if (!free_block_set_pop(free_blocks_, block_id)) {
                if (!free_block_set_pop(retained_free_blocks_, block_id)) {
                    marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "KV pool out of blocks");
                    return MARMOT_ERROR_OUT_OF_MEMORY;
                }
            }
            if (block_id < block_retained_.size()) {
                block_retained_[block_id] = 0;
            }
            marmot_error_t copy_status = copy_block(block_id, existing);
            if (copy_status != MARMOT_SUCCESS) {
                release_block(block_id);
                return copy_status;
            }

            row[logical_block] = block_id;
            block_refcounts_[block_id] += 1;
            if (block_refcounts_[existing] > 0) {
                block_refcounts_[existing] -= 1;
                if (block_refcounts_[existing] == 0) {
                    release_block(existing);
                }
            }
            if (block_id < block_generation_.size()) {
                block_generation_[block_id] = next_generation_++;
                if (next_generation_ == 0) {
                    next_generation_ = 1;
                }
            }
            out_plan.cloned_blocks.push_back(BlockClone{existing, block_id, logical_block});
        }
    }

    for (size_t block = start_block; block <= end_block; ++block) {
        if (row[block] != kInvalidBlock) {
            continue;
        }
        marmot_block_id_t block_id = kInvalidBlock;
        if (!free_block_set_pop(free_blocks_, block_id)) {
            if (!free_block_set_pop(retained_free_blocks_, block_id)) {
                rollback_plan();
                marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "KV pool out of blocks");
                return MARMOT_ERROR_OUT_OF_MEMORY;
            }
        }
        if (block_id < block_retained_.size()) {
            block_retained_[block_id] = 0;
        }
        row[block] = block_id;
        reset_block_scale(block_id);
        block_refcounts_[block_id] += 1;
        if (block_id < block_generation_.size()) {
            block_generation_[block_id] = next_generation_++;
            if (next_generation_ == 0) {
                next_generation_ = 1;
            }
        }
        out_plan.new_blocks.push_back(BlockAlloc{block_id, block});
    }

    for (size_t i = 0; i < token_count; ++i) {
        const size_t pos = start_pos + i;
        const size_t logical_block = pos / opts_.block_size;
        const marmot_block_id_t block_id = row[logical_block];
        const marmot_kv_slot_t offset = static_cast<marmot_kv_slot_t>(pos & block_mask_);
        const marmot_kv_slot_t slot = (static_cast<marmot_kv_slot_t>(block_id) << block_shift_) | offset;
        out_slots[i] = slot;
    }

    out_start_pos = start_pos;
    out_plan.start_pos = start_pos;
    seq_pending_[seq] = 1;
    return MARMOT_SUCCESS;
}

marmot_error_t KVPool::commit_append(const AppendPlan &plan) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (plan.token_count == 0) {
        return MARMOT_SUCCESS;
    }
    if (!seq_in_range(plan.seq) || seq_active_[plan.seq] == 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "KV pool commit requires active sequence slot");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (seq_pending_[plan.seq] == 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "KV pool commit requires pending plan");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
    if (seq_len_[plan.seq] != plan.start_pos) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "KV pool commit start_pos mismatch");
        return MARMOT_ERROR_INVALID_OPERATION;
    }

    seq_len_[plan.seq] += plan.token_count;
    seq_pending_[plan.seq] = 0;
    return MARMOT_SUCCESS;
}

void KVPool::abort_append(const AppendPlan &plan) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (plan.token_count == 0) {
        return;
    }
    if (!seq_in_range(plan.seq) || seq_active_[plan.seq] == 0) {
        return;
    }
    if (seq_pending_[plan.seq] == 0) {
        return;
    }

    marmot_block_id_t *row = block_table_row(plan.seq);
    auto restore_block = [&](marmot_block_id_t block_id) {
        if (block_id >= block_refcounts_.size()) {
            return;
        }
        if (block_refcounts_[block_id] == 0) {
            (void)free_block_set_remove(free_blocks_, block_id);
            (void)free_block_set_remove(retained_free_blocks_, block_id);
        }
        block_refcounts_[block_id] += 1;
    };

    for (const auto &clone : plan.cloned_blocks) {
        row[clone.logical_block] = clone.old_block;
        if (clone.new_block < block_refcounts_.size() && block_refcounts_[clone.new_block] > 0) {
            block_refcounts_[clone.new_block] -= 1;
            if (block_refcounts_[clone.new_block] == 0) {
                release_block(clone.new_block);
            }
        }
        restore_block(clone.old_block);
    }
    for (const auto &alloc : plan.new_blocks) {
        row[alloc.logical_block] = kInvalidBlock;
        if (alloc.block_id < block_refcounts_.size() && block_refcounts_[alloc.block_id] > 0) {
            block_refcounts_[alloc.block_id] -= 1;
            if (block_refcounts_[alloc.block_id] == 0) {
                release_block(alloc.block_id);
            }
        }
    }

    seq_pending_[plan.seq] = 0;
}

marmot_error_t KVPool::prepare_prefix_attach(
    marmot_seq_slot_t seq, const marmot_block_id_t *block_ids, size_t num_blocks, size_t prefix_len,
    PrefixPlan &out_plan
) {
    std::lock_guard<std::mutex> lock(mutex_);

    out_plan = {};
    out_plan.seq = seq;
    out_plan.num_blocks = num_blocks;
    out_plan.prefix_len = prefix_len;

    if (!seq_in_range(seq) || seq_active_[seq] == 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "KV pool prefix attach requires active sequence slot");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (seq_pending_[seq] != 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "KV pool prefix attach requires no pending plan");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
    if (seq_swapped_[seq] != 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "KV pool prefix attach requires resident sequence");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
    if (seq_len_[seq] != 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "KV pool prefix attach requires empty sequence");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
    if (num_blocks == 0) {
        if (prefix_len != 0) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "KV pool prefix attach requires blocks for prefix_len");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        return MARMOT_SUCCESS;
    }
    if (block_ids == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "KV pool prefix attach requires block IDs");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (prefix_len == 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "KV pool prefix attach requires prefix_len > 0");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (prefix_len > opts_.max_seq_len) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "KV pool prefix attach exceeds max_seq_len");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    const size_t required_blocks = ceil_div(prefix_len, opts_.block_size);
    if (required_blocks != num_blocks) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "KV pool prefix attach block count mismatch");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (num_blocks > max_blocks_per_seq_) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "KV pool prefix attach exceeds block table");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    std::vector<uint8_t> seen(opts_.num_blocks, 0);
    for (size_t i = 0; i < num_blocks; ++i) {
        const marmot_block_id_t block_id = block_ids[i];
        if (block_id == kInvalidBlock || block_id >= opts_.num_blocks) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "KV pool prefix attach block ID invalid");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        if (seen[block_id] != 0) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "KV pool prefix attach has duplicate block ID");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        seen[block_id] = 1;
    }

    marmot_block_id_t *row = block_table_row(seq);
    std::fill(row, row + max_blocks_per_seq_, kInvalidBlock);

    out_plan.block_ids.reserve(num_blocks);
    for (size_t i = 0; i < num_blocks; ++i) {
        const marmot_block_id_t block_id = block_ids[i];
        if (block_refcounts_[block_id] == 0) {
            (void)free_block_set_remove(free_blocks_, block_id);
            (void)free_block_set_remove(retained_free_blocks_, block_id);
        }
        row[i] = block_id;
        block_refcounts_[block_id] += 1;
        out_plan.block_ids.push_back(block_id);
    }

    seq_pending_[seq] = 1;
    return MARMOT_SUCCESS;
}

marmot_error_t KVPool::commit_prefix_attach(const PrefixPlan &plan) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (plan.num_blocks == 0) {
        return MARMOT_SUCCESS;
    }
    if (!seq_in_range(plan.seq) || seq_active_[plan.seq] == 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "KV pool commit requires active sequence slot");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (seq_pending_[plan.seq] == 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "KV pool commit requires pending plan");
        return MARMOT_ERROR_INVALID_OPERATION;
    }

    if (plan.prefix_len > opts_.max_seq_len) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "KV pool prefix attach exceeds max_seq_len");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    seq_len_[plan.seq] = plan.prefix_len;
    seq_pending_[plan.seq] = 0;
    return MARMOT_SUCCESS;
}

void KVPool::abort_prefix_attach(const PrefixPlan &plan) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (plan.num_blocks == 0) {
        return;
    }
    if (!seq_in_range(plan.seq) || seq_active_[plan.seq] == 0) {
        return;
    }
    if (seq_pending_[plan.seq] == 0) {
        return;
    }

    marmot_block_id_t *row = block_table_row(plan.seq);
    for (size_t i = 0; i < plan.block_ids.size(); ++i) {
        const marmot_block_id_t block_id = plan.block_ids[i];
        row[i] = kInvalidBlock;
        if (block_id < block_refcounts_.size() && block_refcounts_[block_id] > 0) {
            block_refcounts_[block_id] -= 1;
            if (block_refcounts_[block_id] == 0) {
                release_block(block_id);
            }
        }
    }

    seq_pending_[plan.seq] = 0;
}

void KVPool::release_block(marmot_block_id_t block_id) noexcept {
    if (block_id >= block_retained_.size()) {
        return;
    }
    if (block_retained_[block_id] != 0) {
        free_block_set_push(retained_free_blocks_, block_id);
    } else {
        free_block_set_push(free_blocks_, block_id);
    }
}

void KVPool::release_swap_block(marmot_block_id_t block_id) noexcept {
    if (num_swap_blocks_ == 0) {
        return;
    }
    if (block_id >= num_swap_blocks_) {
        return;
    }
    free_block_set_push(swap_free_blocks_, block_id);
}

void KVPool::set_block_retained(marmot_block_id_t block_id, bool retained) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (block_id >= block_retained_.size()) {
        return;
    }
    const uint8_t value = retained ? 1u : 0u;
    if (block_retained_[block_id] == value) {
        return;
    }
    block_retained_[block_id] = value;
    if (block_refcounts_[block_id] != 0) {
        return;
    }
    if (retained) {
        (void)free_block_set_remove(free_blocks_, block_id);
        free_block_set_push(retained_free_blocks_, block_id);
    } else {
        (void)free_block_set_remove(retained_free_blocks_, block_id);
        free_block_set_push(free_blocks_, block_id);
    }
}

bool KVPool::seq_swapped(marmot_seq_slot_t seq) const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!seq_in_range(seq)) {
        return false;
    }
    return seq_swapped_[seq] != 0;
}

size_t KVPool::free_block_count() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return free_blocks_.count + retained_free_blocks_.count;
}

size_t KVPool::total_block_count() const noexcept {
    return opts_.num_blocks;
}

marmot_error_t KVPool::swap_out_seq(marmot_seq_slot_t seq) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (num_swap_blocks_ == 0 || swap_k_.data() == nullptr || swap_v_.data() == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "KV pool swap not configured");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
    if (!seq_in_range(seq) || seq_active_[seq] == 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "KV pool swap requires active sequence slot");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (seq_pending_[seq] != 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "KV pool swap requires no pending plan");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
    if (seq_swapped_[seq] != 0) {
        return MARMOT_SUCCESS;
    }
    if (ctx_ == nullptr || ctx_->ops == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "KV pool swap requires backend ops");
        return MARMOT_ERROR_INVALID_OPERATION;
    }

    uint8_t *kv_k_data = static_cast<uint8_t *>(kv_k_.data());
    uint8_t *kv_v_data = static_cast<uint8_t *>(kv_v_.data());
    uint8_t *swap_k_data = static_cast<uint8_t *>(swap_k_.data());
    uint8_t *swap_v_data = static_cast<uint8_t *>(swap_v_.data());
    if (kv_k_data == nullptr || kv_v_data == nullptr || swap_k_data == nullptr || swap_v_data == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "KV pool swap missing storage");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
    uint8_t *kv_k_scale_data = static_cast<uint8_t *>(kv_k_scale_.data());
    uint8_t *kv_v_scale_data = static_cast<uint8_t *>(kv_v_scale_.data());
    uint8_t *swap_k_scale_data = static_cast<uint8_t *>(swap_k_scale_.data());
    uint8_t *swap_v_scale_data = static_cast<uint8_t *>(swap_v_scale_.data());
    if (scale_block_bytes_ != 0 &&
        (kv_k_scale_data == nullptr || kv_v_scale_data == nullptr || swap_k_scale_data == nullptr ||
         swap_v_scale_data == nullptr)) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "KV pool swap missing scale storage");
        return MARMOT_ERROR_INVALID_OPERATION;
    }

    marmot_block_id_t *row = block_table_row(seq);
    marmot_block_id_t *swap_row = swap_table_row(seq);
    std::fill(swap_row, swap_row + max_blocks_per_seq_, kInvalidBlock);

    struct SwapMove {
        size_t logical_block;
        marmot_block_id_t device_block;
        marmot_block_id_t swap_block;
    };
    std::vector<SwapMove> moves;
    moves.reserve(max_blocks_per_seq_);

    for (size_t block = 0; block < max_blocks_per_seq_; ++block) {
        const marmot_block_id_t device_block = row[block];
        if (device_block == kInvalidBlock) {
            continue;
        }
        marmot_block_id_t swap_block = kInvalidBlock;
        if (!free_block_set_pop(swap_free_blocks_, swap_block)) {
            for (const auto &move : moves) {
                release_swap_block(move.swap_block);
                swap_row[move.logical_block] = kInvalidBlock;
            }
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "KV pool swap out of host blocks");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }

        const size_t device_offset = static_cast<size_t>(device_block) * block_bytes_;
        const size_t swap_offset = static_cast<size_t>(swap_block) * block_bytes_;
        marmot_error_t status = ctx_->ops->memcpy_from_device(
            ctx_->device_ctx, swap_k_data + swap_offset, kv_k_data + device_offset, block_bytes_
        );
        if (status == MARMOT_SUCCESS) {
            status = ctx_->ops->memcpy_from_device(
                ctx_->device_ctx, swap_v_data + swap_offset, kv_v_data + device_offset, block_bytes_
            );
        }
        if (status == MARMOT_SUCCESS && scale_block_bytes_ != 0) {
            const size_t scale_device_offset = static_cast<size_t>(device_block) * scale_block_bytes_;
            const size_t scale_swap_offset = static_cast<size_t>(swap_block) * scale_block_bytes_;
            status = ctx_->ops->memcpy_from_device(
                ctx_->device_ctx, swap_k_scale_data + scale_swap_offset, kv_k_scale_data + scale_device_offset,
                scale_block_bytes_
            );
            if (status == MARMOT_SUCCESS) {
                status = ctx_->ops->memcpy_from_device(
                    ctx_->device_ctx, swap_v_scale_data + scale_swap_offset, kv_v_scale_data + scale_device_offset,
                    scale_block_bytes_
                );
            }
        }
        if (status != MARMOT_SUCCESS) {
            release_swap_block(swap_block);
            for (const auto &move : moves) {
                release_swap_block(move.swap_block);
                swap_row[move.logical_block] = kInvalidBlock;
            }
            return status;
        }

        swap_row[block] = swap_block;
        moves.push_back(SwapMove{block, device_block, swap_block});
    }

    if (moves.empty()) {
        return MARMOT_SUCCESS;
    }

    for (const auto &move : moves) {
        row[move.logical_block] = kInvalidBlock;
        if (move.device_block < block_refcounts_.size() && block_refcounts_[move.device_block] > 0) {
            block_refcounts_[move.device_block] -= 1;
            if (block_refcounts_[move.device_block] == 0) {
                release_block(move.device_block);
            }
        }
    }

    seq_swapped_[seq] = 1;
    return MARMOT_SUCCESS;
}

marmot_error_t KVPool::swap_in_seq(marmot_seq_slot_t seq) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (num_swap_blocks_ == 0 || swap_k_.data() == nullptr || swap_v_.data() == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "KV pool swap not configured");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
    if (!seq_in_range(seq) || seq_active_[seq] == 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "KV pool swap requires active sequence slot");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (seq_pending_[seq] != 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "KV pool swap requires no pending plan");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
    if (seq_swapped_[seq] == 0) {
        return MARMOT_SUCCESS;
    }
    if (ctx_ == nullptr || ctx_->ops == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "KV pool swap requires backend ops");
        return MARMOT_ERROR_INVALID_OPERATION;
    }

    uint8_t *kv_k_data = static_cast<uint8_t *>(kv_k_.data());
    uint8_t *kv_v_data = static_cast<uint8_t *>(kv_v_.data());
    uint8_t *swap_k_data = static_cast<uint8_t *>(swap_k_.data());
    uint8_t *swap_v_data = static_cast<uint8_t *>(swap_v_.data());
    if (kv_k_data == nullptr || kv_v_data == nullptr || swap_k_data == nullptr || swap_v_data == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "KV pool swap missing storage");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
    uint8_t *kv_k_scale_data = static_cast<uint8_t *>(kv_k_scale_.data());
    uint8_t *kv_v_scale_data = static_cast<uint8_t *>(kv_v_scale_.data());
    uint8_t *swap_k_scale_data = static_cast<uint8_t *>(swap_k_scale_.data());
    uint8_t *swap_v_scale_data = static_cast<uint8_t *>(swap_v_scale_.data());
    if (scale_block_bytes_ != 0 &&
        (kv_k_scale_data == nullptr || kv_v_scale_data == nullptr || swap_k_scale_data == nullptr ||
         swap_v_scale_data == nullptr)) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "KV pool swap missing scale storage");
        return MARMOT_ERROR_INVALID_OPERATION;
    }

    marmot_block_id_t *row = block_table_row(seq);
    marmot_block_id_t *swap_row = swap_table_row(seq);

    auto count_swap_blocks = [&](const marmot_block_id_t *row_ptr) -> size_t {
        size_t count = 0;
        for (size_t block = 0; block < max_blocks_per_seq_; ++block) {
            if (row_ptr[block] != kInvalidBlock) {
                count++;
            }
        }
        return count;
    };

    auto try_swap_exchange = [&]() -> marmot_error_t {
        const size_t target_blocks = count_swap_blocks(swap_row);
        if (target_blocks == 0) {
            seq_swapped_[seq] = 0;
            return MARMOT_SUCCESS;
        }

        marmot_seq_slot_t victim_seq = static_cast<marmot_seq_slot_t>(opts_.max_seqs);
        std::vector<size_t> target_logical;
        target_logical.reserve(max_blocks_per_seq_);
        for (size_t block = 0; block < max_blocks_per_seq_; ++block) {
            if (swap_row[block] != kInvalidBlock) {
                target_logical.push_back(block);
            }
        }

        std::vector<size_t> victim_logical;
        victim_logical.reserve(max_blocks_per_seq_);

        for (marmot_seq_slot_t candidate = 0; candidate < opts_.max_seqs; ++candidate) {
            if (candidate == seq) {
                continue;
            }
            if (seq_active_[candidate] == 0 || seq_pending_[candidate] != 0 || seq_swapped_[candidate] != 0) {
                continue;
            }
            const marmot_block_id_t *candidate_row = block_table_row(candidate);
            size_t candidate_count = 0;
            victim_logical.clear();
            for (size_t block = 0; block < max_blocks_per_seq_; ++block) {
                const marmot_block_id_t block_id = candidate_row[block];
                if (block_id == kInvalidBlock) {
                    continue;
                }
                if (block_id >= block_refcounts_.size() || block_refcounts_[block_id] != 1) {
                    candidate_count = 0;
                    victim_logical.clear();
                    break;
                }
                if (block_id < block_retained_.size() && block_retained_[block_id] != 0) {
                    candidate_count = 0;
                    victim_logical.clear();
                    break;
                }
                victim_logical.push_back(block);
                candidate_count++;
            }
            if (candidate_count == target_blocks) {
                victim_seq = candidate;
                break;
            }
        }

        if (victim_seq == opts_.max_seqs) {
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "KV pool out of blocks");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }

        marmot_block_id_t *victim_row = block_table_row(victim_seq);
        marmot_block_id_t *victim_swap_row = swap_table_row(victim_seq);

        uint8_t *kv_k_data = static_cast<uint8_t *>(kv_k_.data());
        uint8_t *kv_v_data = static_cast<uint8_t *>(kv_v_.data());
        uint8_t *swap_k_data = static_cast<uint8_t *>(swap_k_.data());
        uint8_t *swap_v_data = static_cast<uint8_t *>(swap_v_.data());
        if (kv_k_data == nullptr || kv_v_data == nullptr || swap_k_data == nullptr || swap_v_data == nullptr) {
            marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "KV pool swap missing storage");
            return MARMOT_ERROR_INVALID_OPERATION;
        }
        uint8_t *kv_k_scale_data = static_cast<uint8_t *>(kv_k_scale_.data());
        uint8_t *kv_v_scale_data = static_cast<uint8_t *>(kv_v_scale_.data());
        uint8_t *swap_k_scale_data = static_cast<uint8_t *>(swap_k_scale_.data());
        uint8_t *swap_v_scale_data = static_cast<uint8_t *>(swap_v_scale_.data());
        if (scale_block_bytes_ != 0 &&
            (kv_k_scale_data == nullptr || kv_v_scale_data == nullptr || swap_k_scale_data == nullptr ||
             swap_v_scale_data == nullptr)) {
            marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "KV pool swap missing scale storage");
            return MARMOT_ERROR_INVALID_OPERATION;
        }

        struct ExchangeMove {
            size_t target_logical;
            size_t victim_logical;
            marmot_block_id_t device_block;
            marmot_block_id_t swap_block;
        };
        std::vector<ExchangeMove> moves;
        moves.reserve(target_blocks);

        std::vector<uint8_t> temp_k(block_bytes_);
        std::vector<uint8_t> temp_v(block_bytes_);
        std::vector<uint8_t> temp_k_scale;
        std::vector<uint8_t> temp_v_scale;
        if (scale_block_bytes_ != 0) {
            temp_k_scale.resize(scale_block_bytes_);
            temp_v_scale.resize(scale_block_bytes_);
        }

        auto rollback = [&](size_t rollback_count) -> void {
            for (size_t idx = 0; idx < rollback_count; ++idx) {
                const ExchangeMove &move = moves[rollback_count - 1 - idx];
                const size_t device_offset = static_cast<size_t>(move.device_block) * block_bytes_;
                const size_t swap_offset = static_cast<size_t>(move.swap_block) * block_bytes_;
                std::memcpy(temp_k.data(), swap_k_data + swap_offset, block_bytes_);
                std::memcpy(temp_v.data(), swap_v_data + swap_offset, block_bytes_);
                if (scale_block_bytes_ != 0) {
                    const size_t scale_device_offset = static_cast<size_t>(move.device_block) * scale_block_bytes_;
                    const size_t scale_swap_offset = static_cast<size_t>(move.swap_block) * scale_block_bytes_;
                    std::memcpy(temp_k_scale.data(), swap_k_scale_data + scale_swap_offset, scale_block_bytes_);
                    std::memcpy(temp_v_scale.data(), swap_v_scale_data + scale_swap_offset, scale_block_bytes_);
                    (void)ctx_->ops->memcpy_from_device(
                        ctx_->device_ctx, swap_k_scale_data + scale_swap_offset, kv_k_scale_data + scale_device_offset,
                        scale_block_bytes_
                    );
                    (void)ctx_->ops->memcpy_from_device(
                        ctx_->device_ctx, swap_v_scale_data + scale_swap_offset, kv_v_scale_data + scale_device_offset,
                        scale_block_bytes_
                    );
                    (void)ctx_->ops->memcpy_to_device(
                        ctx_->device_ctx, kv_k_scale_data + scale_device_offset, temp_k_scale.data(), scale_block_bytes_
                    );
                    (void)ctx_->ops->memcpy_to_device(
                        ctx_->device_ctx, kv_v_scale_data + scale_device_offset, temp_v_scale.data(), scale_block_bytes_
                    );
                }
                (void)ctx_->ops->memcpy_from_device(
                    ctx_->device_ctx, swap_k_data + swap_offset, kv_k_data + device_offset, block_bytes_
                );
                (void)ctx_->ops->memcpy_from_device(
                    ctx_->device_ctx, swap_v_data + swap_offset, kv_v_data + device_offset, block_bytes_
                );
                (void)ctx_->ops->memcpy_to_device(
                    ctx_->device_ctx, kv_k_data + device_offset, temp_k.data(), block_bytes_
                );
                (void)ctx_->ops->memcpy_to_device(
                    ctx_->device_ctx, kv_v_data + device_offset, temp_v.data(), block_bytes_
                );
            }
        };

        for (size_t i = 0; i < target_blocks; ++i) {
            const size_t target_block = target_logical[i];
            const size_t victim_block = victim_logical[i];
            const marmot_block_id_t swap_block = swap_row[target_block];
            const marmot_block_id_t device_block = victim_row[victim_block];
            if (swap_block == kInvalidBlock || device_block == kInvalidBlock) {
                marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "KV pool swap exchange invalid block mapping");
                rollback(moves.size());
                return MARMOT_ERROR_INVALID_OPERATION;
            }

            const size_t device_offset = static_cast<size_t>(device_block) * block_bytes_;
            const size_t swap_offset = static_cast<size_t>(swap_block) * block_bytes_;
            std::memcpy(temp_k.data(), swap_k_data + swap_offset, block_bytes_);
            std::memcpy(temp_v.data(), swap_v_data + swap_offset, block_bytes_);
            marmot_error_t status = MARMOT_SUCCESS;
            if (scale_block_bytes_ != 0) {
                const size_t scale_device_offset = static_cast<size_t>(device_block) * scale_block_bytes_;
                const size_t scale_swap_offset = static_cast<size_t>(swap_block) * scale_block_bytes_;
                std::memcpy(temp_k_scale.data(), swap_k_scale_data + scale_swap_offset, scale_block_bytes_);
                std::memcpy(temp_v_scale.data(), swap_v_scale_data + scale_swap_offset, scale_block_bytes_);
                status = ctx_->ops->memcpy_from_device(
                    ctx_->device_ctx, swap_k_scale_data + scale_swap_offset, kv_k_scale_data + scale_device_offset,
                    scale_block_bytes_
                );
                if (status == MARMOT_SUCCESS) {
                    status = ctx_->ops->memcpy_from_device(
                        ctx_->device_ctx, swap_v_scale_data + scale_swap_offset, kv_v_scale_data + scale_device_offset,
                        scale_block_bytes_
                    );
                }
                if (status == MARMOT_SUCCESS) {
                    status = ctx_->ops->memcpy_to_device(
                        ctx_->device_ctx, kv_k_scale_data + scale_device_offset, temp_k_scale.data(), scale_block_bytes_
                    );
                }
                if (status == MARMOT_SUCCESS) {
                    status = ctx_->ops->memcpy_to_device(
                        ctx_->device_ctx, kv_v_scale_data + scale_device_offset, temp_v_scale.data(), scale_block_bytes_
                    );
                }
            }
            if (status == MARMOT_SUCCESS) {
                status = ctx_->ops->memcpy_from_device(
                    ctx_->device_ctx, swap_k_data + swap_offset, kv_k_data + device_offset, block_bytes_
                );
            }
            if (status == MARMOT_SUCCESS) {
                status = ctx_->ops->memcpy_from_device(
                    ctx_->device_ctx, swap_v_data + swap_offset, kv_v_data + device_offset, block_bytes_
                );
            }
            if (status == MARMOT_SUCCESS) {
                status = ctx_->ops->memcpy_to_device(
                    ctx_->device_ctx, kv_k_data + device_offset, temp_k.data(), block_bytes_
                );
            }
            if (status == MARMOT_SUCCESS) {
                status = ctx_->ops->memcpy_to_device(
                    ctx_->device_ctx, kv_v_data + device_offset, temp_v.data(), block_bytes_
                );
            }
            if (status != MARMOT_SUCCESS) {
                rollback(moves.size());
                return status;
            }
            moves.push_back(
                ExchangeMove{
                    .target_logical = target_block,
                    .victim_logical = victim_block,
                    .device_block = device_block,
                    .swap_block = swap_block,
                }
            );
        }

        std::fill(victim_swap_row, victim_swap_row + max_blocks_per_seq_, kInvalidBlock);
        for (const auto &move : moves) {
            row[move.target_logical] = move.device_block;
            swap_row[move.target_logical] = kInvalidBlock;
            victim_row[move.victim_logical] = kInvalidBlock;
            victim_swap_row[move.victim_logical] = move.swap_block;
            if (move.device_block < block_retained_.size()) {
                block_retained_[move.device_block] = 0;
            }
            if (move.device_block < block_generation_.size()) {
                block_generation_[move.device_block] = next_generation_++;
                if (next_generation_ == 0) {
                    next_generation_ = 1;
                }
            }
        }

        seq_swapped_[seq] = 0;
        seq_swapped_[victim_seq] = 1;
        return MARMOT_SUCCESS;
    };

    struct SwapMove {
        size_t logical_block;
        marmot_block_id_t device_block;
        marmot_block_id_t swap_block;
    };
    std::vector<SwapMove> moves;
    moves.reserve(max_blocks_per_seq_);

    for (size_t block = 0; block < max_blocks_per_seq_; ++block) {
        const marmot_block_id_t swap_block = swap_row[block];
        if (swap_block == kInvalidBlock) {
            continue;
        }
        if (row[block] != kInvalidBlock) {
            marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "KV pool swap-in encountered resident block");
            return MARMOT_ERROR_INVALID_OPERATION;
        }
        marmot_block_id_t device_block = kInvalidBlock;
        if (!free_block_set_pop(free_blocks_, device_block)) {
            if (!free_block_set_pop(retained_free_blocks_, device_block)) {
                if (moves.empty()) {
                    marmot_error_t exchange_status = try_swap_exchange();
                    if (exchange_status == MARMOT_SUCCESS) {
                        return MARMOT_SUCCESS;
                    }
                }
                for (const auto &move : moves) {
                    release_block(move.device_block);
                }
                marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "KV pool out of blocks");
                return MARMOT_ERROR_OUT_OF_MEMORY;
            }
        }
        if (device_block < block_retained_.size()) {
            block_retained_[device_block] = 0;
        }

        const size_t device_offset = static_cast<size_t>(device_block) * block_bytes_;
        const size_t swap_offset = static_cast<size_t>(swap_block) * block_bytes_;
        marmot_error_t status = ctx_->ops->memcpy_to_device(
            ctx_->device_ctx, kv_k_data + device_offset, swap_k_data + swap_offset, block_bytes_
        );
        if (status == MARMOT_SUCCESS) {
            status = ctx_->ops->memcpy_to_device(
                ctx_->device_ctx, kv_v_data + device_offset, swap_v_data + swap_offset, block_bytes_
            );
        }
        if (status == MARMOT_SUCCESS && scale_block_bytes_ != 0) {
            const size_t scale_device_offset = static_cast<size_t>(device_block) * scale_block_bytes_;
            const size_t scale_swap_offset = static_cast<size_t>(swap_block) * scale_block_bytes_;
            status = ctx_->ops->memcpy_to_device(
                ctx_->device_ctx, kv_k_scale_data + scale_device_offset, swap_k_scale_data + scale_swap_offset,
                scale_block_bytes_
            );
            if (status == MARMOT_SUCCESS) {
                status = ctx_->ops->memcpy_to_device(
                    ctx_->device_ctx, kv_v_scale_data + scale_device_offset, swap_v_scale_data + scale_swap_offset,
                    scale_block_bytes_
                );
            }
        }
        if (status != MARMOT_SUCCESS) {
            release_block(device_block);
            for (const auto &move : moves) {
                release_block(move.device_block);
            }
            return status;
        }

        moves.push_back(SwapMove{block, device_block, swap_block});
    }

    for (const auto &move : moves) {
        row[move.logical_block] = move.device_block;
        if (move.device_block < block_refcounts_.size()) {
            block_refcounts_[move.device_block] += 1;
        }
        if (move.device_block < block_generation_.size()) {
            block_generation_[move.device_block] = next_generation_++;
            if (next_generation_ == 0) {
                next_generation_ = 1;
            }
        }
        swap_row[move.logical_block] = kInvalidBlock;
        release_swap_block(move.swap_block);
    }

    seq_swapped_[seq] = 0;
    return MARMOT_SUCCESS;
}

size_t KVPool::seq_len(marmot_seq_slot_t seq) const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!seq_in_range(seq)) {
        return 0;
    }
    return seq_len_[seq];
}

marmot_block_id_t KVPool::block_id(marmot_seq_slot_t seq, size_t logical_block) const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!seq_in_range(seq) || logical_block >= max_blocks_per_seq_) {
        return kInvalidBlock;
    }
    const marmot_block_id_t *row = block_table_row(seq);
    return row[logical_block];
}

uint32_t KVPool::block_generation(marmot_block_id_t block_id) const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (block_id >= block_generation_.size()) {
        return 0;
    }
    return block_generation_[block_id];
}

void KVPool::get_tensors(
    marmot_tensor_t **out_k, marmot_tensor_t **out_v, marmot_tensor_t **out_block_table, marmot_tensor_t **out_k_scale,
    marmot_tensor_t **out_v_scale
) noexcept {
    if (out_k != nullptr) {
        *out_k = kv_k_.tensor();
    }
    if (out_v != nullptr) {
        *out_v = kv_v_.tensor();
    }
    if (out_block_table != nullptr) {
        *out_block_table = block_table_.tensor();
    }
    if (out_k_scale != nullptr) {
        *out_k_scale = kv_k_scale_.tensor();
    }
    if (out_v_scale != nullptr) {
        *out_v_scale = kv_v_scale_.tensor();
    }
}

} // namespace marmot::inference
