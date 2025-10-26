#pragma once

#include "marmot/inference/kv_pool.h"
#include "marmot/tensor.h"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <mutex>
#include <vector>

#include "inference/common/storage.hpp"

namespace marmot::inference {

class KVPool {
  public:
    struct Options {
        marmot_backend_type_t backend{MARMOT_BACKEND_CPU};
        size_t max_seqs{1};
        size_t max_seq_len{2048};
        size_t block_size{16};
        size_t num_blocks{256};
        size_t num_swap_blocks{0};
        size_t num_layers{1};
        size_t num_kv_heads{1};
        size_t head_dim{64};
        marmot_dtype_t kv_dtype{MARMOT_DTYPE_FLOAT16};
        bool deterministic_alloc{false};
    };

    struct BlockAlloc {
        marmot_block_id_t block_id{MARMOT_BLOCK_ID_INVALID};
        size_t logical_block{0};
    };

    struct BlockClone {
        marmot_block_id_t old_block{MARMOT_BLOCK_ID_INVALID};
        marmot_block_id_t new_block{MARMOT_BLOCK_ID_INVALID};
        size_t logical_block{0};
    };

    struct AppendPlan {
        marmot_seq_slot_t seq{0};
        size_t start_pos{0};
        size_t token_count{0};
        std::vector<BlockAlloc> new_blocks{};
        std::vector<BlockClone> cloned_blocks{};
    };

    struct PrefixPlan {
        marmot_seq_slot_t seq{0};
        size_t num_blocks{0};
        size_t prefix_len{0};
        std::vector<marmot_block_id_t> block_ids{};
    };

    [[nodiscard]] static std::expected<std::unique_ptr<KVPool>, marmot_error_t>
    create(const marmot_context_t *ctx, const Options &opts);

    KVPool(KVPool &&) noexcept = delete;
    KVPool &operator=(KVPool &&) noexcept = delete;

    KVPool(const KVPool &) = delete;
    KVPool &operator=(const KVPool &) = delete;

    [[nodiscard]] marmot_error_t acquire_seq(marmot_seq_slot_t &out_seq);
    [[nodiscard]] marmot_error_t release_seq(marmot_seq_slot_t seq);

    [[nodiscard]] marmot_error_t prepare_append(
        marmot_seq_slot_t seq, size_t token_count, marmot_kv_slot_t *out_slots, size_t &out_start_pos,
        AppendPlan &out_plan
    );
    [[nodiscard]] marmot_error_t commit_append(const AppendPlan &plan);
    void abort_append(const AppendPlan &plan);

    [[nodiscard]] marmot_error_t prepare_prefix_attach(
        marmot_seq_slot_t seq, const marmot_block_id_t *block_ids, size_t num_blocks, size_t prefix_len,
        PrefixPlan &out_plan
    );
    [[nodiscard]] marmot_error_t commit_prefix_attach(const PrefixPlan &plan);
    void abort_prefix_attach(const PrefixPlan &plan);

    [[nodiscard]] size_t seq_len(marmot_seq_slot_t seq) const noexcept;
    [[nodiscard]] marmot_block_id_t block_id(marmot_seq_slot_t seq, size_t logical_block) const noexcept;
    [[nodiscard]] uint32_t block_generation(marmot_block_id_t block_id) const noexcept;
    [[nodiscard]] bool seq_swapped(marmot_seq_slot_t seq) const noexcept;
    [[nodiscard]] size_t free_block_count() const noexcept;
    [[nodiscard]] size_t total_block_count() const noexcept;
    [[nodiscard]] marmot_error_t swap_out_seq(marmot_seq_slot_t seq);
    [[nodiscard]] marmot_error_t swap_in_seq(marmot_seq_slot_t seq);
    void set_block_retained(marmot_block_id_t block_id, bool retained) noexcept;

    void get_tensors(
        marmot_tensor_t **out_k, marmot_tensor_t **out_v, marmot_tensor_t **out_block_table,
        marmot_tensor_t **out_k_scale = nullptr, marmot_tensor_t **out_v_scale = nullptr
    ) noexcept;

  private:
    KVPool(
        const marmot_context_t *ctx, const Options &opts, size_t max_blocks_per_seq, uint32_t block_shift,
        size_t block_bytes, size_t scale_block_bytes, StorageBlock kv_k, StorageBlock kv_v, StorageBlock kv_k_scale,
        StorageBlock kv_v_scale, StorageBlock block_table, StorageBlock swap_k, StorageBlock swap_v,
        StorageBlock swap_k_scale, StorageBlock swap_v_scale
    );

    [[nodiscard]] bool seq_in_range(marmot_seq_slot_t seq) const noexcept {
        return static_cast<size_t>(seq) < opts_.max_seqs;
    }

    [[nodiscard]] marmot_block_id_t *block_table_row(marmot_seq_slot_t seq) noexcept {
        return block_table_data_ + static_cast<size_t>(seq) * max_blocks_per_seq_;
    }

    [[nodiscard]] const marmot_block_id_t *block_table_row(marmot_seq_slot_t seq) const noexcept {
        return block_table_data_ + static_cast<size_t>(seq) * max_blocks_per_seq_;
    }

    [[nodiscard]] marmot_block_id_t *swap_table_row(marmot_seq_slot_t seq) noexcept {
        return swap_table_data_ + static_cast<size_t>(seq) * max_blocks_per_seq_;
    }

    [[nodiscard]] const marmot_block_id_t *swap_table_row(marmot_seq_slot_t seq) const noexcept {
        return swap_table_data_ + static_cast<size_t>(seq) * max_blocks_per_seq_;
    }

    const marmot_context_t *ctx_{nullptr};
    Options opts_{};
    size_t max_blocks_per_seq_{0};
    uint32_t block_shift_{0};
    uint32_t block_mask_{0};
    size_t block_bytes_{0};
    size_t scale_block_bytes_{0};

    StorageBlock kv_k_{};
    StorageBlock kv_v_{};
    StorageBlock kv_k_scale_{};
    StorageBlock kv_v_scale_{};
    StorageBlock block_table_{};
    marmot_block_id_t *block_table_data_{nullptr};
    StorageBlock swap_k_{};
    StorageBlock swap_v_{};
    StorageBlock swap_k_scale_{};
    StorageBlock swap_v_scale_{};
    marmot_block_id_t *swap_table_data_{nullptr};
    size_t num_swap_blocks_{0};

    std::vector<size_t> seq_len_{};
    std::vector<uint8_t> seq_active_{};
    std::vector<uint8_t> seq_pending_{};
    std::vector<uint8_t> seq_swapped_{};
    std::vector<marmot_seq_slot_t> free_seqs_{};
    struct FreeBlockSet {
        bool deterministic{false};
        std::vector<marmot_block_id_t> blocks{};
        std::vector<int32_t> index{};
        std::vector<uint8_t> mask{};
        size_t count{0};
    };
    static void free_block_set_init(FreeBlockSet &set, size_t capacity, bool deterministic);
    static void free_block_set_push(FreeBlockSet &set, marmot_block_id_t block_id);
    static bool free_block_set_remove(FreeBlockSet &set, marmot_block_id_t block_id);
    static bool free_block_set_pop(FreeBlockSet &set, marmot_block_id_t &out_block);
    FreeBlockSet free_blocks_{};
    FreeBlockSet retained_free_blocks_{};
    FreeBlockSet swap_free_blocks_{};
    std::vector<uint32_t> block_refcounts_{};
    std::vector<uint8_t> block_retained_{};
    std::vector<uint32_t> block_generation_{};
    uint32_t next_generation_{1};
    std::vector<marmot_block_id_t> swap_table_{};

    void release_block(marmot_block_id_t block_id) noexcept;
    void release_swap_block(marmot_block_id_t block_id) noexcept;

    mutable std::mutex mutex_{};
};

} // namespace marmot::inference
