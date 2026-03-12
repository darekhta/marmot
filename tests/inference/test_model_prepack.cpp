#include "marmot/marmot.h"

#include <cassert>
#include <cstdlib>
#include <cstring>

#include "inference/common/model_prepack.hpp"

namespace {

void init_quant_tensor(
    marmot_tensor_t &tensor, marmot_quant_kind_t quant_kind, size_t dim0, size_t dim1, size_t dim2 = 0
) {
    std::memset(&tensor, 0, sizeof(tensor));
    tensor.dtype = MARMOT_DTYPE_FLOAT16;
    tensor.quant_kind = quant_kind;
    tensor.quant_layout = MARMOT_QUANT_LAYOUT_GGUF;
    tensor.shape.ndim = dim2 == 0 ? 2 : 3;
    tensor.shape.shape[0] = dim0;
    tensor.shape.shape[1] = dim1;
    tensor.shape.strides[0] = dim1;
    tensor.shape.strides[1] = 1;
    if (dim2 != 0) {
        tensor.shape.shape[2] = dim2;
        tensor.shape.strides[2] = dim0 * dim1;
    }
}

void test_decode_weight_selection() {
    marmot_tensor_t output{};
    init_quant_tensor(output, MARMOT_QUANT_KIND_Q4_K, 4096, 2048);

    marmot_gguf_tensor_t output_info{
        .name = const_cast<char *>("output.weight"),
        .tensor = &output,
    };
    assert(marmot::inference::should_prepack_cpu_decode_tensor(&output_info, false));

    marmot_gguf_tensor_t embed_info{
        .name = const_cast<char *>("token_embd.weight"),
        .tensor = &output,
    };
    assert(marmot::inference::should_prepack_cpu_decode_tensor(&embed_info, true));
    assert(!marmot::inference::should_prepack_cpu_decode_tensor(&embed_info, false));
}

void test_attention_weight_selection() {
    marmot_tensor_t weight{};
    init_quant_tensor(weight, MARMOT_QUANT_KIND_Q4_K, 4096, 2048);

    marmot_gguf_tensor_t q_info{
        .name = const_cast<char *>("blk.0.attn_q.weight"),
        .tensor = &weight,
    };
    assert(marmot::inference::should_prepack_cpu_attention_tensor(&q_info));

    marmot_gguf_tensor_t k_info{
        .name = const_cast<char *>("blk.0.attn_k.weight"),
        .tensor = &weight,
    };
    assert(marmot::inference::should_prepack_cpu_attention_tensor(&k_info));

    marmot_gguf_tensor_t v_info{
        .name = const_cast<char *>("blk.0.attn_v.weight"),
        .tensor = &weight,
    };
    assert(marmot::inference::should_prepack_cpu_attention_tensor(&v_info));

    marmot_gguf_tensor_t out_info{
        .name = const_cast<char *>("blk.0.attn_output.weight"),
        .tensor = &weight,
    };
    assert(marmot::inference::should_prepack_cpu_attention_tensor(&out_info));

    marmot_gguf_tensor_t norm_info{
        .name = const_cast<char *>("blk.0.attn_norm.weight"),
        .tensor = &weight,
    };
    assert(!marmot::inference::should_prepack_cpu_attention_tensor(&norm_info));
}

void test_moe_prepack_view_gate_up() {
    marmot_tensor_t gate{};
    init_quant_tensor(gate, MARMOT_QUANT_KIND_Q4_K, 8, 6, 2);

    marmot_gguf_tensor_t info{
        .name = const_cast<char *>("blk.0.ffn_gate_exps.weight"),
        .tensor = &gate,
    };
    marmot::inference::CpuMoePrepackView view{};
    assert(marmot::inference::make_cpu_moe_prepack_view(&info, &view));
    assert(view.rows_per_expert == 6);
    assert(view.experts == 2);
    assert(view.tensor.shape.ndim == 2);
    assert(view.tensor.shape.shape[0] == 12);
    assert(view.tensor.shape.shape[1] == 8);
}

void test_moe_prepack_view_down() {
    marmot_tensor_t down{};
    init_quant_tensor(down, MARMOT_QUANT_KIND_Q6_K, 6, 8, 2);

    marmot_gguf_tensor_t info{
        .name = const_cast<char *>("blk.0.ffn_down_exps.weight"),
        .tensor = &down,
    };
    marmot::inference::CpuMoePrepackView view{};
    assert(marmot::inference::make_cpu_moe_prepack_view(&info, &view));
    assert(view.rows_per_expert == 8);
    assert(view.experts == 2);
    assert(view.tensor.shape.shape[0] == 16);
    assert(view.tensor.shape.shape[1] == 6);
}

void test_moe_prepack_view_rejects_non_moe() {
    marmot_tensor_t weight{};
    init_quant_tensor(weight, MARMOT_QUANT_KIND_Q4_K, 8, 6, 2);

    marmot_gguf_tensor_t info{
        .name = const_cast<char *>("blk.0.attn_q.weight"),
        .tensor = &weight,
    };
    marmot::inference::CpuMoePrepackView view{};
    assert(!marmot::inference::make_cpu_moe_prepack_view(&info, &view));
}

void test_moe_prepack_selection() {
    marmot_tensor_t gate{};
    init_quant_tensor(gate, MARMOT_QUANT_KIND_Q4_K, 8, 6, 2);

    marmot_gguf_tensor_t gate_info{
        .name = const_cast<char *>("blk.0.ffn_gate_exps.weight"),
        .tensor = &gate,
    };
    assert(marmot::inference::should_prepack_cpu_moe_bank(&gate_info));

    marmot_gguf_tensor_t up_info{
        .name = const_cast<char *>("blk.0.ffn_up_exps.weight"),
        .tensor = &gate,
    };
    assert(marmot::inference::should_prepack_cpu_moe_bank(&up_info));

    marmot_gguf_tensor_t down_info{
        .name = const_cast<char *>("blk.0.ffn_down_exps.weight"),
        .tensor = &gate,
    };
    assert(!marmot::inference::should_prepack_cpu_moe_bank(&down_info));
}

void test_moe_prepack_view_can_prepack_full_bank() {
    marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_CPU);
    assert(ctx != nullptr);

    constexpr size_t cols = 256;
    constexpr size_t rows_per_expert = 8192;
    constexpr size_t experts = 2;
    marmot_tensor_t weight_q{};
    init_quant_tensor(weight_q, MARMOT_QUANT_KIND_Q4_K, cols, rows_per_expert, experts);
    weight_q.ctx = ctx;
    weight_q.backend = MARMOT_BACKEND_CPU;
    weight_q.memory_location = MARMOT_MEMORY_HOST;

    const size_t row_bytes = marmot::inference::quant_row_bytes(weight_q.quant_kind, cols);
    const size_t bytes = row_bytes * rows_per_expert * experts;
    assert(bytes != 0);
    weight_q.data = std::malloc(bytes);
    assert(weight_q.data != nullptr);
    std::memset(weight_q.data, 0, bytes);
    weight_q.capacity_bytes = bytes;

    marmot_gguf_tensor_t info{
        .name = const_cast<char *>("blk.0.ffn_gate_exps.weight"),
        .tensor = &weight_q,
    };
    marmot::inference::CpuMoePrepackView view{};
    assert(marmot::inference::make_cpu_moe_prepack_view(&info, &view));
    assert(marmot_matmul_prepack_quant_weight(ctx, &view.tensor) == MARMOT_SUCCESS);

    std::free(weight_q.data);
    marmot_destroy(ctx);
}

} // namespace

int main() {
    test_decode_weight_selection();
    test_attention_weight_selection();
    test_moe_prepack_view_gate_up();
    test_moe_prepack_view_down();
    test_moe_prepack_view_rejects_non_moe();
    test_moe_prepack_selection();
    test_moe_prepack_view_can_prepack_full_bank();
    return 0;
}
