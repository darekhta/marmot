#include "marmot/graph/architecture.h"
#include "marmot/types.h"

#include <stdio.h>

#include <assert.h>
#include <string.h>

static void test_architecture_from_string(void) {
    assert(marmot_architecture_from_string("llama") == MARMOT_ARCH_LLAMA);
    assert(marmot_architecture_from_string("qwen2") == MARMOT_ARCH_QWEN2);
    assert(marmot_architecture_from_string("qwen3") == MARMOT_ARCH_QWEN3);
    assert(marmot_architecture_from_string("qwen3moe") == MARMOT_ARCH_QWEN3MOE);
    assert(marmot_architecture_from_string("phi3") == MARMOT_ARCH_PHI3);
    assert(marmot_architecture_from_string("gemma") == MARMOT_ARCH_GEMMA);
    assert(marmot_architecture_from_string("unknown_arch") == MARMOT_ARCH_UNKNOWN);
    assert(marmot_architecture_from_string(nullptr) == MARMOT_ARCH_UNKNOWN);
    printf("  ✓ architecture_from_string\n");
}

static void test_architecture_to_string(void) {
    assert(strcmp(marmot_architecture_to_string(MARMOT_ARCH_LLAMA), "llama") == 0);
    assert(strcmp(marmot_architecture_to_string(MARMOT_ARCH_MISTRAL), "mistral") == 0);
    assert(strcmp(marmot_architecture_to_string(MARMOT_ARCH_QWEN2), "qwen2") == 0);
    assert(strcmp(marmot_architecture_to_string(MARMOT_ARCH_QWEN3), "qwen3") == 0);
    assert(strcmp(marmot_architecture_to_string(MARMOT_ARCH_QWEN3MOE), "qwen3moe") == 0);
    assert(strcmp(marmot_architecture_to_string(MARMOT_ARCH_PHI3), "phi3") == 0);
    assert(strcmp(marmot_architecture_to_string(MARMOT_ARCH_GEMMA), "gemma") == 0);
    assert(strcmp(marmot_architecture_to_string(MARMOT_ARCH_UNKNOWN), "unknown") == 0);
    printf("  ✓ architecture_to_string\n");
}

static void test_architecture_traits(void) {
    const marmot_architecture_traits_t *llama = marmot_get_architecture_traits(MARMOT_ARCH_LLAMA);
    assert(llama != nullptr);
    assert(llama->arch_id == MARMOT_ARCH_LLAMA);
    assert(llama->ffn_type == MARMOT_FFN_SWIGLU);
    assert(llama->has_attention_bias == false);
    assert(llama->has_qk_norm == false);
    assert(llama->rope_type == MARMOT_ROPE_TYPE_NORM);

    const marmot_architecture_traits_t *qwen2 = marmot_get_architecture_traits(MARMOT_ARCH_QWEN2);
    assert(qwen2 != nullptr);
    assert(qwen2->arch_id == MARMOT_ARCH_QWEN2);
    assert(qwen2->ffn_type == MARMOT_FFN_SWIGLU);
    assert(qwen2->has_attention_bias == true);
    assert(qwen2->has_qk_norm == false);
    assert(qwen2->rope_type == MARMOT_ROPE_TYPE_NEOX);

    const marmot_architecture_traits_t *qwen3 = marmot_get_architecture_traits(MARMOT_ARCH_QWEN3);
    assert(qwen3 != nullptr);
    assert(qwen3->arch_id == MARMOT_ARCH_QWEN3);
    assert(qwen3->ffn_type == MARMOT_FFN_SWIGLU);
    assert(qwen3->is_moe == false);
    assert(qwen3->has_attention_bias == false);
    assert(qwen3->has_qk_norm == true);
    assert(qwen3->rope_type == MARMOT_ROPE_TYPE_NEOX);

    const marmot_architecture_traits_t *qwen3moe = marmot_get_architecture_traits(MARMOT_ARCH_QWEN3MOE);
    assert(qwen3moe != nullptr);
    assert(qwen3moe->arch_id == MARMOT_ARCH_QWEN3MOE);
    assert(qwen3moe->ffn_type == MARMOT_FFN_SWIGLU);
    assert(qwen3moe->is_moe == true);
    assert(qwen3moe->has_attention_bias == false);
    assert(qwen3moe->has_qk_norm == true);
    assert(qwen3moe->router_weight_policy == MARMOT_ROUTER_WEIGHT_POLICY_SOFTMAX_SELECTED_SCALED);
    assert(qwen3moe->rope_type == MARMOT_ROPE_TYPE_NEOX);
    assert(qwen3moe->metal_activation_dtype == MARMOT_DTYPE_FLOAT32);

    const marmot_architecture_traits_t *gemma = marmot_get_architecture_traits(MARMOT_ARCH_GEMMA);
    assert(gemma != nullptr);
    assert(gemma->arch_id == MARMOT_ARCH_GEMMA);
    assert(gemma->ffn_type == MARMOT_FFN_GEGLU);
    assert(gemma->has_attention_bias == false);
    assert(gemma->has_qk_norm == false);
    assert(gemma->rope_type == MARMOT_ROPE_TYPE_NEOX);

    assert(marmot_get_architecture_traits(MARMOT_ARCH_UNKNOWN) == nullptr);

    printf("  ✓ architecture_traits\n");
}

static void test_metadata_keys(void) {
    const marmot_metadata_key_map_t *llama_keys = marmot_get_metadata_keys(MARMOT_ARCH_LLAMA);
    assert(llama_keys != nullptr);
    assert(strcmp(llama_keys->context_length, "llama.context_length") == 0);
    assert(strcmp(llama_keys->embedding_length, "llama.embedding_length") == 0);

    const marmot_metadata_key_map_t *qwen2_keys = marmot_get_metadata_keys(MARMOT_ARCH_QWEN2);
    assert(qwen2_keys != nullptr);
    assert(strcmp(qwen2_keys->context_length, "qwen2.context_length") == 0);
    assert(strcmp(qwen2_keys->embedding_length, "qwen2.embedding_length") == 0);

    const marmot_metadata_key_map_t *qwen3_keys = marmot_get_metadata_keys(MARMOT_ARCH_QWEN3);
    assert(qwen3_keys != nullptr);
    assert(strcmp(qwen3_keys->context_length, "qwen3.context_length") == 0);
    assert(strcmp(qwen3_keys->embedding_length, "qwen3.embedding_length") == 0);

    const marmot_metadata_key_map_t *qwen3moe_keys = marmot_get_metadata_keys(MARMOT_ARCH_QWEN3MOE);
    assert(qwen3moe_keys != nullptr);
    assert(strcmp(qwen3moe_keys->context_length, "qwen3moe.context_length") == 0);
    assert(strcmp(qwen3moe_keys->expert_count, "qwen3moe.expert_count") == 0);
    assert(strcmp(qwen3moe_keys->expert_used_count, "qwen3moe.expert_used_count") == 0);
    assert(strcmp(qwen3moe_keys->shared_expert_count, "qwen3moe.expert_shared_count") == 0);

    assert(marmot_get_metadata_keys(MARMOT_ARCH_UNKNOWN) == nullptr);

    printf("  ✓ metadata_keys\n");
}

static void test_mistral_uses_llama_keys(void) {
    const marmot_metadata_key_map_t *mistral_keys = marmot_get_metadata_keys(MARMOT_ARCH_MISTRAL);
    const marmot_metadata_key_map_t *llama_keys = marmot_get_metadata_keys(MARMOT_ARCH_LLAMA);
    assert(mistral_keys == llama_keys);
    printf("  ✓ mistral_uses_llama_keys\n");
}

int main(void) {
    printf("Architecture traits tests:\n");

    test_architecture_from_string();
    test_architecture_to_string();
    test_architecture_traits();
    test_metadata_keys();
    test_mistral_uses_llama_keys();

    printf("\nAll architecture traits tests passed!\n");
    return 0;
}
