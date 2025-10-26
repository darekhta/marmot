// Golden LLM output test data
// Generated with: prompt="Hello", temp=0, max_tokens=10
//
// To regenerate, run each model with deterministic settings and capture output.

#pragma once

#include "marmot/tokenizer.h"

typedef struct {
    const char *model_filename;
    const char *prompt;
    const marmot_token_id_t *expected_tokens;
    size_t expected_len;
} marmot_golden_llm_output_t;

// Golden outputs for "Hello" prompt with temp=0, max_tokens=10
// These are the first 10 tokens generated after "Hello" prompt

// TinyLlama (LLaMA architecture)
static const marmot_token_id_t GOLDEN_TINYLLAMA[] = {
    29892, 2787, 29991, 13, 13, 29945, 29889, 5132, 29901, 13,
};

// Qwen2 0.5B Instruct
static const marmot_token_id_t GOLDEN_QWEN2[] = {
    11, 1246, 646, 358, 7789, 498, 3351, 30, 358, 2776,
};

// Qwen3 0.6B
static const marmot_token_id_t GOLDEN_QWEN3[] = {
    9707, 9707, 9707, 9707, 9707, 9707, 9707, 9707, 9707, 9707,
};

// Gemma 2B
static const marmot_token_id_t GOLDEN_GEMMA[] = {
    109, 235285, 235303, 235262, 780, 2821, 1013, 736, 603, 573,
};

// Phi-3 Mini
static const marmot_token_id_t GOLDEN_PHI3[] = {
    29991, 32007, 32001, 15043, 29991, 1128, 508, 306, 6985, 366,
};

static const marmot_golden_llm_output_t MARMOT_GOLDEN_LLM_OUTPUTS[] = {
    {
        .model_filename = "tinyllama-q4_k_m.gguf",
        .prompt = "Hello",
        .expected_tokens = GOLDEN_TINYLLAMA,
        .expected_len = sizeof(GOLDEN_TINYLLAMA) / sizeof(GOLDEN_TINYLLAMA[0]),
    },
    {
        .model_filename = "qwen2-0_5b-instruct-q4_k_m.gguf",
        .prompt = "Hello",
        .expected_tokens = GOLDEN_QWEN2,
        .expected_len = sizeof(GOLDEN_QWEN2) / sizeof(GOLDEN_QWEN2[0]),
    },
    {
        .model_filename = "Qwen3-0.6B-Q8_0.gguf",
        .prompt = "Hello",
        .expected_tokens = GOLDEN_QWEN3,
        .expected_len = sizeof(GOLDEN_QWEN3) / sizeof(GOLDEN_QWEN3[0]),
    },
    {
        .model_filename = "gemma-2b.Q8_0.gguf",
        .prompt = "Hello",
        .expected_tokens = GOLDEN_GEMMA,
        .expected_len = sizeof(GOLDEN_GEMMA) / sizeof(GOLDEN_GEMMA[0]),
    },
    {
        .model_filename = "Phi-3-mini-4k-instruct-q4.gguf",
        .prompt = "Hello",
        .expected_tokens = GOLDEN_PHI3,
        .expected_len = sizeof(GOLDEN_PHI3) / sizeof(GOLDEN_PHI3[0]),
    },
};

#define MARMOT_GOLDEN_LLM_OUTPUT_COUNT (sizeof(MARMOT_GOLDEN_LLM_OUTPUTS) / sizeof(MARMOT_GOLDEN_LLM_OUTPUTS[0]))
