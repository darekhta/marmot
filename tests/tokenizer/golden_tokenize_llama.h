#ifndef MARMOT_TOKENIZER_GOLDEN_TOKENIZE_LLAMA_H
#define MARMOT_TOKENIZER_GOLDEN_TOKENIZE_LLAMA_H

#include "marmot/tokenizer.h"

// Generated from llama.cpp reference implementation (commit 6f1f6a961a2239b3603867c55f28ddbe811f0293).

typedef struct {
    const char *text;
    marmot_token_id_t expected[8];
    size_t expected_len;
} marmot_tokenizer_golden_tokenize_case_t;

static const marmot_tokenizer_golden_tokenize_case_t marmot_tokenizer_golden_llama_tokenize_cases[] = {
    {.text = "the", .expected = {278}, .expected_len = 1},
    {.text = "hello", .expected = {22172}, .expected_len = 1},
    {.text = "hello world", .expected = {22172, 3186}, .expected_len = 2},
    {.text = "\n", .expected = {29871, 13}, .expected_len = 2},
    {.text = " the", .expected = {259, 1552}, .expected_len = 2},
    {.text = "hello\nworld", .expected = {22172, 13, 11526}, .expected_len = 3},
};

#endif // MARMOT_TOKENIZER_GOLDEN_TOKENIZE_LLAMA_H
