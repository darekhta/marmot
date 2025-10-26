/* clang-format off */
#include "marmot/tokenizer.h"

#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include <cmocka.h>
#include "test_fixture_utils.h"
/* clang-format on */

#define QWEN2_FIXTURE "qwen2-0_5b-instruct-q4_k_m.gguf"

typedef struct {
    const char *text;
    const marmot_token_id_t *expected;
    size_t expected_len;
} qwen2_token_case_t;

// Thread-local fixture path buffer
static _Thread_local char g_fixture_path[MARMOT_TEST_PATH_MAX];

static marmot_tokenizer_t *load_tokenizer(void) {
    const char *path = marmot_test_get_fixture_path(QWEN2_FIXTURE, g_fixture_path, sizeof(g_fixture_path));
    if (!marmot_test_fixture_exists(path)) {
        return nullptr;
    }

    marmot_tokenizer_options_t opts;
    if (marmot_tokenizer_options_init(&opts) != MARMOT_SUCCESS) {
        return nullptr;
    }

    marmot_tokenizer_t *tok = nullptr;
    if (marmot_tokenizer_create_from_gguf_file(path, &opts, &tok) != MARMOT_SUCCESS) {
        return nullptr;
    }
    return tok;
}

static void assert_token_ids(marmot_tokenizer_t *tok, const qwen2_token_case_t *test_case) {
    marmot_tokenizer_encode_options_t enc;
    assert_int_equal(marmot_tokenizer_encode_options_init(&enc), MARMOT_SUCCESS);
    enc.add_bos = false;
    enc.add_eos = false;
    enc.allow_special = false;

    size_t needed = 0;
    assert_int_equal(
        marmot_tokenizer_encode(tok, test_case->text, strlen(test_case->text), &enc, nullptr, &needed), MARMOT_SUCCESS
    );
    assert_int_equal(needed, test_case->expected_len);

    marmot_token_id_t *ids = needed > 0 ? malloc(needed * sizeof(*ids)) : nullptr;
    if (needed > 0) {
        assert_non_null(ids);
    }

    size_t cap = needed;
    assert_int_equal(
        marmot_tokenizer_encode(tok, test_case->text, strlen(test_case->text), &enc, ids, &cap), MARMOT_SUCCESS
    );
    assert_int_equal(cap, needed);
    assert_memory_equal(ids, test_case->expected, needed * sizeof(*ids));

    free(ids);
}

static void test_qwen2_tokenization_cases(void **state) {
    (void)state;

    marmot_tokenizer_t *tok = load_tokenizer();
    if (tok == nullptr) {
        skip();
    }

    static const marmot_token_id_t k_case0[] = {
        7985, 264, 3364, 911, 264, 326, 57909, 53416, 879, 15803, 4627, 13,
    };
    static const marmot_token_id_t k_case1[] = {
        7985, 264, 220, 21, 1331, 18380, 3364, 911, 264, 5562, 13, 3972, 448, 279, 3409, 54685, 13,
    };
    static const marmot_token_id_t k_case2[] = {
        840, 20772, 30128, 21321, 0,
    };

    static const qwen2_token_case_t cases[] = {
        {
            .text = "Write a story about a lighthouse keeper who loves music.",
            .expected = k_case0,
            .expected_len = sizeof(k_case0) / sizeof(k_case0[0]),
        },
        {
            .text = "Write a 6-sentence story about a dog. End with the word DONE.",
            .expected = k_case1,
            .expected_len = sizeof(k_case1) / sizeof(k_case1[0]),
        },
        {
            .text = "Explain quantum physics!",
            .expected = k_case2,
            .expected_len = sizeof(k_case2) / sizeof(k_case2[0]),
        },
    };

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        assert_token_ids(tok, &cases[i]);
    }

    marmot_tokenizer_destroy(tok);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_qwen2_tokenization_cases),
    };

    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
