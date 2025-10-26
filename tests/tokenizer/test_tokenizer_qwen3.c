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

#define QWEN3_FIXTURE "Qwen3-0.6B-Q8_0.gguf"

// Thread-local fixture path buffer
static _Thread_local char g_fixture_path[MARMOT_TEST_PATH_MAX];

static marmot_tokenizer_t *load_tokenizer(void) {
    const char *path = marmot_test_get_fixture_path(QWEN3_FIXTURE, g_fixture_path, sizeof(g_fixture_path));
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

static void test_qwen3_think_tags_visible(void **state) {
    (void)state;

    marmot_tokenizer_t *tok = load_tokenizer();
    if (tok == nullptr) {
        skip();
    }

    marmot_token_id_t think_id = MARMOT_TOKEN_ID_INVALID;
    marmot_token_id_t end_id = MARMOT_TOKEN_ID_INVALID;

    assert_int_equal(marmot_tokenizer_piece_to_token(tok, "<think>", strlen("<think>"), &think_id), MARMOT_SUCCESS);
    assert_int_equal(marmot_tokenizer_piece_to_token(tok, "</think>", strlen("</think>"), &end_id), MARMOT_SUCCESS);

    const marmot_token_id_t ids[] = {think_id, end_id};

    marmot_tokenizer_decode_options_t dec;
    assert_int_equal(marmot_tokenizer_decode_options_init(&dec), MARMOT_SUCCESS);
    dec.skip_special = true;

    size_t out_len = 0;
    assert_int_equal(marmot_tokenizer_decode(tok, ids, 2, &dec, nullptr, &out_len), MARMOT_SUCCESS);
    assert_true(out_len > 0);

    char *out = malloc(out_len);
    assert_non_null(out);
    size_t cap = out_len;
    assert_int_equal(marmot_tokenizer_decode(tok, ids, 2, &dec, out, &cap), MARMOT_SUCCESS);
    assert_int_equal(cap, out_len);
    assert_string_equal(out, "<think></think>");

    free(out);
    marmot_tokenizer_destroy(tok);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_qwen3_think_tags_visible),
    };

    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
