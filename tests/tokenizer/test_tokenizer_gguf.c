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

#include "golden_tokenize_llama.h"

// Thread-local fixture path buffer
static _Thread_local char g_fixture_path[MARMOT_TEST_PATH_MAX];

static marmot_tokenizer_t *load_tokenizer_for_fixture(const char *filename) {
    const char *path = marmot_test_get_fixture_path(filename, g_fixture_path, sizeof(g_fixture_path));
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

// Generic test: encode/decode roundtrip - works with ANY tokenizer
static void test_encode_decode_roundtrip_all_fixtures(void **state) {
    (void)state;

    const marmot_test_fixture_t *fixture;
    MARMOT_TEST_FOR_EACH_FIXTURE(fixture) {
        marmot_tokenizer_t *tok = load_tokenizer_for_fixture(fixture->filename);
        if (tok == nullptr) {
            continue; // Skip missing fixtures
        }

        const char *cases[] = {"the", " the", "\n", "hello world", "hello\nworld"};
        for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
            const char *text = cases[i];
            size_t text_len = strlen(text);

            marmot_tokenizer_encode_options_t enc;
            assert_int_equal(marmot_tokenizer_encode_options_init(&enc), MARMOT_SUCCESS);

            size_t needed = 0;
            assert_int_equal(marmot_tokenizer_encode(tok, text, text_len, &enc, nullptr, &needed), MARMOT_SUCCESS);
            assert_true(needed > 0 || text_len == 0);

            marmot_token_id_t *ids = needed > 0 ? malloc(needed * sizeof(*ids)) : nullptr;
            if (needed > 0) {
                assert_non_null(ids);
            }

            size_t cap = needed;
            assert_int_equal(marmot_tokenizer_encode(tok, text, text_len, &enc, ids, &cap), MARMOT_SUCCESS);
            assert_int_equal(cap, needed);

            marmot_tokenizer_decode_options_t dec;
            assert_int_equal(marmot_tokenizer_decode_options_init(&dec), MARMOT_SUCCESS);

            size_t out_len = 0;
            assert_int_equal(marmot_tokenizer_decode(tok, ids, needed, &dec, nullptr, &out_len), MARMOT_SUCCESS);
            assert_true(out_len >= 1);

            char *out = malloc(out_len);
            assert_non_null(out);
            size_t out_cap = out_len;
            assert_int_equal(marmot_tokenizer_decode(tok, ids, needed, &dec, out, &out_cap), MARMOT_SUCCESS);
            assert_int_equal(out_cap, out_len);
            assert_string_equal(out, text);

            free(out);
            free(ids);
        }

        marmot_tokenizer_destroy(tok);
    }
}

// Generic test: buffer too small error - works with ANY tokenizer
static void test_encode_buffer_too_small_all_fixtures(void **state) {
    (void)state;

    const marmot_test_fixture_t *fixture;
    MARMOT_TEST_FOR_EACH_FIXTURE(fixture) {
        marmot_tokenizer_t *tok = load_tokenizer_for_fixture(fixture->filename);
        if (tok == nullptr) {
            continue; // Skip missing fixtures
        }

        // Use text that requires multiple tokens for all tokenizers
        const char *text = "hello world";

        marmot_tokenizer_encode_options_t enc;
        assert_int_equal(marmot_tokenizer_encode_options_init(&enc), MARMOT_SUCCESS);

        size_t needed = 0;
        assert_int_equal(marmot_tokenizer_encode(tok, text, strlen(text), &enc, nullptr, &needed), MARMOT_SUCCESS);
        assert_true(needed > 1); // "hello world" should need multiple tokens

        marmot_token_id_t small[1] = {0};
        size_t cap = 1;
        assert_int_equal(
            marmot_tokenizer_encode(tok, text, strlen(text), &enc, small, &cap), MARMOT_ERROR_INVALID_ARGUMENT
        );
        assert_int_equal(cap, needed);

        marmot_tokenizer_destroy(tok);
    }
}

// ============================================================================
// LLaMA-specific tests (tinyllama fixture only)
// ============================================================================

#define LLAMA_FIXTURE "tinyllama-q4_k_m.gguf"

static marmot_tokenizer_t *load_llama_tokenizer(void) {
    return load_tokenizer_for_fixture(LLAMA_FIXTURE);
}

static void test_llama_vocab_and_special_ids(void **state) {
    (void)state;

    marmot_tokenizer_t *tok = load_llama_tokenizer();
    if (tok == nullptr) {
        skip();
    }

    assert_int_equal(marmot_tokenizer_vocab_size(tok), 32000);

    marmot_tokenizer_special_ids_t ids;
    assert_int_equal(marmot_tokenizer_get_special_ids(tok, &ids), MARMOT_SUCCESS);
    assert_true(ids.has_bos);
    assert_true(ids.has_eos);
    assert_true(ids.has_unk);
    assert_true(ids.has_pad);
    assert_int_equal(ids.bos_id, 1);
    assert_int_equal(ids.eos_id, 2);
    assert_int_equal(ids.unk_id, 0);

    marmot_tokenizer_destroy(tok);
}

static void test_llama_piece_roundtrip(void **state) {
    (void)state;

    marmot_tokenizer_t *tok = load_llama_tokenizer();
    if (tok == nullptr) {
        skip();
    }

    marmot_token_id_t id = MARMOT_TOKEN_ID_INVALID;
    assert_int_equal(marmot_tokenizer_piece_to_token(tok, "▁the", strlen("▁the"), &id), MARMOT_SUCCESS);
    assert_int_equal(id, 278);

    size_t len = 0;
    assert_int_equal(marmot_tokenizer_token_to_piece(tok, id, nullptr, &len), MARMOT_SUCCESS);
    assert_true(len > 0);

    char *buf = malloc(len);
    assert_non_null(buf);
    size_t cap = len;
    assert_int_equal(marmot_tokenizer_token_to_piece(tok, id, buf, &cap), MARMOT_SUCCESS);
    assert_int_equal(cap, len);
    assert_string_equal(buf, "▁the");
    free(buf);

    marmot_tokenizer_destroy(tok);
}

static void test_llama_encode_allow_special(void **state) {
    (void)state;

    marmot_tokenizer_t *tok = load_llama_tokenizer();
    if (tok == nullptr) {
        skip();
    }

    marmot_tokenizer_special_ids_t special_ids;
    assert_int_equal(marmot_tokenizer_get_special_ids(tok, &special_ids), MARMOT_SUCCESS);
    assert_true(special_ids.has_bos);
    assert_true(special_ids.has_eos);

    marmot_tokenizer_encode_options_t enc;
    assert_int_equal(marmot_tokenizer_encode_options_init(&enc), MARMOT_SUCCESS);

    size_t hello_needed = 0;
    assert_int_equal(
        marmot_tokenizer_encode(tok, "hello", strlen("hello"), &enc, nullptr, &hello_needed), MARMOT_SUCCESS
    );
    assert_true(hello_needed > 0);

    marmot_token_id_t *hello_ids = malloc(hello_needed * sizeof(*hello_ids));
    assert_non_null(hello_ids);
    size_t hello_cap = hello_needed;
    assert_int_equal(
        marmot_tokenizer_encode(tok, "hello", strlen("hello"), &enc, hello_ids, &hello_cap), MARMOT_SUCCESS
    );
    assert_int_equal(hello_cap, hello_needed);

    size_t bos_piece_len = 0;
    assert_int_equal(marmot_tokenizer_token_to_piece(tok, special_ids.bos_id, nullptr, &bos_piece_len), MARMOT_SUCCESS);
    assert_true(bos_piece_len > 0);
    char *bos_piece = malloc(bos_piece_len);
    assert_non_null(bos_piece);
    size_t bos_piece_cap = bos_piece_len;
    assert_int_equal(
        marmot_tokenizer_token_to_piece(tok, special_ids.bos_id, bos_piece, &bos_piece_cap), MARMOT_SUCCESS
    );
    assert_int_equal(bos_piece_cap, bos_piece_len);

    const size_t bos_hello_len = strlen(bos_piece) + strlen("hello");
    char *bos_hello = malloc(bos_hello_len + 1);
    assert_non_null(bos_hello);
    strcpy(bos_hello, bos_piece);
    strcat(bos_hello, "hello");

    enc.allow_special = true;
    size_t bos_needed = 0;
    assert_int_equal(
        marmot_tokenizer_encode(tok, bos_hello, strlen(bos_hello), &enc, nullptr, &bos_needed), MARMOT_SUCCESS
    );
    assert_int_equal(bos_needed, hello_needed + 1);

    marmot_token_id_t *bos_ids = malloc(bos_needed * sizeof(*bos_ids));
    assert_non_null(bos_ids);
    size_t bos_cap = bos_needed;
    assert_int_equal(
        marmot_tokenizer_encode(tok, bos_hello, strlen(bos_hello), &enc, bos_ids, &bos_cap), MARMOT_SUCCESS
    );
    assert_int_equal(bos_cap, bos_needed);
    assert_int_equal(bos_ids[0], special_ids.bos_id);
    for (size_t i = 0; i < hello_needed; ++i) {
        assert_int_equal(bos_ids[i + 1], hello_ids[i]);
    }

    size_t eos_piece_len = 0;
    assert_int_equal(marmot_tokenizer_token_to_piece(tok, special_ids.eos_id, nullptr, &eos_piece_len), MARMOT_SUCCESS);
    assert_true(eos_piece_len > 0);
    char *eos_piece = malloc(eos_piece_len);
    assert_non_null(eos_piece);
    size_t eos_piece_cap = eos_piece_len;
    assert_int_equal(
        marmot_tokenizer_token_to_piece(tok, special_ids.eos_id, eos_piece, &eos_piece_cap), MARMOT_SUCCESS
    );
    assert_int_equal(eos_piece_cap, eos_piece_len);

    const size_t hello_eos_len = strlen("hello") + strlen(eos_piece);
    char *hello_eos = malloc(hello_eos_len + 1);
    assert_non_null(hello_eos);
    strcpy(hello_eos, "hello");
    strcat(hello_eos, eos_piece);

    size_t eos_needed = 0;
    assert_int_equal(
        marmot_tokenizer_encode(tok, hello_eos, strlen(hello_eos), &enc, nullptr, &eos_needed), MARMOT_SUCCESS
    );
    assert_int_equal(eos_needed, hello_needed + 1);

    marmot_token_id_t *eos_ids = malloc(eos_needed * sizeof(*eos_ids));
    assert_non_null(eos_ids);
    size_t eos_cap = eos_needed;
    assert_int_equal(
        marmot_tokenizer_encode(tok, hello_eos, strlen(hello_eos), &enc, eos_ids, &eos_cap), MARMOT_SUCCESS
    );
    assert_int_equal(eos_cap, eos_needed);
    for (size_t i = 0; i < hello_needed; ++i) {
        assert_int_equal(eos_ids[i], hello_ids[i]);
    }
    assert_int_equal(eos_ids[eos_needed - 1], special_ids.eos_id);

    free(eos_ids);
    free(hello_eos);
    free(eos_piece);
    free(bos_ids);
    free(bos_hello);
    free(bos_piece);
    free(hello_ids);
    marmot_tokenizer_destroy(tok);
}

static void test_llama_known_encodings(void **state) {
    (void)state;

    marmot_tokenizer_t *tok = load_llama_tokenizer();
    if (tok == nullptr) {
        skip();
    }

    marmot_tokenizer_encode_options_t enc;
    assert_int_equal(marmot_tokenizer_encode_options_init(&enc), MARMOT_SUCCESS);

    for (size_t i = 0; i < ARRAY_LENGTH(marmot_tokenizer_golden_llama_tokenize_cases); ++i) {
        const marmot_tokenizer_golden_tokenize_case_t golden = marmot_tokenizer_golden_llama_tokenize_cases[i];
        size_t needed = 0;
        assert_int_equal(
            marmot_tokenizer_encode(tok, golden.text, strlen(golden.text), &enc, nullptr, &needed), MARMOT_SUCCESS
        );
        assert_int_equal(needed, golden.expected_len);

        marmot_token_id_t ids[8] = {0};
        size_t cap = 8;
        assert_int_equal(
            marmot_tokenizer_encode(tok, golden.text, strlen(golden.text), &enc, ids, &cap), MARMOT_SUCCESS
        );
        assert_int_equal(cap, golden.expected_len);

        for (size_t j = 0; j < golden.expected_len; ++j) {
            assert_int_equal(ids[j], golden.expected[j]);
        }
    }

    marmot_tokenizer_destroy(tok);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        // Generic tests (all fixtures)
        cmocka_unit_test(test_encode_decode_roundtrip_all_fixtures),
        cmocka_unit_test(test_encode_buffer_too_small_all_fixtures),
        // LLaMA-specific tests (tinyllama only)
        cmocka_unit_test(test_llama_vocab_and_special_ids),
        cmocka_unit_test(test_llama_piece_roundtrip),
        cmocka_unit_test(test_llama_encode_allow_special),
        cmocka_unit_test(test_llama_known_encodings),
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
