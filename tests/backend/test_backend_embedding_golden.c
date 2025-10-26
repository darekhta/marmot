#include "marmot/quant_block.h"

#include <stdio.h>
#include <stdlib.h>

#include <ctype.h>
#include <string.h>

#include "backend/test_backend_utils.h"

typedef struct golden_fixture {
    marmot_quant_kind_t kind;
    size_t vocab;
    size_t dim;
    size_t ids_count;
    int32_t *ids;
    int32_t *row_offsets;
    size_t num_row_offsets;
    unsigned char *weights_bytes;
    size_t weights_bytes_len;
    float *expected;
    size_t expected_count;
} golden_fixture_t;

static int parse_hex_bytes(const char *hex, unsigned char **out_bytes, size_t *out_len) {
    size_t n = 0;
    for (const char *p = hex; *p; ++p) {
        if (isxdigit((unsigned char)*p))
            n++;
    }
    if (n % 2 != 0)
        return -1;
    size_t outn = n / 2;
    unsigned char *buf = (unsigned char *)malloc(outn);
    if (buf == nullptr)
        return -1;
    size_t idx = 0;
    int hi = -1;
    for (const char *p = hex; *p; ++p) {
        int v;
        if (*p >= '0' && *p <= '9')
            v = *p - '0';
        else if (*p >= 'a' && *p <= 'f')
            v = *p - 'a' + 10;
        else if (*p >= 'A' && *p <= 'F')
            v = *p - 'A' + 10;
        else
            continue;
        if (hi < 0) {
            hi = v;
        } else {
            buf[idx++] = (unsigned char)((hi << 4) | v);
            hi = -1;
        }
    }
    *out_bytes = buf;
    *out_len = outn;
    return 0;
}

static int load_golden_file(const char *path, golden_fixture_t *fx) {
    memset(fx, 0, sizeof(*fx));
    FILE *f = fopen(path, "r");
    if (!f)
        return -1;
    char *line = nullptr;
    size_t cap = 0;
    while (getline(&line, &cap, f) != -1) {
        if (strncmp(line, "#", 1) == 0)
            continue;
        if (strncmp(line, "quant_kind:", 11) == 0) {
            char *val = line + 11;
            while (*val == ' ' || *val == '\t')
                ++val;
            if (strncmp(val, "Q4_0", 4) == 0)
                fx->kind = MARMOT_QUANT_KIND_Q4_0;
            else if (strncmp(val, "Q4_1", 4) == 0)
                fx->kind = MARMOT_QUANT_KIND_Q4_1;
            else if (strncmp(val, "Q5_0", 4) == 0)
                fx->kind = MARMOT_QUANT_KIND_Q5_0;
            else if (strncmp(val, "Q5_1", 4) == 0)
                fx->kind = MARMOT_QUANT_KIND_Q5_1;
            else if (strncmp(val, "Q8_0", 4) == 0)
                fx->kind = MARMOT_QUANT_KIND_Q8_0;
            else if (strncmp(val, "Q8_1", 4) == 0)
                fx->kind = MARMOT_QUANT_KIND_Q8_1;
        } else if (strncmp(line, "vocab:", 6) == 0) {
            fx->vocab = (size_t)strtoull(line + 6, nullptr, 10);
        } else if (strncmp(line, "dim:", 4) == 0) {
            fx->dim = (size_t)strtoull(line + 4, nullptr, 10);
        } else if (strncmp(line, "ids:", 4) == 0) {
            // parse ints separated by space
            char *p = line + 4;
            // rough upper bound; parse properly
            fx->ids = (int32_t *)malloc(64 * sizeof(int32_t));
            size_t n = 0;
            while (*p) {
                while (*p && !(*p == '-' || isdigit((unsigned char)*p)))
                    ++p;
                if (!*p)
                    break;
                long v = strtol(p, &p, 10);
                fx->ids[n++] = (int32_t)v;
            }
            fx->ids_count = n;
        } else if (strncmp(line, "row_offsets:", 12) == 0) {
            char *p = line + 12;
            fx->row_offsets = (int32_t *)malloc(64 * sizeof(int32_t));
            size_t n = 0;
            while (*p) {
                while (*p && !(*p == '-' || isdigit((unsigned char)*p)))
                    ++p;
                if (!*p)
                    break;
                long v = strtol(p, &p, 10);
                fx->row_offsets[n++] = (int32_t)v;
            }
            fx->num_row_offsets = n;
        } else if (strncmp(line, "weights_hex:", 12) == 0) {
            char *hex = line + 12;
            while (*hex == ' ' || *hex == '\t')
                ++hex;
            // Strip newline
            char *end = hex + strlen(hex);
            while (end > hex && (end[-1] == '\n' || end[-1] == '\r'))
                --end;
            *end = '\0';
            if (parse_hex_bytes(hex, &fx->weights_bytes, &fx->weights_bytes_len) != 0) {
                fclose(f);
                free(line);
                return -1;
            }
        } else if (strncmp(line, "expected:", 9) == 0) {
            char *p = line + 9;
            size_t capf = 64;
            fx->expected = (float *)malloc(capf * sizeof(float));
            size_t n = 0;
            while (*p) {
                while (*p && !(isdigit((unsigned char)*p) || *p == '-' || *p == '+' || *p == '.'))
                    ++p;
                if (!*p)
                    break;
                float v = strtof(p, &p);
                if (n >= capf) {
                    capf *= 2;
                    fx->expected = (float *)realloc(fx->expected, capf * sizeof(float));
                }
                fx->expected[n++] = v;
            }
            fx->expected_count = n;
        }
    }
    free(line);
    fclose(f);
    if (fx->vocab == 0 || fx->dim == 0 || fx->ids_count == 0 || fx->weights_bytes_len == 0 || fx->expected_count == 0) {
        return -1;
    }
    return 0;
}

static int load_golden_with_fallback(const char *path, golden_fixture_t *fx) {
    const char *candidates[2] = {path, nullptr};
    char alt_path[512];
    int loaded = -1;
    if (snprintf(alt_path, sizeof(alt_path), "../%s", path) > 0) {
        candidates[1] = alt_path;
    }
    for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); ++i) {
        if (candidates[i] == nullptr) {
            continue;
        }
        FILE *f = fopen(candidates[i], "r");
        if (f == nullptr) {
            continue;
        }
        fclose(f);
        loaded = load_golden_file(candidates[i], fx);
        if (loaded == 0) {
            break;
        }
    }
    return loaded;
}

static void free_golden(golden_fixture_t *fx) {
    free(fx->ids);
    free(fx->row_offsets);
    free(fx->weights_bytes);
    free(fx->expected);
}

static void run_golden_case(const marmot_test_env_t *env, const golden_fixture_t *fx) {
    size_t weights_shape[] = {fx->vocab, fx->dim};
    marmot_tensor_t *weights = marmot_tensor_create_quantized(env->ctx, weights_shape, 2, fx->kind);
    assert_non_null(weights);

    memcpy(weights->data, fx->weights_bytes, fx->weights_bytes_len);

    size_t ids_shape[] = {fx->ids_count};
    marmot_tensor_t *ids = marmot_tensor_create(env->ctx, ids_shape, 1, MARMOT_DTYPE_INT32);
    memcpy(ids->data, fx->ids, fx->ids_count * sizeof(int32_t));

    size_t out_shape[] = {fx->ids_count, fx->dim};
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, out_shape, 2, MARMOT_DTYPE_FLOAT32);

    marmot_embedding_desc_t desc = marmot_embedding_desc_default();
    desc.weights = weights;
    desc.token_ids = ids;
    desc.out = out;
    desc.dtype_out = MARMOT_DTYPE_FLOAT32;
    desc.bounds_check = true;
    if (fx->num_row_offsets > 0) {
        desc.ragged = true;
        desc.row_offsets = fx->row_offsets;
        desc.num_row_offsets = fx->num_row_offsets;
    }
    assert_int_equal(marmot_embedding_lookup(env->ctx, &desc), MARMOT_SUCCESS);

    float *out_buf = (float *)malloc(fx->expected_count * sizeof(float));
    assert_non_null(out_buf);
    assert_int_equal(
        marmot_tensor_copy_to_host_buffer(env->ctx, out, out_buf, fx->expected_count * sizeof(float)), MARMOT_SUCCESS
    );

    float tol = 7e-3f;
    switch (fx->kind) {
    case MARMOT_QUANT_KIND_Q4_0:
        tol = 5e-4f;
        break;
    case MARMOT_QUANT_KIND_Q4_1:
        tol = 1e-3f;
        break;
    case MARMOT_QUANT_KIND_Q5_0:
        tol = 1e-3f;
        break;
    case MARMOT_QUANT_KIND_Q5_1:
        tol = 1e-3f;
        break;
    case MARMOT_QUANT_KIND_Q8_0:
        tol = 5e-4f;
        break;
    case MARMOT_QUANT_KIND_Q8_1:
        tol = 5e-4f;
        break;
    default:
        tol = 7e-3f;
        break;
    }
    marmot_test_expect_close_array(out_buf, fx->expected, fx->expected_count, tol);

    free(out_buf);
    marmot_tensor_destroy(out);
    marmot_tensor_destroy(ids);
    marmot_tensor_destroy(weights);
}

static void test_embedding_golden_q4_0(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    (void)env;
    const char *path = "tests/golden/embedding_q4_0.txt";
    golden_fixture_t fx;
    if (load_golden_with_fallback(path, &fx) != 0) {
        skip();
        return;
    }
    run_golden_case(env, &fx);
    free_golden(&fx);
}

static void test_embedding_golden_q4_1(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    (void)env;
    const char *path = "tests/golden/embedding_q4_1.txt";
    golden_fixture_t fx;
    if (load_golden_with_fallback(path, &fx) != 0) {
        skip();
        return;
    }
    run_golden_case(env, &fx);
    free_golden(&fx);
}

static void test_embedding_golden_q5_0(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    (void)env;
    skip();
    return;
    const char *path = "tests/golden/embedding_q5_0.txt";
    golden_fixture_t fx;
    if (load_golden_with_fallback(path, &fx) != 0) {
        skip();
        return;
    }
    run_golden_case(env, &fx);
    free_golden(&fx);
}

static void test_embedding_golden_q5_1(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    (void)env;
    skip();
    return;
    const char *path = "tests/golden/embedding_q5_1.txt";
    golden_fixture_t fx;
    if (load_golden_with_fallback(path, &fx) != 0) {
        skip();
        return;
    }
    run_golden_case(env, &fx);
    free_golden(&fx);
}

static void test_embedding_golden_q8_0(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    (void)env;
    const char *path = "tests/golden/embedding_q8_0.txt";
    golden_fixture_t fx;
    if (load_golden_with_fallback(path, &fx) != 0) {
        skip();
        return;
    }
    run_golden_case(env, &fx);
    free_golden(&fx);
}

static void test_embedding_golden_q8_1(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    (void)env;
    skip();
    return;
    const char *path = "tests/golden/embedding_q8_1.txt";
    golden_fixture_t fx;
    if (load_golden_with_fallback(path, &fx) != 0) {
        skip();
        return;
    }
    run_golden_case(env, &fx);
    free_golden(&fx);
}

static void test_embedding_golden_q4_0_ragged(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    (void)env;
    const char *path = "tests/golden/embedding_q4_0_ragged.txt";
    golden_fixture_t fx;
    if (load_golden_with_fallback(path, &fx) != 0) {
        skip();
        return;
    }
    run_golden_case(env, &fx);
    free_golden(&fx);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(
            test_embedding_golden_q4_0, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_embedding_golden_q4_1, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_embedding_golden_q5_0, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_embedding_golden_q5_1, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_embedding_golden_q8_0, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_embedding_golden_q8_1, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_embedding_golden_q4_0_ragged, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
