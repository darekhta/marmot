#include "bench_param_sweep.h"

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

void marmot_bench_range_init(marmot_bench_param_range_t *range) {
    range->values = nullptr;
    range->count = 0;
    range->capacity = 0;
}

static marmot_error_t range_ensure_capacity(marmot_bench_param_range_t *range, size_t needed) {
    if (range->capacity >= needed) {
        return MARMOT_SUCCESS;
    }

    size_t new_cap = range->capacity == 0 ? 8 : range->capacity * 2;
    while (new_cap < needed) {
        new_cap *= 2;
    }

    size_t *new_values = realloc(range->values, new_cap * sizeof(size_t));
    if (new_values == nullptr) {
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    range->values = new_values;
    range->capacity = new_cap;
    return MARMOT_SUCCESS;
}

static marmot_error_t range_push(marmot_bench_param_range_t *range, size_t value) {
    marmot_error_t err = range_ensure_capacity(range, range->count + 1);
    if (err != MARMOT_SUCCESS) {
        return err;
    }
    range->values[range->count++] = value;
    return MARMOT_SUCCESS;
}

marmot_error_t marmot_bench_parse_range(const char *str, marmot_bench_param_range_t *out) {
    marmot_bench_range_init(out);

    if (str == nullptr || *str == '\0') {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    // Check for range syntax: contains '-' but not at start (negative number)
    const char *dash = strchr(str, '-');
    const char *star = strchr(str, '*');
    const char *plus_op = nullptr;

    // Find '+' that's not at start (could be part of range syntax)
    for (const char *p = str; *p; p++) {
        if (*p == '+' && p != str) {
            plus_op = p;
            break;
        }
    }

    if (dash != nullptr && dash != str) {
        // Range syntax: start-end or start-end*mult or start-end+step
        size_t start = (size_t)strtoull(str, nullptr, 10);
        size_t end = (size_t)strtoull(dash + 1, nullptr, 10);

        if (start > end || start == 0) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }

        if (star != nullptr) {
            // Multiplicative: 1-8*2 -> 1, 2, 4, 8
            size_t mult = (size_t)strtoull(star + 1, nullptr, 10);
            if (mult <= 1) {
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            for (size_t v = start; v <= end; v *= mult) {
                marmot_error_t err = range_push(out, v);
                if (err != MARMOT_SUCCESS) {
                    marmot_bench_range_free(out);
                    return err;
                }
                if (v > end / mult) break; // Prevent overflow
            }
        } else if (plus_op != nullptr && plus_op > dash) {
            // Additive: 100-1000+100 -> 100, 200, ..., 1000
            size_t step = (size_t)strtoull(plus_op + 1, nullptr, 10);
            if (step == 0) {
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            for (size_t v = start; v <= end; v += step) {
                marmot_error_t err = range_push(out, v);
                if (err != MARMOT_SUCCESS) {
                    marmot_bench_range_free(out);
                    return err;
                }
            }
        } else {
            // Default: doubling 512-2048 -> 512, 1024, 2048
            for (size_t v = start; v <= end; v *= 2) {
                marmot_error_t err = range_push(out, v);
                if (err != MARMOT_SUCCESS) {
                    marmot_bench_range_free(out);
                    return err;
                }
            }
        }
    } else {
        // Comma-separated: 512,1024,2048
        char *copy = strdup(str);
        if (copy == nullptr) {
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }

        char *token = strtok(copy, ",");
        while (token != nullptr) {
            // Skip whitespace
            while (*token && isspace((unsigned char)*token)) token++;

            size_t val = (size_t)strtoull(token, nullptr, 10);
            marmot_error_t err = range_push(out, val);
            if (err != MARMOT_SUCCESS) {
                free(copy);
                marmot_bench_range_free(out);
                return err;
            }
            token = strtok(nullptr, ",");
        }
        free(copy);
    }

    if (out->count == 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    return MARMOT_SUCCESS;
}

marmot_error_t marmot_bench_parse_range_or_default(
    const char *str, size_t default_val, marmot_bench_param_range_t *out
) {
    if (str == nullptr || *str == '\0') {
        marmot_bench_range_init(out);
        return range_push(out, default_val);
    }
    return marmot_bench_parse_range(str, out);
}

void marmot_bench_range_free(marmot_bench_param_range_t *range) {
    if (range == nullptr) {
        return;
    }
    free(range->values);
    range->values = nullptr;
    range->count = 0;
    range->capacity = 0;
}
