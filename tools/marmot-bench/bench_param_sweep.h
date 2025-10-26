#ifndef MARMOT_BENCH_PARAM_SWEEP_H
#define MARMOT_BENCH_PARAM_SWEEP_H

#include "marmot/error.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Parameter range for sweep functionality
// Supports: "1,2,4,8" or "1-8" or "1-8*2" or "100-1000+100"
typedef struct {
    size_t *values;
    size_t count;
    size_t capacity;
} marmot_bench_param_range_t;

// Initialize range to empty
void marmot_bench_range_init(marmot_bench_param_range_t *range);

// Parse range string into values array
// Syntax:
//   "512,1024,2048"  -> explicit values
//   "512-2048"       -> 512, 1024, 2048 (doubling)
//   "1-8*2"          -> 1, 2, 4, 8 (multiplicative)
//   "100-1000+100"   -> 100, 200, 300, ..., 1000 (additive)
[[nodiscard]] marmot_error_t marmot_bench_parse_range(const char *str, marmot_bench_param_range_t *out);

// Parse range or use default if str is NULL
[[nodiscard]] marmot_error_t marmot_bench_parse_range_or_default(
    const char *str, size_t default_val, marmot_bench_param_range_t *out
);

// Free range resources
void marmot_bench_range_free(marmot_bench_param_range_t *range);

#ifdef __cplusplus
}
#endif

#endif // MARMOT_BENCH_PARAM_SWEEP_H
