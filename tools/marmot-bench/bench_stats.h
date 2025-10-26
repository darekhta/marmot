#ifndef MARMOT_BENCH_STATS_H
#define MARMOT_BENCH_STATS_H

#include "bench_core.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void marmot_bench_compute_stats(
    const double *samples_us,
    size_t count,
    double confidence_level,
    marmot_bench_stats_t *out_stats
);

double marmot_bench_percentile(const double *sorted_samples, size_t count, double percentile);

void marmot_bench_confidence_interval(
    double mean,
    double stddev,
    size_t count,
    double confidence_level,
    double *out_low,
    double *out_high
);

int marmot_bench_compare_double(const void *a, const void *b);

#ifdef __cplusplus
}
#endif

#endif
