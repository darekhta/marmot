#include "bench_stats.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

int marmot_bench_compare_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db)
        return -1;
    if (da > db)
        return 1;
    return 0;
}

double marmot_bench_percentile(const double *sorted_samples, size_t count, double percentile) {
    if (count == 0)
        return 0.0;
    if (count == 1)
        return sorted_samples[0];

    double idx = percentile * (double)(count - 1);
    size_t lo = (size_t)floor(idx);
    size_t hi = (size_t)ceil(idx);

    if (lo == hi || hi >= count) {
        return sorted_samples[lo];
    }

    double frac = idx - (double)lo;
    return sorted_samples[lo] * (1.0 - frac) + sorted_samples[hi] * frac;
}

void marmot_bench_confidence_interval(
    double mean, double stddev, size_t count, double confidence_level, double *out_low, double *out_high
) {
    if (count < 2) {
        *out_low = mean;
        *out_high = mean;
        return;
    }

    double z = 1.96;
    if (confidence_level >= 0.99) {
        z = 2.576;
    } else if (confidence_level >= 0.95) {
        z = 1.96;
    } else if (confidence_level >= 0.90) {
        z = 1.645;
    }

    double margin = z * stddev / sqrt((double)count);
    *out_low = mean - margin;
    *out_high = mean + margin;
}

void marmot_bench_compute_stats(
    const double *samples_us, size_t count, double confidence_level, marmot_bench_stats_t *out_stats
) {
    memset(out_stats, 0, sizeof(*out_stats));
    out_stats->sample_count = count;

    if (count == 0) {
        return;
    }

    double *sorted = malloc(count * sizeof(double));
    if (sorted == nullptr) {
        return;
    }
    memcpy(sorted, samples_us, count * sizeof(double));
    qsort(sorted, count, sizeof(double), marmot_bench_compare_double);

    out_stats->min_us = sorted[0];
    out_stats->max_us = sorted[count - 1];

    double sum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        sum += sorted[i];
    }
    out_stats->mean_us = sum / (double)count;

    double sum_sq = 0.0;
    for (size_t i = 0; i < count; ++i) {
        double diff = sorted[i] - out_stats->mean_us;
        sum_sq += diff * diff;
    }
    if (count > 1) {
        out_stats->stddev_us = sqrt(sum_sq / (double)(count - 1));
    }

    out_stats->p50_us = marmot_bench_percentile(sorted, count, 0.50);
    out_stats->p95_us = marmot_bench_percentile(sorted, count, 0.95);
    out_stats->p99_us = marmot_bench_percentile(sorted, count, 0.99);

    marmot_bench_confidence_interval(
        out_stats->mean_us, out_stats->stddev_us, count, confidence_level, &out_stats->ci_low_us,
        &out_stats->ci_high_us
    );

    free(sorted);
}
