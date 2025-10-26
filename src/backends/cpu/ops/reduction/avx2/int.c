#include "cpu_backend_internal.h"

#if MARMOT_ENABLE_AVX2

static marmot_error_t cpu_reduce_avx2_require_numeric(const void *base, double *out_value, size_t length) {
    if (base == nullptr || out_value == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null pointer in AVX2 int reduction kernel");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (length == 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "Reduction over zero elements is undefined");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_reduce_avx2_require_arg(const void *base, double *out_value, uint64_t *out_index, size_t length) {
    marmot_error_t status = cpu_reduce_avx2_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    if (out_index == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Arg reduction requires index output");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    return MARMOT_SUCCESS;
}

static inline double cpu_reduce_avx2_horizontal_sum_epi32(__m256i v) {
    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    return (double)_mm_cvtsi128_si32(sum128);
}

static inline __m256i cpu_reduce_avx2_accumulate_s16(__m256i acc, __m256i values) {
    __m128i lo = _mm256_castsi256_si128(values);
    __m128i hi = _mm256_extracti128_si256(values, 1);
    __m256i lo32 = _mm256_cvtepi16_epi32(lo);
    __m256i hi32 = _mm256_cvtepi16_epi32(hi);
    acc = _mm256_add_epi32(acc, lo32);
    acc = _mm256_add_epi32(acc, hi32);
    return acc;
}

static inline __m256i cpu_reduce_avx2_accumulate_u16(__m256i acc, __m256i values) {
    __m128i lo = _mm256_castsi256_si128(values);
    __m128i hi = _mm256_extracti128_si256(values, 1);
    __m256i lo32 = _mm256_cvtepu16_epi32(lo);
    __m256i hi32 = _mm256_cvtepu16_epi32(hi);
    acc = _mm256_add_epi32(acc, lo32);
    acc = _mm256_add_epi32(acc, hi32);
    return acc;
}

// -----------------------------------------------------------------------------
// INT16
// -----------------------------------------------------------------------------

static double cpu_reduce_avx2_i16_sum_impl(const int16_t *data, size_t length) {
    __m256i acc = _mm256_setzero_si256();
    size_t i = 0;
    for (; i + 16 <= length; i += 16) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(data + i));
        acc = cpu_reduce_avx2_accumulate_s16(acc, v);
    }
    double sum = cpu_reduce_avx2_horizontal_sum_epi32(acc);
    for (; i < length; ++i) {
        sum += (double)data[i];
    }
    return sum;
}

static marmot_error_t
cpu_reduce_i16_avx2_sum(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_avx2_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    *out_value = cpu_reduce_avx2_i16_sum_impl((const int16_t *)base, length);
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_reduce_i16_avx2_mean(const void *device_ctx, const void *base, size_t length, double *out_value) {
    marmot_error_t status = cpu_reduce_i16_avx2_sum(device_ctx, base, length, out_value);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    *out_value /= (double)length;
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_reduce_i16_avx2_prod(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_avx2_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const int16_t *data = (const int16_t *)base;
    double prod = 1.0;
    for (size_t i = 0; i < length; ++i) {
        prod *= (double)data[i];
    }
    *out_value = prod;
    return MARMOT_SUCCESS;
}

static inline int16_t
cpu_reduce_avx2_i16_finalize_best(__m256i best_vec, const int16_t *data, size_t start, size_t length, bool find_max) {
    int16_t lanes[16];
    _mm256_storeu_si256((__m256i *)lanes, best_vec);
    int16_t best = lanes[0];
    for (int lane = 1; lane < 16; ++lane) {
        int16_t value = lanes[lane];
        best = find_max ? (value > best ? value : best) : (value < best ? value : best);
    }
    for (size_t i = start; i < length; ++i) {
        int16_t value = data[i];
        best = find_max ? (value > best ? value : best) : (value < best ? value : best);
    }
    return best;
}

static int16_t cpu_reduce_avx2_i16_best(const int16_t *data, size_t length, bool find_max) {
    size_t i = 0;
    __m256i best_vec = _mm256_set1_epi16(find_max ? INT16_MIN : INT16_MAX);
    for (; i + 16 <= length; i += 16) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(data + i));
        best_vec = find_max ? _mm256_max_epi16(best_vec, v) : _mm256_min_epi16(best_vec, v);
    }
    return cpu_reduce_avx2_i16_finalize_best(best_vec, data, i, length, find_max);
}

static marmot_error_t
cpu_reduce_i16_avx2_max(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_avx2_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const int16_t *data = (const int16_t *)base;
    int16_t best = cpu_reduce_avx2_i16_best(data, length, true);
    *out_value = (double)best;
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_reduce_i16_avx2_min(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_avx2_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const int16_t *data = (const int16_t *)base;
    int16_t best = cpu_reduce_avx2_i16_best(data, length, false);
    *out_value = (double)best;
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_reduce_i16_avx2_argmax(
    const void *device_ctx, const void *base, size_t length, double *out_value, uint64_t *out_index
) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_avx2_require_arg(base, out_value, out_index, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const int16_t *data = (const int16_t *)base;
    int16_t best = cpu_reduce_avx2_i16_best(data, length, true);
    size_t idx = 0;
    for (; idx < length; ++idx) {
        if (data[idx] == best) {
            break;
        }
    }
    *out_value = (double)best;
    *out_index = (uint64_t)idx;
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_reduce_i16_avx2_argmin(
    const void *device_ctx, const void *base, size_t length, double *out_value, uint64_t *out_index
) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_avx2_require_arg(base, out_value, out_index, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const int16_t *data = (const int16_t *)base;
    int16_t best = cpu_reduce_avx2_i16_best(data, length, false);
    size_t idx = 0;
    for (; idx < length; ++idx) {
        if (data[idx] == best) {
            break;
        }
    }
    *out_value = (double)best;
    *out_index = (uint64_t)idx;
    return MARMOT_SUCCESS;
}

const cpu_reduce_traits_t cpu_reduce_i16_avx2_traits = {
    .dtype = MARMOT_DTYPE_INT16,
    .impl_kind = CPU_REDUCE_IMPL_AVX2,
    .ops = {
        .sum = cpu_reduce_i16_avx2_sum,
        .mean = cpu_reduce_i16_avx2_mean,
        .prod = cpu_reduce_i16_avx2_prod,
        .min = cpu_reduce_i16_avx2_min,
        .max = cpu_reduce_i16_avx2_max,
        .argmax = cpu_reduce_i16_avx2_argmax,
        .argmin = cpu_reduce_i16_avx2_argmin,
        .impl_name = "avx2-i16",
    },
};

CPU_REDUCTION_REGISTER_TRAITS(cpu_reduce_i16_avx2_traits)

// -----------------------------------------------------------------------------
// UINT16
// -----------------------------------------------------------------------------

static double cpu_reduce_avx2_u16_sum_impl(const uint16_t *data, size_t length) {
    __m256i acc = _mm256_setzero_si256();
    size_t i = 0;
    for (; i + 16 <= length; i += 16) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(data + i));
        acc = cpu_reduce_avx2_accumulate_u16(acc, v);
    }
    double sum = cpu_reduce_avx2_horizontal_sum_epi32(acc);
    for (; i < length; ++i) {
        sum += (double)data[i];
    }
    return sum;
}

static marmot_error_t
cpu_reduce_u16_avx2_sum(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_avx2_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    *out_value = cpu_reduce_avx2_u16_sum_impl((const uint16_t *)base, length);
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_reduce_u16_avx2_mean(const void *device_ctx, const void *base, size_t length, double *out_value) {
    marmot_error_t status = cpu_reduce_u16_avx2_sum(device_ctx, base, length, out_value);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    *out_value /= (double)length;
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_reduce_u16_avx2_prod(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_avx2_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const uint16_t *data = (const uint16_t *)base;
    double prod = 1.0;
    for (size_t i = 0; i < length; ++i) {
        prod *= (double)data[i];
    }
    *out_value = prod;
    return MARMOT_SUCCESS;
}

static inline uint16_t
cpu_reduce_avx2_u16_finalize_best(__m256i best_vec, const uint16_t *data, size_t start, size_t length, bool find_max) {
    uint16_t lanes[16];
    _mm256_storeu_si256((__m256i *)lanes, best_vec);
    uint16_t best = lanes[0];
    for (int lane = 1; lane < 16; ++lane) {
        uint16_t value = lanes[lane];
        best = find_max ? (value > best ? value : best) : (value < best ? value : best);
    }
    for (size_t i = start; i < length; ++i) {
        uint16_t value = data[i];
        best = find_max ? (value > best ? value : best) : (value < best ? value : best);
    }
    return best;
}

static uint16_t cpu_reduce_avx2_u16_best(const uint16_t *data, size_t length, bool find_max) {
    size_t i = 0;
    __m256i best_vec = _mm256_set1_epi16(find_max ? 0 : (int)UINT16_MAX);
    for (; i + 16 <= length; i += 16) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(data + i));
        best_vec = find_max ? _mm256_max_epu16(best_vec, v) : _mm256_min_epu16(best_vec, v);
    }
    return cpu_reduce_avx2_u16_finalize_best(best_vec, data, i, length, find_max);
}

static marmot_error_t
cpu_reduce_u16_avx2_max(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_avx2_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const uint16_t *data = (const uint16_t *)base;
    uint16_t best = cpu_reduce_avx2_u16_best(data, length, true);
    *out_value = (double)best;
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_reduce_u16_avx2_min(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_avx2_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const uint16_t *data = (const uint16_t *)base;
    uint16_t best = cpu_reduce_avx2_u16_best(data, length, false);
    *out_value = (double)best;
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_reduce_u16_avx2_argmax(
    const void *device_ctx, const void *base, size_t length, double *out_value, uint64_t *out_index
) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_avx2_require_arg(base, out_value, out_index, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const uint16_t *data = (const uint16_t *)base;
    uint16_t best = cpu_reduce_avx2_u16_best(data, length, true);
    size_t idx = 0;
    for (; idx < length; ++idx) {
        if (data[idx] == best) {
            break;
        }
    }
    *out_value = (double)best;
    *out_index = (uint64_t)idx;
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_reduce_u16_avx2_argmin(
    const void *device_ctx, const void *base, size_t length, double *out_value, uint64_t *out_index
) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_avx2_require_arg(base, out_value, out_index, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const uint16_t *data = (const uint16_t *)base;
    uint16_t best = cpu_reduce_avx2_u16_best(data, length, false);
    size_t idx = 0;
    for (; idx < length; ++idx) {
        if (data[idx] == best) {
            break;
        }
    }
    *out_value = (double)best;
    *out_index = (uint64_t)idx;
    return MARMOT_SUCCESS;
}

const cpu_reduce_traits_t cpu_reduce_u16_avx2_traits = {
    .dtype = MARMOT_DTYPE_UINT16,
    .impl_kind = CPU_REDUCE_IMPL_AVX2,
    .ops = {
        .sum = cpu_reduce_u16_avx2_sum,
        .mean = cpu_reduce_u16_avx2_mean,
        .prod = cpu_reduce_u16_avx2_prod,
        .min = cpu_reduce_u16_avx2_min,
        .max = cpu_reduce_u16_avx2_max,
        .argmax = cpu_reduce_u16_avx2_argmax,
        .argmin = cpu_reduce_u16_avx2_argmin,
        .impl_name = "avx2-u16",
    },
};

CPU_REDUCTION_REGISTER_TRAITS(cpu_reduce_u16_avx2_traits)

// -----------------------------------------------------------------------------
// INT8
// -----------------------------------------------------------------------------

static double cpu_reduce_avx2_i8_sum_impl(const int8_t *data, size_t length) {
    __m256i acc = _mm256_setzero_si256();
    size_t i = 0;
    for (; i + 32 <= length; i += 32) {
        __m256i chunk = _mm256_loadu_si256((const __m256i *)(data + i));
        __m256i lo16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(chunk));
        __m256i hi16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(chunk, 1));
        acc = cpu_reduce_avx2_accumulate_s16(acc, lo16);
        acc = cpu_reduce_avx2_accumulate_s16(acc, hi16);
    }
    if (i + 16 <= length) {
        __m128i chunk = _mm_loadu_si128((const __m128i *)(data + i));
        __m256i wide = _mm256_cvtepi8_epi16(chunk);
        acc = cpu_reduce_avx2_accumulate_s16(acc, wide);
        i += 16;
    }
    double sum = cpu_reduce_avx2_horizontal_sum_epi32(acc);
    for (; i < length; ++i) {
        sum += (double)data[i];
    }
    return sum;
}

static marmot_error_t
cpu_reduce_i8_avx2_sum(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_avx2_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    *out_value = cpu_reduce_avx2_i8_sum_impl((const int8_t *)base, length);
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_reduce_i8_avx2_mean(const void *device_ctx, const void *base, size_t length, double *out_value) {
    marmot_error_t status = cpu_reduce_i8_avx2_sum(device_ctx, base, length, out_value);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    *out_value /= (double)length;
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_reduce_i8_avx2_prod(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_avx2_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const int8_t *data = (const int8_t *)base;
    double prod = 1.0;
    for (size_t i = 0; i < length; ++i) {
        prod *= (double)data[i];
    }
    *out_value = prod;
    return MARMOT_SUCCESS;
}

static inline int8_t
cpu_reduce_avx2_i8_finalize_best(__m256i best_vec, const int8_t *data, size_t start, size_t length, bool find_max) {
    int8_t lanes[32];
    _mm256_storeu_si256((__m256i *)lanes, best_vec);
    int8_t best = lanes[0];
    for (int lane = 1; lane < 32; ++lane) {
        int8_t value = lanes[lane];
        best = find_max ? (value > best ? value : best) : (value < best ? value : best);
    }
    for (size_t i = start; i < length; ++i) {
        int8_t value = data[i];
        best = find_max ? (value > best ? value : best) : (value < best ? value : best);
    }
    return best;
}

static int8_t cpu_reduce_avx2_i8_best(const int8_t *data, size_t length, bool find_max) {
    size_t i = 0;
    __m256i best_vec = _mm256_set1_epi8(find_max ? INT8_MIN : INT8_MAX);
    for (; i + 32 <= length; i += 32) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(data + i));
        best_vec = find_max ? _mm256_max_epi8(best_vec, v) : _mm256_min_epi8(best_vec, v);
    }
    return cpu_reduce_avx2_i8_finalize_best(best_vec, data, i, length, find_max);
}

static marmot_error_t
cpu_reduce_i8_avx2_max(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_avx2_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const int8_t *data = (const int8_t *)base;
    int8_t best = cpu_reduce_avx2_i8_best(data, length, true);
    *out_value = (double)best;
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_reduce_i8_avx2_min(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_avx2_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const int8_t *data = (const int8_t *)base;
    int8_t best = cpu_reduce_avx2_i8_best(data, length, false);
    *out_value = (double)best;
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_reduce_i8_avx2_argmax(
    const void *device_ctx, const void *base, size_t length, double *out_value, uint64_t *out_index
) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_avx2_require_arg(base, out_value, out_index, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const int8_t *data = (const int8_t *)base;
    int8_t best = cpu_reduce_avx2_i8_best(data, length, true);
    size_t idx = 0;
    for (; idx < length; ++idx) {
        if (data[idx] == best) {
            break;
        }
    }
    *out_value = (double)best;
    *out_index = (uint64_t)idx;
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_reduce_i8_avx2_argmin(
    const void *device_ctx, const void *base, size_t length, double *out_value, uint64_t *out_index
) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_avx2_require_arg(base, out_value, out_index, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const int8_t *data = (const int8_t *)base;
    int8_t best = cpu_reduce_avx2_i8_best(data, length, false);
    size_t idx = 0;
    for (; idx < length; ++idx) {
        if (data[idx] == best) {
            break;
        }
    }
    *out_value = (double)best;
    *out_index = (uint64_t)idx;
    return MARMOT_SUCCESS;
}

const cpu_reduce_traits_t cpu_reduce_i8_avx2_traits = {
    .dtype = MARMOT_DTYPE_INT8,
    .impl_kind = CPU_REDUCE_IMPL_AVX2,
    .ops = {
        .sum = cpu_reduce_i8_avx2_sum,
        .mean = cpu_reduce_i8_avx2_mean,
        .prod = cpu_reduce_i8_avx2_prod,
        .min = cpu_reduce_i8_avx2_min,
        .max = cpu_reduce_i8_avx2_max,
        .argmax = cpu_reduce_i8_avx2_argmax,
        .argmin = cpu_reduce_i8_avx2_argmin,
        .impl_name = "avx2-i8",
    },
};

CPU_REDUCTION_REGISTER_TRAITS(cpu_reduce_i8_avx2_traits)

// -----------------------------------------------------------------------------
// UINT8
// -----------------------------------------------------------------------------

static double cpu_reduce_avx2_u8_sum_impl(const uint8_t *data, size_t length) {
    __m256i acc = _mm256_setzero_si256();
    size_t i = 0;
    for (; i + 32 <= length; i += 32) {
        __m256i chunk = _mm256_loadu_si256((const __m256i *)(data + i));
        __m256i lo16 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(chunk));
        __m256i hi16 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(chunk, 1));
        acc = cpu_reduce_avx2_accumulate_u16(acc, lo16);
        acc = cpu_reduce_avx2_accumulate_u16(acc, hi16);
    }
    if (i + 16 <= length) {
        __m128i chunk = _mm_loadu_si128((const __m128i *)(data + i));
        __m256i wide = _mm256_cvtepu8_epi16(chunk);
        acc = cpu_reduce_avx2_accumulate_u16(acc, wide);
        i += 16;
    }
    double sum = cpu_reduce_avx2_horizontal_sum_epi32(acc);
    for (; i < length; ++i) {
        sum += (double)data[i];
    }
    return sum;
}

static marmot_error_t
cpu_reduce_u8_avx2_sum(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_avx2_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    *out_value = cpu_reduce_avx2_u8_sum_impl((const uint8_t *)base, length);
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_reduce_u8_avx2_mean(const void *device_ctx, const void *base, size_t length, double *out_value) {
    marmot_error_t status = cpu_reduce_u8_avx2_sum(device_ctx, base, length, out_value);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    *out_value /= (double)length;
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_reduce_u8_avx2_prod(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_avx2_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const uint8_t *data = (const uint8_t *)base;
    double prod = 1.0;
    for (size_t i = 0; i < length; ++i) {
        prod *= (double)data[i];
    }
    *out_value = prod;
    return MARMOT_SUCCESS;
}

static inline uint8_t
cpu_reduce_avx2_u8_finalize_best(__m256i best_vec, const uint8_t *data, size_t start, size_t length, bool find_max) {
    uint8_t lanes[32];
    _mm256_storeu_si256((__m256i *)lanes, best_vec);
    uint8_t best = lanes[0];
    for (int lane = 1; lane < 32; ++lane) {
        uint8_t value = lanes[lane];
        best = find_max ? (value > best ? value : best) : (value < best ? value : best);
    }
    for (size_t i = start; i < length; ++i) {
        uint8_t value = data[i];
        best = find_max ? (value > best ? value : best) : (value < best ? value : best);
    }
    return best;
}

static uint8_t cpu_reduce_avx2_u8_best(const uint8_t *data, size_t length, bool find_max) {
    size_t i = 0;
    __m256i best_vec = _mm256_set1_epi8(find_max ? 0 : (int)UINT8_MAX);
    for (; i + 32 <= length; i += 32) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(data + i));
        best_vec = find_max ? _mm256_max_epu8(best_vec, v) : _mm256_min_epu8(best_vec, v);
    }
    return cpu_reduce_avx2_u8_finalize_best(best_vec, data, i, length, find_max);
}

static marmot_error_t
cpu_reduce_u8_avx2_max(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_avx2_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const uint8_t *data = (const uint8_t *)base;
    uint8_t best = cpu_reduce_avx2_u8_best(data, length, true);
    *out_value = (double)best;
    return MARMOT_SUCCESS;
}

static marmot_error_t
cpu_reduce_u8_avx2_min(const void *device_ctx, const void *base, size_t length, double *out_value) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_avx2_require_numeric(base, out_value, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const uint8_t *data = (const uint8_t *)base;
    uint8_t best = cpu_reduce_avx2_u8_best(data, length, false);
    *out_value = (double)best;
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_reduce_u8_avx2_argmax(
    const void *device_ctx, const void *base, size_t length, double *out_value, uint64_t *out_index
) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_avx2_require_arg(base, out_value, out_index, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const uint8_t *data = (const uint8_t *)base;
    uint8_t best = cpu_reduce_avx2_u8_best(data, length, true);
    size_t idx = 0;
    for (; idx < length; ++idx) {
        if (data[idx] == best) {
            break;
        }
    }
    *out_value = (double)best;
    *out_index = (uint64_t)idx;
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_reduce_u8_avx2_argmin(
    const void *device_ctx, const void *base, size_t length, double *out_value, uint64_t *out_index
) {
    (void)device_ctx;
    marmot_error_t status = cpu_reduce_avx2_require_arg(base, out_value, out_index, length);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    const uint8_t *data = (const uint8_t *)base;
    uint8_t best = cpu_reduce_avx2_u8_best(data, length, false);
    size_t idx = 0;
    for (; idx < length; ++idx) {
        if (data[idx] == best) {
            break;
        }
    }
    *out_value = (double)best;
    *out_index = (uint64_t)idx;
    return MARMOT_SUCCESS;
}

const cpu_reduce_traits_t cpu_reduce_u8_avx2_traits = {
    .dtype = MARMOT_DTYPE_UINT8,
    .impl_kind = CPU_REDUCE_IMPL_AVX2,
    .ops = {
        .sum = cpu_reduce_u8_avx2_sum,
        .mean = cpu_reduce_u8_avx2_mean,
        .prod = cpu_reduce_u8_avx2_prod,
        .min = cpu_reduce_u8_avx2_min,
        .max = cpu_reduce_u8_avx2_max,
        .argmax = cpu_reduce_u8_avx2_argmax,
        .argmin = cpu_reduce_u8_avx2_argmin,
        .impl_name = "avx2-u8",
    },
};

CPU_REDUCTION_REGISTER_TRAITS(cpu_reduce_u8_avx2_traits)

#endif // MARMOT_ENABLE_AVX2
