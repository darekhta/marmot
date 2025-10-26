#include "marmot/tensor.h"

#include "cpu_backend_internal.h"
#include "ops/elementwise/elementwise_int_common.h"

// -----------------------------------------------------------------------------
// Scalar integer kernels
// -----------------------------------------------------------------------------

#define CPU_EW_DEFINE_INT_POW_SIGNED(name, base_t)                                                                     \
    static base_t cpu_ew_pow_##name(base_t base, base_t exp) {                                                         \
        int64_t e = (int64_t)exp;                                                                                      \
        if (e < 0) {                                                                                                   \
            if (base == (base_t)1) {                                                                                   \
                return (base_t)1;                                                                                      \
            }                                                                                                          \
            if (base == (base_t) - 1) {                                                                                \
                return (base_t)((e & 1LL) ? -1 : 1);                                                                   \
            }                                                                                                          \
            return (base_t)0;                                                                                          \
        }                                                                                                              \
        base_t result = (base_t)1;                                                                                     \
        base_t factor = base;                                                                                          \
        while (e > 0) {                                                                                                \
            if (e & 1LL) {                                                                                             \
                result = (base_t)(result * factor);                                                                    \
            }                                                                                                          \
            e >>= 1LL;                                                                                                 \
            if (e != 0) {                                                                                              \
                factor = (base_t)(factor * factor);                                                                    \
            }                                                                                                          \
        }                                                                                                              \
        return result;                                                                                                 \
    }

#define CPU_EW_DEFINE_INT_POW_UNSIGNED(name, base_t)                                                                   \
    static base_t cpu_ew_pow_##name(base_t base, base_t exp) {                                                         \
        uint64_t e = (uint64_t)exp;                                                                                    \
        base_t result = (base_t)1;                                                                                     \
        base_t factor = base;                                                                                          \
        while (e > 0) {                                                                                                \
            if (e & 1ULL) {                                                                                            \
                result = (base_t)(result * factor);                                                                    \
            }                                                                                                          \
            e >>= 1ULL;                                                                                                \
            if (e != 0) {                                                                                              \
                factor = (base_t)(factor * factor);                                                                    \
            }                                                                                                          \
        }                                                                                                              \
        return result;                                                                                                 \
    }

CPU_EW_DEFINE_INT_POW_SIGNED(i8, int8_t)
CPU_EW_DEFINE_INT_POW_SIGNED(i16, int16_t)
CPU_EW_DEFINE_INT_POW_SIGNED(i32, int32_t)
CPU_EW_DEFINE_INT_POW_SIGNED(i64, int64_t)
CPU_EW_DEFINE_INT_POW_UNSIGNED(u8, uint8_t)
CPU_EW_DEFINE_INT_POW_UNSIGNED(u16, uint16_t)
CPU_EW_DEFINE_INT_POW_UNSIGNED(u32, uint32_t)
CPU_EW_DEFINE_INT_POW_UNSIGNED(u64, uint64_t)

#undef CPU_EW_DEFINE_INT_POW_UNSIGNED
#undef CPU_EW_DEFINE_INT_POW_SIGNED

#define CPU_EW_DEFINE_INT_ARITH(op, suffix, struct_t, base_t, expr)                                                    \
    marmot_error_t cpu_##op##_##suffix##_scalar(                                                                       \
        const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out               \
    ) {                                                                                                                \
        (void)device_ctx;                                                                                              \
        const size_t n = marmot_tensor_num_elements(a);                                                                \
        const struct_t *lhs = (const struct_t *)a->data;                                                               \
        const struct_t *rhs = (const struct_t *)b->data;                                                               \
        struct_t *dst = (struct_t *)out->data;                                                                         \
        for (size_t i = 0; i < n; ++i) {                                                                               \
            base_t aval = lhs[i].value;                                                                                \
            base_t bval = rhs[i].value;                                                                                \
            dst[i].value = (base_t)(expr);                                                                             \
        }                                                                                                              \
        return MARMOT_SUCCESS;                                                                                         \
    }

#define CPU_EW_DEFINE_INT_DIV(op, suffix, struct_t, base_t)                                                            \
    marmot_error_t cpu_##op##_##suffix##_scalar(                                                                       \
        const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out               \
    ) {                                                                                                                \
        (void)device_ctx;                                                                                              \
        const size_t n = marmot_tensor_num_elements(a);                                                                \
        const struct_t *lhs = (const struct_t *)a->data;                                                               \
        const struct_t *rhs = (const struct_t *)b->data;                                                               \
        struct_t *dst = (struct_t *)out->data;                                                                         \
        for (size_t i = 0; i < n; ++i) {                                                                               \
            base_t denom = rhs[i].value;                                                                               \
            if (denom == (base_t)0) {                                                                                  \
                marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Integer division by zero");                           \
                return MARMOT_ERROR_INVALID_ARGUMENT;                                                                  \
            }                                                                                                          \
            dst[i].value = (base_t)(lhs[i].value / denom);                                                             \
        }                                                                                                              \
        return MARMOT_SUCCESS;                                                                                         \
    }

#define CPU_EW_DEFINE_INT_MOD(op, suffix, struct_t, base_t)                                                            \
    marmot_error_t cpu_##op##_##suffix##_scalar(                                                                       \
        const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out               \
    ) {                                                                                                                \
        (void)device_ctx;                                                                                              \
        const size_t n = marmot_tensor_num_elements(a);                                                                \
        const struct_t *lhs = (const struct_t *)a->data;                                                               \
        const struct_t *rhs = (const struct_t *)b->data;                                                               \
        struct_t *dst = (struct_t *)out->data;                                                                         \
        for (size_t i = 0; i < n; ++i) {                                                                               \
            base_t denom = rhs[i].value;                                                                               \
            if (denom == (base_t)0) {                                                                                  \
                marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Integer modulo by zero");                             \
                return MARMOT_ERROR_INVALID_ARGUMENT;                                                                  \
            }                                                                                                          \
            dst[i].value = (base_t)(lhs[i].value % denom);                                                             \
        }                                                                                                              \
        return MARMOT_SUCCESS;                                                                                         \
    }

#define CPU_EW_DEFINE_INT_POW(op, suffix, struct_t, base_t, pow_fn)                                                    \
    marmot_error_t cpu_##op##_##suffix##_scalar(                                                                       \
        const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out               \
    ) {                                                                                                                \
        (void)device_ctx;                                                                                              \
        const size_t n = marmot_tensor_num_elements(a);                                                                \
        const struct_t *lhs = (const struct_t *)a->data;                                                               \
        const struct_t *rhs = (const struct_t *)b->data;                                                               \
        struct_t *dst = (struct_t *)out->data;                                                                         \
        for (size_t i = 0; i < n; ++i) {                                                                               \
            dst[i].value = (base_t)pow_fn(lhs[i].value, rhs[i].value);                                                 \
        }                                                                                                              \
        return MARMOT_SUCCESS;                                                                                         \
    }

#define CPU_EW_DEFINE_INT_BITWISE(op, suffix, struct_t, base_t, expr)                                                  \
    marmot_error_t cpu_##op##_##suffix##_scalar(                                                                       \
        const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out               \
    ) {                                                                                                                \
        (void)device_ctx;                                                                                              \
        const size_t n = marmot_tensor_num_elements(a);                                                                \
        const struct_t *lhs = (const struct_t *)a->data;                                                               \
        const struct_t *rhs = (const struct_t *)b->data;                                                               \
        struct_t *dst = (struct_t *)out->data;                                                                         \
        for (size_t i = 0; i < n; ++i) {                                                                               \
            dst[i].value = (base_t)(expr);                                                                             \
        }                                                                                                              \
        return MARMOT_SUCCESS;                                                                                         \
    }

#define CPU_EW_DEFINE_INT_SHIFT_LEFT(op, suffix, struct_t, base_t)                                                     \
    marmot_error_t cpu_##op##_##suffix##_scalar(                                                                       \
        const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out               \
    ) {                                                                                                                \
        (void)device_ctx;                                                                                              \
        const size_t n = marmot_tensor_num_elements(a);                                                                \
        const struct_t *lhs = (const struct_t *)a->data;                                                               \
        const struct_t *rhs = (const struct_t *)b->data;                                                               \
        struct_t *dst = (struct_t *)out->data;                                                                         \
        const unsigned bits = (unsigned)(sizeof(base_t) * 8U);                                                         \
        for (size_t i = 0; i < n; ++i) {                                                                               \
            unsigned amount = cpu_ew_shift_amount((int64_t)rhs[i].value, bits);                                        \
            if (amount >= bits) {                                                                                      \
                dst[i].value = (base_t)0;                                                                              \
            } else {                                                                                                   \
                dst[i].value = (base_t)(lhs[i].value << amount);                                                       \
            }                                                                                                          \
        }                                                                                                              \
        return MARMOT_SUCCESS;                                                                                         \
    }

#define CPU_EW_DEFINE_SIGNED_SHIFT_RIGHT(op, suffix, struct_t, base_t)                                                 \
    marmot_error_t cpu_##op##_##suffix##_scalar(                                                                       \
        const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out               \
    ) {                                                                                                                \
        (void)device_ctx;                                                                                              \
        const size_t n = marmot_tensor_num_elements(a);                                                                \
        const struct_t *lhs = (const struct_t *)a->data;                                                               \
        const struct_t *rhs = (const struct_t *)b->data;                                                               \
        struct_t *dst = (struct_t *)out->data;                                                                         \
        const unsigned bits = (unsigned)(sizeof(base_t) * 8U);                                                         \
        for (size_t i = 0; i < n; ++i) {                                                                               \
            unsigned amount = cpu_ew_shift_amount((int64_t)rhs[i].value, bits);                                        \
            if (amount >= bits) {                                                                                      \
                base_t v = lhs[i].value;                                                                               \
                dst[i].value = (base_t)((v < 0) ? -1 : 0);                                                             \
            } else {                                                                                                   \
                dst[i].value = (base_t)(lhs[i].value >> amount);                                                       \
            }                                                                                                          \
        }                                                                                                              \
        return MARMOT_SUCCESS;                                                                                         \
    }

#define CPU_EW_DEFINE_SIGNED_SHIFT_RIGHT_LOGICAL(op, suffix, struct_t, base_t, unsigned_t)                             \
    marmot_error_t cpu_##op##_##suffix##_scalar(                                                                       \
        const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out               \
    ) {                                                                                                                \
        (void)device_ctx;                                                                                              \
        const size_t n = marmot_tensor_num_elements(a);                                                                \
        const struct_t *lhs = (const struct_t *)a->data;                                                               \
        const struct_t *rhs = (const struct_t *)b->data;                                                               \
        struct_t *dst = (struct_t *)out->data;                                                                         \
        const unsigned bits = (unsigned)(sizeof(base_t) * 8U);                                                         \
        for (size_t i = 0; i < n; ++i) {                                                                               \
            unsigned amount = cpu_ew_shift_amount((int64_t)rhs[i].value, bits);                                        \
            if (amount >= bits) {                                                                                      \
                dst[i].value = (base_t)0;                                                                              \
            } else {                                                                                                   \
                unsigned_t shifted = ((unsigned_t)lhs[i].value) >> amount;                                             \
                dst[i].value = (base_t)shifted;                                                                        \
            }                                                                                                          \
        }                                                                                                              \
        return MARMOT_SUCCESS;                                                                                         \
    }

#define CPU_EW_DEFINE_UNSIGNED_SHIFT_RIGHT(op, suffix, struct_t, base_t)                                               \
    marmot_error_t cpu_##op##_##suffix##_scalar(                                                                       \
        const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out               \
    ) {                                                                                                                \
        (void)device_ctx;                                                                                              \
        const size_t n = marmot_tensor_num_elements(a);                                                                \
        const struct_t *lhs = (const struct_t *)a->data;                                                               \
        const struct_t *rhs = (const struct_t *)b->data;                                                               \
        struct_t *dst = (struct_t *)out->data;                                                                         \
        const unsigned bits = (unsigned)(sizeof(base_t) * 8U);                                                         \
        for (size_t i = 0; i < n; ++i) {                                                                               \
            unsigned amount = cpu_ew_shift_amount((int64_t)rhs[i].value, bits);                                        \
            if (amount >= bits) {                                                                                      \
                dst[i].value = (base_t)0;                                                                              \
            } else {                                                                                                   \
                dst[i].value = (base_t)(lhs[i].value >> amount);                                                       \
            }                                                                                                          \
        }                                                                                                              \
        return MARMOT_SUCCESS;                                                                                         \
    }

#define CPU_EW_DEFINE_INT_COMPARE(op, suffix, struct_t, base_t, pred)                                                  \
    marmot_error_t cpu_##op##_##suffix##_scalar(                                                                       \
        const void *device_ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out               \
    ) {                                                                                                                \
        (void)device_ctx;                                                                                              \
        const size_t n = marmot_tensor_num_elements(a);                                                                \
        const struct_t *lhs = (const struct_t *)a->data;                                                               \
        const struct_t *rhs = (const struct_t *)b->data;                                                               \
        uint8_t *dst = (uint8_t *)out->data;                                                                           \
        for (size_t i = 0; i < n; ++i) {                                                                               \
            base_t aval = lhs[i].value;                                                                                \
            base_t bval = rhs[i].value;                                                                                \
            dst[i] = (uint8_t)(pred);                                                                                  \
        }                                                                                                              \
        return MARMOT_SUCCESS;                                                                                         \
    }

#define CPU_EW_DEFINE_SIGNED_INT_KERNELS(suffix, struct_t, base_t, unsigned_t, pow_fn)                                 \
    CPU_EW_DEFINE_INT_ARITH(add, suffix, struct_t, base_t, aval + bval)                                                \
    CPU_EW_DEFINE_INT_ARITH(sub, suffix, struct_t, base_t, aval - bval)                                                \
    CPU_EW_DEFINE_INT_ARITH(mul, suffix, struct_t, base_t, aval *bval)                                                 \
    CPU_EW_DEFINE_INT_DIV(div, suffix, struct_t, base_t)                                                               \
    CPU_EW_DEFINE_INT_POW(pow, suffix, struct_t, base_t, pow_fn)                                                       \
    CPU_EW_DEFINE_INT_MOD(mod, suffix, struct_t, base_t)                                                               \
    CPU_EW_DEFINE_INT_ARITH(min, suffix, struct_t, base_t, (aval < bval ? aval : bval))                                \
    CPU_EW_DEFINE_INT_ARITH(max, suffix, struct_t, base_t, (aval > bval ? aval : bval))                                \
    CPU_EW_DEFINE_INT_BITWISE(bitwise_and, suffix, struct_t, base_t, lhs[i].value &rhs[i].value)                       \
    CPU_EW_DEFINE_INT_BITWISE(bitwise_or, suffix, struct_t, base_t, lhs[i].value | rhs[i].value)                       \
    CPU_EW_DEFINE_INT_BITWISE(bitwise_xor, suffix, struct_t, base_t, lhs[i].value ^ rhs[i].value)                      \
    CPU_EW_DEFINE_INT_SHIFT_LEFT(bitwise_shl, suffix, struct_t, base_t)                                                \
    CPU_EW_DEFINE_SIGNED_SHIFT_RIGHT(bitwise_shr, suffix, struct_t, base_t)                                            \
    CPU_EW_DEFINE_SIGNED_SHIFT_RIGHT_LOGICAL(bitwise_shr_logical, suffix, struct_t, base_t, unsigned_t)                \
    CPU_EW_DEFINE_INT_COMPARE(compare_eq, suffix, struct_t, base_t, aval == bval)                                      \
    CPU_EW_DEFINE_INT_COMPARE(compare_ne, suffix, struct_t, base_t, aval != bval)                                      \
    CPU_EW_DEFINE_INT_COMPARE(compare_lt, suffix, struct_t, base_t, aval < bval)                                       \
    CPU_EW_DEFINE_INT_COMPARE(compare_le, suffix, struct_t, base_t, aval <= bval)                                      \
    CPU_EW_DEFINE_INT_COMPARE(compare_gt, suffix, struct_t, base_t, aval > bval)                                       \
    CPU_EW_DEFINE_INT_COMPARE(compare_ge, suffix, struct_t, base_t, aval >= bval)

#define CPU_EW_DEFINE_UNSIGNED_INT_KERNELS(suffix, struct_t, base_t, pow_fn)                                           \
    CPU_EW_DEFINE_INT_ARITH(add, suffix, struct_t, base_t, aval + bval)                                                \
    CPU_EW_DEFINE_INT_ARITH(sub, suffix, struct_t, base_t, aval - bval)                                                \
    CPU_EW_DEFINE_INT_ARITH(mul, suffix, struct_t, base_t, aval *bval)                                                 \
    CPU_EW_DEFINE_INT_DIV(div, suffix, struct_t, base_t)                                                               \
    CPU_EW_DEFINE_INT_POW(pow, suffix, struct_t, base_t, pow_fn)                                                       \
    CPU_EW_DEFINE_INT_MOD(mod, suffix, struct_t, base_t)                                                               \
    CPU_EW_DEFINE_INT_ARITH(min, suffix, struct_t, base_t, (aval < bval ? aval : bval))                                \
    CPU_EW_DEFINE_INT_ARITH(max, suffix, struct_t, base_t, (aval > bval ? aval : bval))                                \
    CPU_EW_DEFINE_INT_BITWISE(bitwise_and, suffix, struct_t, base_t, lhs[i].value &rhs[i].value)                       \
    CPU_EW_DEFINE_INT_BITWISE(bitwise_or, suffix, struct_t, base_t, lhs[i].value | rhs[i].value)                       \
    CPU_EW_DEFINE_INT_BITWISE(bitwise_xor, suffix, struct_t, base_t, lhs[i].value ^ rhs[i].value)                      \
    CPU_EW_DEFINE_INT_SHIFT_LEFT(bitwise_shl, suffix, struct_t, base_t)                                                \
    CPU_EW_DEFINE_UNSIGNED_SHIFT_RIGHT(bitwise_shr, suffix, struct_t, base_t)                                          \
    CPU_EW_DEFINE_UNSIGNED_SHIFT_RIGHT(bitwise_shr_logical, suffix, struct_t, base_t)                                  \
    CPU_EW_DEFINE_INT_COMPARE(compare_eq, suffix, struct_t, base_t, aval == bval)                                      \
    CPU_EW_DEFINE_INT_COMPARE(compare_ne, suffix, struct_t, base_t, aval != bval)                                      \
    CPU_EW_DEFINE_INT_COMPARE(compare_lt, suffix, struct_t, base_t, aval < bval)                                       \
    CPU_EW_DEFINE_INT_COMPARE(compare_le, suffix, struct_t, base_t, aval <= bval)                                      \
    CPU_EW_DEFINE_INT_COMPARE(compare_gt, suffix, struct_t, base_t, aval > bval)                                       \
    CPU_EW_DEFINE_INT_COMPARE(compare_ge, suffix, struct_t, base_t, aval >= bval)

CPU_EW_DEFINE_SIGNED_INT_KERNELS(i8, marmot_int8_t, int8_t, uint8_t, cpu_ew_pow_i8)
CPU_EW_DEFINE_SIGNED_INT_KERNELS(i16, marmot_int16_t, int16_t, uint16_t, cpu_ew_pow_i16)
CPU_EW_DEFINE_SIGNED_INT_KERNELS(i32, marmot_int32_t, int32_t, uint32_t, cpu_ew_pow_i32)
CPU_EW_DEFINE_SIGNED_INT_KERNELS(i64, marmot_int64_t, int64_t, uint64_t, cpu_ew_pow_i64)

CPU_EW_DEFINE_UNSIGNED_INT_KERNELS(u8, marmot_uint8_t, uint8_t, cpu_ew_pow_u8)
CPU_EW_DEFINE_UNSIGNED_INT_KERNELS(u16, marmot_uint16_t, uint16_t, cpu_ew_pow_u16)
CPU_EW_DEFINE_UNSIGNED_INT_KERNELS(u32, marmot_uint32_t, uint32_t, cpu_ew_pow_u32)
CPU_EW_DEFINE_UNSIGNED_INT_KERNELS(u64, marmot_uint64_t, uint64_t, cpu_ew_pow_u64)

#undef CPU_EW_DEFINE_UNSIGNED_INT_KERNELS
#undef CPU_EW_DEFINE_SIGNED_INT_KERNELS
#undef CPU_EW_DEFINE_INT_COMPARE
#undef CPU_EW_DEFINE_UNSIGNED_SHIFT_RIGHT
#undef CPU_EW_DEFINE_SIGNED_SHIFT_RIGHT_LOGICAL
#undef CPU_EW_DEFINE_SIGNED_SHIFT_RIGHT
#undef CPU_EW_DEFINE_INT_SHIFT_LEFT
#undef CPU_EW_DEFINE_INT_BITWISE
#undef CPU_EW_DEFINE_INT_POW
#undef CPU_EW_DEFINE_INT_MOD
#undef CPU_EW_DEFINE_INT_DIV
#undef CPU_EW_DEFINE_INT_ARITH
