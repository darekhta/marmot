#include "metal_backend_internal.h"
#include "utils/dtype_ref.h"

#ifdef __APPLE__

static NSString *metal_convert_kernel_name(marmot_dtype_t src_dtype, marmot_dtype_t dst_dtype) {
    if (src_dtype == MARMOT_DTYPE_FLOAT32 && dst_dtype == MARMOT_DTYPE_FLOAT16)
        return @"convert_f32_to_f16";
    if (src_dtype == MARMOT_DTYPE_FLOAT32 && dst_dtype == MARMOT_DTYPE_BFLOAT16)
        return @"convert_f32_to_bf16";
    if (src_dtype == MARMOT_DTYPE_FLOAT16 && dst_dtype == MARMOT_DTYPE_FLOAT32)
        return @"convert_f16_to_f32";
    if (src_dtype == MARMOT_DTYPE_FLOAT16 && dst_dtype == MARMOT_DTYPE_BFLOAT16)
        return @"convert_f16_to_bf16";
    if (src_dtype == MARMOT_DTYPE_BFLOAT16 && dst_dtype == MARMOT_DTYPE_FLOAT32)
        return @"convert_bf16_to_f32";
    if (src_dtype == MARMOT_DTYPE_BFLOAT16 && dst_dtype == MARMOT_DTYPE_FLOAT16)
        return @"convert_bf16_to_f16";
#if MARMOT_ENABLE_FP8
    if (src_dtype == MARMOT_DTYPE_FLOAT32 && dst_dtype == MARMOT_DTYPE_FLOAT8_E4M3)
        return @"convert_f32_to_fp8_e4m3";
    if (src_dtype == MARMOT_DTYPE_FLOAT8_E4M3 && dst_dtype == MARMOT_DTYPE_FLOAT32)
        return @"convert_fp8_e4m3_to_f32";
    if (src_dtype == MARMOT_DTYPE_FLOAT32 && dst_dtype == MARMOT_DTYPE_FLOAT8_E5M2)
        return @"convert_f32_to_fp8_e5m2";
    if (src_dtype == MARMOT_DTYPE_FLOAT8_E5M2 && dst_dtype == MARMOT_DTYPE_FLOAT32)
        return @"convert_fp8_e5m2_to_f32";
#endif
    if (src_dtype == MARMOT_DTYPE_FLOAT32 && dst_dtype == MARMOT_DTYPE_INT64)
        return @"convert_f32_to_i64";
    if (src_dtype == MARMOT_DTYPE_INT64 && dst_dtype == MARMOT_DTYPE_FLOAT32)
        return @"convert_i64_to_f32";
    return nullptr;
}

static marmot_error_t metal_run_convert_kernel(
    metal_context_t *ctx, NSString *kernel_name, marmot_dtype_t dst_dtype, void *dst, marmot_dtype_t src_dtype,
    const void *src, size_t n
) {
    if (kernel_name == nil) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    (void)dst_dtype;
    (void)src_dtype;

    if (n == 0) {
        return MARMOT_SUCCESS;
    }

    id<MTLBuffer> bufferSrc = metal_buffer_lookup(ctx, (void *)src);
    id<MTLBuffer> bufferDst = metal_buffer_lookup(ctx, dst);
    size_t src_bytes = marmot_dtype_size(src_dtype) * n;
    size_t dst_bytes = marmot_dtype_size(dst_dtype) * n;
    bool dst_temp = false;
    if (bufferSrc == nil) {
        bufferSrc = [ctx->device newBufferWithLength:src_bytes options:MTLResourceStorageModeShared];
        if (bufferSrc != nil) {
            memcpy([bufferSrc contents], src, src_bytes);
        }
    }
    if (bufferDst == nil) {
        bufferDst = [ctx->device newBufferWithLength:dst_bytes options:MTLResourceStorageModeShared];
        if (bufferDst != nil) {
            dst_temp = true;
        }
    }
    if (bufferSrc == nil || bufferDst == nil) {
        if (bufferSrc != nil) {
            [bufferSrc release];
        }
        if (bufferDst != nil) {
            [bufferDst release];
        }
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    NSString *kernel_fn = [[NSString alloc] initWithString:kernel_name];
    NSError *error = nil;
    id<MTLFunction> function = [ctx->library newFunctionWithName:kernel_fn];
    if (function == nil) {
        [bufferSrc release];
        [bufferDst release];
        [kernel_fn release];
        marmot_set_error(MARMOT_ERROR_BACKEND_INIT_FAILED, "Missing Metal conversion kernel");
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    id<MTLComputePipelineState> pipeline = [ctx->device newComputePipelineStateWithFunction:function error:&error];
    [function release];
    [kernel_fn release];
    if (pipeline == nil) {
        [bufferSrc release];
        [bufferDst release];
        (void)error;
        marmot_set_error(MARMOT_ERROR_BACKEND_INIT_FAILED, "Failed to create Metal conversion pipeline");
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
    if (encoder == nil) {
        [pipeline release];
        [bufferSrc release];
        [bufferDst release];
        marmot_set_error(MARMOT_ERROR_BACKEND_INIT_FAILED, "Failed to acquire Metal conversion encoder");
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    uint32_t count = (uint32_t)n;
    [encoder setBuffer:bufferSrc offset:0 atIndex:0];
    [encoder setBuffer:bufferDst offset:0 atIndex:1];
    [encoder setBytes:&count length:sizeof(uint32_t) atIndex:2];

    NSUInteger threadGroupSize = metal_threadgroup_size_1d(pipeline, (NSUInteger)n);
    MTLSize grid = MTLSizeMake((NSUInteger)n, 1, 1);
    MTLSize threads = MTLSizeMake(threadGroupSize, 1, 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads];

    metal_command_stream_flush(ctx, true);
    if (dst_temp) {
        memcpy(dst, [bufferDst contents], dst_bytes);
    }

    [pipeline release];
    [bufferSrc release];
    [bufferDst release];
    return MARMOT_SUCCESS;
}

static marmot_error_t metal_convert_try_two_step(
    metal_context_t *ctx, marmot_dtype_t dst_dtype, void *dst, marmot_dtype_t src_dtype, const void *src, size_t n
) {
#if MARMOT_ENABLE_FP8
    marmot_dtype_t intermediate = MARMOT_DTYPE_COUNT;
    bool requires_two_step = false;

    if ((src_dtype == MARMOT_DTYPE_FLOAT16 || src_dtype == MARMOT_DTYPE_BFLOAT16) &&
        (dst_dtype == MARMOT_DTYPE_FLOAT8_E4M3 || dst_dtype == MARMOT_DTYPE_FLOAT8_E5M2)) {
        intermediate = MARMOT_DTYPE_FLOAT32;
        requires_two_step = true;
    } else if ((dst_dtype == MARMOT_DTYPE_FLOAT16 || dst_dtype == MARMOT_DTYPE_BFLOAT16) &&
               (src_dtype == MARMOT_DTYPE_FLOAT8_E4M3 || src_dtype == MARMOT_DTYPE_FLOAT8_E5M2)) {
        intermediate = MARMOT_DTYPE_FLOAT32;
        requires_two_step = true;
    }

    if (!requires_two_step) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    size_t tmp_bytes = marmot_dtype_size(intermediate) * n;
    id<MTLBuffer> tmp_buffer = [ctx->device newBufferWithLength:tmp_bytes options:MTLResourceStorageModeShared];
    if (tmp_buffer == nil) {
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Metal dtype conversion intermediate allocation failed");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }
    void *tmp_ptr = tmp_buffer.contents;

    NSString *first_kernel = metal_convert_kernel_name(src_dtype, intermediate);
    marmot_error_t first = metal_run_convert_kernel(ctx, first_kernel, intermediate, tmp_ptr, src_dtype, src, n);
    if (first == MARMOT_ERROR_NOT_IMPLEMENTED || first_kernel == nil) {
        [tmp_buffer release];
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (first != MARMOT_SUCCESS) {
        [tmp_buffer release];
        return first;
    }

    NSString *second_kernel = metal_convert_kernel_name(intermediate, dst_dtype);
    marmot_error_t second = metal_run_convert_kernel(ctx, second_kernel, dst_dtype, dst, intermediate, tmp_ptr, n);
    [tmp_buffer release];
    if (second == MARMOT_ERROR_NOT_IMPLEMENTED || second_kernel == nil) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    return second;
#else
    (void)ctx;
    (void)dst_dtype;
    (void)dst;
    (void)src_dtype;
    (void)src;
    (void)n;
    return MARMOT_ERROR_NOT_IMPLEMENTED;
#endif
}

static bool metal_convert_buffers_overlap(const void *dst, size_t dst_bytes, const void *src, size_t src_bytes) {
    if (dst == nullptr || src == nullptr || dst_bytes == 0 || src_bytes == 0) {
        return false;
    }
    uintptr_t dst_begin = (uintptr_t)dst;
    uintptr_t src_begin = (uintptr_t)src;
    if (dst_bytes > UINTPTR_MAX - dst_begin || src_bytes > UINTPTR_MAX - src_begin) {
        return true;
    }
    uintptr_t dst_end = dst_begin + dst_bytes;
    uintptr_t src_end = src_begin + src_bytes;
    return dst_begin < src_end && src_begin < dst_end;
}

marmot_error_t metal_convert_dispatch(
    const void *device_ctx, marmot_dtype_t dst_dtype, void *dst, marmot_dtype_t src_dtype, const void *src, size_t n
) {
    metal_context_t *ctx = (metal_context_t *)device_ctx;
    if (dst == nullptr || src == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null pointer in Metal dtype conversion");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (n == 0) {
        return MARMOT_SUCCESS;
    }

    if (dst_dtype >= MARMOT_DTYPE_COUNT || src_dtype >= MARMOT_DTYPE_COUNT) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Unsupported Metal dtype conversion");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    size_t src_stride = marmot_dtype_size(src_dtype);
    size_t dst_stride = marmot_dtype_size(dst_dtype);
    if (src_stride == 0 || dst_stride == 0) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Unsupported Metal dtype conversion");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    size_t src_bytes = src_stride * n;
    size_t dst_bytes = dst_stride * n;
    if (metal_convert_buffers_overlap(dst, dst_bytes, src, src_bytes)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Overlapping buffers in Metal dtype conversion");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (src_dtype == dst_dtype) {
        memcpy(dst, src, src_bytes);
        if (ctx != nullptr) {
            metal_residency_invalidate(ctx, dst);
        }
        return MARMOT_SUCCESS;
    }

    if (ctx == nullptr) {
        marmot_set_error(MARMOT_ERROR_BACKEND_INIT_FAILED, "Metal dtype conversion requires initialized context");
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    id<MTLBuffer> src_buf = metal_buffer_lookup(ctx, (void *)src);
    id<MTLBuffer> dst_buf = metal_buffer_lookup(ctx, dst);
    bool host_only = (src_buf == nil && dst_buf == nil);
    if (src_buf != nil) {
        [src_buf release];
    }
    if (dst_buf != nil) {
        [dst_buf release];
    }

    if (src_dtype == MARMOT_DTYPE_FLOAT64 || dst_dtype == MARMOT_DTYPE_FLOAT64) {
        if (dst_dtype == MARMOT_DTYPE_FLOAT32 && src_dtype == MARMOT_DTYPE_FLOAT64) {
            const double *src_f64 = (const double *)src;
            float *dst_f32 = (float *)dst;
            for (size_t i = 0; i < n; ++i) {
                dst_f32[i] = (float)src_f64[i];
            }
        } else if (dst_dtype == MARMOT_DTYPE_FLOAT16 && src_dtype == MARMOT_DTYPE_FLOAT64) {
            const double *src_f64 = (const double *)src;
            marmot_float16_t *dst_f16 = (marmot_float16_t *)dst;
            for (size_t i = 0; i < n; ++i) {
                dst_f16[i] = marmot_f32_to_f16_ref((float)src_f64[i]);
            }
        } else if (dst_dtype == MARMOT_DTYPE_BFLOAT16 && src_dtype == MARMOT_DTYPE_FLOAT64) {
            const double *src_f64 = (const double *)src;
            marmot_bfloat16_t *dst_bf16 = (marmot_bfloat16_t *)dst;
            for (size_t i = 0; i < n; ++i) {
                dst_bf16[i] = marmot_f32_to_bf16_ref((float)src_f64[i]);
            }
        }
#if MARMOT_ENABLE_FP8
        else if (dst_dtype == MARMOT_DTYPE_FLOAT8_E4M3 && src_dtype == MARMOT_DTYPE_FLOAT64) {
            const double *src_f64 = (const double *)src;
            marmot_float8_e4m3_t *dst_fp8 = (marmot_float8_e4m3_t *)dst;
            for (size_t i = 0; i < n; ++i) {
                dst_fp8[i] = marmot_f32_to_fp8_e4m3_ref((float)src_f64[i]);
            }
        } else if (dst_dtype == MARMOT_DTYPE_FLOAT8_E5M2 && src_dtype == MARMOT_DTYPE_FLOAT64) {
            const double *src_f64 = (const double *)src;
            marmot_float8_e5m2_t *dst_fp8 = (marmot_float8_e5m2_t *)dst;
            for (size_t i = 0; i < n; ++i) {
                dst_fp8[i] = marmot_f32_to_fp8_e5m2_ref((float)src_f64[i]);
            }
        } else if (dst_dtype == MARMOT_DTYPE_FLOAT64 && src_dtype == MARMOT_DTYPE_FLOAT8_E4M3) {
            const marmot_float8_e4m3_t *src_fp8 = (const marmot_float8_e4m3_t *)src;
            double *dst_f64 = (double *)dst;
            for (size_t i = 0; i < n; ++i) {
                dst_f64[i] = (double)marmot_fp8_e4m3_to_f32_ref(src_fp8[i]);
            }
        } else if (dst_dtype == MARMOT_DTYPE_FLOAT64 && src_dtype == MARMOT_DTYPE_FLOAT8_E5M2) {
            const marmot_float8_e5m2_t *src_fp8 = (const marmot_float8_e5m2_t *)src;
            double *dst_f64 = (double *)dst;
            for (size_t i = 0; i < n; ++i) {
                dst_f64[i] = (double)marmot_fp8_e5m2_to_f32_ref(src_fp8[i]);
            }
        }
#endif
        else if (dst_dtype == MARMOT_DTYPE_FLOAT64 && src_dtype == MARMOT_DTYPE_FLOAT32) {
            const float *src_f32 = (const float *)src;
            double *dst_f64 = (double *)dst;
            for (size_t i = 0; i < n; ++i) {
                dst_f64[i] = (double)src_f32[i];
            }
        } else if (dst_dtype == MARMOT_DTYPE_FLOAT64 && src_dtype == MARMOT_DTYPE_FLOAT16) {
            const marmot_float16_t *src_f16 = (const marmot_float16_t *)src;
            double *dst_f64 = (double *)dst;
            for (size_t i = 0; i < n; ++i) {
                dst_f64[i] = (double)marmot_f16_to_f32_ref(src_f16[i]);
            }
        } else if (dst_dtype == MARMOT_DTYPE_FLOAT64 && src_dtype == MARMOT_DTYPE_BFLOAT16) {
            const marmot_bfloat16_t *src_bf16 = (const marmot_bfloat16_t *)src;
            double *dst_f64 = (double *)dst;
            for (size_t i = 0; i < n; ++i) {
                dst_f64[i] = (double)marmot_bf16_to_f32_ref(src_bf16[i]);
            }
        } else if (dst_dtype == MARMOT_DTYPE_INT64 && src_dtype == MARMOT_DTYPE_FLOAT64) {
            const double *src_f64 = (const double *)src;
            marmot_int64_t *dst_i64 = (marmot_int64_t *)dst;
            for (size_t i = 0; i < n; ++i) {
                dst_i64[i].value = (int64_t)src_f64[i];
            }
        } else if (dst_dtype == MARMOT_DTYPE_FLOAT64 && src_dtype == MARMOT_DTYPE_INT64) {
            const marmot_int64_t *src_i64 = (const marmot_int64_t *)src;
            double *dst_f64 = (double *)dst;
            for (size_t i = 0; i < n; ++i) {
                dst_f64[i] = (double)src_i64[i].value;
            }
        } else {
            marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Unsupported Metal dtype conversion");
            return MARMOT_ERROR_UNSUPPORTED_DTYPE;
        }

        if (ctx != nullptr) {
            metal_residency_invalidate(ctx, dst);
        }
        return MARMOT_SUCCESS;
    }

    if (host_only) {
        if (dst_dtype == MARMOT_DTYPE_FLOAT16 && src_dtype == MARMOT_DTYPE_FLOAT32) {
            const float *src_f32 = (const float *)src;
            marmot_float16_t *dst_f16 = (marmot_float16_t *)dst;
            for (size_t i = 0; i < n; ++i) {
                dst_f16[i] = marmot_f32_to_f16_ref(src_f32[i]);
            }
        } else if (dst_dtype == MARMOT_DTYPE_FLOAT32 && src_dtype == MARMOT_DTYPE_FLOAT16) {
            const marmot_float16_t *src_f16 = (const marmot_float16_t *)src;
            float *dst_f32 = (float *)dst;
            for (size_t i = 0; i < n; ++i) {
                dst_f32[i] = marmot_f16_to_f32_ref(src_f16[i]);
            }
        } else if (dst_dtype == MARMOT_DTYPE_BFLOAT16 && src_dtype == MARMOT_DTYPE_FLOAT32) {
            const float *src_f32 = (const float *)src;
            marmot_bfloat16_t *dst_bf16 = (marmot_bfloat16_t *)dst;
            for (size_t i = 0; i < n; ++i) {
                dst_bf16[i] = marmot_f32_to_bf16_ref(src_f32[i]);
            }
        } else if (dst_dtype == MARMOT_DTYPE_FLOAT32 && src_dtype == MARMOT_DTYPE_BFLOAT16) {
            const marmot_bfloat16_t *src_bf16 = (const marmot_bfloat16_t *)src;
            float *dst_f32 = (float *)dst;
            for (size_t i = 0; i < n; ++i) {
                dst_f32[i] = marmot_bf16_to_f32_ref(src_bf16[i]);
            }
        } else if (dst_dtype == MARMOT_DTYPE_FLOAT16 && src_dtype == MARMOT_DTYPE_BFLOAT16) {
            const marmot_bfloat16_t *src_bf16 = (const marmot_bfloat16_t *)src;
            marmot_float16_t *dst_f16 = (marmot_float16_t *)dst;
            for (size_t i = 0; i < n; ++i) {
                dst_f16[i] = marmot_f32_to_f16_ref(marmot_bf16_to_f32_ref(src_bf16[i]));
            }
        } else if (dst_dtype == MARMOT_DTYPE_BFLOAT16 && src_dtype == MARMOT_DTYPE_FLOAT16) {
            const marmot_float16_t *src_f16 = (const marmot_float16_t *)src;
            marmot_bfloat16_t *dst_bf16 = (marmot_bfloat16_t *)dst;
            for (size_t i = 0; i < n; ++i) {
                dst_bf16[i] = marmot_f32_to_bf16_ref(marmot_f16_to_f32_ref(src_f16[i]));
            }
        } else if (dst_dtype == MARMOT_DTYPE_INT64 && src_dtype == MARMOT_DTYPE_FLOAT32) {
            const float *src_f32 = (const float *)src;
            marmot_int64_t *dst_i64 = (marmot_int64_t *)dst;
            for (size_t i = 0; i < n; ++i) {
                dst_i64[i].value = (int64_t)src_f32[i];
            }
        } else if (dst_dtype == MARMOT_DTYPE_FLOAT32 && src_dtype == MARMOT_DTYPE_INT64) {
            const marmot_int64_t *src_i64 = (const marmot_int64_t *)src;
            float *dst_f32 = (float *)dst;
            for (size_t i = 0; i < n; ++i) {
                dst_f32[i] = (float)src_i64[i].value;
            }
        }
#if MARMOT_ENABLE_FP8
        else if (dst_dtype == MARMOT_DTYPE_FLOAT8_E4M3 && src_dtype == MARMOT_DTYPE_FLOAT32) {
            const float *src_f32 = (const float *)src;
            marmot_float8_e4m3_t *dst_fp8 = (marmot_float8_e4m3_t *)dst;
            for (size_t i = 0; i < n; ++i) {
                dst_fp8[i] = marmot_f32_to_fp8_e4m3_ref(src_f32[i]);
            }
        } else if (dst_dtype == MARMOT_DTYPE_FLOAT32 && src_dtype == MARMOT_DTYPE_FLOAT8_E4M3) {
            const marmot_float8_e4m3_t *src_fp8 = (const marmot_float8_e4m3_t *)src;
            float *dst_f32 = (float *)dst;
            for (size_t i = 0; i < n; ++i) {
                dst_f32[i] = marmot_fp8_e4m3_to_f32_ref(src_fp8[i]);
            }
        } else if (dst_dtype == MARMOT_DTYPE_FLOAT8_E5M2 && src_dtype == MARMOT_DTYPE_FLOAT32) {
            const float *src_f32 = (const float *)src;
            marmot_float8_e5m2_t *dst_fp8 = (marmot_float8_e5m2_t *)dst;
            for (size_t i = 0; i < n; ++i) {
                dst_fp8[i] = marmot_f32_to_fp8_e5m2_ref(src_f32[i]);
            }
        } else if (dst_dtype == MARMOT_DTYPE_FLOAT32 && src_dtype == MARMOT_DTYPE_FLOAT8_E5M2) {
            const marmot_float8_e5m2_t *src_fp8 = (const marmot_float8_e5m2_t *)src;
            float *dst_f32 = (float *)dst;
            for (size_t i = 0; i < n; ++i) {
                dst_f32[i] = marmot_fp8_e5m2_to_f32_ref(src_fp8[i]);
            }
        }
#endif
        else {
            marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Unsupported Metal dtype conversion");
            return MARMOT_ERROR_UNSUPPORTED_DTYPE;
        }

        if (ctx != nullptr) {
            metal_residency_invalidate(ctx, dst);
        }
        return MARMOT_SUCCESS;
    }

    if (dst_dtype == MARMOT_DTYPE_FLOAT16 && src_dtype == MARMOT_DTYPE_FLOAT32) {
        const float *src_f32 = (const float *)src;
        marmot_float16_t *dst_f16 = (marmot_float16_t *)dst;
        for (size_t i = 0; i < n; ++i) {
            dst_f16[i] = marmot_f32_to_f16_ref(src_f32[i]);
        }
    } else if (dst_dtype == MARMOT_DTYPE_FLOAT32 && src_dtype == MARMOT_DTYPE_FLOAT16) {
        const marmot_float16_t *src_f16 = (const marmot_float16_t *)src;
        float *dst_f32 = (float *)dst;
        for (size_t i = 0; i < n; ++i) {
            dst_f32[i] = marmot_f16_to_f32_ref(src_f16[i]);
        }
    } else if (dst_dtype == MARMOT_DTYPE_BFLOAT16 && src_dtype == MARMOT_DTYPE_FLOAT32) {
        const float *src_f32 = (const float *)src;
        marmot_bfloat16_t *dst_bf16 = (marmot_bfloat16_t *)dst;
        for (size_t i = 0; i < n; ++i) {
            dst_bf16[i] = marmot_f32_to_bf16_ref(src_f32[i]);
        }
    } else if (dst_dtype == MARMOT_DTYPE_FLOAT32 && src_dtype == MARMOT_DTYPE_BFLOAT16) {
        const marmot_bfloat16_t *src_bf16 = (const marmot_bfloat16_t *)src;
        float *dst_f32 = (float *)dst;
        for (size_t i = 0; i < n; ++i) {
            dst_f32[i] = marmot_bf16_to_f32_ref(src_bf16[i]);
        }
    } else if (dst_dtype == MARMOT_DTYPE_FLOAT16 && src_dtype == MARMOT_DTYPE_BFLOAT16) {
        const marmot_bfloat16_t *src_bf16 = (const marmot_bfloat16_t *)src;
        marmot_float16_t *dst_f16 = (marmot_float16_t *)dst;
        for (size_t i = 0; i < n; ++i) {
            dst_f16[i] = marmot_f32_to_f16_ref(marmot_bf16_to_f32_ref(src_bf16[i]));
        }
    } else if (dst_dtype == MARMOT_DTYPE_BFLOAT16 && src_dtype == MARMOT_DTYPE_FLOAT16) {
        const marmot_float16_t *src_f16 = (const marmot_float16_t *)src;
        marmot_bfloat16_t *dst_bf16 = (marmot_bfloat16_t *)dst;
        for (size_t i = 0; i < n; ++i) {
            dst_bf16[i] = marmot_f32_to_bf16_ref(marmot_f16_to_f32_ref(src_f16[i]));
        }
    } else if (dst_dtype == MARMOT_DTYPE_INT64 && src_dtype == MARMOT_DTYPE_FLOAT32) {
        const float *src_f32 = (const float *)src;
        marmot_int64_t *dst_i64 = (marmot_int64_t *)dst;
        for (size_t i = 0; i < n; ++i) {
            dst_i64[i].value = (int64_t)src_f32[i];
        }
    } else if (dst_dtype == MARMOT_DTYPE_FLOAT32 && src_dtype == MARMOT_DTYPE_INT64) {
        const marmot_int64_t *src_i64 = (const marmot_int64_t *)src;
        float *dst_f32 = (float *)dst;
        for (size_t i = 0; i < n; ++i) {
            dst_f32[i] = (float)src_i64[i].value;
        }
    }
#if MARMOT_ENABLE_FP8
    else if (dst_dtype == MARMOT_DTYPE_FLOAT8_E4M3 && src_dtype == MARMOT_DTYPE_FLOAT32) {
        const float *src_f32 = (const float *)src;
        marmot_float8_e4m3_t *dst_fp8 = (marmot_float8_e4m3_t *)dst;
        for (size_t i = 0; i < n; ++i) {
            dst_fp8[i] = marmot_f32_to_fp8_e4m3_ref(src_f32[i]);
        }
    } else if (dst_dtype == MARMOT_DTYPE_FLOAT32 && src_dtype == MARMOT_DTYPE_FLOAT8_E4M3) {
        const marmot_float8_e4m3_t *src_fp8 = (const marmot_float8_e4m3_t *)src;
        float *dst_f32 = (float *)dst;
        for (size_t i = 0; i < n; ++i) {
            dst_f32[i] = marmot_fp8_e4m3_to_f32_ref(src_fp8[i]);
        }
    } else if (dst_dtype == MARMOT_DTYPE_FLOAT8_E5M2 && src_dtype == MARMOT_DTYPE_FLOAT32) {
        const float *src_f32 = (const float *)src;
        marmot_float8_e5m2_t *dst_fp8 = (marmot_float8_e5m2_t *)dst;
        for (size_t i = 0; i < n; ++i) {
            dst_fp8[i] = marmot_f32_to_fp8_e5m2_ref(src_f32[i]);
        }
    } else if (dst_dtype == MARMOT_DTYPE_FLOAT32 && src_dtype == MARMOT_DTYPE_FLOAT8_E5M2) {
        const marmot_float8_e5m2_t *src_fp8 = (const marmot_float8_e5m2_t *)src;
        float *dst_f32 = (float *)dst;
        for (size_t i = 0; i < n; ++i) {
            dst_f32[i] = marmot_fp8_e5m2_to_f32_ref(src_fp8[i]);
        }
    }
#endif
    else {
        goto gpu_path;
    }

    metal_residency_invalidate(ctx, dst);
    return MARMOT_SUCCESS;

gpu_path:
    marmot_error_t direct = metal_run_convert_kernel(
        ctx, metal_convert_kernel_name(src_dtype, dst_dtype), dst_dtype, dst, src_dtype, src, n
    );
    if (direct == MARMOT_SUCCESS) {
        metal_residency_invalidate(ctx, dst);
        return MARMOT_SUCCESS;
    }

    if (direct != MARMOT_ERROR_NOT_IMPLEMENTED) {
        return direct;
    }

    marmot_error_t two_step = metal_convert_try_two_step(ctx, dst_dtype, dst, src_dtype, src, n);
    if (two_step == MARMOT_SUCCESS) {
        metal_residency_invalidate(ctx, dst);
        return MARMOT_SUCCESS;
    }
    if (two_step != MARMOT_ERROR_NOT_IMPLEMENTED) {
        return two_step;
    }

    marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Unsupported Metal dtype conversion");
    return MARMOT_ERROR_UNSUPPORTED_DTYPE;
}

#endif // __APPLE__
