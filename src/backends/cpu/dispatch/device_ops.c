#include <stdlib.h>

#include <string.h>

#include "cpu_backend_internal.h"
#include "ops/convert/convert_registry.h"

// ===================================================================
// CPU Backend Operations Table
// ===================================================================

static const marmot_device_ops_t cpu_ops = {
    .init = cpu_init,
    .destroy = cpu_destroy,
    .configure = cpu_configure,
    .alloc = cpu_alloc,
    .free = cpu_free,
    .memcpy_to_device = cpu_memcpy_to_device,
    .memcpy_from_device = cpu_memcpy_from_device,
    .synchronize = cpu_synchronize,
    .allocator_usage = cpu_allocator_usage,
    .graph_batch_begin = nullptr,
    .graph_batch_end = nullptr,
    .on_host_ptr_freed = nullptr,
    .on_host_range_freed = nullptr,
};

const marmot_device_ops_t *marmot_get_cpu_ops(void) {
    return &cpu_ops;
}

static bool cpu_convert_buffers_overlap(const void *dst, size_t dst_bytes, const void *src, size_t src_bytes) {
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

marmot_error_t cpu_convert_dispatch(
    const void *device_ctx, marmot_dtype_t dst_dtype, void *dst, marmot_dtype_t src_dtype, const void *src, size_t n
) {
    if (dst == nullptr || src == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null pointer in CPU dtype conversion");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (n == 0) {
        return MARMOT_SUCCESS;
    }

    if (src_dtype >= MARMOT_DTYPE_COUNT || dst_dtype >= MARMOT_DTYPE_COUNT) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Unsupported dtype conversion for CPU backend");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    size_t src_stride = marmot_dtype_size(src_dtype);
    size_t dst_stride = marmot_dtype_size(dst_dtype);
    if (src_stride == 0 || dst_stride == 0) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Unsupported dtype conversion for CPU backend");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    size_t src_bytes = src_stride * n;
    size_t dst_bytes = dst_stride * n;
    if (cpu_convert_buffers_overlap(dst, dst_bytes, src, src_bytes)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Overlapping buffers in CPU dtype conversion");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (src_dtype == dst_dtype) {
        memcpy(dst, src, src_bytes);
        return MARMOT_SUCCESS;
    }

    cpu_context_t *ctx = get_cpu_context(device_ctx);
    const char *impl_name = nullptr;
    cpu_convert_fn fn = cpu_convert_resolve_fn(ctx, dst_dtype, src_dtype, &impl_name);
    if (fn == nullptr) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Unsupported dtype conversion for CPU backend");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    (void)impl_name;
    fn(device_ctx, dst, src, n);
    return MARMOT_SUCCESS;
}
