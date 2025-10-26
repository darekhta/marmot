#include "cpu_backend_internal.h"

// ==================================================================
// Shared CPU Backend Helper Functions
// ==================================================================

// Execute an F32 kernel on F16/BF16 data by converting to/from F32
marmot_error_t cpu_op_via_f32_conversion(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *output, marmot_dtype_t input_dtype,
    marmot_dtype_t output_dtype, f32_kernel_fn_t f32_kernel, void *extra_args
) {
    size_t total_elements = marmot_tensor_num_elements(input);

    // Allocate aligned F32 buffers for SIMD optimization
    float *x_f32 = (float *)marmot_aligned_alloc(64, total_elements * sizeof(float));
    float *out_f32 = (float *)marmot_aligned_alloc(64, total_elements * sizeof(float));

    if (x_f32 == nullptr || out_f32 == nullptr) {
        free(x_f32);
        free(out_f32);
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to allocate F32 conversion buffers");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    // Convert input to F32
    if (input_dtype == MARMOT_DTYPE_FLOAT16) {
        cpu_convert_f16_to_f32(device_ctx, x_f32, (const marmot_float16_t *)input->data, total_elements);
    } else if (input_dtype == MARMOT_DTYPE_BFLOAT16) {
        cpu_convert_bf16_to_f32(device_ctx, x_f32, (const marmot_bfloat16_t *)input->data, total_elements);
    } else if (input_dtype == MARMOT_DTYPE_FLOAT32) {
        memcpy(x_f32, input->data, total_elements * sizeof(float));
    } else {
        free(x_f32);
        free(out_f32);
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Unsupported input dtype for via-F32 conversion");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    // Execute F32 kernel
    marmot_error_t err = f32_kernel(device_ctx, x_f32, out_f32, total_elements, extra_args);
    if (err != MARMOT_SUCCESS) {
        free(x_f32);
        free(out_f32);
        return err;
    }

    // Convert output from F32
    if (output_dtype == MARMOT_DTYPE_FLOAT16) {
        cpu_convert_f32_to_f16(device_ctx, (marmot_float16_t *)output->data, out_f32, total_elements);
    } else if (output_dtype == MARMOT_DTYPE_BFLOAT16) {
        cpu_convert_f32_to_bf16(device_ctx, (marmot_bfloat16_t *)output->data, out_f32, total_elements);
    } else if (output_dtype == MARMOT_DTYPE_FLOAT32) {
        memcpy(output->data, out_f32, total_elements * sizeof(float));
    } else {
        free(x_f32);
        free(out_f32);
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Unsupported output dtype for via-F32 conversion");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    free(x_f32);
    free(out_f32);
    return MARMOT_SUCCESS;
}
