#ifndef MARMOT_CORE_TENSOR_UTILS_H
#define MARMOT_CORE_TENSOR_UTILS_H

#include "marmot/tensor.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

bool marmot_tensor_is_contiguous(const marmot_tensor_t *tensor);
bool marmot_tensor_is_row_strided(const marmot_tensor_t *tensor);
bool marmot_tensors_same_shape(const marmot_tensor_t *lhs, const marmot_tensor_t *rhs);
bool marmot_tensor_is_block_quantized_weight(const marmot_tensor_t *tensor);

bool marmot_buffers_overlap(const void *dst, size_t dst_bytes, const void *src, size_t src_bytes);
bool marmot_is_power_of_two_u32(uint32_t value);

#endif
