#ifndef CPU_MATMUL_EPILOGUE_H
#define CPU_MATMUL_EPILOGUE_H

#include "marmot/device.h"
#include "marmot/tensor.h"

marmot_error_t
cpu_matmul_apply_epilogue(const void *device_ctx, marmot_tensor_t *out, const marmot_matmul_epilogue_t *epilogue);

#endif // CPU_MATMUL_EPILOGUE_H
