#ifndef MARMOT_OPS_H
#define MARMOT_OPS_H

#include "ops_types.h"
#include "tensor.h"

// Umbrella header for all Marmot ops; include ops/* headers directly for faster builds.
#include "ops/conversion.h"
#include "ops/elementwise.h"
#include "ops/manipulation.h"
#include "ops/matmul.h"
#include "ops/neural.h"
#include "ops/paged_attention.h"
#include "ops/quantization.h"
#include "ops/reduction.h"
#include "ops/rope.h"
#include "ops/unary.h"

#endif // MARMOT_OPS_H
