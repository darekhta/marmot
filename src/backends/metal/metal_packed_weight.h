#pragma once

#include "metal_backend_internal.h"

#ifdef __APPLE__

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

typedef enum {
    METAL_PACKED_LAYOUT_SOA = 0,
    METAL_PACKED_LAYOUT_TILED_SOA = 1,
} metal_packed_layout_t;

typedef struct {
    size_t tile_cols;
    size_t tile_k;
    size_t tile_stride;
    size_t tile_section;
    size_t tiles_per_row;
    size_t tiles_per_col;
    bool use_vec4;
    size_t element_size;
} metal_packed_layout_config_t;

@interface MarmotMetalPackedWeightRecord : NSObject
@property(nonatomic, retain) id<MTLBuffer> packedBuffer;
@property(nonatomic, assign) size_t packedByteLength;
@property(nonatomic, assign) size_t sourceRows;
@property(nonatomic, assign) size_t sourceCols;
@property(nonatomic, assign) marmot_dtype_t sourceDtype;
@property(nonatomic, assign) uint64_t sourceVersion;
@property(nonatomic, assign) metal_packed_layout_t layout;
@property(nonatomic, assign) metal_packed_layout_config_t config;
@end

marmot_error_t metal_packed_weight_acquire(
    metal_context_t *ctx, const marmot_tensor_t *weight, MarmotMetalPackedWeightRecord **out_record
);
void metal_packed_weight_invalidate(metal_context_t *ctx, const void *ptr);
void metal_packed_weight_clear(metal_context_t *ctx);

#endif // __APPLE__
