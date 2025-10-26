#include "internal/metal_matmul_internal.h"
#include "internal/metal_matmul_qkv_shared.h"
#include "metal_packed_weight.h"

#ifdef __APPLE__

typedef struct {
    uint32_t N;
    uint32_t K;
    uint32_t M;
    uint32_t has_bias;
    uint32_t has_residual;
    uint32_t activation;
    uint32_t use_packed_weights;
    uint32_t packed_tile_cols;
    uint32_t packed_tile_k;
    uint32_t packed_tiles_per_row;
    uint32_t packed_tiles_per_col;
    uint32_t packed_tile_stride;
    uint32_t packed_tile_section;
    uint32_t packed_use_vec4;
    uint32_t has_bias_q;
    uint32_t has_bias_k;
    uint32_t has_bias_v;
    uint32_t rope_enabled;
    uint32_t rope_apply_q;
    uint32_t rope_apply_k;
    uint32_t rope_head_dim;
    float rope_attn_scale;
    metal_matmul_activation_params_t params;
} metal_matmul_qkv_uniforms_t;

typedef struct {
    uint32_t rows;
    uint32_t dim;
    uint32_t head_dim;
    uint32_t apply_q;
    uint32_t apply_k;
    uint32_t rope_type;
    float attn_scale;
} metal_matmul_qkv_rope_uniforms_t;

static const char *metal_matmul_qkv_rope_kernel_name(marmot_dtype_t dtype) {
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT32:
        return "rope_qk_f32";
    case MARMOT_DTYPE_FLOAT16:
        return "rope_qk_f16";
    case MARMOT_DTYPE_BFLOAT16:
        return "rope_qk_bf16";
    default:
        return nullptr;
    }
}

marmot_error_t metal_matmul_qkv_run_kernel(
    metal_context_t *ctx, const marmot_matmul_qkv_desc_t *desc, const metal_matmul_qkv_dims_t *dims,
    const char *kernel_name
) {
    if (ctx == nullptr || desc == nullptr || dims == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    const marmot_rope_params_t *rope = desc->rope_params;
    const bool wants_rope = rope != nullptr && (rope->apply_to_q || rope->apply_to_k);
    const bool rope_inline = wants_rope && rope->rope_type == MARMOT_ROPE_TYPE_NORM;
    const uint32_t rope_head_dim = metal_matmul_qkv_resolve_head_dim(dims->M, rope);
    const marmot_tensor_t *input = desc->input;
    const marmot_tensor_t *weight = desc->fused.weight;
    const marmot_tensor_t *bias = desc->fused.bias;
    if (kernel_name == nullptr) {
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    const size_t bytesInput = marmot_tensor_size_bytes(input);
    const size_t bytesWeight = marmot_tensor_size_bytes(weight);
    const size_t bytesBias = bias != nullptr ? marmot_tensor_size_bytes(bias) : 0;
    const size_t bytesOutQ = marmot_tensor_size_bytes(desc->out_q);
    const size_t bytesOutK = marmot_tensor_size_bytes(desc->out_k);
    const size_t bytesOutV = marmot_tensor_size_bytes(desc->out_v);

    id<MTLBuffer> bufferInput = metal_buffer_acquire(ctx, input->data, bytesInput);
    id<MTLBuffer> bufferWeight = metal_buffer_acquire(ctx, weight->data, bytesWeight);
    id<MTLBuffer> bufferBias = bias != nullptr ? metal_buffer_acquire(ctx, bias->data, bytesBias) : nil;

    bool q_private = false;
    bool k_private = false;
    bool v_private = false;
    bool q_new = false;
    bool k_new = false;
    bool v_new = false;
    id<MTLBuffer> bufferQ = metal_residency_acquire_compute(ctx, desc->out_q, desc->out_q->dtype, &q_new);
    id<MTLBuffer> bufferK = metal_residency_acquire_compute(ctx, desc->out_k, desc->out_k->dtype, &k_new);
    id<MTLBuffer> bufferV = metal_residency_acquire_compute(ctx, desc->out_v, desc->out_v->dtype, &v_new);
    if (bufferQ == nil) {
        bufferQ = metal_buffer_acquire(ctx, desc->out_q->data, bytesOutQ);
    } else {
        q_private = true;
    }
    if (bufferK == nil) {
        bufferK = metal_buffer_acquire(ctx, desc->out_k->data, bytesOutK);
    } else {
        k_private = true;
    }
    if (bufferV == nil) {
        bufferV = metal_buffer_acquire(ctx, desc->out_v->data, bytesOutV);
    } else {
        v_private = true;
    }
    if (bufferInput == nil || bufferWeight == nil || bufferQ == nil || bufferK == nil || bufferV == nil ||
        (bias != nullptr && bufferBias == nil)) {
        if (bufferInput != nil)
            [bufferInput release];
        if (bufferWeight != nil)
            [bufferWeight release];
        if (bufferBias != nil)
            [bufferBias release];
        if (bufferQ != nil)
            [bufferQ release];
        if (bufferK != nil)
            [bufferK release];
        if (bufferV != nil)
            [bufferV release];
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    bool use_packed_weight = false;
    MarmotMetalPackedWeightRecord *packed_record = nil;
    id<MTLBuffer> bufferWeightPacked = nil;
    bool dtype_can_pack = weight->dtype == MARMOT_DTYPE_FLOAT32 || weight->dtype == MARMOT_DTYPE_FLOAT16 ||
        weight->dtype == MARMOT_DTYPE_BFLOAT16;
    if (dtype_can_pack && weight->quant_kind == MARMOT_QUANT_KIND_GENERIC) {
        MarmotMetalPackedWeightRecord *record = nil;
        marmot_error_t pack_status = metal_packed_weight_acquire(ctx, weight, &record);
        if (pack_status == MARMOT_SUCCESS && record != nil) {
            bufferWeightPacked = [record.packedBuffer retain];
            if (bufferWeightPacked != nil) {
                use_packed_weight = true;
                packed_record = record;
            } else {
                [record release];
            }
        } else {
            if (record != nil) {
                [record release];
            }
            if (pack_status != MARMOT_SUCCESS && pack_status != MARMOT_ERROR_NOT_IMPLEMENTED) {
                [bufferInput release];
                [bufferWeight release];
                if (bufferBias != nil)
                    [bufferBias release];
                [bufferQ release];
                [bufferK release];
                [bufferV release];
                return pack_status;
            }
        }
    }
    if (use_packed_weight && bufferWeight != nil) {
        [bufferWeight release];
        bufferWeight = nil;
    }

    metal_matmul_qkv_epilogue_config_t ep_cfg = {};
    marmot_error_t prep_status =
        metal_matmul_qkv_prepare_epilogue(desc->out_q, desc->epilogue, nullptr, rope, true, true, true, true, &ep_cfg);
    if (prep_status != MARMOT_SUCCESS) {
        if (bufferWeightPacked != nil)
            [bufferWeightPacked release];
        if (packed_record != nil)
            [packed_record release];
        [bufferInput release];
        [bufferWeight release];
        if (bufferBias != nil)
            [bufferBias release];
        [bufferQ release];
        [bufferK release];
        [bufferV release];
        return prep_status;
    }

    const bool ep_has_effect = (ep_cfg.ep != nullptr);
    const marmot_matmul_epilogue_t *ep_to_apply = ep_cfg.ep;
    size_t ep_feature_dim = ep_cfg.feature_dim;
    bool ep_bias_scalar = ep_cfg.bias_scalar;
    bool ep_inline = false;
    bool ep_inline_has_residual = false;
    uint32_t inline_activation = (uint32_t)MARMOT_DEVICE_UNARY_IDENTITY;
    metal_matmul_activation_params_t inline_params = {.alpha = 0.0f, .beta = 0.0f, .gamma = 0.0f, .delta = 0.0f};
    id<MTLBuffer> bufferResidual = nil;
    id<MTLBuffer> bufferPositions = nil;
    id<MTLBuffer> bufferFreqs = nil;
    (void)ep_has_effect;

    id<MTLComputePipelineState> pipeline = metal_pipeline_get(ctx, kernel_name);
    if (pipeline == nil) {
        [bufferInput release];
        if (bufferWeightPacked != nil)
            [bufferWeightPacked release];
        if (packed_record != nil)
            [packed_record release];
        [bufferWeight release];
        if (bufferBias != nil)
            [bufferBias release];
        [bufferQ release];
        [bufferK release];
        [bufferV release];
        [bufferResidual release];
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
    if (encoder == nil) {
        [pipeline release];
        [bufferInput release];
        if (bufferWeightPacked != nil)
            [bufferWeightPacked release];
        if (packed_record != nil)
            [packed_record release];
        [bufferWeight release];
        if (bufferBias != nil)
            [bufferBias release];
        [bufferQ release];
        [bufferK release];
        [bufferV release];
        [bufferResidual release];
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    float rope_attn_scale = 1.0f;
    if (rope_inline) {
        bufferPositions = metal_matmul_create_positions_buffer(ctx, rope->positions, dims->N);
        if (bufferPositions == nil) {
            [pipeline release];
            [bufferInput release];
            if (bufferWeightPacked != nil)
                [bufferWeightPacked release];
            if (packed_record != nil)
                [packed_record release];
            [bufferWeight release];
            if (bufferBias != nil)
                [bufferBias release];
            [bufferQ release];
            [bufferK release];
            [bufferV release];
            [bufferResidual release];
            metal_command_stream_discard(ctx);
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        bufferFreqs = metal_matmul_prepare_freq_buffer(ctx, rope_head_dim, rope, &rope_attn_scale);
        if (bufferFreqs == nil) {
            [bufferPositions release];
            [pipeline release];
            [bufferInput release];
            if (bufferWeightPacked != nil)
                [bufferWeightPacked release];
            if (packed_record != nil)
                [packed_record release];
            [bufferWeight release];
            if (bufferBias != nil)
                [bufferBias release];
            [bufferQ release];
            [bufferK release];
            [bufferV release];
            [bufferResidual release];
            metal_command_stream_discard(ctx);
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
    }

    [encoder setBuffer:bufferInput offset:0 atIndex:0];
    [encoder setBuffer:bufferWeight offset:0 atIndex:1];
    if (bias != nullptr && bufferBias != nil) {
        [encoder setBuffer:bufferBias offset:0 atIndex:2];
    } else {
        [encoder setBuffer:metal_get_dummy_buffer(ctx) offset:0 atIndex:2];
    }
    if (ep_inline_has_residual && bufferResidual != nil) {
        [encoder setBuffer:bufferResidual offset:0 atIndex:3];
    } else {
        [encoder setBuffer:metal_get_dummy_buffer(ctx) offset:0 atIndex:3];
    }
    [encoder setBuffer:bufferQ offset:0 atIndex:4];
    [encoder setBuffer:bufferK offset:0 atIndex:5];
    [encoder setBuffer:bufferV offset:0 atIndex:6];
    if (rope_inline && bufferPositions != nil && bufferFreqs != nil) {
        [encoder setBuffer:bufferPositions offset:0 atIndex:7];
        [encoder setBuffer:bufferFreqs offset:0 atIndex:8];
    } else {
        [encoder setBuffer:metal_get_dummy_buffer(ctx) offset:0 atIndex:7];
        [encoder setBuffer:metal_get_dummy_buffer(ctx) offset:0 atIndex:8];
    }

    metal_matmul_qkv_uniforms_t uniforms = {
        .N = (uint32_t)dims->N,
        .K = (uint32_t)dims->K,
        .M = (uint32_t)dims->M,
        .has_bias = bias != nullptr ? 1u : 0u,
        .has_residual = ep_inline_has_residual ? 1u : 0u,
        .activation = inline_activation,
        .use_packed_weights = use_packed_weight ? 1u : 0u,
        .packed_tile_cols = use_packed_weight && packed_record != nil ? (uint32_t)packed_record.config.tile_cols : 0u,
        .packed_tile_k = use_packed_weight && packed_record != nil ? (uint32_t)packed_record.config.tile_k : 0u,
        .packed_tiles_per_row =
            use_packed_weight && packed_record != nil ? (uint32_t)packed_record.config.tiles_per_row : 0u,
        .packed_tiles_per_col =
            use_packed_weight && packed_record != nil ? (uint32_t)packed_record.config.tiles_per_col : 0u,
        .packed_tile_stride =
            use_packed_weight && packed_record != nil ? (uint32_t)packed_record.config.tile_stride : 0u,
        .packed_tile_section =
            use_packed_weight && packed_record != nil ? (uint32_t)packed_record.config.tile_section : 0u,
        .packed_use_vec4 = use_packed_weight && packed_record != nil && packed_record.config.use_vec4 ? 1u : 0u,
        .has_bias_q = bias != nullptr ? 1u : 0u,
        .has_bias_k = bias != nullptr ? 1u : 0u,
        .has_bias_v = bias != nullptr ? 1u : 0u,
        .rope_enabled = rope_inline && bufferPositions != nil && bufferFreqs != nil ? 1u : 0u,
        .rope_apply_q = rope_inline && rope->apply_to_q ? 1u : 0u,
        .rope_apply_k = rope_inline && rope->apply_to_k ? 1u : 0u,
        .rope_head_dim = rope_head_dim,
        .rope_attn_scale = rope_attn_scale,
        .params = inline_params,
    };
    [encoder setBytes:&uniforms length:sizeof(uniforms) atIndex:9];

    metal_profiling_set_label(ctx, "matmul_qkv");
    metal_profiling_begin(ctx);
    const NSUInteger tileM = 32;
    const NSUInteger tileN = 32;
    NSUInteger groupsX = (NSUInteger)((dims->M + tileN - 1) / tileN);
    NSUInteger groupsY = (NSUInteger)((dims->N + tileM - 1) / tileM);
    if (groupsX == 0) {
        groupsX = 1;
    }
    if (groupsY == 0) {
        groupsY = 1;
    }
    MTLSize threadgroups = MTLSizeMake(groupsX, groupsY, 1);
    MTLSize threads = MTLSizeMake(tileN, tileM, 1);
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads];
    metal_profiling_end(ctx);
    metal_command_stream_flush(ctx, false);

    marmot_error_t ep_status = MARMOT_SUCCESS;
    if (!ep_inline && ep_to_apply != nullptr) {
        size_t total_elements = dims->N * dims->M;
        ep_status = metal_matmul_apply_epilogue(
            ctx, desc->out_q, bufferQ, total_elements, ep_feature_dim, ep_bias_scalar, ep_to_apply
        );
        if (ep_status == MARMOT_SUCCESS) {
            ep_status = metal_matmul_apply_epilogue(
                ctx, desc->out_k, bufferK, total_elements, ep_feature_dim, ep_bias_scalar, ep_to_apply
            );
        }
        if (ep_status == MARMOT_SUCCESS) {
            ep_status = metal_matmul_apply_epilogue(
                ctx, desc->out_v, bufferV, total_elements, ep_feature_dim, ep_bias_scalar, ep_to_apply
            );
        }
        if (ep_status != MARMOT_SUCCESS) {
            [pipeline release];
            [bufferInput release];
            if (bufferWeightPacked != nil)
                [bufferWeightPacked release];
            if (packed_record != nil)
                [packed_record release];
            [bufferWeight release];
            if (bufferBias != nil)
                [bufferBias release];
            [bufferQ release];
            [bufferK release];
            [bufferV release];
            [bufferResidual release];
            if (bufferPositions != nil)
                [bufferPositions release];
            if (bufferFreqs != nil)
                [bufferFreqs release];
            metal_command_stream_discard(ctx);
            return ep_status;
        }
        metal_command_stream_flush(ctx, false);
    }

    if (bufferWeightPacked != nil)
        [bufferWeightPacked release];
    if (packed_record != nil)
        [packed_record release];
    [pipeline release];
    [bufferInput release];
    [bufferWeight release];
    if (bufferBias != nil)
        [bufferBias release];
    [bufferQ release];
    [bufferK release];
    [bufferV release];
    [bufferResidual release];
    if (bufferPositions != nil)
        [bufferPositions release];
    if (bufferFreqs != nil)
        [bufferFreqs release];

    if (q_private) {
        metal_residency_mark_dirty(ctx, desc->out_q, desc->out_q->dtype);
    }
    if (k_private) {
        metal_residency_mark_dirty(ctx, desc->out_k, desc->out_k->dtype);
    }
    if (v_private) {
        metal_residency_mark_dirty(ctx, desc->out_v, desc->out_v->dtype);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t metal_matmul_qkv_apply_rope_gpu(
    metal_context_t *ctx, const marmot_matmul_qkv_desc_t *desc, const metal_matmul_qkv_dims_t *dims
) {
    if (ctx == nullptr || desc == nullptr || dims == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    const marmot_rope_params_t *rope = desc->rope_params;
    if (rope == nullptr || (!rope->apply_to_q && !rope->apply_to_k)) {
        return MARMOT_SUCCESS;
    }
    if (dims->M == 0 || dims->N == 0) {
        return MARMOT_SUCCESS;
    }
    const uint32_t rope_head_dim = metal_matmul_qkv_resolve_head_dim(dims->M, rope);
    const marmot_tensor_t *positions = rope->positions;
    if (positions == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    id<MTLBuffer> bufferPositions = metal_matmul_create_positions_buffer(ctx, positions, dims->N);
    if (bufferPositions == nil) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    float rope_attn_scale = 1.0f;
    id<MTLBuffer> bufferFreqs = metal_matmul_prepare_freq_buffer(ctx, rope_head_dim, rope, &rope_attn_scale);
    if (bufferFreqs == nil) {
        [bufferPositions release];
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    bool q_new = false;
    bool k_new = false;
    id<MTLBuffer> bufferQ = metal_residency_acquire_compute(ctx, desc->out_q, desc->out_q->dtype, &q_new);
    if (bufferQ == nil) {
        bufferQ = metal_buffer_acquire(ctx, desc->out_q->data, marmot_tensor_size_bytes(desc->out_q));
    }
    id<MTLBuffer> bufferK = metal_residency_acquire_compute(ctx, desc->out_k, desc->out_k->dtype, &k_new);
    if (bufferK == nil) {
        bufferK = metal_buffer_acquire(ctx, desc->out_k->data, marmot_tensor_size_bytes(desc->out_k));
    }
    if (bufferQ == nil || bufferK == nil) {
        if (bufferQ != nil)
            [bufferQ release];
        if (bufferK != nil)
            [bufferK release];
        [bufferPositions release];
        [bufferFreqs release];
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    const char *rope_kernel = metal_matmul_qkv_rope_kernel_name(desc->out_q->dtype);
    if (rope_kernel == nullptr) {
        [bufferQ release];
        [bufferK release];
        [bufferPositions release];
        [bufferFreqs release];
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    id<MTLComputePipelineState> pipeline = metal_pipeline_get(ctx, rope_kernel);
    if (pipeline == nil) {
        [bufferQ release];
        [bufferK release];
        [bufferPositions release];
        [bufferFreqs release];
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
    if (encoder == nil) {
        [pipeline release];
        [bufferQ release];
        [bufferK release];
        [bufferPositions release];
        [bufferFreqs release];
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    const size_t pairs = dims->M / 2;
    const size_t total = dims->N * pairs;
    if (total == 0) {
        [pipeline release];
        [bufferQ release];
        [bufferK release];
        [bufferPositions release];
        [bufferFreqs release];
        return MARMOT_SUCCESS;
    }

    [encoder setBuffer:bufferQ offset:0 atIndex:0];
    [encoder setBuffer:bufferK offset:0 atIndex:1];
    [encoder setBuffer:bufferPositions offset:0 atIndex:2];
    [encoder setBuffer:bufferFreqs offset:0 atIndex:3];

    metal_matmul_qkv_rope_uniforms_t rope_uniforms = {
        .rows = (uint32_t)dims->N,
        .dim = (uint32_t)dims->M,
        .head_dim = rope_head_dim,
        .apply_q = rope->apply_to_q ? 1u : 0u,
        .apply_k = rope->apply_to_k ? 1u : 0u,
        .rope_type = (uint32_t)rope->rope_type,
        .attn_scale = rope_attn_scale,
    };
    [encoder setBytes:&rope_uniforms length:sizeof(rope_uniforms) atIndex:4];

    metal_profiling_set_label(ctx, "rope_qk");
    metal_profiling_begin(ctx);
    NSUInteger threads_per_group = pipeline.maxTotalThreadsPerThreadgroup;
    if (threads_per_group == 0) {
        threads_per_group = 64;
    }
    if (threads_per_group > total) {
        threads_per_group = (NSUInteger)total;
    }
    MTLSize threads = MTLSizeMake(threads_per_group, 1, 1);
    NSUInteger groups = (NSUInteger)((total + threads.width - 1) / threads.width);
    if (groups == 0) {
        groups = 1;
    }
    MTLSize threadgroups = MTLSizeMake(groups, 1, 1);
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads];
    metal_profiling_end(ctx);
    metal_command_stream_flush(ctx, false);

    [pipeline release];
    [bufferQ release];
    [bufferK release];
    [bufferPositions release];
    [bufferFreqs release];

    if (q_new) {
        metal_residency_mark_dirty(ctx, desc->out_q, desc->out_q->dtype);
    }
    if (k_new) {
        metal_residency_mark_dirty(ctx, desc->out_k, desc->out_k->dtype);
    }
    return MARMOT_SUCCESS;
}

#endif // __APPLE__
