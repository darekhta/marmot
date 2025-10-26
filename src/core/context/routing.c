#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <strings.h>

#include "context_internal.h"
#include "core/dispatch/kernel_query.h"

static const char *marmot_routing_env(void) {
    return getenv("MARMOT_ROUTING");
}

marmot_routing_policy_t marmot_routing_policy_from_env(void) {
    const char *value = marmot_routing_env();
    if (value == nullptr || *value == '\0') {
        return MARMOT_ROUTING_AUTO;
    }

    if (strcasecmp(value, "cpu") == 0 || strcasecmp(value, "always_cpu") == 0) {
        return MARMOT_ROUTING_ALWAYS_CPU;
    }
    if (strcasecmp(value, "gpu") == 0 || strcasecmp(value, "always_gpu") == 0) {
        return MARMOT_ROUTING_ALWAYS_GPU;
    }
    if (strcasecmp(value, "auto") == 0) {
        return MARMOT_ROUTING_AUTO;
    }

    fprintf(stderr, "[marmot] warning: unrecognised MARMOT_ROUTING value '%s', defaulting to AUTO\n", value);
    return MARMOT_ROUTING_AUTO;
}

static const char *marmot_quant_activation_env(void) {
    return getenv("MARMOT_QUANT_ACT");
}

marmot_quant_activation_mode_t marmot_quant_activation_mode_from_env(void) {
    const char *value = marmot_quant_activation_env();
    if (value == nullptr || *value == '\0') {
        return MARMOT_QUANT_ACTIVATION_AUTO;
    }

    if (strcasecmp(value, "direct") == 0 || strcasecmp(value, "fp") == 0) {
        return MARMOT_QUANT_ACTIVATION_FORCE_DIRECT;
    }
    if (strcasecmp(value, "auto") == 0) {
        return MARMOT_QUANT_ACTIVATION_AUTO;
    }
    if (strcasecmp(value, "packed") == 0 || strcasecmp(value, "q8") == 0) {
        fprintf(
            stderr, "[marmot] warning: MARMOT_QUANT_ACT='%s' is no longer supported; use 'auto' or 'direct' instead\n",
            value
        );
        return MARMOT_QUANT_ACTIVATION_AUTO;
    }

    fprintf(stderr, "[marmot] warning: unrecognised MARMOT_QUANT_ACT value '%s', defaulting to AUTO\n", value);
    return MARMOT_QUANT_ACTIVATION_AUTO;
}

void marmot_context_apply_default_policy(marmot_context_t *ctx) {
    if (ctx == nullptr) {
        return;
    }

    marmot_backend_preferences_t prefs = {0};
    if (marmot_backend_get_default_preferences(ctx->backend_type, &ctx->device_caps, &prefs)) {
        ctx->policy = prefs.policy;
        ctx->routing_policy = prefs.routing_policy;
    } else {
        marmot_dtype_t quant_dtype = MARMOT_DTYPE_FLOAT32;
        if (ctx->backend_type == MARMOT_BACKEND_METAL && ctx->device_caps.has_fp16_compute) {
            quant_dtype = MARMOT_DTYPE_FLOAT16;
        }
        ctx->policy = (marmot_context_policy_t){
            .embedding_quant_output_dtype = quant_dtype,
            .quant_activation_mode = MARMOT_QUANT_ACTIVATION_AUTO,
            .variant_flags_mask = UINT32_MAX,
            .embedding_prefer_gpu_private = (ctx->backend_type == MARMOT_BACKEND_METAL),
            .embedding_allow_quant_decode_on_the_fly = true,
            .matmul_requires_temp_tensors = (ctx->backend_type == MARMOT_BACKEND_METAL),
            .matmul_prefer_packed_weights = false,
        };
        ctx->routing_policy =
            (ctx->backend_type == MARMOT_BACKEND_CPU) ? MARMOT_ROUTING_ALWAYS_CPU : MARMOT_ROUTING_AUTO;
    }

    const char *quant_env = marmot_quant_activation_env();
    if (quant_env != nullptr && *quant_env != '\0') {
        ctx->policy.quant_activation_mode = marmot_quant_activation_mode_from_env();
    }

    if (ctx->backend_type == MARMOT_BACKEND_CPU) {
        ctx->routing_policy = MARMOT_ROUTING_ALWAYS_CPU;
        return;
    }

    const char *routing_env = marmot_routing_env();
    if (routing_env != nullptr && *routing_env != '\0') {
        ctx->routing_policy = marmot_routing_policy_from_env();
    }
}
