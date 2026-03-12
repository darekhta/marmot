#include "marmot/dispatch.h"
#include "marmot/ops/conversion.h"
#include "marmot/ops/matmul.h"
#include "marmot/ops/neural.h"
#include "marmot/tensor.h"

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>

#include <float.h>
#include <inttypes.h>
#include <string.h>
#include <time.h>

#include "core/tensor/tensor_utils.h"
#include "cpu_backend_internal.h"
#include "ops/matmul/quantized/matmul_quant_internal.h"
#include "utils/dtype_ref.h"

static const float kCpuMoeRsqrt2 = 0.7071067811865475f;
static const size_t kCpuMoeDecodeMaxTokens = 4;
static const size_t kCpuMoeDecodeMaxRoutes = 64;

static bool cpu_moe_value_dtype_supported(marmot_dtype_t dtype) {
    return dtype == MARMOT_DTYPE_FLOAT32 || dtype == MARMOT_DTYPE_FLOAT16;
}

static float cpu_moe_load_value(const void *data, marmot_dtype_t dtype, size_t index) {
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT32:
        return ((const float *)data)[index];
    case MARMOT_DTYPE_FLOAT16:
        return marmot_f16_to_f32_ref(((const marmot_float16_t *)data)[index]);
    default:
        return 0.0f;
    }
}

static void cpu_moe_store_value(void *data, marmot_dtype_t dtype, size_t index, float value) {
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT32:
        ((float *)data)[index] = value;
        break;
    case MARMOT_DTYPE_FLOAT16:
        ((marmot_float16_t *)data)[index] = marmot_f32_to_f16_ref(value);
        break;
    default:
        break;
    }
}

static marmot_tensor_t *cpu_moe_create_f32_tensor_like(const marmot_context_t *ctx, const marmot_tensor_t *src) {
    return marmot_tensor_create(ctx, src->shape.shape, src->shape.ndim, MARMOT_DTYPE_FLOAT32);
}

static marmot_error_t
cpu_moe_copy_tensor_to_f32(const marmot_context_t *ctx, const marmot_tensor_t *src, marmot_tensor_t *dst) {
    if (ctx == nullptr || src == nullptr || dst == nullptr || dst->dtype != MARMOT_DTYPE_FLOAT32) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "MoE FP32 conversion requires valid tensors");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const void *src_data = marmot_tensor_data(ctx, (marmot_tensor_t *)src);
    float *dst_data = marmot_tensor_data_f32_mut(ctx, dst);
    if (src_data == nullptr || dst_data == nullptr) {
        return marmot_get_last_error();
    }

    return marmot_convert(ctx, MARMOT_DTYPE_FLOAT32, dst_data, src->dtype, src_data, marmot_tensor_numel(src));
}

static marmot_error_t
cpu_moe_copy_tensor_from_f32(const marmot_context_t *ctx, marmot_tensor_t *dst, const marmot_tensor_t *src) {
    if (ctx == nullptr || dst == nullptr || src == nullptr || src->dtype != MARMOT_DTYPE_FLOAT32) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "MoE output conversion requires valid tensors");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const float *src_data = marmot_tensor_data_f32(ctx, (marmot_tensor_t *)src);
    void *dst_data = marmot_tensor_data_mut(ctx, dst);
    if (src_data == nullptr || dst_data == nullptr) {
        return marmot_get_last_error();
    }

    return marmot_convert(ctx, dst->dtype, dst_data, MARMOT_DTYPE_FLOAT32, src_data, marmot_tensor_numel(dst));
}

typedef struct {
    size_t *expert_counts;
    size_t *expert_offsets;
    size_t *expert_cursor;
    size_t *expert_order;
    uint32_t *sorted_tokens;
    float *sorted_weights;
    float *route_hidden_batch;
    float *hidden_batch;
    float *gate_batch;
    float *up_batch;
    float *down_batch;
    float *route_down_batch;
    size_t expert_cap;
    size_t route_cap;
    size_t route_hidden_batch_cap;
    size_t hidden_batch_cap;
    size_t ff_batch_cap;
    size_t down_batch_cap;
    size_t route_down_batch_cap;
} cpu_moe_workspace_t;

static thread_local cpu_moe_workspace_t cpu_moe_tls = {0};

typedef struct {
    atomic_uint_fast64_t gate_up_ns;
    atomic_uint_fast64_t activation_ns;
    atomic_uint_fast64_t down_ns;
    atomic_uint_fast64_t expert_calls;
} cpu_moe_profile_accum_t;

static uint64_t cpu_moe_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static bool cpu_moe_profile_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("MARMOT_MOE_PROFILE");
        cached = (env != nullptr && env[0] != '\0' && strcmp(env, "0") != 0) ? 1 : 0;
    }
    return cached != 0;
}

static void cpu_moe_log_grouped_profile(
    const size_t *expert_counts, size_t experts, size_t tokens, size_t experts_per_token, size_t route_count
) {
    if (!cpu_moe_profile_enabled() || expert_counts == nullptr || experts == 0) {
        return;
    }

    size_t active_experts = 0;
    size_t max_batch = 0;
    size_t hot_experts = 0;
    size_t bucket_1 = 0;
    size_t bucket_2 = 0;
    size_t bucket_3_4 = 0;
    size_t bucket_5_8 = 0;
    size_t bucket_9_16 = 0;
    size_t bucket_17_plus = 0;

    for (size_t expert = 0; expert < experts; ++expert) {
        const size_t count = expert_counts[expert];
        if (count == 0) {
            continue;
        }
        active_experts++;
        if (count > max_batch) {
            max_batch = count;
        }
        if (count >= 8) {
            hot_experts++;
        }
        if (count == 1) {
            bucket_1++;
        } else if (count == 2) {
            bucket_2++;
        } else if (count <= 4) {
            bucket_3_4++;
        } else if (count <= 8) {
            bucket_5_8++;
        } else if (count <= 16) {
            bucket_9_16++;
        } else {
            bucket_17_plus++;
        }
    }

    const double avg_active_batch = active_experts != 0 ? (double)route_count / (double)active_experts : 0.0;
    fprintf(
        stderr,
        "[cpu moe] tokens=%zu topk=%zu routes=%zu experts=%zu active=%zu avg_active_batch=%.2f max_batch=%zu "
        "hot(>=8)=%zu buckets{1=%zu,2=%zu,3-4=%zu,5-8=%zu,9-16=%zu,17+=%zu}\n",
        tokens, experts_per_token, route_count, experts, active_experts, avg_active_batch, max_batch, hot_experts,
        bucket_1, bucket_2, bucket_3_4, bucket_5_8, bucket_9_16, bucket_17_plus
    );
}

static void cpu_moe_log_grouped_timing(
    const cpu_moe_profile_accum_t *profile, size_t active_experts, size_t route_count, uint64_t route_fill_ns,
    uint64_t route_hidden_gather_ns, uint64_t expert_wall_ns, uint64_t scatter_ns
) {
    if (!cpu_moe_profile_enabled() || profile == nullptr) {
        return;
    }

    const uint64_t gate_up_ns = atomic_load(&profile->gate_up_ns);
    const uint64_t activation_ns = atomic_load(&profile->activation_ns);
    const uint64_t down_ns = atomic_load(&profile->down_ns);
    const uint64_t expert_calls = atomic_load(&profile->expert_calls);
    fprintf(
        stderr,
        "[cpu moe] grouped_prefill routes=%zu active=%zu fill=%.3fms gather=%.3fms expert_wall=%.3fms "
        "gate_up(sum)=%.3fms activation(sum)=%.3fms down(sum)=%.3fms scatter=%.3fms expert_calls=%" PRIu64 "\n",
        route_count, active_experts, (double)route_fill_ns / 1000000.0, (double)route_hidden_gather_ns / 1000000.0,
        (double)expert_wall_ns / 1000000.0, (double)gate_up_ns / 1000000.0, (double)activation_ns / 1000000.0,
        (double)down_ns / 1000000.0, (double)scatter_ns / 1000000.0, expert_calls
    );
}

static bool cpu_moe_workspace_ensure_routes(cpu_moe_workspace_t *ws, size_t experts, size_t routes) {
    if (ws == nullptr) {
        return false;
    }
    if (ws->expert_cap < experts) {
        size_t *counts = (size_t *)realloc(ws->expert_counts, experts * sizeof(size_t));
        size_t *offsets = (size_t *)realloc(ws->expert_offsets, experts * sizeof(size_t));
        size_t *cursor = (size_t *)realloc(ws->expert_cursor, experts * sizeof(size_t));
        size_t *order = (size_t *)realloc(ws->expert_order, experts * sizeof(size_t));
        if (counts == nullptr || offsets == nullptr || cursor == nullptr || order == nullptr) {
            free(counts);
            free(offsets);
            free(cursor);
            free(order);
            return false;
        }
        ws->expert_counts = counts;
        ws->expert_offsets = offsets;
        ws->expert_cursor = cursor;
        ws->expert_order = order;
        ws->expert_cap = experts;
    }
    if (ws->route_cap < routes) {
        uint32_t *sorted_tokens = (uint32_t *)realloc(ws->sorted_tokens, routes * sizeof(uint32_t));
        float *sorted_weights = (float *)realloc(ws->sorted_weights, routes * sizeof(float));
        if (sorted_tokens == nullptr || sorted_weights == nullptr) {
            free(sorted_tokens);
            free(sorted_weights);
            return false;
        }
        ws->sorted_tokens = sorted_tokens;
        ws->sorted_weights = sorted_weights;
        ws->route_cap = routes;
    }
    return true;
}

static bool
cpu_moe_workspace_ensure_batches(cpu_moe_workspace_t *ws, size_t max_batch, size_t hidden, size_t ff_length) {
    if (ws == nullptr) {
        return false;
    }

    const size_t hidden_elems = max_batch * hidden;
    const size_t ff_elems = max_batch * ff_length;
    if (ws->hidden_batch_cap < hidden_elems) {
        float *hidden_batch = (float *)realloc(ws->hidden_batch, hidden_elems * sizeof(float));
        float *down_batch = (float *)realloc(ws->down_batch, hidden_elems * sizeof(float));
        if (hidden_batch == nullptr || down_batch == nullptr) {
            free(hidden_batch);
            free(down_batch);
            return false;
        }
        ws->hidden_batch = hidden_batch;
        ws->down_batch = down_batch;
        ws->hidden_batch_cap = hidden_elems;
        ws->down_batch_cap = hidden_elems;
    }
    if (ws->ff_batch_cap < ff_elems) {
        float *gate_batch = (float *)realloc(ws->gate_batch, ff_elems * sizeof(float));
        float *up_batch = (float *)realloc(ws->up_batch, ff_elems * sizeof(float));
        if (gate_batch == nullptr || up_batch == nullptr) {
            free(gate_batch);
            free(up_batch);
            return false;
        }
        ws->gate_batch = gate_batch;
        ws->up_batch = up_batch;
        ws->ff_batch_cap = ff_elems;
    }
    return true;
}

static bool cpu_moe_workspace_ensure_route_hidden(cpu_moe_workspace_t *ws, size_t route_count, size_t hidden) {
    if (ws == nullptr) {
        return false;
    }
    const size_t route_elems = route_count * hidden;
    if (ws->route_hidden_batch_cap >= route_elems) {
        return true;
    }
    float *route_hidden_batch = (float *)realloc(ws->route_hidden_batch, route_elems * sizeof(float));
    if (route_hidden_batch == nullptr) {
        return false;
    }
    ws->route_hidden_batch = route_hidden_batch;
    ws->route_hidden_batch_cap = route_elems;
    return true;
}

static bool cpu_moe_workspace_ensure_route_down(cpu_moe_workspace_t *ws, size_t route_count, size_t hidden) {
    if (ws == nullptr) {
        return false;
    }
    const size_t route_elems = route_count * hidden;
    if (ws->route_down_batch_cap >= route_elems) {
        return true;
    }
    float *route_down_batch = (float *)realloc(ws->route_down_batch, route_elems * sizeof(float));
    if (route_down_batch == nullptr) {
        return false;
    }
    ws->route_down_batch = route_down_batch;
    ws->route_down_batch_cap = route_elems;
    return true;
}

static float cpu_moe_silu(float x) {
    return x / (1.0f + expf(-x));
}

static float cpu_moe_gelu(float x) {
    return 0.5f * x * (1.0f + erff(x * kCpuMoeRsqrt2));
}

static size_t cpu_moe_quant_row_bytes(marmot_quant_kind_t kind, size_t cols) {
    const marmot_quant_kind_traits_t *traits = marmot_get_quant_kind_traits(kind);
    if (traits == nullptr || traits->block_values == 0) {
        return 0;
    }
    const size_t block_bytes = traits->header_bytes + traits->payload_bytes;
    const size_t blocks_per_row = (cols + traits->block_values - 1) / traits->block_values;
    return blocks_per_row * block_bytes;
}

static size_t cpu_moe_expert_slice_bytes(const marmot_tensor_t *tensor, size_t rows, size_t cols) {
    if (tensor == nullptr) {
        return 0;
    }
    if (marmot_tensor_is_block_quantized_weight(tensor)) {
        return rows * cpu_moe_quant_row_bytes(tensor->quant_kind, cols);
    }
    return rows * cols * marmot_dtype_size(tensor->dtype);
}

static void cpu_moe_init_dense_view(
    marmot_tensor_t *view, const marmot_tensor_t *base, size_t rows, size_t cols, size_t byte_offset
) {
    memset(view, 0, sizeof(*view));
    view->shape.ndim = 2;
    view->shape.shape[0] = rows;
    view->shape.shape[1] = cols;
    view->shape.strides[0] = cols;
    view->shape.strides[1] = 1;
    view->dtype = base->dtype;
    view->data = (uint8_t *)base->data + byte_offset;
    view->capacity_bytes = rows * cols * marmot_dtype_size(base->dtype);
    view->owns_data = false;
    view->quant_kind = base->quant_kind;
    view->quant_layout = base->quant_layout;
    view->backend = base->backend;
    view->ctx = base->ctx;
    view->memory_location = base->memory_location;
    view->needs_sync = base->needs_sync;
}

static void
cpu_moe_init_f32_view(marmot_tensor_t *view, const marmot_context_t *ctx, float *data, size_t rows, size_t cols) {
    memset(view, 0, sizeof(*view));
    view->shape.ndim = 2;
    view->shape.shape[0] = rows;
    view->shape.shape[1] = cols;
    view->shape.strides[0] = cols;
    view->shape.strides[1] = 1;
    view->dtype = MARMOT_DTYPE_FLOAT32;
    view->data = data;
    view->capacity_bytes = rows * cols * sizeof(float);
    view->owns_data = false;
    view->backend = ctx != nullptr ? ctx->backend_type : MARMOT_BACKEND_CPU;
    view->ctx = (marmot_context_t *)ctx;
    view->memory_location = MARMOT_MEMORY_HOST;
}

static void cpu_moe_init_quant_view(
    marmot_tensor_t *view, const marmot_tensor_t *base, size_t rows, size_t cols, size_t byte_offset
) {
    memset(view, 0, sizeof(*view));
    view->shape.ndim = 2;
    view->shape.shape[0] = rows;
    view->shape.shape[1] = cols;
    view->shape.strides[0] = cols;
    view->shape.strides[1] = 1;
    view->dtype = base->dtype;
    view->data = (uint8_t *)base->data + byte_offset;
    view->capacity_bytes = cpu_moe_expert_slice_bytes(base, rows, cols);
    view->owns_data = false;
    view->quant_kind = base->quant_kind;
    view->quant_layout = base->quant_layout;
    view->backend = base->backend;
    view->ctx = base->ctx;
    view->memory_location = base->memory_location;
    view->needs_sync = base->needs_sync;
}

static void cpu_moe_init_expert_view(
    marmot_tensor_t *view, const marmot_tensor_t *base, size_t rows, size_t cols, size_t expert_idx
) {
    const size_t slice_bytes = cpu_moe_expert_slice_bytes(base, rows, cols);
    const size_t byte_offset = expert_idx * slice_bytes;
    if (marmot_tensor_is_block_quantized_weight(base)) {
        cpu_moe_init_quant_view(view, base, rows, cols, byte_offset);
        return;
    }
    cpu_moe_init_dense_view(view, base, rows, cols, byte_offset);
}

static bool cpu_moe_should_use_decode_path(const marmot_moe_experts_desc_t *desc) {
    if (desc == nullptr) {
        return false;
    }
    const size_t tokens = desc->hidden_states->shape.shape[0];
    const size_t experts_per_token = desc->topk_ids->shape.shape[1];
    return tokens <= kCpuMoeDecodeMaxTokens && tokens * experts_per_token <= kCpuMoeDecodeMaxRoutes;
}

static bool cpu_moe_can_dual_quant_project(const marmot_tensor_t *gate_view, const marmot_tensor_t *up_view) {
    if (gate_view == nullptr || up_view == nullptr) {
        return false;
    }
    if (!marmot_tensor_is_block_quantized_weight(gate_view) || !marmot_tensor_is_block_quantized_weight(up_view)) {
        return false;
    }
    return gate_view->dtype == up_view->dtype && gate_view->quant_kind == up_view->quant_kind &&
        gate_view->quant_layout == up_view->quant_layout && gate_view->shape.shape[0] == up_view->shape.shape[0] &&
        gate_view->shape.shape[1] == up_view->shape.shape[1];
}

static bool cpu_moe_all_quantized(const marmot_moe_experts_desc_t *desc) {
    return desc != nullptr && marmot_tensor_is_block_quantized_weight(desc->gate_exps) &&
        marmot_tensor_is_block_quantized_weight(desc->up_exps) &&
        marmot_tensor_is_block_quantized_weight(desc->down_exps);
}

static marmot_error_t cpu_moe_project_down(
    const marmot_context_t *ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, marmot_tensor_t *out,
    uint32_t hints
) {
    if (ctx != nullptr && ctx->backend_type == MARMOT_BACKEND_CPU && marmot_tensor_is_block_quantized_weight(weight)) {
        return cpu_matmul_quantized_with_hints(ctx->device_ctx, input, weight, nullptr, out, hints);
    }
    return marmot_linear(ctx, input, weight, nullptr, out);
}

static bool cpu_moe_should_parallel_grouped(
    const marmot_context_t *ctx, const marmot_moe_experts_desc_t *desc, size_t route_count
) {
    if (ctx == nullptr || desc == nullptr || !cpu_moe_all_quantized(desc) || route_count < 32) {
        return false;
    }
    const cpu_context_t *cpu_ctx = get_cpu_context(ctx->device_ctx);
    return cpu_ctx != nullptr && cpu_ctx->num_threads > 1;
}

static size_t cpu_moe_parallel_worker_count(const marmot_context_t *ctx) {
    if (ctx == nullptr) {
        return 1;
    }
    const cpu_context_t *cpu_ctx = get_cpu_context(ctx->device_ctx);
    if (cpu_ctx == nullptr || cpu_ctx->num_threads <= 1) {
        return 1;
    }
    return (size_t)cpu_ctx->num_threads;
}

static size_t cpu_moe_build_expert_order(size_t *expert_order, const size_t *expert_counts, size_t experts) {
    if (expert_order == nullptr || expert_counts == nullptr) {
        return 0;
    }

    size_t active = 0;
    for (size_t expert = 0; expert < experts; ++expert) {
        if (expert_counts[expert] == 0) {
            continue;
        }
        expert_order[active++] = expert;
    }

    for (size_t i = 1; i < active; ++i) {
        const size_t expert = expert_order[i];
        const size_t count = expert_counts[expert];
        size_t j = i;
        while (j > 0) {
            const size_t prev_expert = expert_order[j - 1];
            const size_t prev_count = expert_counts[prev_expert];
            if (prev_count > count || (prev_count == count && prev_expert < expert)) {
                break;
            }
            expert_order[j] = prev_expert;
            --j;
        }
        expert_order[j] = expert;
    }
    return active;
}

static marmot_error_t cpu_moe_compute_grouped_expert(
    const marmot_context_t *ctx, const marmot_moe_experts_desc_t *desc, const float *route_hidden, size_t hidden,
    size_t ff_length, size_t expert, size_t count, float *route_down_out, cpu_moe_profile_accum_t *profile
) {
    cpu_moe_workspace_t *ws = &cpu_moe_tls;
    if (!cpu_moe_workspace_ensure_batches(ws, count, hidden, ff_length)) {
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to allocate MoE compute workspace");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    marmot_tensor_t hidden_view;
    marmot_tensor_t gate_view;
    marmot_tensor_t up_view;
    marmot_tensor_t down_view;
    marmot_tensor_t gate_out_view;
    marmot_tensor_t up_out_view;
    marmot_tensor_t down_out_view;
    cpu_moe_init_f32_view(&hidden_view, ctx, (float *)route_hidden, count, hidden);
    cpu_moe_init_expert_view(&gate_view, desc->gate_exps, ff_length, hidden, expert);
    cpu_moe_init_expert_view(&up_view, desc->up_exps, ff_length, hidden, expert);
    cpu_moe_init_expert_view(&down_view, desc->down_exps, hidden, ff_length, expert);
    cpu_moe_init_f32_view(&gate_out_view, ctx, ws->gate_batch, count, ff_length);
    cpu_moe_init_f32_view(&up_out_view, ctx, ws->up_batch, count, ff_length);
    cpu_moe_init_f32_view(&down_out_view, ctx, route_down_out, count, hidden);

    marmot_error_t status = MARMOT_SUCCESS;
    const uint64_t gate_up_start_ns = profile != nullptr ? cpu_moe_now_ns() : 0;
    if (cpu_moe_can_dual_quant_project(&gate_view, &up_view)) {
        status = cpu_matmul_quantized_dual_output(
            ctx->device_ctx, &hidden_view, &gate_view, &up_view, &gate_out_view, &up_out_view
        );
    } else {
        status = marmot_linear(ctx, &hidden_view, &gate_view, nullptr, &gate_out_view);
        if (status == MARMOT_SUCCESS) {
            status = marmot_linear(ctx, &hidden_view, &up_view, nullptr, &up_out_view);
        }
    }
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    if (profile != nullptr) {
        atomic_fetch_add(&profile->gate_up_ns, cpu_moe_now_ns() - gate_up_start_ns);
    }

    const size_t ff_elems = count * ff_length;
    const uint64_t activation_start_ns = profile != nullptr ? cpu_moe_now_ns() : 0;
    for (size_t i = 0; i < ff_elems; ++i) {
        const float gate = ws->gate_batch[i];
        const float activated = desc->ffn_type == MARMOT_FFN_GEGLU ? cpu_moe_gelu(gate) : cpu_moe_silu(gate);
        ws->gate_batch[i] = activated * ws->up_batch[i];
    }
    if (profile != nullptr) {
        atomic_fetch_add(&profile->activation_ns, cpu_moe_now_ns() - activation_start_ns);
    }

    const uint64_t down_start_ns = profile != nullptr ? cpu_moe_now_ns() : 0;
    status = cpu_moe_project_down(ctx, &gate_out_view, &down_view, &down_out_view, 0);
    if (profile != nullptr) {
        atomic_fetch_add(&profile->down_ns, cpu_moe_now_ns() - down_start_ns);
        atomic_fetch_add(&profile->expert_calls, 1);
    }
    return status;
}

typedef struct {
    const marmot_context_t *ctx;
    const marmot_moe_experts_desc_t *desc;
    const float *route_hidden_batch;
    const size_t *expert_counts;
    const size_t *expert_offsets;
    const size_t *expert_order;
    size_t active_expert_count;
    size_t hidden;
    size_t ff_length;
    float *route_down_batch;
    cpu_moe_profile_accum_t *profile;
    atomic_size_t next_expert;
    _Atomic marmot_error_t first_error;
} cpu_moe_grouped_parallel_ctx_t;

static void cpu_moe_execute_grouped_parallel_worker(void *context, size_t worker_idx) {
    (void)worker_idx;
    cpu_moe_grouped_parallel_ctx_t *parallel_ctx = (cpu_moe_grouped_parallel_ctx_t *)context;
    if (parallel_ctx == nullptr) {
        return;
    }

    cpu_quant_matmul_set_thread_cap_override(1);
    for (;;) {
        if (atomic_load(&parallel_ctx->first_error) != MARMOT_SUCCESS) {
            break;
        }
        const size_t ordered_idx = atomic_fetch_add(&parallel_ctx->next_expert, 1);
        if (ordered_idx >= parallel_ctx->active_expert_count) {
            break;
        }
        const size_t expert = parallel_ctx->expert_order[ordered_idx];
        const size_t count = parallel_ctx->expert_counts[expert];
        if (count == 0) {
            continue;
        }
        const size_t offset = parallel_ctx->expert_offsets[expert];
        marmot_error_t status = cpu_moe_compute_grouped_expert(
            parallel_ctx->ctx, parallel_ctx->desc, parallel_ctx->route_hidden_batch + offset * parallel_ctx->hidden,
            parallel_ctx->hidden, parallel_ctx->ff_length, expert, count,
            parallel_ctx->route_down_batch + offset * parallel_ctx->hidden, parallel_ctx->profile
        );
        if (status != MARMOT_SUCCESS) {
            marmot_error_t expected = MARMOT_SUCCESS;
            (void)atomic_compare_exchange_strong(&parallel_ctx->first_error, &expected, status);
            break;
        }
    }
    cpu_quant_matmul_set_thread_cap_override(0);
}

static marmot_error_t cpu_moe_execute_decode(
    const marmot_context_t *ctx, const marmot_moe_experts_desc_t *desc, const float *hidden_data,
    const marmot_int32_t *topk_ids, const float *topk_weights, float *out_data
) {
    const size_t tokens = desc->hidden_states->shape.shape[0];
    const size_t hidden = desc->hidden_states->shape.shape[1];
    const size_t ff_length = desc->gate_exps->shape.shape[1];
    const size_t experts_per_token = desc->topk_ids->shape.shape[1];
    const size_t hidden_stride = desc->hidden_states->shape.strides[0];
    const size_t topk_id_stride = desc->topk_ids->shape.strides[0];
    const size_t topk_weight_stride = desc->topk_weights->shape.strides[0];
    const size_t out_stride = desc->out->shape.strides[0];
    const size_t route_count = tokens * experts_per_token;

    uint32_t active_experts[64] = {0};
    size_t active_counts[64] = {0};
    size_t active_offsets[64] = {0};
    size_t active_cursor[64] = {0};
    size_t active_count = 0;
    size_t max_batch = 0;

    cpu_moe_workspace_t *ws = &cpu_moe_tls;
    for (size_t token = 0; token < tokens; ++token) {
        memset(out_data + token * out_stride, 0, hidden * sizeof(float));
        for (size_t expert_slot = 0; expert_slot < experts_per_token; ++expert_slot) {
            const int32_t expert_idx_i32 = topk_ids[token * topk_id_stride + expert_slot].value;
            if (expert_idx_i32 < 0) {
                marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "MoE expert id is out of range");
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            const uint32_t expert_idx = (uint32_t)expert_idx_i32;
            size_t slot = 0;
            for (; slot < active_count; ++slot) {
                if (active_experts[slot] == expert_idx) {
                    break;
                }
            }
            if (slot == active_count) {
                active_experts[active_count] = expert_idx;
                active_counts[active_count] = 0;
                active_count++;
            }
            active_counts[slot]++;
            if (active_counts[slot] > max_batch) {
                max_batch = active_counts[slot];
            }
        }
    }

    if (!cpu_moe_workspace_ensure_batches(ws, max_batch, hidden, ff_length)) {
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to allocate MoE decode workspace");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    size_t prefix = 0;
    for (size_t slot = 0; slot < active_count; ++slot) {
        active_offsets[slot] = prefix;
        active_cursor[slot] = prefix;
        prefix += active_counts[slot];
    }

    if (!cpu_moe_workspace_ensure_routes(ws, active_count, route_count)) {
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to allocate MoE decode routing workspace");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    for (size_t token = 0; token < tokens; ++token) {
        float token_weight_norm = desc->weights_scale;
        if (desc->router_weight_policy == MARMOT_ROUTER_WEIGHT_POLICY_RENORMALIZE_SELECTED) {
            float weight_sum = 0.0f;
            for (size_t expert_slot = 0; expert_slot < experts_per_token; ++expert_slot) {
                weight_sum += topk_weights[token * topk_weight_stride + expert_slot];
            }
            token_weight_norm = weight_sum > FLT_MIN ? desc->weights_scale / weight_sum : 0.0f;
        }
        for (size_t expert_slot = 0; expert_slot < experts_per_token; ++expert_slot) {
            const uint32_t expert_idx = (uint32_t)topk_ids[token * topk_id_stride + expert_slot].value;
            size_t slot = 0;
            for (; slot < active_count; ++slot) {
                if (active_experts[slot] == expert_idx) {
                    break;
                }
            }
            if (slot == active_count) {
                marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "MoE decode expert lookup mismatch");
                return MARMOT_ERROR_INVALID_OPERATION;
            }
            const size_t pos = active_cursor[slot]++;
            ws->sorted_tokens[pos] = (uint32_t)token;
            ws->sorted_weights[pos] = topk_weights[token * topk_weight_stride + expert_slot] * token_weight_norm;
        }
    }

    for (size_t slot = 0; slot < active_count; ++slot) {
        const size_t count = active_counts[slot];
        if (count == 0) {
            continue;
        }

        const size_t offset = active_offsets[slot];
        const size_t expert_idx = active_experts[slot];
        for (size_t row = 0; row < count; ++row) {
            const size_t token = ws->sorted_tokens[offset + row];
            memcpy(ws->hidden_batch + row * hidden, hidden_data + token * hidden_stride, hidden * sizeof(float));
        }

        marmot_tensor_t hidden_view;
        marmot_tensor_t gate_view;
        marmot_tensor_t up_view;
        marmot_tensor_t down_view;
        marmot_tensor_t gate_out_view;
        marmot_tensor_t up_out_view;
        marmot_tensor_t down_out_view;
        cpu_moe_init_f32_view(&hidden_view, ctx, ws->hidden_batch, count, hidden);
        cpu_moe_init_expert_view(&gate_view, desc->gate_exps, ff_length, hidden, expert_idx);
        cpu_moe_init_expert_view(&up_view, desc->up_exps, ff_length, hidden, expert_idx);
        cpu_moe_init_expert_view(&down_view, desc->down_exps, hidden, ff_length, expert_idx);
        cpu_moe_init_f32_view(&gate_out_view, ctx, ws->gate_batch, count, ff_length);
        cpu_moe_init_f32_view(&up_out_view, ctx, ws->up_batch, count, ff_length);
        cpu_moe_init_f32_view(&down_out_view, ctx, ws->down_batch, count, hidden);

        marmot_error_t status = MARMOT_SUCCESS;
        if (cpu_moe_can_dual_quant_project(&gate_view, &up_view)) {
            status = cpu_matmul_quantized_dual_output(
                ctx->device_ctx, &hidden_view, &gate_view, &up_view, &gate_out_view, &up_out_view
            );
        } else {
            status = marmot_linear(ctx, &hidden_view, &gate_view, nullptr, &gate_out_view);
            if (status == MARMOT_SUCCESS) {
                status = marmot_linear(ctx, &hidden_view, &up_view, nullptr, &up_out_view);
            }
        }
        if (status != MARMOT_SUCCESS) {
            return status;
        }

        for (size_t row = 0; row < count; ++row) {
            const float *gate_row = ws->gate_batch + row * ff_length;
            const float *up_row = ws->up_batch + row * ff_length;
            float *fused_row = ws->gate_batch + row * ff_length;
            for (size_t col = 0; col < ff_length; ++col) {
                const float gate = gate_row[col];
                const float activated = desc->ffn_type == MARMOT_FFN_GEGLU ? cpu_moe_gelu(gate) : cpu_moe_silu(gate);
                fused_row[col] = activated * up_row[col];
            }
        }

        status =
            cpu_moe_project_down(ctx, &gate_out_view, &down_view, &down_out_view, CPU_QUANT_MATMUL_HINT_PREFER_RAW);
        if (status != MARMOT_SUCCESS) {
            return status;
        }

        for (size_t row = 0; row < count; ++row) {
            const size_t token = ws->sorted_tokens[offset + row];
            const float weight = ws->sorted_weights[offset + row];
            const float *src = ws->down_batch + row * hidden;
            float *dst = out_data + token * out_stride;
            for (size_t col = 0; col < hidden; ++col) {
                dst[col] += weight * src[col];
            }
        }
    }

    return MARMOT_SUCCESS;
}

marmot_error_t cpu_topk_impl(const void *device_ctx, const marmot_topk_desc_t *desc) {
    (void)device_ctx;

    if (desc == nullptr || desc->x == nullptr || desc->values_out == nullptr || desc->indices_out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "TopK descriptor is incomplete");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (!cpu_moe_value_dtype_supported(desc->x->dtype) || desc->values_out->dtype != desc->x->dtype ||
        desc->indices_out->dtype != MARMOT_DTYPE_INT32) {
        marmot_set_error(
            MARMOT_ERROR_UNSUPPORTED_DTYPE,
            "TopK supports FLOAT16/FLOAT32 values with matching output dtype and INT32 indices only"
        );
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (desc->x->shape.ndim != 2 || desc->values_out->shape.ndim != 2 || desc->indices_out->shape.ndim != 2) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "TopK expects 2D tensors");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (desc->axis != 1 && desc->axis != -1) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "TopK currently supports only the last axis");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    const size_t rows = desc->x->shape.shape[0];
    const size_t cols = desc->x->shape.shape[1];
    const size_t k = desc->k;
    if (k == 0 || k > cols || desc->values_out->shape.shape[0] != rows || desc->indices_out->shape.shape[0] != rows ||
        desc->values_out->shape.shape[1] != k || desc->indices_out->shape.shape[1] != k) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "TopK shapes do not match");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    const marmot_context_t *ctx = desc->x->ctx != nullptr ? desc->x->ctx : desc->values_out->ctx;
    if (ctx == nullptr) {
        ctx = desc->indices_out->ctx;
    }

    const void *input = ctx != nullptr ? marmot_tensor_data(ctx, (marmot_tensor_t *)desc->x) : desc->x->data;
    void *values_out = ctx != nullptr ? marmot_tensor_data_mut(ctx, desc->values_out) : desc->values_out->data;
    marmot_int32_t *indices_out =
        ctx != nullptr ? marmot_tensor_data_i32_mut(ctx, desc->indices_out) : (marmot_int32_t *)desc->indices_out->data;
    if (input == nullptr || values_out == nullptr || indices_out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "TopK tensor buffers are null");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const size_t input_stride = desc->x->shape.strides[0];
    const size_t values_stride = desc->values_out->shape.strides[0];
    const size_t indices_stride = desc->indices_out->shape.strides[0];

    float *best_values = (float *)malloc(k * sizeof(float));
    marmot_int32_t *best_indices = (marmot_int32_t *)malloc(k * sizeof(marmot_int32_t));
    if (best_values == nullptr || best_indices == nullptr) {
        free(best_values);
        free(best_indices);
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to allocate TopK scratch");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    for (size_t row = 0; row < rows; ++row) {
        for (size_t i = 0; i < k; ++i) {
            best_values[i] = -FLT_MAX;
            best_indices[i].value = -1;
        }

        const size_t input_base = row * input_stride;
        for (size_t col = 0; col < cols; ++col) {
            const float value = cpu_moe_load_value(input, desc->x->dtype, input_base + col);
            size_t insert_pos = k;
            for (size_t i = 0; i < k; ++i) {
                const bool better = value > best_values[i];
                const bool same = value == best_values[i] && (int32_t)col < best_indices[i].value;
                if (better || same) {
                    insert_pos = i;
                    break;
                }
            }
            if (insert_pos == k) {
                continue;
            }
            for (size_t i = k - 1; i > insert_pos; --i) {
                best_values[i] = best_values[i - 1];
                best_indices[i] = best_indices[i - 1];
            }
            best_values[insert_pos] = value;
            best_indices[insert_pos].value = (int32_t)col;
        }

        const size_t values_base = row * values_stride;
        marmot_int32_t *row_indices = indices_out + row * indices_stride;
        for (size_t i = 0; i < k; ++i) {
            cpu_moe_store_value(values_out, desc->values_out->dtype, values_base + i, best_values[i]);
            row_indices[i] = best_indices[i];
        }
    }

    free(best_values);
    free(best_indices);
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_moe_execute_grouped(
    const marmot_context_t *ctx, const marmot_moe_experts_desc_t *desc, const float *hidden_data,
    const marmot_int32_t *topk_ids, const float *topk_weights, float *out_data
) {
    const size_t tokens = desc->hidden_states->shape.shape[0];
    const size_t hidden = desc->hidden_states->shape.shape[1];
    const size_t ff_length = desc->gate_exps->shape.shape[1];
    const size_t experts = desc->gate_exps->shape.shape[2];
    const size_t experts_per_token = desc->topk_ids->shape.shape[1];
    const size_t hidden_stride = desc->hidden_states->shape.strides[0];
    const size_t topk_id_stride = desc->topk_ids->shape.strides[0];
    const size_t topk_weight_stride = desc->topk_weights->shape.strides[0];
    const size_t out_stride = desc->out->shape.strides[0];
    const size_t route_count = tokens * experts_per_token;

    cpu_moe_workspace_t *ws = &cpu_moe_tls;
    if (!cpu_moe_workspace_ensure_routes(ws, experts, route_count)) {
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to allocate MoE routing workspace");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }
    memset(ws->expert_counts, 0, experts * sizeof(size_t));

    for (size_t token = 0; token < tokens; ++token) {
        for (size_t expert_slot = 0; expert_slot < experts_per_token; ++expert_slot) {
            const int32_t expert_idx = topk_ids[token * topk_id_stride + expert_slot].value;
            if (expert_idx < 0 || (size_t)expert_idx >= experts) {
                marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "MoE expert id is out of range");
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            ws->expert_counts[(size_t)expert_idx]++;
        }
    }

    size_t prefix = 0;
    size_t max_batch = 0;
    size_t active_experts = 0;
    for (size_t expert = 0; expert < experts; ++expert) {
        ws->expert_offsets[expert] = prefix;
        ws->expert_cursor[expert] = prefix;
        const size_t count = ws->expert_counts[expert];
        prefix += count;
        if (count > max_batch) {
            max_batch = count;
        }
        if (count != 0) {
            active_experts++;
        }
    }
    if (prefix != route_count) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "MoE routing workspace prefix sum mismatch");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
    cpu_moe_log_grouped_profile(ws->expert_counts, experts, tokens, experts_per_token, route_count);
    if (max_batch == 0) {
        memset(out_data, 0, tokens * out_stride * sizeof(float));
        return MARMOT_SUCCESS;
    }

    if (!cpu_moe_workspace_ensure_route_down(ws, route_count, hidden)) {
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to allocate MoE route output workspace");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }
    if (!cpu_moe_workspace_ensure_route_hidden(ws, route_count, hidden)) {
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to allocate MoE routed hidden workspace");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }
    const size_t ordered_active_experts = cpu_moe_build_expert_order(ws->expert_order, ws->expert_counts, experts);
    const uint64_t route_fill_start_ns = cpu_moe_profile_enabled() ? cpu_moe_now_ns() : 0;
    for (size_t token = 0; token < tokens; ++token) {
        float token_weight_norm = desc->weights_scale;
        if (desc->router_weight_policy == MARMOT_ROUTER_WEIGHT_POLICY_RENORMALIZE_SELECTED) {
            float weight_sum = 0.0f;
            for (size_t expert_slot = 0; expert_slot < experts_per_token; ++expert_slot) {
                weight_sum += topk_weights[token * topk_weight_stride + expert_slot];
            }
            token_weight_norm = weight_sum > FLT_MIN ? desc->weights_scale / weight_sum : 0.0f;
        }
        for (size_t expert_slot = 0; expert_slot < experts_per_token; ++expert_slot) {
            const size_t expert_idx = (size_t)topk_ids[token * topk_id_stride + expert_slot].value;
            const size_t pos = ws->expert_cursor[expert_idx]++;
            ws->sorted_tokens[pos] = (uint32_t)token;
            ws->sorted_weights[pos] = topk_weights[token * topk_weight_stride + expert_slot] * token_weight_norm;
        }
    }
    const uint64_t route_fill_ns = cpu_moe_profile_enabled() ? cpu_moe_now_ns() - route_fill_start_ns : 0;

    const uint64_t route_hidden_start_ns = cpu_moe_profile_enabled() ? cpu_moe_now_ns() : 0;
    for (size_t route = 0; route < route_count; ++route) {
        const size_t token = ws->sorted_tokens[route];
        memcpy(ws->route_hidden_batch + route * hidden, hidden_data + token * hidden_stride, hidden * sizeof(float));
    }
    const uint64_t route_hidden_gather_ns = cpu_moe_profile_enabled() ? cpu_moe_now_ns() - route_hidden_start_ns : 0;

    for (size_t token = 0; token < tokens; ++token) {
        memset(out_data + token * out_stride, 0, hidden * sizeof(float));
    }

    cpu_moe_profile_accum_t profile = {0};
    const uint64_t expert_wall_start_ns = cpu_moe_profile_enabled() ? cpu_moe_now_ns() : 0;
    if (cpu_moe_should_parallel_grouped(ctx, desc, route_count)) {
        const size_t worker_count = cpu_moe_parallel_worker_count(ctx);
        cpu_moe_grouped_parallel_ctx_t parallel_ctx = {
            .ctx = ctx,
            .desc = desc,
            .route_hidden_batch = ws->route_hidden_batch,
            .expert_counts = ws->expert_counts,
            .expert_offsets = ws->expert_offsets,
            .expert_order = ws->expert_order,
            .active_expert_count = ordered_active_experts,
            .hidden = hidden,
            .ff_length = ff_length,
            .route_down_batch = ws->route_down_batch,
            .profile = &profile,
            .next_expert = 0,
            .first_error = MARMOT_SUCCESS,
        };
        marmot_dispatch_parallel_for(
            MARMOT_DISPATCH_PRIORITY_HIGH, worker_count, &parallel_ctx, cpu_moe_execute_grouped_parallel_worker
        );
        const marmot_error_t status = atomic_load(&parallel_ctx.first_error);
        if (status != MARMOT_SUCCESS) {
            return status;
        }
    } else {
        for (size_t ordered_idx = 0; ordered_idx < ordered_active_experts; ++ordered_idx) {
            const size_t expert = ws->expert_order[ordered_idx];
            const size_t count = ws->expert_counts[expert];
            if (count == 0) {
                continue;
            }
            const size_t offset = ws->expert_offsets[expert];
            marmot_error_t status = cpu_moe_compute_grouped_expert(
                ctx, desc, ws->route_hidden_batch + offset * hidden, hidden, ff_length, expert, count,
                ws->route_down_batch + offset * hidden, &profile
            );
            if (status != MARMOT_SUCCESS) {
                return status;
            }
        }
    }
    const uint64_t expert_wall_ns = cpu_moe_profile_enabled() ? cpu_moe_now_ns() - expert_wall_start_ns : 0;

    const uint64_t scatter_start_ns = cpu_moe_profile_enabled() ? cpu_moe_now_ns() : 0;
    for (size_t route = 0; route < route_count; ++route) {
        const size_t token = ws->sorted_tokens[route];
        const float weight = ws->sorted_weights[route];
        const float *src = ws->route_down_batch + route * hidden;
        float *dst = out_data + token * out_stride;
        for (size_t col = 0; col < hidden; ++col) {
            dst[col] += weight * src[col];
        }
    }
    const uint64_t scatter_ns = cpu_moe_profile_enabled() ? cpu_moe_now_ns() - scatter_start_ns : 0;
    cpu_moe_log_grouped_timing(
        &profile, active_experts, route_count, route_fill_ns, route_hidden_gather_ns, expert_wall_ns, scatter_ns
    );

    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_moe_experts_impl_f32(const marmot_context_t *ctx, const marmot_moe_experts_desc_t *desc) {
    const float *hidden_data = marmot_tensor_data_f32(ctx, (marmot_tensor_t *)desc->hidden_states);
    const marmot_int32_t *topk_ids = marmot_tensor_data_i32(ctx, (marmot_tensor_t *)desc->topk_ids);
    const float *topk_weights = marmot_tensor_data_f32(ctx, (marmot_tensor_t *)desc->topk_weights);
    const void *gate_data = marmot_tensor_data(ctx, (marmot_tensor_t *)desc->gate_exps);
    const void *up_data = marmot_tensor_data(ctx, (marmot_tensor_t *)desc->up_exps);
    const void *down_data = marmot_tensor_data(ctx, (marmot_tensor_t *)desc->down_exps);
    float *out_data = marmot_tensor_data_f32_mut(ctx, desc->out);
    if (hidden_data == nullptr || topk_ids == nullptr || topk_weights == nullptr || gate_data == nullptr ||
        up_data == nullptr || down_data == nullptr || out_data == nullptr) {
        return marmot_get_last_error();
    }
    if (cpu_moe_should_use_decode_path(desc)) {
        return cpu_moe_execute_decode(ctx, desc, hidden_data, topk_ids, topk_weights, out_data);
    }
    return cpu_moe_execute_grouped(ctx, desc, hidden_data, topk_ids, topk_weights, out_data);
}

marmot_error_t cpu_moe_experts_impl(const void *device_ctx, const marmot_moe_experts_desc_t *desc) {
    (void)device_ctx;

    if (desc == nullptr || desc->hidden_states == nullptr || desc->gate_exps == nullptr || desc->up_exps == nullptr ||
        desc->down_exps == nullptr || desc->topk_ids == nullptr || desc->topk_weights == nullptr ||
        desc->out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "MoE descriptor is incomplete");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const marmot_context_t *ctx = desc->hidden_states->ctx != nullptr ? desc->hidden_states->ctx : desc->out->ctx;
    if (ctx == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "MoE execution requires tensors bound to a context");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    const bool gate_quantized = marmot_tensor_is_block_quantized_weight(desc->gate_exps);
    const bool up_quantized = marmot_tensor_is_block_quantized_weight(desc->up_exps);
    const bool down_quantized = marmot_tensor_is_block_quantized_weight(desc->down_exps);
    const bool all_quantized = gate_quantized && up_quantized && down_quantized;
    const bool no_quantized = !gate_quantized && !up_quantized && !down_quantized;
    if (!all_quantized && !no_quantized) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "MoE expert quantization must be consistent across weights");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (!cpu_moe_value_dtype_supported(desc->hidden_states->dtype) ||
        desc->topk_weights->dtype != desc->hidden_states->dtype || desc->out->dtype != desc->hidden_states->dtype ||
        desc->topk_ids->dtype != MARMOT_DTYPE_INT32) {
        marmot_set_error(
            MARMOT_ERROR_UNSUPPORTED_DTYPE,
            "MoE experts supports FLOAT16/FLOAT32 activations with matching output/router weights and INT32 expert ids"
        );
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (no_quantized &&
        (desc->gate_exps->dtype != desc->hidden_states->dtype || desc->up_exps->dtype != desc->hidden_states->dtype ||
         desc->down_exps->dtype != desc->hidden_states->dtype)) {
        marmot_set_error(
            MARMOT_ERROR_UNSUPPORTED_DTYPE, "Dense MoE experts require activation and weight dtypes to match"
        );
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    if (desc->hidden_states->dtype == MARMOT_DTYPE_FLOAT32) {
        return cpu_moe_experts_impl_f32(ctx, desc);
    }

    marmot_tensor_t *hidden_f32 = cpu_moe_create_f32_tensor_like(ctx, desc->hidden_states);
    marmot_tensor_t *topk_weights_f32 = cpu_moe_create_f32_tensor_like(ctx, desc->topk_weights);
    marmot_tensor_t *out_f32 = cpu_moe_create_f32_tensor_like(ctx, desc->out);
    marmot_tensor_t *gate_f32 = nullptr;
    marmot_tensor_t *up_f32 = nullptr;
    marmot_tensor_t *down_f32 = nullptr;
    if (hidden_f32 == nullptr || topk_weights_f32 == nullptr || out_f32 == nullptr) {
        marmot_tensor_destroy(out_f32);
        marmot_tensor_destroy(topk_weights_f32);
        marmot_tensor_destroy(hidden_f32);
        return marmot_get_last_error();
    }
    if (no_quantized) {
        gate_f32 = cpu_moe_create_f32_tensor_like(ctx, desc->gate_exps);
        up_f32 = cpu_moe_create_f32_tensor_like(ctx, desc->up_exps);
        down_f32 = cpu_moe_create_f32_tensor_like(ctx, desc->down_exps);
        if (gate_f32 == nullptr || up_f32 == nullptr || down_f32 == nullptr) {
            marmot_tensor_destroy(down_f32);
            marmot_tensor_destroy(up_f32);
            marmot_tensor_destroy(gate_f32);
            marmot_tensor_destroy(out_f32);
            marmot_tensor_destroy(topk_weights_f32);
            marmot_tensor_destroy(hidden_f32);
            return marmot_get_last_error();
        }
    }

    marmot_error_t status = cpu_moe_copy_tensor_to_f32(ctx, desc->hidden_states, hidden_f32);
    if (status == MARMOT_SUCCESS) {
        status = cpu_moe_copy_tensor_to_f32(ctx, desc->topk_weights, topk_weights_f32);
    }
    if (status == MARMOT_SUCCESS && no_quantized) {
        status = cpu_moe_copy_tensor_to_f32(ctx, desc->gate_exps, gate_f32);
    }
    if (status == MARMOT_SUCCESS && no_quantized) {
        status = cpu_moe_copy_tensor_to_f32(ctx, desc->up_exps, up_f32);
    }
    if (status == MARMOT_SUCCESS && no_quantized) {
        status = cpu_moe_copy_tensor_to_f32(ctx, desc->down_exps, down_f32);
    }
    if (status == MARMOT_SUCCESS) {
        marmot_moe_experts_desc_t desc_f32 = *desc;
        desc_f32.hidden_states = hidden_f32;
        desc_f32.topk_weights = topk_weights_f32;
        desc_f32.out = out_f32;
        if (no_quantized) {
            desc_f32.gate_exps = gate_f32;
            desc_f32.up_exps = up_f32;
            desc_f32.down_exps = down_f32;
        }
        status = cpu_moe_experts_impl_f32(ctx, &desc_f32);
    }
    if (status == MARMOT_SUCCESS) {
        status = cpu_moe_copy_tensor_from_f32(ctx, desc->out, out_f32);
    }

    marmot_tensor_destroy(down_f32);
    marmot_tensor_destroy(up_f32);
    marmot_tensor_destroy(gate_f32);
    marmot_tensor_destroy(out_f32);
    marmot_tensor_destroy(topk_weights_f32);
    marmot_tensor_destroy(hidden_f32);
    return status;
}
