#include <metal_stdlib>
using namespace metal;

#include "common/defines.h"

#include "common/math_utils.h"

#include "common/dtype_utils.h"

#include "common/quant_blocks.h"

inline int ggml_nearest_int(float x) {
    return int(rint(x));
}

inline int clamp_int(int value, int lo, int hi) {
    return value < lo ? lo : (value > hi ? hi : value);
}

inline float make_qkx2_quants_metal(
    int n, int nmax, thread const float *x, thread const float *weights, thread uchar *L, thread float *the_min,
    thread uchar *Laux, float rmin, float rdelta, int nstep, bool use_mad
) {
    float min_val = x[0];
    float max_val = x[0];
    float sum_w = weights[0];
    float sum_x = sum_w * x[0];
    for (int i = 1; i < n; ++i) {
        float xi = x[i];
        if (xi < min_val) {
            min_val = xi;
        }
        if (xi > max_val) {
            max_val = xi;
        }
        float w = weights[i];
        sum_w += w;
        sum_x += w * xi;
    }
    if (min_val > 0.0f) {
        min_val = 0.0f;
    }
    if (max_val == min_val) {
        for (int i = 0; i < n; ++i) {
            L[i] = 0;
        }
        *the_min = -min_val;
        return 0.0f;
    }
    float iscale = float(nmax) / (max_val - min_val);
    float scale = 1.0f / iscale;
    float best_error = 0.0f;
    for (int i = 0; i < n; ++i) {
        int l = ggml_nearest_int(iscale * (x[i] - min_val));
        l = clamp_int(l, 0, nmax);
        L[i] = uchar(l);
        float diff = scale * float(l) + min_val - x[i];
        diff = use_mad ? fabs(diff) : diff * diff;
        float w = weights[i];
        best_error += w * diff;
    }
    if (nstep < 1) {
        *the_min = -min_val;
        return scale;
    }
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta * float(is) + float(nmax)) / (max_val - min_val);
        float sum_l = 0.0f;
        float sum_l2 = 0.0f;
        float sum_xl = 0.0f;
        for (int i = 0; i < n; ++i) {
            int l = ggml_nearest_int(iscale * (x[i] - min_val));
            l = clamp_int(l, 0, nmax);
            Laux[i] = uchar(l);
            float w = weights[i];
            float fl = float(l);
            sum_l += w * fl;
            sum_l2 += w * fl * fl;
            sum_xl += w * fl * x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0.0f) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
            float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
            if (this_min > 0.0f) {
                this_min = 0.0f;
                this_scale = sum_xl / sum_l2;
            }
            float cur_error = 0.0f;
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * float(Laux[i]) + this_min - x[i];
                diff = use_mad ? fabs(diff) : diff * diff;
                float w = weights[i];
                cur_error += w * diff;
            }
            if (cur_error < best_error) {
                for (int i = 0; i < n; ++i) {
                    L[i] = Laux[i];
                }
                best_error = cur_error;
                scale = this_scale;
                min_val = this_min;
            }
        }
    }
    *the_min = -min_val;
    return scale;
}

constant float kGroupMaxEps = 1e-15f;

inline float make_q3_quants_metal(int n, int nmax, thread const float *x, thread char *L, bool do_rmse) {
    float max_val = 0.0f;
    float abs_max = 0.0f;
    for (int i = 0; i < n; ++i) {
        float ax = fabs(x[i]);
        if (ax > abs_max) {
            abs_max = ax;
            max_val = x[i];
        }
    }
    if (abs_max == 0.0f) {
        for (int i = 0; i < n; ++i) {
            L[i] = 0;
        }
        return 0.0f;
    }

    float iscale = -float(nmax) / max_val;
    if (!do_rmse) {
        for (int i = 0; i < n; ++i) {
            int l = ggml_nearest_int(iscale * x[i]);
            l = clamp_int(l, -nmax, nmax - 1);
            L[i] = char(l + nmax);
        }
        return 1.0f / iscale;
    }

    float sumlx = 0.0f;
    float suml2 = 0.0f;
    for (int i = 0; i < n; ++i) {
        int l = ggml_nearest_int(iscale * x[i]);
        l = clamp_int(l, -nmax, nmax - 1);
        L[i] = char(l);
        float w = x[i] * x[i];
        sumlx += w * x[i] * float(l);
        suml2 += w * float(l) * float(l);
    }

    for (int itry = 0; itry < 5; ++itry) {
        int n_changed = 0;
        for (int i = 0; i < n; ++i) {
            float w = x[i] * x[i];
            float current_l = float(L[i]);
            float slx = sumlx - w * x[i] * current_l;
            if (slx > 0.0f) {
                float sl2 = suml2 - w * current_l * current_l;
                if (sl2 > 0.0f) {
                    float ratio = x[i] * sl2 / slx;
                    int new_l = ggml_nearest_int(ratio);
                    new_l = clamp_int(new_l, -nmax, nmax - 1);
                    if (new_l != int(L[i])) {
                        float new_lf = float(new_l);
                        float trial_slx = slx + w * x[i] * new_lf;
                        float trial_sl2 = sl2 + w * new_lf * new_lf;
                        if (trial_sl2 > 0.0f && trial_slx * trial_slx * suml2 > sumlx * sumlx * trial_sl2) {
                            sumlx = trial_slx;
                            suml2 = trial_sl2;
                            L[i] = char(new_l);
                            n_changed++;
                        }
                    }
                }
            }
        }
        if (n_changed == 0) {
            break;
        }
    }

    for (int i = 0; i < n; ++i) {
        L[i] = char(L[i] + nmax);
    }

    return suml2 > 0.0f ? (sumlx / suml2) : 0.0f;
}

constant uint kQKGroups = kQK_K / 16u;
constant uint kQK32Groups = kQK_K / 32u;

inline int clamp_qx_value_metal(int value, int nmax) {
    int hi = nmax - 1;
    if (value > hi) {
        return hi;
    }
    int lo = -nmax;
    if (value < lo) {
        return lo;
    }
    return value;
}

inline float make_qx_quants_metal(int n, int nmax, thread const float *x, thread char *L) {
    float max_val = 0.0f;
    float abs_max = 0.0f;
    for (int i = 0; i < n; ++i) {
        float ax = fabs(x[i]);
        if (ax > abs_max) {
            abs_max = ax;
            max_val = x[i];
        }
    }

    if (abs_max < kGroupMaxEps) {
        for (int i = 0; i < n; ++i) {
            L[i] = 0;
        }
        return 0.0f;
    }

    float iscale = -float(nmax) / max_val;
    float sumlx = 0.0f;
    float suml2 = 0.0f;
    for (int i = 0; i < n; ++i) {
        int l = ggml_nearest_int(iscale * x[i]);
        l = clamp_qx_value_metal(l, nmax);
        L[i] = char(l + nmax);
        float w = x[i] * x[i];
        float lf = float(l);
        sumlx += w * x[i] * lf;
        suml2 += w * lf * lf;
    }

    float scale = suml2 > 0.0f ? (sumlx / suml2) : 0.0f;
    float best = scale * sumlx;

    for (int is = -9; is <= 9; ++is) {
        if (is == 0) {
            continue;
        }
        float trial_iscale = -(float(nmax) + 0.1f * float(is)) / max_val;
        float trial_sumlx = 0.0f;
        float trial_suml2 = 0.0f;
        for (int i = 0; i < n; ++i) {
            int l = ggml_nearest_int(trial_iscale * x[i]);
            l = clamp_qx_value_metal(l, nmax);
            float w = x[i] * x[i];
            float lf = float(l);
            trial_sumlx += w * x[i] * lf;
            trial_suml2 += w * lf * lf;
        }
        if (trial_suml2 > 0.0f && trial_sumlx * trial_sumlx > best * trial_suml2) {
            for (int i = 0; i < n; ++i) {
                int l = ggml_nearest_int(trial_iscale * x[i]);
                l = clamp_qx_value_metal(l, nmax);
                L[i] = char(l + nmax);
            }
            scale = trial_sumlx / trial_suml2;
            best = scale * trial_sumlx;
        }
    }

    return scale;
}

// -----------------------------------------------------------------------------
// Quantization kernels (Q4_0, Q4_1, INT8)
// -----------------------------------------------------------------------------

// Q4_0 Format: Symmetric quantization, 32-weight blocks, 20 bytes/block
// Block layout: [4 bytes float32 scale][16 bytes packed INT4]
// Formula: weight_i = scale * quantized_i (quantized_i in [-7, 7])

kernel void quantize_q4_0(
    constant float *input [[buffer(0)]], device uint8_t *output [[buffer(1)]], constant uint &num_rows [[buffer(2)]],
    constant uint &row_size [[buffer(3)]], uint block_id [[thread_position_in_grid]]
) {
    const uint block_size = 32;
    const uint block_stride = 18;

    // For 2D tensors: calculate row and block within row
    const uint blocks_per_row = (row_size + block_size - 1) / block_size;
    const uint row_idx = block_id / blocks_per_row;
    const uint block_idx_in_row = block_id % blocks_per_row;

    if (row_idx >= num_rows) {
        return;
    }

    const uint row_start = row_idx * row_size;
    const uint block_start = row_start + block_idx_in_row * block_size;
    const uint block_end = min(block_start + block_size, row_start + row_size);
    const uint block_len = block_end - block_start;

    float amax = 0.0f;
    float max = 0.0f;
    for (uint i = block_start; i < block_end; i++) {
        const float v = input[i];
        const float abs_v = fabs(v);
        if (abs_v > amax) {
            amax = abs_v;
            max = v;
        }
    }

    const float d = max / -8.0f;
    float scale = d != 0.0f ? d : 1.0f;
    const float id = scale != 0.0f ? 1.0f / scale : 0.0f;

    device uchar *block_base = output + block_id * block_stride;
    device half *scale_ptr = (device half *)(block_base);
    scale_ptr[0] = half(scale);

    device uchar *qs = block_base + sizeof(half);
    for (uint i = 0; i < 16; ++i) {
        qs[i] = 0;
    }

    const uint half_size = block_size / 2;
    for (uint j = 0; j < half_size; ++j) {
        float x0 = 0.0f, x1 = 0.0f;
        if (j < block_len) {
            x0 = input[block_start + j] * id;
        }
        if (j + half_size < block_len) {
            x1 = input[block_start + j + half_size] * id;
        }

        const uchar xi0 = uchar(char(x0 + 8.5f));
        const uchar xi1 = uchar(char(x1 + 8.5f));
        const uchar q0 = xi0 > 15 ? 15 : xi0;
        const uchar q1 = xi1 > 15 ? 15 : xi1;

        qs[j] = (q1 << 4) | q0;
    }
}

kernel void dequantize_q4_0(
    constant uint8_t *input [[buffer(0)]], device float *output [[buffer(1)]], constant uint &num_rows [[buffer(2)]],
    constant uint &row_size [[buffer(3)]], uint block_id [[thread_position_in_grid]]
) {
    const uint block_size = 32;
    const uint block_stride = 18;

    // For 2D tensors: calculate row and block within row
    const uint blocks_per_row = (row_size + block_size - 1) / block_size;
    const uint row_idx = block_id / blocks_per_row;
    const uint block_idx_in_row = block_id % blocks_per_row;

    if (row_idx >= num_rows) {
        return;
    }

    const uint row_start = row_idx * row_size;
    const uint block_start = row_start + block_idx_in_row * block_size;
    const uint block_end = min(block_start + block_size, row_start + row_size);
    const uint block_len = block_end - block_start;

    constant uchar *block_base = input + block_id * block_stride;
    constant half *scale_ptr = (constant half *)(block_base);
    const float scale = float(scale_ptr[0]);
    constant uchar *qs = block_base + sizeof(half);

    const uint half_size = block_size / 2;
    for (uint j = 0; j < half_size; ++j) {
        const uchar packed = qs[j];
        const uchar q0 = packed & 0x0F;
        const uchar q1 = packed >> 4;

        if (j < block_len) {
            output[block_start + j] = (char(q0) - 8) * scale;
        }
        if (j + half_size < block_len) {
            output[block_start + j + half_size] = (char(q1) - 8) * scale;
        }
    }
}

// Q4_1 Format: Asymmetric quantization, 32-weight blocks, 24 bytes/block
// Block layout: [4 bytes scale][4 bytes min][16 bytes packed UINT4]
// Formula: weight_i = scale * quantized_i + min (quantized_i in [0, 15])

kernel void quantize_q4_1(
    constant float *input [[buffer(0)]], device uint8_t *output [[buffer(1)]], constant uint &num_rows [[buffer(2)]],
    constant uint &row_size [[buffer(3)]], uint block_id [[thread_position_in_grid]]
) {
    const uint block_size = 32;
    const uint block_stride = 20;

    const uint blocks_per_row = (row_size + block_size - 1) / block_size;
    const uint row_idx = block_id / blocks_per_row;
    const uint block_idx_in_row = block_id % blocks_per_row;

    if (row_idx >= num_rows) {
        return;
    }

    const uint row_start = row_idx * row_size;
    const uint block_start = row_start + block_idx_in_row * block_size;
    const uint block_end = min(block_start + block_size, row_start + row_size);
    const uint block_len = block_end - block_start;

    float min_val = input[block_start];
    float max_val = input[block_start];
    for (uint i = block_start + 1; i < block_end; i++) {
        min_val = fmin(min_val, input[i]);
        max_val = fmax(max_val, input[i]);
    }

    float scale = (max_val - min_val) / 15.0f;
    if (scale < 1e-8f) {
        scale = 1.0f;
    }

    device uchar *block_base = output + block_id * block_stride;
    device half *header = (device half *)(block_base);
    header[0] = half(scale);
    header[1] = half(min_val);

    device uchar *qs = block_base + sizeof(half) * 2;
    for (uint i = 0; i < 16; ++i) {
        qs[i] = 0;
    }

    const float id = scale != 0.0f ? 1.0f / scale : 0.0f;
    const uint half_size = block_size / 2;

    for (uint j = 0; j < half_size; ++j) {
        float x0 = 0.0f, x1 = 0.0f;
        if (j < block_len) {
            x0 = (input[block_start + j] - min_val) * id;
        }
        if (j + half_size < block_len) {
            x1 = (input[block_start + j + half_size] - min_val) * id;
        }

        const uchar xi0 = uchar(char(x0 + 0.5f));
        const uchar xi1 = uchar(char(x1 + 0.5f));
        const uchar q0 = xi0 > 15 ? 15 : xi0;
        const uchar q1 = xi1 > 15 ? 15 : xi1;

        qs[j] = (q1 << 4) | q0;
    }
}

kernel void dequantize_q4_1(
    constant uint8_t *input [[buffer(0)]], device float *output [[buffer(1)]], constant uint &num_rows [[buffer(2)]],
    constant uint &row_size [[buffer(3)]], uint block_id [[thread_position_in_grid]]
) {
    const uint block_size = 32;
    const uint block_stride = 20;

    // For 2D tensors: calculate row and block within row
    const uint blocks_per_row = (row_size + block_size - 1) / block_size;
    const uint row_idx = block_id / blocks_per_row;
    const uint block_idx_in_row = block_id % blocks_per_row;

    if (row_idx >= num_rows) {
        return;
    }

    const uint row_start = row_idx * row_size;
    const uint block_start = row_start + block_idx_in_row * block_size;
    const uint block_end = min(block_start + block_size, row_start + row_size);
    const uint block_len = block_end - block_start;

    constant uchar *block_base = input + block_id * block_stride;
    constant half *header = (constant half *)(block_base);
    const float scale = float(header[0]);
    const float min_val = float(header[1]);
    constant uchar *qs = block_base + sizeof(half) * 2;

    const uint half_size = block_size / 2;
    for (uint j = 0; j < half_size; ++j) {
        const uchar packed = qs[j];
        const uchar q0 = packed & 0x0F;
        const uchar q1 = packed >> 4;

        if (j < block_len) {
            output[block_start + j] = min_val + float(q0) * scale;
        }
        if (j + half_size < block_len) {
            output[block_start + j + half_size] = min_val + float(q1) * scale;
        }
    }
}

// Q5_0 Format: Symmetric quantization, 32-weight blocks, 24 bytes/block
// Block layout: [4 bytes scale][20 bytes packed INT5]

kernel void quantize_q5_0(
    constant float *input [[buffer(0)]], device uint8_t *output [[buffer(1)]], constant uint &num_rows [[buffer(2)]],
    constant uint &row_size [[buffer(3)]], uint block_id [[thread_position_in_grid]]
) {
    const uint block_size = 32;
    const uint block_stride = 22;

    const uint blocks_per_row = (row_size + block_size - 1) / block_size;
    const uint row_idx = block_id / blocks_per_row;
    const uint block_idx_in_row = block_id % blocks_per_row;

    if (row_idx >= num_rows) {
        return;
    }

    const uint row_start = row_idx * row_size;
    const uint block_start = row_start + block_idx_in_row * block_size;
    const uint block_end = min(block_start + block_size, row_start + row_size);
    const uint block_len = block_end - block_start;

    float amax = 0.0f;
    float max = 0.0f;
    for (uint i = block_start; i < block_end; ++i) {
        const float v = input[i];
        const float abs_v = fabs(v);
        if (abs_v > amax) {
            amax = abs_v;
            max = v;
        }
    }

    const float d = max / -16.0f;
    float scale = d != 0.0f ? d : 1.0f;
    const float id = scale != 0.0f ? 1.0f / scale : 0.0f;

    device uchar *block_base = output + block_id * block_stride;
    device half *scale_ptr = (device half *)(block_base);
    scale_ptr[0] = half(scale);

    device uchar *qh = block_base + sizeof(half);
    device uchar *qs = qh + 4;
    for (uint i = 0; i < 4; ++i) {
        qh[i] = 0;
    }
    for (uint i = 0; i < 16; ++i) {
        qs[i] = 0;
    }

    uint qh_bits = 0;
    const uint half_size = block_size / 2;

    for (uint j = 0; j < half_size; ++j) {
        float x0 = 0.0f, x1 = 0.0f;
        if (j < block_len) {
            x0 = input[block_start + j] * id;
        }
        if (j + half_size < block_len) {
            x1 = input[block_start + j + half_size] * id;
        }

        const uchar xi0 = uchar(char(x0 + 16.5f));
        const uchar xi1 = uchar(char(x1 + 16.5f));
        const uchar q0 = xi0 > 31 ? 31 : xi0;
        const uchar q1 = xi1 > 31 ? 31 : xi1;

        qs[j] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);

        qh_bits |= ((q0 & 0x10) >> 4) << (j + 0);
        qh_bits |= ((q1 & 0x10) >> 4) << (j + half_size);
    }

    for (uint i = 0; i < 4; ++i) {
        qh[i] = uchar((qh_bits >> (i * 8)) & 0xFF);
    }
}

kernel void dequantize_q5_0(
    constant uint8_t *input [[buffer(0)]], device float *output [[buffer(1)]], constant uint &num_rows [[buffer(2)]],
    constant uint &row_size [[buffer(3)]], uint block_id [[thread_position_in_grid]]
) {
    const uint block_size = 32;
    const uint block_stride = 22;

    // For 2D tensors: calculate row and block within row
    const uint blocks_per_row = (row_size + block_size - 1) / block_size;
    const uint row_idx = block_id / blocks_per_row;
    const uint block_idx_in_row = block_id % blocks_per_row;

    if (row_idx >= num_rows) {
        return;
    }

    const uint row_start = row_idx * row_size;
    const uint block_start = row_start + block_idx_in_row * block_size;
    const uint block_end = min(block_start + block_size, row_start + row_size);
    const uint block_len = block_end - block_start;

    constant uchar *block_base = input + block_id * block_stride;
    constant half *scale_ptr = (constant half *)(block_base);
    const float scale = float(scale_ptr[0]);
    constant uchar *qh = block_base + sizeof(half);
    constant uchar *qs = qh + 4;

    uint qh_bits = 0;
    for (uint i = 0; i < 4; ++i) {
        qh_bits |= uint(qh[i]) << (i * 8);
    }

    const uint half_size = block_size / 2;
    for (uint j = 0; j < half_size; ++j) {
        const uchar qs_byte = qs[j];
        const uchar x0 = qs_byte & 0x0F;
        const uchar x1 = qs_byte >> 4;

        const uchar xh0 = uchar((qh_bits >> (j + 0)) & 0x1);
        const uchar xh1 = uchar((qh_bits >> (j + half_size)) & 0x1);

        const uchar q0 = x0 | (xh0 << 4);
        const uchar q1 = x1 | (xh1 << 4);

        if (j < block_len) {
            output[block_start + j] = (char(q0) - 16) * scale;
        }
        if (j + half_size < block_len) {
            output[block_start + j + half_size] = (char(q1) - 16) * scale;
        }
    }
}

// Q5_1 Format: Asymmetric quantization, 32-weight blocks, 28 bytes/block
// Block layout: [4 bytes scale][4 bytes min][20 bytes packed UINT5]

kernel void quantize_q5_1(
    constant float *input [[buffer(0)]], device uint8_t *output [[buffer(1)]], constant uint &num_rows [[buffer(2)]],
    constant uint &row_size [[buffer(3)]], uint block_id [[thread_position_in_grid]]
) {
    const uint block_size = 32;
    const uint block_stride = 24;

    const uint blocks_per_row = (row_size + block_size - 1) / block_size;
    const uint row_idx = block_id / blocks_per_row;
    const uint block_idx_in_row = block_id % blocks_per_row;

    if (row_idx >= num_rows) {
        return;
    }

    const uint row_start = row_idx * row_size;
    const uint block_start = row_start + block_idx_in_row * block_size;
    const uint block_end = min(block_start + block_size, row_start + row_size);
    const uint block_len = block_end - block_start;

    float min_val = input[block_start];
    float max_val = input[block_start];
    for (uint i = block_start + 1; i < block_end; ++i) {
        const float value = input[i];
        min_val = fmin(min_val, value);
        max_val = fmax(max_val, value);
    }

    float scale = (max_val - min_val) / 31.0f;
    if (scale < 1e-8f) {
        scale = 1.0f;
    }

    device uchar *block_base = output + block_id * block_stride;
    device half *header = (device half *)(block_base);
    header[0] = half(scale);
    header[1] = half(min_val);

    device uchar *qh = block_base + sizeof(half) * 2;
    device uchar *qs = qh + 4;
    for (uint i = 0; i < 4; ++i) {
        qh[i] = 0;
    }
    for (uint i = 0; i < 16; ++i) {
        qs[i] = 0;
    }

    const float id = scale != 0.0f ? 1.0f / scale : 0.0f;
    uint qh_bits = 0;
    const uint half_size = block_size / 2;

    for (uint j = 0; j < half_size; ++j) {
        float x0 = 0.0f, x1 = 0.0f;
        if (j < block_len) {
            x0 = (input[block_start + j] - min_val) * id;
        }
        if (j + half_size < block_len) {
            x1 = (input[block_start + j + half_size] - min_val) * id;
        }

        const uchar xi0 = uchar(char(x0 + 0.5f));
        const uchar xi1 = uchar(char(x1 + 0.5f));
        const uchar q0 = xi0 > 31 ? 31 : xi0;
        const uchar q1 = xi1 > 31 ? 31 : xi1;

        qs[j] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);

        qh_bits |= ((q0 & 0x10) >> 4) << (j + 0);
        qh_bits |= ((q1 & 0x10) >> 4) << (j + half_size);
    }

    for (uint i = 0; i < 4; ++i) {
        qh[i] = uchar((qh_bits >> (i * 8)) & 0xFF);
    }
}

kernel void dequantize_q5_1(
    constant uint8_t *input [[buffer(0)]], device float *output [[buffer(1)]], constant uint &num_rows [[buffer(2)]],
    constant uint &row_size [[buffer(3)]], uint block_id [[thread_position_in_grid]]
) {
    const uint block_size = 32;
    const uint block_stride = 24;

    // For 2D tensors: calculate row and block within row
    const uint blocks_per_row = (row_size + block_size - 1) / block_size;
    const uint row_idx = block_id / blocks_per_row;
    const uint block_idx_in_row = block_id % blocks_per_row;

    if (row_idx >= num_rows) {
        return;
    }

    const uint row_start = row_idx * row_size;
    const uint block_start = row_start + block_idx_in_row * block_size;
    const uint block_end = min(block_start + block_size, row_start + row_size);
    const uint block_len = block_end - block_start;

    constant uchar *block_base = input + block_id * block_stride;
    constant half *header = (constant half *)(block_base);
    const float scale = float(header[0]);
    const float min_val = float(header[1]);
    constant uchar *qh = block_base + sizeof(half) * 2;
    constant uchar *qs = qh + 4;

    uint qh_bits = 0;
    for (uint i = 0; i < 4; ++i) {
        qh_bits |= uint(qh[i]) << (i * 8);
    }

    const uint half_size = block_size / 2;
    for (uint j = 0; j < half_size; ++j) {
        const uchar qs_byte = qs[j];
        const uchar xh_0 = uchar(((qh_bits >> (j + 0)) << 4) & 0x10);
        const uchar xh_1 = uchar((qh_bits >> (j + 12)) & 0x10);

        const uchar x0 = (qs_byte & 0x0F) | xh_0;
        const uchar x1 = (qs_byte >> 4) | xh_1;

        if (j < block_len) {
            output[block_start + j] = float(x0) * scale + min_val;
        }
        if (j + half_size < block_len) {
            output[block_start + j + half_size] = float(x1) * scale + min_val;
        }
    }
}

// Q8_0 Format: Symmetric quantization, 32-weight blocks, 34 bytes/block
// Block layout: [2 bytes scale (fp16)][32 bytes INT8]

kernel void quantize_q8_0(
    constant float *input [[buffer(0)]], device int8_t *output [[buffer(1)]], constant uint &num_rows [[buffer(2)]],
    constant uint &row_size [[buffer(3)]], uint block_id [[thread_position_in_grid]]
) {
    const uint block_size = 32;
    const uint block_stride = 34;

    const uint blocks_per_row = (row_size + block_size - 1) / block_size;
    const uint row_idx = block_id / blocks_per_row;
    const uint block_idx_in_row = block_id % blocks_per_row;

    if (row_idx >= num_rows) {
        return;
    }

    const uint row_start = row_idx * row_size;
    const uint block_start = row_start + block_idx_in_row * block_size;
    const uint block_end = min(block_start + block_size, row_start + row_size);
    const uint block_len = block_end - block_start;

    float max_abs = 0.0f;
    for (uint i = block_start; i < block_end; ++i) {
        max_abs = fmax(max_abs, fabs(input[i]));
    }
    float scale = max_abs / 127.0f;
    if (scale < 1e-8f) {
        scale = 1.0f;
    }

    device uchar *raw = (device uchar *)output;
    device half *scale_ptr = (device half *)(raw + block_id * block_stride);
    scale_ptr[0] = half(scale);

    device int8_t *weights = (device int8_t *)(raw + block_id * block_stride + sizeof(half));
    for (uint j = 0; j < block_size; ++j) {
        weights[j] = 0;
    }

    for (uint i = 0; i < block_len; ++i) {
        float value = input[block_start + i] / scale;
        int q = int(round(value));
        q = clamp(q, -127, 127);
        weights[i] = int8_t(q);
    }
}

kernel void dequantize_q8_0(
    constant int8_t *input [[buffer(0)]], device float *output [[buffer(1)]], constant uint &num_rows [[buffer(2)]],
    constant uint &row_size [[buffer(3)]], uint block_id [[thread_position_in_grid]]
) {
    const uint block_size = 32;
    const uint block_stride = 34;

    // For 2D tensors: calculate row and block within row
    const uint blocks_per_row = (row_size + block_size - 1) / block_size;
    const uint row_idx = block_id / blocks_per_row;
    const uint block_idx_in_row = block_id % blocks_per_row;

    if (row_idx >= num_rows) {
        return;
    }

    const uint row_start = row_idx * row_size;
    const uint block_start = row_start + block_idx_in_row * block_size;
    const uint block_end = min(block_start + block_size, row_start + row_size);
    const uint block_len = block_end - block_start;

    constant uchar *raw = (constant uchar *)input;
    constant half *scale_ptr = (constant half *)(raw + block_id * block_stride);
    const float scale = float(scale_ptr[0]);
    constant int8_t *weights = (constant int8_t *)(raw + block_id * block_stride + sizeof(half));

    for (uint i = 0; i < block_len; ++i) {
        output[block_start + i] = float(weights[i]) * scale;
    }
}

// Generic INT8 quantization helpers
kernel void compute_quant_params_int8(
    constant float *input [[buffer(0)]],
    device float *params [[buffer(1)]], // [scale, zero_point]
    constant uint &num_elements [[buffer(2)]], constant uint &is_unsigned [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id != 0) {
        return;
    }

    float min_val = input[0];
    float max_val = input[0];
    for (uint i = 1; i < num_elements; i++) {
        min_val = fmin(min_val, input[i]);
        max_val = fmax(max_val, input[i]);
    }

    float qmin = is_unsigned ? 0.0f : -128.0f;
    float qmax = is_unsigned ? 255.0f : 127.0f;

    float scale = (max_val - min_val) / (qmax - qmin);
    if (scale < 1e-8f) {
        scale = 1.0f;
    }

    float zero_point = -min_val / scale + qmin;
    zero_point = clamp(zero_point, qmin, qmax);

    params[0] = scale;
    params[1] = zero_point;
}

kernel void quantize_int8(
    constant float *input [[buffer(0)]], device char *output [[buffer(1)]], constant float &scale [[buffer(2)]],
    constant float &zero_point [[buffer(3)]], constant int &qmin [[buffer(4)]], constant int &qmax [[buffer(5)]],
    constant uint &num_elements [[buffer(6)]], uint id [[thread_position_in_grid]]
) {
    if (id < num_elements) {
        float scaled = input[id] / scale + zero_point;
        int q = int(round(scaled));
        q = clamp(q, qmin, qmax);
        output[id] = char(q);
    }
}

kernel void dequantize_int8(
    constant char *input [[buffer(0)]], device float *output [[buffer(1)]], constant float &scale [[buffer(2)]],
    constant float &zero_point [[buffer(3)]], constant uint &num_elements [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < num_elements) {
        output[id] = (float(input[id]) - zero_point) * scale;
    }
}

kernel void quantize_uint8(
    constant float *input [[buffer(0)]], device uchar *output [[buffer(1)]], constant float &scale [[buffer(2)]],
    constant float &zero_point [[buffer(3)]], constant int &qmin [[buffer(4)]], constant int &qmax [[buffer(5)]],
    constant uint &num_elements [[buffer(6)]], uint id [[thread_position_in_grid]]
) {
    if (id < num_elements) {
        float scaled = input[id] / scale + zero_point;
        int q = int(round(scaled));
        q = clamp(q, qmin, qmax);
        output[id] = uchar(q);
    }
}

kernel void dequantize_uint8(
    constant uchar *input [[buffer(0)]], device float *output [[buffer(1)]], constant float &scale [[buffer(2)]],
    constant float &zero_point [[buffer(3)]], constant uint &num_elements [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < num_elements) {
        output[id] = (float(input[id]) - zero_point) * scale;
    }
}

// -----------------------------------------------------------------------------
// Q8_1 quantization / dequantization
// -----------------------------------------------------------------------------

kernel void quantize_q8_1(
    constant float *input [[buffer(0)]], device int8_t *output [[buffer(1)]], constant uint &num_rows [[buffer(2)]],
    constant uint &row_size [[buffer(3)]], uint block_id [[thread_position_in_grid]]
) {
    const uint block_size = 32;
    const uint block_stride = 36;

    const uint blocks_per_row = (row_size + block_size - 1) / block_size;
    const uint row_idx = block_id / blocks_per_row;
    const uint block_idx_in_row = block_id % blocks_per_row;

    if (row_idx >= num_rows) {
        return;
    }

    const uint row_start = row_idx * row_size;
    const uint block_start = row_start + block_idx_in_row * block_size;
    const uint block_end = min(block_start + block_size, row_start + row_size);
    const uint block_len = block_end - block_start;

    float max_abs = 0.0f;
    for (uint i = block_start; i < block_end; ++i) {
        max_abs = fmax(max_abs, fabs(input[i]));
    }
    float scale = max_abs / 127.0f;
    if (scale < 1e-8f) {
        scale = 1.0f;
    }

    device q8_1_block *block = reinterpret_cast<device q8_1_block *>(output + block_id * block_stride);
    block->scale = half(scale);
    block->sum = half(0.0f);
    for (uint j = 0; j < block_size; ++j) {
        block->qs[j] = 0;
    }

    int sum_q = 0;
    for (uint i = 0; i < block_len; ++i) {
        float value = input[block_start + i] / scale;
        int q = int(round(value));
        q = clamp(q, -127, 127);
        block->qs[i] = char(q);
        sum_q += q;
    }
    block->sum = half(float(sum_q) * scale);
}

kernel void dequantize_q8_1(
    constant int8_t *input [[buffer(0)]], device float *output [[buffer(1)]], constant uint &num_rows [[buffer(2)]],
    constant uint &row_size [[buffer(3)]], uint block_id [[thread_position_in_grid]]
) {
    const uint block_size = 32;
    const uint block_stride = 36;

    const uint blocks_per_row = (row_size + block_size - 1) / block_size;
    const uint row_idx = block_id / blocks_per_row;
    const uint block_idx_in_row = block_id % blocks_per_row;

    if (row_idx >= num_rows) {
        return;
    }

    const uint row_start = row_idx * row_size;
    const uint block_start = row_start + block_idx_in_row * block_size;
    const uint block_end = min(block_start + block_size, row_start + row_size);
    const uint block_len = block_end - block_start;

    constant q8_1_block *block = reinterpret_cast<constant q8_1_block *>(input + block_id * block_stride);
    const float scale = float(block->scale);
    (void)block->sum;

    for (uint i = 0; i < block_len; ++i) {
        output[block_start + i] = scale * float(block->qs[i]);
    }
}

inline uchar2 get_scale_min_k4_helper(int idx, constant uchar *scales) {
    uchar d = 0;
    uchar m = 0;
    if (idx < 4) {
        d = scales[idx] & 63;
        m = scales[idx + 4] & 63;
    } else {
        d = (scales[idx + 4] & 0xF) | ((scales[idx - 4] >> 6) << 4);
        m = (scales[idx + 4] >> 4) | ((scales[idx] >> 6) << 4);
    }
    return uchar2(d, m);
}

inline int decode_q3_scale(int idx, constant uchar *scales) {
    int base = 0;
    if (idx < 8) {
        base = int(scales[idx] & 0xF);
    } else {
        base = int((scales[idx - 8] >> 4) & 0xF);
    }
    int high = int((scales[8 + (idx % 4)] >> (2 * (idx / 4))) & 0x3);
    return (base | (high << 4)) - 32;
}

// -----------------------------------------------------------------------------
// K-quant (Q2_K) quantization
// -----------------------------------------------------------------------------

kernel void quantize_q2_k(
    constant float *input [[buffer(0)]], device uint8_t *output [[buffer(1)]], constant uint &num_rows [[buffer(2)]],
    constant uint &row_size [[buffer(3)]], uint block_id [[thread_position_in_grid]]
) {
    const uint block_size = kQK_K;
    const uint block_stride = 84;

    const uint blocks_per_row = (row_size + block_size - 1) / block_size;
    const uint row_idx = block_id / blocks_per_row;
    const uint block_idx_in_row = block_id % blocks_per_row;

    if (row_idx >= num_rows) {
        return;
    }

    const uint row_start = row_idx * row_size;
    const uint row_end = row_start + row_size;
    const uint block_start = row_start + block_idx_in_row * block_size;
    if (block_start >= row_end) {
        return;
    }
    device q2_k_block *block = reinterpret_cast<device q2_k_block *>(output + block_id * block_stride);

    thread float x_local[kQK_K];
    for (uint i = 0; i < block_size; ++i) {
        uint idx = block_start + i;
        x_local[i] = (idx < row_end) ? input[idx] : 0.0f;
    }

    thread uchar L[kQK_K];
    thread uchar Laux[16];
    thread float weights[16];
    thread float scales[16];
    thread float mins[16];

    const float q4scale = 15.0f;
    float max_scale = 0.0f;
    float max_min = 0.0f;

    for (uint j = 0; j < kQKGroups; ++j) {
        for (uint l = 0; l < 16; ++l) {
            weights[l] = fabs(x_local[16 * j + l]);
        }
        float min_val = 0.0f;
        float scale =
            make_qkx2_quants_metal(16, 3, &x_local[16 * j], weights, &L[16 * j], &min_val, Laux, -0.5f, 0.1f, 15, true);
        scales[j] = scale;
        mins[j] = min_val;
        max_scale = fmax(max_scale, scale);
        max_min = fmax(max_min, min_val);
    }

    thread uchar scales_tmp[16];
    for (uint j = 0; j < 16; ++j) {
        scales_tmp[j] = 0;
    }

    float d_scale = 0.0f;
    if (max_scale > 0.0f) {
        float iscale = q4scale / max_scale;
        for (uint j = 0; j < kQKGroups; ++j) {
            int l = ggml_nearest_int(iscale * scales[j]);
            l = clamp_int(l, 0, 15);
            scales_tmp[j] = uchar(l & 0xF);
        }
        d_scale = max_scale / q4scale;
    }

    float dmin_scale = 0.0f;
    if (max_min > 0.0f) {
        float iscale = q4scale / max_min;
        for (uint j = 0; j < kQKGroups; ++j) {
            int l = ggml_nearest_int(iscale * mins[j]);
            l = clamp_int(l, 0, 15);
            scales_tmp[j] = uchar(scales_tmp[j] | uchar(l << 4));
        }
        dmin_scale = max_min / q4scale;
    }

    block->d = half(d_scale);
    block->dmin = half(dmin_scale);
    for (uint j = 0; j < kQKGroups; ++j) {
        block->scales[j] = scales_tmp[j];
    }

    for (uint j = 0; j < kQKGroups; ++j) {
        const float scale_q = float(scales_tmp[j] & 0xF);
        const float min_q = float(scales_tmp[j] >> 4);
        const float dl = d_scale * scale_q;
        const float ml = dmin_scale * min_q;
        if (dl == 0.0f) {
            continue;
        }
        for (uint ii = 0; ii < 16; ++ii) {
            const float value = (x_local[16 * j + ii] + ml) / dl;
            int l = ggml_nearest_int(value);
            l = clamp_int(l, 0, 3);
            L[16 * j + ii] = uchar(l);
        }
    }

    for (uint idx = 0; idx < kQK_K / 4; ++idx) {
        block->qs[idx] = 0;
    }
    for (uint j = 0; j < kQK_K; j += 128) {
        for (uint l = 0; l < 32; ++l) {
            uint idx = j / 4 + l;
            uchar packed = (L[j + l] & 0x3) | ((L[j + l + 32] & 0x3) << 2) | ((L[j + l + 64] & 0x3) << 4) |
                ((L[j + l + 96] & 0x3) << 6);
            block->qs[idx] = packed;
        }
    }
}

// -----------------------------------------------------------------------------
// K-quant (Q2_K) dequantization
// -----------------------------------------------------------------------------

kernel void dequantize_q2_k(
    constant uint8_t *input [[buffer(0)]], device float *output [[buffer(1)]], constant uint &num_rows [[buffer(2)]],
    constant uint &row_size [[buffer(3)]], uint block_id [[thread_position_in_grid]]
) {
    const uint block_size = kQK_K;
    const uint block_stride = 84;

    const uint blocks_per_row = (row_size + block_size - 1u) / block_size;
    const uint row_idx = block_id / blocks_per_row;
    const uint block_idx_in_row = block_id % blocks_per_row;
    if (row_idx >= num_rows) {
        return;
    }

    const uint row_start = row_idx * row_size;
    const uint row_end = row_start + row_size;
    const uint block_start = row_start + block_idx_in_row * block_size;
    if (block_start >= row_end) {
        return;
    }
    const uint block_len = min(block_start + block_size, row_end) - block_start;

    constant q2_k_block *block = reinterpret_cast<constant q2_k_block *>(input + block_id * block_stride);
    constant uchar *scales = block->scales;
    constant uchar *qs = block->qs;

    float d_scale = float(block->d);
    float d_min_scale = float(block->dmin);

    uint produced = 0;
    uint scale_index = 0;
    for (uint chunk = 0; chunk < block_size && produced < block_len; chunk += 128u) {
        constant uchar *q_chunk = qs + (chunk / 4u);
        int shift = 0;
        for (uint group = 0; group < 4u && produced < block_len; ++group) {
            uchar sc = scales[scale_index++];
            float dl = d_scale * float(sc & 0xF);
            float ml = d_min_scale * float(sc >> 4);
            for (uint l = 0; l < 16u && produced < block_len; ++l) {
                float qv = float((q_chunk[l] >> shift) & 0x3);
                output[block_start + produced++] = dl * qv - ml;
            }
            if (produced >= block_len) {
                break;
            }
            sc = scales[scale_index++];
            dl = d_scale * float(sc & 0xF);
            ml = d_min_scale * float(sc >> 4);
            for (uint l = 0; l < 16u && produced < block_len; ++l) {
                float qv = float((q_chunk[l + 16u] >> shift) & 0x3);
                output[block_start + produced++] = dl * qv - ml;
            }
            shift += 2;
        }
    }
}

// -----------------------------------------------------------------------------
// K-quant (Q3_K) quantization
// -----------------------------------------------------------------------------

kernel void quantize_q3_k(
    constant float *input [[buffer(0)]], device uint8_t *output [[buffer(1)]], constant uint &num_rows [[buffer(2)]],
    constant uint &row_size [[buffer(3)]], uint block_id [[thread_position_in_grid]]
) {
    const uint block_size = kQK_K;
    const uint block_stride = 110;

    const uint blocks_per_row = (row_size + block_size - 1) / block_size;
    const uint row_idx = block_id / blocks_per_row;
    const uint block_idx_in_row = block_id % blocks_per_row;

    if (row_idx >= num_rows) {
        return;
    }

    const uint row_start = row_idx * row_size;
    const uint row_end = row_start + row_size;
    const uint block_start = row_start + block_idx_in_row * block_size;
    if (block_start >= row_end) {
        return;
    }
    const uint block_len = min(block_start + block_size, row_end) - block_start;

    device q3_k_block *block = reinterpret_cast<device q3_k_block *>(output + block_id * block_stride);

    for (uint i = 0; i < kQK_K / 8u; ++i) {
        block->hmask[i] = 0;
    }
    for (uint i = 0; i < kQK_K / 4u; ++i) {
        block->qs[i] = 0;
    }
    for (uint i = 0; i < kQK_K_ScaleBytes; ++i) {
        block->scales[i] = 0;
    }
    block->d = half(0.0f);

    thread float x_local[kQK_K];
    for (uint i = 0; i < block_size; ++i) {
        uint idx = block_start + i;
        x_local[i] = (idx < row_end) ? input[idx] : 0.0f;
    }

    thread char L[kQK_K];
    thread uchar Laux[32];
    (void)Laux; // Silence unused warning (kept for future parity helpers)
    thread float scales[kQKGroups];

    float max_scale = 0.0f;
    float abs_max_scale = 0.0f;
    for (uint j = 0; j < kQKGroups; ++j) {
        float scale = make_q3_quants_metal(16, 4, &x_local[16 * j], &L[16 * j], true);
        scales[j] = scale;
        float abs_scale = fabs(scale);
        if (abs_scale > abs_max_scale) {
            abs_max_scale = abs_scale;
            max_scale = scale;
        }
    }

    if (max_scale != 0.0f) {
        float iscale = -32.0f / max_scale;
        for (uint j = 0; j < kQKGroups; ++j) {
            int l = ggml_nearest_int(iscale * scales[j]);
            l = clamp_int(l, -32, 31);
            int shifted = l + 32;
            if (j < 8) {
                block->scales[j] = uchar(shifted & 0xF);
            } else {
                block->scales[j - 8] |= uchar((shifted & 0xF) << 4);
            }
            int high = shifted >> 4;
            block->scales[8 + (j % 4)] |= uchar(high << (2 * (j / 4)));
        }
        block->d = half(1.0f / iscale);
    }

    const float d_scale = float(block->d);
    for (uint j = 0; j < kQKGroups; ++j) {
        int sc = 0;
        if (j < 8) {
            sc = block->scales[j] & 0xF;
        } else {
            sc = (block->scales[j - 8] >> 4) & 0xF;
        }
        int high = (block->scales[8 + (j % 4)] >> (2 * (j / 4))) & 0x3;
        sc |= high << 4;
        sc -= 32;
        float d = d_scale * float(sc);
        if (d == 0.0f) {
            continue;
        }
        for (uint ii = 0; ii < 16; ++ii) {
            int l = ggml_nearest_int(x_local[16 * j + ii] / d);
            l = clamp_int(l, -4, 3);
            L[16 * j + ii] = char(l + 4);
        }
    }

    int m = 0;
    uchar mask = 1;
    for (uint j = 0; j < block_size; ++j) {
        if (L[j] > 3) {
            block->hmask[m] |= mask;
            L[j] = char(L[j] - 4);
        }
        m++;
        if (m == int(kQK_K / 8u)) {
            m = 0;
            mask <<= 1;
        }
    }

    for (uint j = 0; j < kQK_K; j += 128u) {
        for (uint l = 0; l < 32u; ++l) {
            uint idx = j / 4u + l;
            block->qs[idx] = uchar(
                (L[j + l] & 0x3) | ((L[j + l + 32] & 0x3) << 2) | ((L[j + l + 64] & 0x3) << 4) |
                ((L[j + l + 96] & 0x3) << 6)
            );
        }
    }

    if (block_len < block_size) {
        const uint q_start = block_len / 4u;
        for (uint i = q_start; i < kQK_K / 4u; ++i) {
            block->qs[i] = 0;
        }
        const uint h_start = block_len / 8u;
        for (uint i = h_start; i < kQK_K / 8u; ++i) {
            block->hmask[i] = 0;
        }
    }
}

// -----------------------------------------------------------------------------
// K-quant (Q3_K) dequantization
// -----------------------------------------------------------------------------

kernel void dequantize_q3_k(
    constant uint8_t *input [[buffer(0)]], device float *output [[buffer(1)]], constant uint &num_rows [[buffer(2)]],
    constant uint &row_size [[buffer(3)]], uint block_id [[thread_position_in_grid]]
) {
    const uint block_size = kQK_K;
    const uint block_stride = 110;

    const uint blocks_per_row = (row_size + block_size - 1u) / block_size;
    const uint row_idx = block_id / blocks_per_row;
    const uint block_idx_in_row = block_id % blocks_per_row;
    if (row_idx >= num_rows) {
        return;
    }

    const uint row_start = row_idx * row_size;
    const uint row_end = row_start + row_size;
    const uint block_start = row_start + block_idx_in_row * block_size;
    if (block_start >= row_end) {
        return;
    }
    const uint block_len = min(block_start + block_size, row_end) - block_start;

    constant q3_k_block *block = reinterpret_cast<constant q3_k_block *>(input + block_id * block_stride);
    constant uchar *qs = block->qs;
    constant uchar *hm = block->hmask;
    constant uchar *scales = block->scales;
    float d_all = float(block->d);

    uint produced = 0;
    uint scale_index = 0;
    uchar mask = 1;

    for (uint chunk = 0; chunk < block_size && produced < block_len; chunk += 128u) {
        constant uchar *q_chunk = qs + (chunk / 4u);
        int shift = 0;
        for (uint group = 0; group < 4u && produced < block_len; ++group) {
            int sc = decode_q3_scale(scale_index++, scales);
            float dl = d_all * float(sc);
            for (uint l = 0; l < 16u && produced < block_len; ++l) {
                int value = int((q_chunk[l] >> shift) & 0x3);
                if ((hm[l] & mask) == 0) {
                    value -= 4;
                }
                output[block_start + produced++] = dl * float(value);
            }
            if (produced >= block_len) {
                break;
            }
            sc = decode_q3_scale(scale_index++, scales);
            dl = d_all * float(sc);
            for (uint l = 0; l < 16u && produced < block_len; ++l) {
                int value = int((q_chunk[l + 16u] >> shift) & 0x3);
                if ((hm[l + 16u] & mask) == 0) {
                    value -= 4;
                }
                output[block_start + produced++] = dl * float(value);
            }
            shift += 2;
            mask <<= 1;
        }
    }
}

// -----------------------------------------------------------------------------
// K-quant (Q4_K) quantization
// -----------------------------------------------------------------------------

kernel void quantize_q4_k(
    constant float *input [[buffer(0)]], device uint8_t *output [[buffer(1)]], constant uint &num_rows [[buffer(2)]],
    constant uint &row_size [[buffer(3)]], uint block_id [[thread_position_in_grid]]
) {
    const uint block_size = kQK_K;
    const uint block_stride = 144;

    const uint blocks_per_row = (row_size + block_size - 1) / block_size;
    const uint row_idx = block_id / blocks_per_row;
    const uint block_idx_in_row = block_id % blocks_per_row;

    if (row_idx >= num_rows) {
        return;
    }

    const uint row_start = row_idx * row_size;
    const uint row_end = row_start + row_size;
    const uint block_start = row_start + block_idx_in_row * block_size;
    if (block_start >= row_end) {
        return;
    }
    const uint block_len = min(block_start + block_size, row_end) - block_start;

    device q4_k_block *block = reinterpret_cast<device q4_k_block *>(output + block_id * block_stride);

    for (uint i = 0; i < kQK_K / 2u; ++i) {
        block->qs[i] = 0;
    }
    for (uint i = 0; i < kQK_K_ScaleBytes; ++i) {
        block->scales[i] = 0;
    }
    block->d = half(0.0f);
    block->dmin = half(0.0f);

    thread float x_local[kQK_K];
    for (uint i = 0; i < block_size; ++i) {
        uint idx = block_start + i;
        x_local[i] = (idx < row_end) ? input[idx] : 0.0f;
    }

    thread uchar L[kQK_K];
    thread uchar Laux[32];
    thread float weights[32];
    thread float scales[kQK32Groups];
    thread float mins[kQK32Groups];

    float max_scale = 0.0f;
    float max_min = 0.0f;

    for (uint j = 0; j < kQK32Groups; ++j) {
        float sum_x2 = 0.0f;
        for (uint l = 0; l < 32; ++l) {
            float val = x_local[32 * j + l];
            sum_x2 += val * val;
        }
        float av_x = sqrt(sum_x2 * (1.0f / 32.0f));
        for (uint l = 0; l < 32; ++l) {
            weights[l] = av_x + fabs(x_local[32 * j + l]);
        }
        float min_val = 0.0f;
        float scale = make_qkx2_quants_metal(
            32, 15, &x_local[32 * j], weights, &L[32 * j], &min_val, Laux, -1.0f, 0.1f, 20, false
        );
        scales[j] = scale;
        mins[j] = min_val;
        if (scale > max_scale) {
            max_scale = scale;
        }
        if (min_val > max_min) {
            max_min = min_val;
        }
    }

    float inv_scale = max_scale > 0.0f ? 63.0f / max_scale : 0.0f;
    float inv_min = max_min > 0.0f ? 63.0f / max_min : 0.0f;

    for (uint j = 0; j < kQK32Groups; ++j) {
        uint8_t ls = (uint8_t)ggml_nearest_int(inv_scale * scales[j]);
        uint8_t lm = (uint8_t)ggml_nearest_int(inv_min * mins[j]);
        ls = ls > 63 ? 63 : ls;
        lm = lm > 63 ? 63 : lm;
        if (j < 4) {
            block->scales[j] = uchar(ls & 0x3F);
            block->scales[j + 4] = uchar(lm & 0x3F);
        } else {
            block->scales[j + 4] = uchar((ls & 0xF) | ((lm & 0xF) << 4));
            block->scales[j - 4] = uchar(block->scales[j - 4] | ((ls >> 4) << 6));
            block->scales[j] = uchar(block->scales[j] | ((lm >> 4) << 6));
        }
    }

    block->d = half(max_scale / 63.0f);
    block->dmin = half(max_min / 63.0f);

    const float d_scale = float(block->d);
    const float d_min_scale = float(block->dmin);

    for (uint j = 0; j < kQK32Groups; ++j) {
        uint8_t sc = 0;
        uint8_t m = 0;
        if (j < 4) {
            sc = block->scales[j] & 0x3F;
            m = block->scales[j + 4] & 0x3F;
        } else {
            sc = (block->scales[j + 4] & 0xF) | ((block->scales[j - 4] >> 6) << 4);
            m = (block->scales[j + 4] >> 4) | ((block->scales[j] >> 6) << 4);
        }
        float d = d_scale * float(sc);
        if (d == 0.0f) {
            continue;
        }
        float dm = d_min_scale * float(m);
        for (uint ii = 0; ii < 32; ++ii) {
            float value = (x_local[32 * j + ii] + dm) / d;
            int l = ggml_nearest_int(value);
            l = clamp_int(l, 0, 15);
            L[32 * j + ii] = uchar(l);
        }
    }

    for (uint j = 0; j < kQK_K; j += 64u) {
        for (uint l = 0; l < 32u; ++l) {
            uint idx = (j / 2u) + l;
            block->qs[idx] = uchar((L[j + l] & 0xF) | ((L[j + l + 32] & 0xF) << 4));
        }
    }

    if (block_len < block_size) {
        const uint q_start = block_len / 2u;
        for (uint i = q_start; i < kQK_K / 2u; ++i) {
            block->qs[i] = 0;
        }
    }
}

// -----------------------------------------------------------------------------
// K-quant (Q4_K) dequantization
// -----------------------------------------------------------------------------

kernel void dequantize_q4_k(
    constant uint8_t *input [[buffer(0)]], device float *output [[buffer(1)]], constant uint &num_rows [[buffer(2)]],
    constant uint &row_size [[buffer(3)]], uint block_id [[thread_position_in_grid]]
) {
    const uint block_size = kQK_K;
    const uint block_stride = 144;

    const uint blocks_per_row = (row_size + block_size - 1u) / block_size;
    const uint row_idx = block_id / blocks_per_row;
    const uint block_idx_in_row = block_id % blocks_per_row;
    if (row_idx >= num_rows) {
        return;
    }

    const uint row_start = row_idx * row_size;
    const uint row_end = row_start + row_size;
    const uint block_start = row_start + block_idx_in_row * block_size;
    if (block_start >= row_end) {
        return;
    }
    const uint block_len = min(block_start + block_size, row_end) - block_start;

    constant q4_k_block *block = reinterpret_cast<constant q4_k_block *>(input + block_id * block_stride);
    constant uchar *scales = block->scales;
    constant uchar *qs = block->qs;
    float d_scale = float(block->d);
    float d_min_scale = float(block->dmin);

    uint produced = 0;
    int scale_index = 0;
    for (uint chunk = 0; chunk < block_size && produced < block_len; chunk += 64u) {
        constant uchar *q_chunk = qs + (chunk / 2u);
        uchar2 sm1 = get_scale_min_k4_helper(scale_index++, scales);
        uchar2 sm2 = get_scale_min_k4_helper(scale_index++, scales);
        float d1 = d_scale * float(sm1.x);
        float m1 = d_min_scale * float(sm1.y);
        float d2 = d_scale * float(sm2.x);
        float m2 = d_min_scale * float(sm2.y);
        for (uint l = 0; l < 32u && produced < block_len; ++l) {
            float qv = float(q_chunk[l] & 0xF);
            output[block_start + produced++] = d1 * qv - m1;
        }
        for (uint l = 0; l < 32u && produced < block_len; ++l) {
            float qv = float((q_chunk[l] >> 4) & 0xF);
            output[block_start + produced++] = d2 * qv - m2;
        }
    }
}

// -----------------------------------------------------------------------------
// K-quant (Q5_K) quantization
// -----------------------------------------------------------------------------

kernel void quantize_q5_k(
    constant float *input [[buffer(0)]], device uint8_t *output [[buffer(1)]], constant uint &num_rows [[buffer(2)]],
    constant uint &row_size [[buffer(3)]], uint block_id [[thread_position_in_grid]]
) {
    const uint block_size = kQK_K;
    const uint block_stride = 176;

    const uint blocks_per_row = (row_size + block_size - 1) / block_size;
    const uint row_idx = block_id / blocks_per_row;
    const uint block_idx_in_row = block_id % blocks_per_row;

    if (row_idx >= num_rows) {
        return;
    }

    const uint row_start = row_idx * row_size;
    const uint row_end = row_start + row_size;
    const uint block_start = row_start + block_idx_in_row * block_size;
    if (block_start >= row_end) {
        return;
    }
    const uint block_len = min(block_start + block_size, row_end) - block_start;

    device q5_k_block *block = reinterpret_cast<device q5_k_block *>(output + block_id * block_stride);

    for (uint i = 0; i < kQK_K / 2u; ++i) {
        block->qs[i] = 0;
    }
    for (uint i = 0; i < kQK_K / 8u; ++i) {
        block->qh[i] = 0;
    }
    for (uint i = 0; i < kQK_K_ScaleBytes; ++i) {
        block->scales[i] = 0;
    }
    block->d = half(0.0f);
    block->dmin = half(0.0f);

    thread float x_local[kQK_K];
    for (uint i = 0; i < block_size; ++i) {
        uint idx = block_start + i;
        x_local[i] = (idx < row_end) ? input[idx] : 0.0f;
    }

    thread uchar L[kQK_K];
    thread uchar Laux[32];
    thread float weights[32];
    thread float scales[kQK32Groups];
    thread float mins[kQK32Groups];

    float max_scale = 0.0f;
    float max_min = 0.0f;

    for (uint j = 0; j < kQK32Groups; ++j) {
        float sum_x2 = 0.0f;
        for (uint l = 0; l < 32; ++l) {
            float val = x_local[32 * j + l];
            sum_x2 += val * val;
        }
        float av_x = sqrt(sum_x2 * (1.0f / 32.0f));
        for (uint l = 0; l < 32; ++l) {
            weights[l] = av_x + fabs(x_local[32 * j + l]);
        }
        float min_val = 0.0f;
        float scale = make_qkx2_quants_metal(
            32, 31, &x_local[32 * j], weights, &L[32 * j], &min_val, Laux, -0.5f, 0.1f, 15, false
        );
        scales[j] = scale;
        mins[j] = min_val;
        if (scale > max_scale) {
            max_scale = scale;
        }
        if (min_val > max_min) {
            max_min = min_val;
        }
    }

    float inv_scale = max_scale > 0.0f ? 63.0f / max_scale : 0.0f;
    float inv_min = max_min > 0.0f ? 63.0f / max_min : 0.0f;

    for (uint j = 0; j < kQK32Groups; ++j) {
        uint8_t ls = (uint8_t)ggml_nearest_int(inv_scale * scales[j]);
        uint8_t lm = (uint8_t)ggml_nearest_int(inv_min * mins[j]);
        ls = ls > 63 ? 63 : ls;
        lm = lm > 63 ? 63 : lm;
        if (j < 4) {
            block->scales[j] = uchar(ls & 0x3F);
            block->scales[j + 4] = uchar(lm & 0x3F);
        } else {
            block->scales[j + 4] = uchar((ls & 0xF) | ((lm & 0xF) << 4));
            block->scales[j - 4] = uchar(block->scales[j - 4] | ((ls >> 4) << 6));
            block->scales[j] = uchar(block->scales[j] | ((lm >> 4) << 6));
        }
    }

    block->d = half(max_scale / 63.0f);
    block->dmin = half(max_min / 63.0f);

    const float d_scale = float(block->d);
    const float d_min_scale = float(block->dmin);

    for (uint j = 0; j < kQK32Groups; ++j) {
        uint8_t sc = 0;
        uint8_t m = 0;
        if (j < 4) {
            sc = block->scales[j] & 0x3F;
            m = block->scales[j + 4] & 0x3F;
        } else {
            sc = (block->scales[j + 4] & 0xF) | ((block->scales[j - 4] >> 6) << 4);
            m = (block->scales[j + 4] >> 4) | ((block->scales[j] >> 6) << 4);
        }
        float d = d_scale * float(sc);
        if (d == 0.0f) {
            continue;
        }
        float dm = d_min_scale * float(m);
        for (uint ii = 0; ii < 32; ++ii) {
            float value = (x_local[32 * j + ii] + dm) / d;
            int l = ggml_nearest_int(value);
            l = clamp_int(l, 0, 31);
            L[32 * j + ii] = uchar(l);
        }
    }

    uint8_t mask_low = 1;
    uint8_t mask_high = 2;
    for (uint n = 0; n < kQK_K; n += 64u) {
        uint base_qs = (n / 2u);
        for (uint j = 0; j < 32u; ++j) {
            int l1 = int(L[n + j]);
            if (l1 > 15) {
                l1 -= 16;
                block->qh[j] |= mask_low;
            }
            int l2 = int(L[n + j + 32u]);
            if (l2 > 15) {
                l2 -= 16;
                block->qh[j] |= mask_high;
            }
            block->qs[base_qs + j] = uchar((l1 & 0xF) | ((l2 & 0xF) << 4));
        }
        mask_low <<= 2;
        mask_high <<= 2;
    }

    if (block_len < block_size) {
        const uint q_start = block_len / 2u;
        for (uint i = q_start; i < kQK_K / 2u; ++i) {
            block->qs[i] = 0;
        }
        const uint h_start = block_len / 8u;
        for (uint i = h_start; i < kQK_K / 8u; ++i) {
            block->qh[i] = 0;
        }
    }
}

// -----------------------------------------------------------------------------
// K-quant (Q5_K) dequantization
// -----------------------------------------------------------------------------

kernel void dequantize_q5_k(
    constant uint8_t *input [[buffer(0)]], device float *output [[buffer(1)]], constant uint &num_rows [[buffer(2)]],
    constant uint &row_size [[buffer(3)]], uint block_id [[thread_position_in_grid]]
) {
    const uint block_size = kQK_K;
    const uint block_stride = 176;

    const uint blocks_per_row = (row_size + block_size - 1u) / block_size;
    const uint row_idx = block_id / blocks_per_row;
    const uint block_idx_in_row = block_id % blocks_per_row;
    if (row_idx >= num_rows) {
        return;
    }

    const uint row_start = row_idx * row_size;
    const uint row_end = row_start + row_size;
    const uint block_start = row_start + block_idx_in_row * block_size;
    if (block_start >= row_end) {
        return;
    }
    const uint block_len = min(block_start + block_size, row_end) - block_start;

    constant q5_k_block *block = reinterpret_cast<constant q5_k_block *>(input + block_id * block_stride);
    constant uchar *scales = block->scales;
    constant uchar *ql = block->qs;
    constant uchar *qh = block->qh;
    float d_scale = float(block->d);
    float d_min_scale = float(block->dmin);

    uint produced = 0;
    int scale_index = 0;
    uchar mask1 = 1;
    uchar mask2 = 2;

    for (uint chunk = 0; chunk < block_size && produced < block_len; chunk += 64u) {
        constant uchar *ql_chunk = ql + (chunk / 2u);
        uchar2 sm1 = get_scale_min_k4_helper(scale_index++, scales);
        uchar2 sm2 = get_scale_min_k4_helper(scale_index++, scales);
        float d1 = d_scale * float(sm1.x);
        float m1 = d_min_scale * float(sm1.y);
        float d2 = d_scale * float(sm2.x);
        float m2 = d_min_scale * float(sm2.y);

        for (uint l = 0; l < 32u && produced < block_len; ++l) {
            uint8_t base = ql_chunk[l] & 0xF;
            if (qh[l] & mask1) {
                base |= 16u;
            }
            output[block_start + produced++] = d1 * float(base) - m1;
        }
        for (uint l = 0; l < 32u && produced < block_len; ++l) {
            uint8_t base = (ql_chunk[l] >> 4) & 0xF;
            if (qh[l] & mask2) {
                base |= 16u;
            }
            output[block_start + produced++] = d2 * float(base) - m2;
        }

        mask1 <<= 2;
        mask2 <<= 2;
    }
}

// -----------------------------------------------------------------------------
// K-quant (Q6_K) quantization
// -----------------------------------------------------------------------------

kernel void quantize_q6_k(
    constant float *input [[buffer(0)]], device uint8_t *output [[buffer(1)]], constant uint &num_rows [[buffer(2)]],
    constant uint &row_size [[buffer(3)]], uint block_id [[thread_position_in_grid]]
) {
    const uint block_size = kQK_K;
    const uint block_stride = 210;

    const uint blocks_per_row = (row_size + block_size - 1) / block_size;
    const uint row_idx = block_id / blocks_per_row;
    const uint block_idx_in_row = block_id % blocks_per_row;

    if (row_idx >= num_rows) {
        return;
    }

    const uint row_start = row_idx * row_size;
    const uint row_end = row_start + row_size;
    const uint block_start = row_start + block_idx_in_row * block_size;
    if (block_start >= row_end) {
        return;
    }
    const uint block_len = min(block_start + block_size, row_end) - block_start;

    device q6_k_block *block = reinterpret_cast<device q6_k_block *>(output + block_id * block_stride);

    for (uint i = 0; i < kQK_K / 2u; ++i) {
        block->ql[i] = 0;
    }
    for (uint i = 0; i < kQK_K / 4u; ++i) {
        block->qh[i] = 0;
    }
    for (uint i = 0; i < kQK_K / 16u; ++i) {
        block->scales[i] = 0;
    }
    block->d = half(0.0f);

    thread float x_local[kQK_K];
    for (uint i = 0; i < block_size; ++i) {
        uint idx = block_start + i;
        x_local[i] = (idx < row_end) ? input[idx] : 0.0f;
    }

    thread char L[kQK_K];
    thread float scales[kQK_K / 16u];

    float max_scale = 0.0f;
    float max_abs_scale = 0.0f;

    for (uint ib = 0; ib < kQK_K / 16u; ++ib) {
        float scale = make_qx_quants_metal(16, 32, &x_local[16 * ib], &L[16 * ib]);
        scales[ib] = scale;
        float abs_scale = fabs(scale);
        if (abs_scale > max_abs_scale) {
            max_abs_scale = abs_scale;
            max_scale = scale;
        }
    }

    if (max_abs_scale < kGroupMaxEps) {
        return;
    }

    float iscale = -128.0f / max_scale;
    block->d = half(1.0f / iscale);

    for (uint ib = 0; ib < kQK_K / 16u; ++ib) {
        int sc = ggml_nearest_int(iscale * scales[ib]);
        if (sc > 127) {
            sc = 127;
        } else if (sc < -128) {
            sc = -128;
        }
        block->scales[ib] = char(sc);
    }

    const float d_base = float(block->d);
    for (uint ib = 0; ib < kQK_K / 16u; ++ib) {
        float d = d_base * float(block->scales[ib]);
        if (d == 0.0f) {
            continue;
        }
        for (uint ii = 0; ii < 16u; ++ii) {
            float value = x_local[16 * ib + ii] / d;
            int l = ggml_nearest_int(value);
            l = clamp_qx_value_metal(l, 32);
            L[16 * ib + ii] = char(l + 32);
        }
    }

    uint ql_idx = 0;
    uint qh_idx = 0;
    for (uint n = 0; n < kQK_K; n += 128u) {
        for (uint j = 0; j < 32u; ++j) {
            uchar v0 = uchar(L[n + j]);
            uchar v1 = uchar(L[n + j + 32u]);
            uchar v2 = uchar(L[n + j + 64u]);
            uchar v3 = uchar(L[n + j + 96u]);

            block->ql[ql_idx + j] = uchar((v0 & 0xF) | ((v2 & 0xF) << 4));
            block->ql[ql_idx + j + 32u] = uchar((v1 & 0xF) | ((v3 & 0xF) << 4));

            uchar high_bits = uchar((v0 >> 4) | ((v1 >> 4) << 2) | ((v2 >> 4) << 4) | ((v3 >> 4) << 6));
            block->qh[qh_idx + j] = high_bits;
        }
        ql_idx += 64u;
        qh_idx += 32u;
    }

    if (block_len < block_size) {
        const uint ql_start = block_len / 2u;
        for (uint i = ql_start; i < kQK_K / 2u; ++i) {
            block->ql[i] = 0;
        }
        const uint qh_start = block_len / 4u;
        for (uint i = qh_start; i < kQK_K / 4u; ++i) {
            block->qh[i] = 0;
        }
    }
}

// -----------------------------------------------------------------------------
// K-quant (Q6_K) dequantization
// -----------------------------------------------------------------------------

kernel void dequantize_q6_k(
    constant uint8_t *input [[buffer(0)]], device float *output [[buffer(1)]], constant uint &num_rows [[buffer(2)]],
    constant uint &row_size [[buffer(3)]], uint block_id [[thread_position_in_grid]]
) {
    const uint block_size = kQK_K;
    const uint block_stride = 210;

    const uint blocks_per_row = (row_size + block_size - 1) / block_size;
    const uint row_idx = block_id / blocks_per_row;
    const uint block_idx_in_row = block_id % blocks_per_row;

    if (row_idx >= num_rows) {
        return;
    }

    const uint row_start = row_idx * row_size;
    const uint row_end = row_start + row_size;
    const uint block_start = row_start + block_idx_in_row * block_size;
    if (block_start >= row_end) {
        return;
    }
    const uint block_len = min(block_start + block_size, row_end) - block_start;

    constant q6_k_block *block = reinterpret_cast<constant q6_k_block *>(input + block_id * block_stride);

    float d = float(block->d);
    constant uchar *ql = block->ql;
    constant uchar *qh = block->qh;
    constant char *sc = block->scales;

    for (uint n = 0; n < block_size && n < block_len; n += 128u) {
        for (uint j = 0; j < 32u; ++j) {
            uint idx0 = j;
            uint idx1 = j + 32u;
            uchar ql0 = ql[idx0];
            uchar ql1 = ql[idx1];
            uchar qh_byte = qh[j];

            int q1 = int((ql0 & 0xF) | ((qh_byte & 0x3) << 4)) - 32;
            int q2 = int((ql1 & 0xF) | (((qh_byte >> 2) & 0x3) << 4)) - 32;
            int q3 = int(((ql0 >> 4) & 0xF) | (((qh_byte >> 4) & 0x3) << 4)) - 32;
            int q4 = int(((ql1 >> 4) & 0xF) | (((qh_byte >> 6) & 0x3) << 4)) - 32;

            uint is = j / 16u;
            float s0 = float(sc[is + 0]);
            float s1 = float(sc[is + 2]);
            float s2 = float(sc[is + 4]);
            float s3 = float(sc[is + 6]);

            uint out_idx = block_start + n + j;
            if (out_idx < row_end && out_idx < block_start + block_len) {
                output[out_idx] = d * s0 * float(q1);
            }
            if (out_idx + 32u < row_end && out_idx + 32u < block_start + block_len) {
                output[out_idx + 32u] = d * s1 * float(q2);
            }
            if (out_idx + 64u < row_end && out_idx + 64u < block_start + block_len) {
                output[out_idx + 64u] = d * s2 * float(q3);
            }
            if (out_idx + 96u < row_end && out_idx + 96u < block_start + block_len) {
                output[out_idx + 96u] = d * s3 * float(q4);
            }
        }
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

// -----------------------------------------------------------------------------
// K-quant (Q8_K) quantization
// -----------------------------------------------------------------------------

kernel void quantize_q8_k(
    constant float *input [[buffer(0)]], device int8_t *output [[buffer(1)]], constant uint &num_rows [[buffer(2)]],
    constant uint &row_size [[buffer(3)]], uint block_id [[thread_position_in_grid]]
) {
    const uint block_size = kQK_K;
    const uint block_stride = 292;

    const uint blocks_per_row = (row_size + block_size - 1) / block_size;
    const uint row_idx = block_id / blocks_per_row;
    const uint block_idx_in_row = block_id % blocks_per_row;

    if (row_idx >= num_rows) {
        return;
    }

    const uint row_start = row_idx * row_size;
    const uint row_end = row_start + row_size;
    const uint block_start = row_start + block_idx_in_row * block_size;
    if (block_start >= row_end) {
        return;
    }
    const uint block_len = min(block_start + block_size, row_end) - block_start;

    device q8_k_block *block = reinterpret_cast<device q8_k_block *>(output + block_id * block_stride);

    for (uint i = 0; i < kQK_K; ++i) {
        block->qs[i] = 0;
    }
    for (uint i = 0; i < kQK_K / 16u; ++i) {
        block->bsums[i] = 0;
    }
    block->d = 0.0f;

    thread float x_local[kQK_K];
    for (uint i = 0; i < block_size; ++i) {
        uint idx = block_start + i;
        x_local[i] = (idx < row_end) ? input[idx] : 0.0f;
    }

    float max_val = 0.0f;
    float max_abs = 0.0f;
    for (uint i = 0; i < block_len; ++i) {
        float ax = fabs(x_local[i]);
        if (ax > max_abs) {
            max_abs = ax;
            max_val = x_local[i];
        }
    }

    if (max_abs == 0.0f) {
        return;
    }

    float iscale = -127.0f / max_val;
    block->d = -max_val / 127.0f;

    for (uint i = 0; i < block_len; ++i) {
        int v = ggml_nearest_int(iscale * x_local[i]);
        if (v > 127) {
            v = 127;
        } else if (v < -128) {
            v = -128;
        }
        block->qs[i] = char(v);
    }

    for (uint i = 0; i < kQK_K / 16u; ++i) {
        int sum = 0;
        for (uint j = 0; j < 16u; ++j) {
            sum += int(block->qs[i * 16u + j]);
        }
        block->bsums[i] = short(sum);
    }
}

// -----------------------------------------------------------------------------
// K-quant (Q8_K) dequantization
// -----------------------------------------------------------------------------

kernel void dequantize_q8_k(
    constant int8_t *input [[buffer(0)]], device float *output [[buffer(1)]], constant uint &num_rows [[buffer(2)]],
    constant uint &row_size [[buffer(3)]], uint block_id [[thread_position_in_grid]]
) {
    const uint block_size = kQK_K;
    const uint block_stride = 292;

    const uint blocks_per_row = (row_size + block_size - 1) / block_size;
    const uint row_idx = block_id / blocks_per_row;
    const uint block_idx_in_row = block_id % blocks_per_row;

    if (row_idx >= num_rows) {
        return;
    }

    const uint row_start = row_idx * row_size;
    const uint row_end = row_start + row_size;
    const uint block_start = row_start + block_idx_in_row * block_size;
    if (block_start >= row_end) {
        return;
    }
    const uint block_len = min(block_start + block_size, row_end) - block_start;

    constant q8_k_block *block = reinterpret_cast<constant q8_k_block *>(input + block_id * block_stride);

    float d = block->d;
    for (uint i = 0; i < block_len; ++i) {
        output[block_start + i] = d * float(block->qs[i]);
    }
}
