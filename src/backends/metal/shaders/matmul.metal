#include <metal_stdlib>
using namespace metal;

#include "common/defines.h"

#include "common/math_utils.h"

#include "common/dtype_utils.h"

// -----------------------------------------------------------------------------
// Tiled Matrix Multiplication Kernels
// -----------------------------------------------------------------------------
// Implements efficient blocked matrix multiplication with transpose support
// Following MLX-inspired approach: transpose = different memory access pattern

#define MARMOT_MATMUL_TILE_M 16u
#define MARMOT_MATMUL_TILE_N 16u
#define MARMOT_MATMUL_TILE_K 16u

// PyTorch Convention: input(N×K) @ weight(M×K).T = output(N×M)
// We provide two kernel variants:
// - _nt: Normal × Transposed (standard for PyTorch matmul)
// - _nn: Normal × Normal (for compatibility)

// Macro for Normal × Transposed variant (PyTorch convention)
// Computes: input(N×K) @ weight(M×K).T = output(N×M)
// Weight is read transposed: weight[m,k] accessed as weight[m*K + k] but treated as weight[k,m]
#define DEFINE_MATMUL_NT_KERNEL(NAME, READ_PTR, WRITE_PTR, READ_FN, WRITE_FN)                                          \
    kernel void NAME##_nt(                                                                                             \
        device const READ_PTR *input [[buffer(0)]], device const READ_PTR *weight [[buffer(1)]],                       \
        device WRITE_PTR *out [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],           \
        constant uint &M [[buffer(5)]], uint2 tg_pos [[threadgroup_position_in_grid]],                                 \
        uint2 tid [[thread_position_in_threadgroup]]                                                                   \
    ) {                                                                                                                \
        if (tid.x >= MARMOT_MATMUL_TILE_N || tid.y >= MARMOT_MATMUL_TILE_M) {                                          \
            return;                                                                                                    \
        }                                                                                                              \
                                                                                                                       \
        threadgroup float tileInput[MARMOT_MATMUL_TILE_M][MARMOT_MATMUL_TILE_K];                                       \
        threadgroup float tileWeight[MARMOT_MATMUL_TILE_K][MARMOT_MATMUL_TILE_N];                                      \
                                                                                                                       \
        uint n = tg_pos.y * MARMOT_MATMUL_TILE_M + tid.y;                                                              \
        uint m = tg_pos.x * MARMOT_MATMUL_TILE_N + tid.x;                                                              \
                                                                                                                       \
        float acc = 0.0f;                                                                                              \
        for (uint k0 = 0; k0 < K; k0 += MARMOT_MATMUL_TILE_K) {                                                        \
            uint k_input = k0 + tid.x;                                                                                 \
            if (n < N && k_input < K) {                                                                                \
                tileInput[tid.y][tid.x] = READ_FN(input[n * K + k_input]);                                             \
            } else {                                                                                                   \
                tileInput[tid.y][tid.x] = 0.0f;                                                                        \
            }                                                                                                          \
                                                                                                                       \
            uint k_weight = k0 + tid.y;                                                                                \
            if (k_weight < K && m < M) {                                                                               \
                tileWeight[tid.y][tid.x] = READ_FN(weight[m * K + k_weight]);                                          \
            } else {                                                                                                   \
                tileWeight[tid.y][tid.x] = 0.0f;                                                                       \
            }                                                                                                          \
                                                                                                                       \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                                           \
                                                                                                                       \
            for (uint k1 = 0; k1 < MARMOT_MATMUL_TILE_K && (k0 + k1) < K; ++k1) {                                      \
                acc += tileInput[tid.y][k1] * tileWeight[k1][tid.x];                                                   \
            }                                                                                                          \
                                                                                                                       \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                                           \
        }                                                                                                              \
                                                                                                                       \
        if (n < N && m < M) {                                                                                          \
            out[n * M + m] = WRITE_FN(acc);                                                                            \
        }                                                                                                              \
    }

// Macro for Normal × Normal variant (GGML/legacy convention)
// Computes: A(M×K) @ B(K×N) = C(M×N)
#define DEFINE_MATMUL_NN_KERNEL(NAME, READ_PTR, WRITE_PTR, READ_FN, WRITE_FN)                                          \
    kernel void NAME##_nn(                                                                                             \
        device const READ_PTR *a [[buffer(0)]], device const READ_PTR *b [[buffer(1)]],                                \
        device WRITE_PTR *out [[buffer(2)]], constant uint &M [[buffer(3)]], constant uint &K [[buffer(4)]],           \
        constant uint &N [[buffer(5)]], uint2 tg_pos [[threadgroup_position_in_grid]],                                 \
        uint2 tid [[thread_position_in_threadgroup]]                                                                   \
    ) {                                                                                                                \
        if (tid.x >= MARMOT_MATMUL_TILE_N || tid.y >= MARMOT_MATMUL_TILE_M) {                                          \
            return;                                                                                                    \
        }                                                                                                              \
                                                                                                                       \
        threadgroup float tileA[MARMOT_MATMUL_TILE_M][MARMOT_MATMUL_TILE_K];                                           \
        threadgroup float tileB[MARMOT_MATMUL_TILE_K][MARMOT_MATMUL_TILE_N];                                           \
                                                                                                                       \
        uint global_row = tg_pos.y * MARMOT_MATMUL_TILE_M + tid.y;                                                     \
        uint global_col = tg_pos.x * MARMOT_MATMUL_TILE_N + tid.x;                                                     \
                                                                                                                       \
        float acc = 0.0f;                                                                                              \
        for (uint k0 = 0; k0 < K; k0 += MARMOT_MATMUL_TILE_K) {                                                        \
            uint a_col = k0 + tid.x;                                                                                   \
            if (global_row < M && a_col < K) {                                                                         \
                tileA[tid.y][tid.x] = READ_FN(a[global_row * K + a_col]);                                              \
            } else {                                                                                                   \
                tileA[tid.y][tid.x] = 0.0f;                                                                            \
            }                                                                                                          \
                                                                                                                       \
            uint b_row = k0 + tid.y;                                                                                   \
            if (b_row < K && global_col < N) {                                                                         \
                tileB[tid.y][tid.x] = READ_FN(b[b_row * N + global_col]);                                              \
            } else {                                                                                                   \
                tileB[tid.y][tid.x] = 0.0f;                                                                            \
            }                                                                                                          \
                                                                                                                       \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                                           \
                                                                                                                       \
            for (uint k1 = 0; k1 < MARMOT_MATMUL_TILE_K && (k0 + k1) < K; ++k1) {                                      \
                acc += tileA[tid.y][k1] * tileB[k1][tid.x];                                                            \
            }                                                                                                          \
                                                                                                                       \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                                           \
        }                                                                                                              \
                                                                                                                       \
        if (global_row < M && global_col < N) {                                                                        \
            out[global_row * N + global_col] = WRITE_FN(acc);                                                          \
        }                                                                                                              \
    }

// Generate both variants for each data type
DEFINE_MATMUL_NT_KERNEL(matmul_f32, float, float, read_float, write_float)
DEFINE_MATMUL_NN_KERNEL(matmul_f32, float, float, read_float, write_float)

DEFINE_MATMUL_NT_KERNEL(matmul_f16, half, half, read_half, write_half)
DEFINE_MATMUL_NN_KERNEL(matmul_f16, half, half, read_half, write_half)

DEFINE_MATMUL_NT_KERNEL(matmul_bf16, ushort, ushort, read_bf16, write_bf16)
DEFINE_MATMUL_NN_KERNEL(matmul_bf16, ushort, ushort, read_bf16, write_bf16)

DEFINE_MATMUL_NT_KERNEL(matmul_fp8_e4m3, uchar, uchar, read_fp8_e4m3, write_fp8_e4m3)
DEFINE_MATMUL_NN_KERNEL(matmul_fp8_e4m3, uchar, uchar, read_fp8_e4m3, write_fp8_e4m3)

DEFINE_MATMUL_NT_KERNEL(matmul_fp8_e5m2, uchar, uchar, read_fp8_e5m2, write_fp8_e5m2)
DEFINE_MATMUL_NN_KERNEL(matmul_fp8_e5m2, uchar, uchar, read_fp8_e5m2, write_fp8_e5m2)

#undef DEFINE_MATMUL_NT_KERNEL
#undef DEFINE_MATMUL_NN_KERNEL
#undef MARMOT_MATMUL_TILE_M
#undef MARMOT_MATMUL_TILE_N
#undef MARMOT_MATMUL_TILE_K
