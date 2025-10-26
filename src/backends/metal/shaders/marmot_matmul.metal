#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

using namespace metal;

#include "common/math_utils.h"

#include "common/dtype_utils.h"

#define MARMOT_METAL_UNROLL _Pragma("clang loop unroll(full)")
#define MARMOT_METAL_NO_UNROLL _Pragma("clang loop unroll(disable)")
#define MARMOT_METAL_CONST static constant constexpr const

constant bool marmot_align_m [[function_constant(200)]];
constant bool marmot_align_n [[function_constant(201)]];
constant bool marmot_align_k [[function_constant(202)]];

typedef struct marmot_gemm_params {
    int M;
    int N;
    int K;
    int lda;
    int ldb;
    int ldd;
    int tiles_n;
    int tiles_m;
    int swizzle_log;
    int gemm_k_iterations_aligned;
} marmot_gemm_params_t;

template <typename T>
struct marmot_value_to_float;

template <>
struct marmot_value_to_float<float> {
    METAL_FUNC static float convert(float value) {
        return value;
    }
};

template <>
struct marmot_value_to_float<half> {
    METAL_FUNC static float convert(half value) {
        return float(value);
    }
};

template <>
struct marmot_value_to_float<ushort> {
    METAL_FUNC static float convert(ushort value) {
        return read_bf16(value);
    }
};

template <typename T>
struct marmot_value_from_float;

template <>
struct marmot_value_from_float<float> {
    METAL_FUNC static float convert(float value) {
        return value;
    }
};

template <>
struct marmot_value_from_float<half> {
    METAL_FUNC static half convert(float value) {
        return half(value);
    }
};

template <>
struct marmot_value_from_float<ushort> {
    METAL_FUNC static ushort convert(float value) {
        return write_bf16(value);
    }
};

template <typename T, short BROWS, short BCOLS, short dst_ld, bool reduction_dim, short tgp_size>
struct marmot_block_loader {
    static_assert(BROWS > 0);
    static_assert(BCOLS > 0);
    static_assert(tgp_size > 0);

    MARMOT_METAL_CONST short vec_size = (BROWS * BCOLS) / tgp_size;
    static_assert(vec_size > 0);
    static_assert((BROWS * BCOLS) % tgp_size == 0);

    MARMOT_METAL_CONST short tcols = BCOLS / vec_size;
    static_assert(tcols > 0);
    static_assert(BCOLS % vec_size == 0);

    MARMOT_METAL_CONST short trows = tgp_size / tcols;
    static_assert(trows > 0);
    static_assert(tgp_size % tcols == 0);

    const int src_ld;
    const int tile_stride;

    const short thread_idx;
    const short bi;
    const short bj;

    threadgroup T *dst;
    const device T *src;

    struct alignas(sizeof(T)) read_vec {
        uint8_t v[sizeof(T) * vec_size];
    };

    METAL_FUNC marmot_block_loader(
        const device T *src_, int src_ld_, threadgroup T *dst_, ushort simd_group_id, ushort simd_lane_id
    )
        : src_ld(src_ld_), tile_stride(reduction_dim ? BCOLS : BROWS * src_ld),
          thread_idx(short(simd_group_id * 32u + simd_lane_id)), bi(short(thread_idx / tcols)),
          bj(short(vec_size * (thread_idx % tcols))), dst(dst_ + bi * dst_ld + bj), src(src_ + bi * src_ld + bj) {}

    METAL_FUNC void load_unsafe() const {
        MARMOT_METAL_UNROLL
        for (short i = 0; i < BROWS; i += trows) {
            *((threadgroup read_vec *)(&dst[i * dst_ld])) = *((const device read_vec *)(&src[i * src_ld]));
        }
    }

    METAL_FUNC void load_safe(short2 src_tile_dim) const {
        src_tile_dim -= short2(bj, bi);

        if (src_tile_dim.x <= 0 || src_tile_dim.y <= 0) {
            MARMOT_METAL_UNROLL
            for (short i = 0; i < BROWS; i += trows) {
                MARMOT_METAL_UNROLL
                for (short j = 0; j < vec_size; j++) {
                    dst[i * dst_ld + j] = T(0);
                }
            }
            return;
        }

        bool valid[vec_size];
        T values[vec_size];

        MARMOT_METAL_UNROLL
        for (short i = 0; i < BROWS; i += trows) {
            MARMOT_METAL_UNROLL
            for (short j = 0; j < vec_size; j++) {
                valid[j] = (i < src_tile_dim.y) && (j < src_tile_dim.x);
            }

            MARMOT_METAL_UNROLL
            for (short j = 0; j < vec_size; j++) {
                values[j] = src[valid[j] ? (i * src_ld + j) : 0];
            }

            MARMOT_METAL_UNROLL
            for (short j = 0; j < vec_size; j++) {
                dst[i * dst_ld + j] = valid[j] ? values[j] : T(0);
            }
        }
    }

    METAL_FUNC void next() {
        src += tile_stride;
    }
};

template <typename T>
struct marmot_mma_frag_8x8 {
    MARMOT_METAL_CONST short frag_rows = 8;
    MARMOT_METAL_CONST short frag_cols = 8;
    MARMOT_METAL_CONST short elems_per_frag = (frag_rows * frag_cols) / 32;

    MARMOT_METAL_CONST short elem_rows = 1;
    MARMOT_METAL_CONST short elem_cols = 2;
    static_assert(elem_rows * elem_cols == elems_per_frag);

    using mat_type = simdgroup_matrix<T, frag_rows, frag_cols>;
    using frag_type = vec<T, elems_per_frag>;

    METAL_FUNC static constexpr short2 coord(ushort simd_lane_id) {
        const short quad = short(simd_lane_id >> 2);
        const short row = short((quad & 4) | ((simd_lane_id >> 1) & 3));
        const short col = short(((quad & 2) << 1) | ((simd_lane_id & 1) << 1));
        return short2{col, row};
    }

    template <typename InT, typename PtrT, int StrX, int StrY>
    METAL_FUNC static constexpr void load(thread frag_type &dst, PtrT src) {
        MARMOT_METAL_UNROLL
        for (short i = 0; i < elem_rows; i++) {
            MARMOT_METAL_UNROLL
            for (short j = 0; j < elem_cols; j++) {
                dst[i * elem_cols + j] = T(marmot_value_to_float<InT>::convert(src[i * StrX + j * StrY]));
            }
        }
    }

    template <typename InT, typename PtrT, int StrX, int StrY>
    METAL_FUNC static constexpr void load_safe(thread frag_type &dst, PtrT src, int lim_x, int lim_y) {
        MARMOT_METAL_UNROLL
        for (short i = 0; i < elem_rows; i++) {
            MARMOT_METAL_UNROLL
            for (short j = 0; j < elem_cols; j++) {
                if (i < lim_x && j < lim_y) {
                    dst[i * elem_cols + j] = T(marmot_value_to_float<InT>::convert(src[i * StrX + j * StrY]));
                } else {
                    dst[i * elem_cols + j] = T(0);
                }
            }
        }
    }

    template <typename OutT, typename PtrT, int StrX, int StrY>
    METAL_FUNC static constexpr void store(const thread frag_type &src, PtrT dst) {
        MARMOT_METAL_UNROLL
        for (short i = 0; i < elem_rows; i++) {
            MARMOT_METAL_UNROLL
            for (short j = 0; j < elem_cols; j++) {
                dst[i * StrX + j * StrY] = marmot_value_from_float<OutT>::convert(float(src[i * elem_cols + j]));
            }
        }
    }

    template <typename OutT, typename PtrT>
    METAL_FUNC static constexpr void store(const thread frag_type &src, PtrT dst, int str_x, int str_y) {
        MARMOT_METAL_UNROLL
        for (short i = 0; i < elem_rows; i++) {
            MARMOT_METAL_UNROLL
            for (short j = 0; j < elem_cols; j++) {
                dst[i * str_x + j * str_y] = marmot_value_from_float<OutT>::convert(float(src[i * elem_cols + j]));
            }
        }
    }

    template <typename OutT, typename PtrT>
    METAL_FUNC static constexpr void store_safe(
        const thread frag_type &src, PtrT dst, int lim_x, int lim_y, int off_x, int off_y, int str_x, int str_y
    ) {
        MARMOT_METAL_UNROLL
        for (short i = 0; i < elem_rows; i++) {
            MARMOT_METAL_UNROLL
            for (short j = 0; j < elem_cols; j++) {
                if ((off_x + i) < lim_x && (off_y + j) < lim_y) {
                    dst[i * str_x + j * str_y] = marmot_value_from_float<OutT>::convert(float(src[i * elem_cols + j]));
                }
            }
        }
    }

    METAL_FUNC static constexpr void
    mma(thread frag_type &D, thread frag_type &A, thread frag_type &B, thread frag_type &C) {
        mat_type Dm;
        mat_type Am;
        mat_type Bm;
        mat_type Cm;

        reinterpret_cast<thread frag_type &>(Am.thread_elements()) = A;
        reinterpret_cast<thread frag_type &>(Bm.thread_elements()) = B;
        reinterpret_cast<thread frag_type &>(Cm.thread_elements()) = C;

        simdgroup_multiply_accumulate(Dm, Am, Bm, Cm);

        D = reinterpret_cast<thread frag_type &>(Dm.thread_elements());
    }
};

template <typename T, int TileRows, int TileCols, typename FragT = marmot_mma_frag_8x8<T>>
struct marmot_mma_tile {
    using frag_t = FragT;
    using elem_type = T;

    MARMOT_METAL_CONST short frag_rows = frag_t::frag_rows;
    MARMOT_METAL_CONST short frag_cols = frag_t::frag_cols;
    MARMOT_METAL_CONST short elems_per_frag = frag_t::elems_per_frag;

    MARMOT_METAL_CONST short tile_rows = TileRows;
    MARMOT_METAL_CONST short tile_cols = TileCols;
    MARMOT_METAL_CONST short num_frags = tile_rows * tile_cols;
    MARMOT_METAL_CONST short elems_per_tile = num_frags * elems_per_frag;

    using frag_type = typename frag_t::frag_type;

    frag_type frags[num_frags] = {frag_type(0)};

    METAL_FUNC marmot_mma_tile() thread {}

    METAL_FUNC constexpr void clear() {
        MARMOT_METAL_UNROLL
        for (short i = 0; i < num_frags; i++) {
            frags[i] = frag_type(0);
        }
    }

    METAL_FUNC constexpr thread frag_type &frag_at(short i, short j) {
        return frags[i * tile_cols + j];
    }

    METAL_FUNC constexpr const thread frag_type &frag_at(short i, short j) const {
        return frags[i * tile_cols + j];
    }

    METAL_FUNC thread elem_type *elems() {
        return reinterpret_cast<thread elem_type *>(frags);
    }

    METAL_FUNC const thread elem_type *elems() const {
        return reinterpret_cast<const thread elem_type *>(frags);
    }

    template <typename InT, int w_x, int w_y, int str_x, int str_y>
    METAL_FUNC void load(const threadgroup InT *src) {
        MARMOT_METAL_UNROLL
        for (short i = 0; i < tile_rows; ++i) {
            MARMOT_METAL_UNROLL
            for (short j = 0; j < tile_cols; ++j) {
                frag_t::template load<InT, const threadgroup InT *, str_x, str_y>(
                    frag_at(i, j), &(src[(i * frag_rows) * w_x * str_x + (j * frag_cols) * w_y * str_y])
                );
            }
        }
    }

    template <typename OutT, int w_x, int w_y>
    METAL_FUNC void store(device OutT *dst, int ld) const {
        MARMOT_METAL_UNROLL
        for (short i = 0; i < tile_rows; ++i) {
            MARMOT_METAL_UNROLL
            for (short j = 0; j < tile_cols; ++j) {
                frag_t::template store<OutT, device OutT *>(
                    frag_at(i, j), &(dst[(i * frag_rows) * w_x * ld + (j * frag_cols) * w_y]), ld, 1
                );
            }
        }
    }

    template <typename OutT, int w_x, int w_y>
    METAL_FUNC void store_safe(device OutT *dst, int ld, short2 dst_tile_dims) const {
        MARMOT_METAL_UNROLL
        for (short i = 0; i < tile_rows; ++i) {
            MARMOT_METAL_UNROLL
            for (short j = 0; j < tile_cols; ++j) {
                const int off_x = int(i * frag_rows) * w_x;
                const int off_y = int(j * frag_cols) * w_y;
                frag_t::template store_safe<OutT, device OutT *>(
                    frag_at(i, j), &(dst[(i * frag_rows) * w_x * ld + (j * frag_cols) * w_y]), dst_tile_dims.y,
                    dst_tile_dims.x, off_x, off_y, ld, 1
                );
            }
        }
    }
};

template <typename T, typename U, int M, int N, int K>
METAL_FUNC void marmot_tile_matmad(
    thread marmot_mma_tile<T, M, N> &D, thread marmot_mma_tile<U, M, K> &A, thread marmot_mma_tile<U, K, N> &B,
    thread marmot_mma_tile<T, M, N> &C
) {
    MARMOT_METAL_UNROLL
    for (short m = 0; m < M; ++m) {
        MARMOT_METAL_UNROLL
        for (short n = 0; n < N; ++n) {
            const short n_serp = (m & 1) ? short(N - 1 - n) : n;
            MARMOT_METAL_UNROLL
            for (short k = 0; k < K; ++k) {
                marmot_mma_frag_8x8<T>::mma(
                    D.frag_at(m, n_serp), A.frag_at(m, k), B.frag_at(k, n_serp), C.frag_at(m, n_serp)
                );
            }
        }
    }
}

template <
    typename T, typename U, int BM, int BN, int BK, int WM, int WN, bool transpose_a, bool transpose_b, short lda_tgp,
    short ldb_tgp, typename AccumType = float>
struct marmot_block_mma {
    MARMOT_METAL_CONST short frag_size = 8;

    MARMOT_METAL_CONST short tm_stride = frag_size * WM;
    MARMOT_METAL_CONST short tn_stride = frag_size * WN;

    MARMOT_METAL_CONST short TM = BM / (frag_size * WM);
    MARMOT_METAL_CONST short TN = BN / (frag_size * WN);

    MARMOT_METAL_CONST short A_str_m = transpose_a ? 1 : lda_tgp;
    MARMOT_METAL_CONST short A_str_k = transpose_a ? lda_tgp : 1;

    MARMOT_METAL_CONST short B_str_k = transpose_b ? 1 : ldb_tgp;
    MARMOT_METAL_CONST short B_str_n = transpose_b ? ldb_tgp : 1;

    MARMOT_METAL_CONST short tile_stride_a = frag_size * A_str_k;
    MARMOT_METAL_CONST short tile_stride_b = frag_size * B_str_k;

    using tile_a_t = marmot_mma_tile<AccumType, TM, 1>;
    using tile_b_t = marmot_mma_tile<AccumType, 1, TN>;
    using tile_c_t = marmot_mma_tile<AccumType, TM, TN>;

    tile_a_t Atile;
    tile_b_t Btile;
    tile_c_t Ctile;

    short sm;
    short sn;
    short As_offset;
    short Bs_offset;

    METAL_FUNC marmot_block_mma(ushort simd_group_id, ushort simd_lane_id) {
        short tm = short(frag_size * (simd_group_id / WN));
        short tn = short(frag_size * (simd_group_id % WN));

        short2 sc = marmot_mma_frag_8x8<AccumType>::coord(simd_lane_id);
        sm = sc.y;
        sn = sc.x;

        As_offset = short((tm + sm) * A_str_m + (sn)*A_str_k);
        Bs_offset = short((sm)*B_str_k + (tn + sn) * B_str_n);

        sm = short(sm + tm);
        sn = short(sn + tn);
    }

    METAL_FUNC void mma(const threadgroup T *As, const threadgroup T *Bs) {
        As += As_offset;
        Bs += Bs_offset;

        MARMOT_METAL_UNROLL
        for (short kk = 0; kk < BK; kk += frag_size) {
            simdgroup_barrier(mem_flags::mem_none);
            Atile.template load<T, WM, 1, A_str_m, A_str_k>(As);

            simdgroup_barrier(mem_flags::mem_none);
            Btile.template load<T, 1, WN, B_str_k, B_str_n>(Bs);

            simdgroup_barrier(mem_flags::mem_none);
            marmot_tile_matmad(Ctile, Atile, Btile, Ctile);

            As += tile_stride_a;
            Bs += tile_stride_b;
        }
    }

    METAL_FUNC void store_result(device U *D, int ldd) {
        D += sm * ldd + sn;
        Ctile.template store<U, WM, WN>(D, ldd);
    }

    METAL_FUNC void store_result_safe(device U *D, int ldd, short2 dst_tile_dims) {
        D += sm * ldd + sn;
        dst_tile_dims -= short2(sn, sm);
        if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0) {
            return;
        }
        Ctile.template store_safe<U, WM, WN>(D, ldd, dst_tile_dims);
    }
};

template <typename T, typename U, int BM, int BN, int BK, int WM, int WN, bool transpose_a, bool transpose_b>
static inline void marmot_gemm_run(
    const device T *A, const device T *B, device U *D, const constant marmot_gemm_params_t &params, uint simd_lane_id,
    uint simd_group_id, uint3 tid, threadgroup T *As, threadgroup T *Bs
) {
    const int tid_y = (tid.y << params.swizzle_log) + (tid.x & ((1 << params.swizzle_log) - 1));
    const int tid_x = tid.x >> params.swizzle_log;
    if (tid_x >= params.tiles_n || tid_y >= params.tiles_m) {
        return;
    }

    const int c_row = tid_y * BM;
    const int c_col = tid_x * BN;
    const size_t c_row_long = size_t(c_row);
    const size_t c_col_long = size_t(c_col);

    A += transpose_a ? c_row_long : c_row_long * size_t(params.lda);
    B += transpose_b ? c_col_long * size_t(params.ldb) : c_col_long;
    D += c_row_long * size_t(params.ldd) + c_col_long;

    constexpr short pad_a = short(16 / sizeof(T));
    constexpr short pad_b = short(16 / sizeof(T));

    constexpr short lda_tgp = transpose_a ? short(BM + pad_a) : short(BK + pad_a);
    constexpr short ldb_tgp = transpose_b ? short(BK + pad_b) : short(BN + pad_b);

    constexpr short tgp_size = short(WM * WN * 32);

    using loader_a_t =
        marmot_block_loader<T, transpose_a ? BK : BM, transpose_a ? BM : BK, lda_tgp, !transpose_a, tgp_size>;
    using loader_b_t =
        marmot_block_loader<T, transpose_b ? BN : BK, transpose_b ? BK : BN, ldb_tgp, transpose_b, tgp_size>;

    loader_a_t loader_a{A, params.lda, As, ushort(simd_group_id), ushort(simd_lane_id)};
    loader_b_t loader_b{B, params.ldb, Bs, ushort(simd_group_id), ushort(simd_lane_id)};

    using mma_t = marmot_block_mma<T, U, BM, BN, BK, WM, WN, transpose_a, transpose_b, lda_tgp, ldb_tgp, float>;
    mma_t mma_op{ushort(simd_group_id), ushort(simd_lane_id)};

    const int m_limit = params.M - c_row;
    const int n_limit = params.N - c_col;
    const short tgp_bm = short(min(BM, m_limit));
    const short tgp_bn = short(min(BN, n_limit));

    const bool m_full = marmot_align_m || (tgp_bm == BM);
    const bool n_full = marmot_align_n || (tgp_bn == BN);

    const short2 tile_dims_A = transpose_a ? short2(tgp_bm, BK) : short2(BK, tgp_bm);
    const short2 tile_dims_B = transpose_b ? short2(BK, tgp_bn) : short2(tgp_bn, BK);

    const int gemm_k_iterations = params.gemm_k_iterations_aligned;

    MARMOT_METAL_NO_UNROLL
    for (int k = 0; k < gemm_k_iterations; k++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (m_full) {
            loader_a.load_unsafe();
        } else {
            loader_a.load_safe(tile_dims_A);
        }

        if (n_full) {
            loader_b.load_unsafe();
        } else {
            loader_b.load_safe(tile_dims_B);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        mma_op.mma(As, Bs);

        loader_a.next();
        loader_b.next();
    }

    if (!marmot_align_k) {
        const int rem_bk = params.K - gemm_k_iterations * BK;
        if (rem_bk > 0) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const short2 tile_dims_A_last = transpose_a ? short2(tgp_bm, rem_bk) : short2(rem_bk, tgp_bm);
            const short2 tile_dims_B_last = transpose_b ? short2(rem_bk, tgp_bn) : short2(tgp_bn, rem_bk);

            loader_a.load_safe(tile_dims_A_last);
            loader_b.load_safe(tile_dims_B_last);

            threadgroup_barrier(mem_flags::mem_threadgroup);
            mma_op.mma(As, Bs);
        }
    }

    threadgroup_barrier(mem_flags::mem_none);

    if (m_full && n_full) {
        mma_op.store_result(D, params.ldd);
    } else {
        mma_op.store_result_safe(D, params.ldd, short2(tgp_bn, tgp_bm));
    }
}

#define MARMOT_GEMM_KERNEL(name, iname, itype, oname, otype, bm, bn, bk, wm, wn, ta, tb)                               \
    [[kernel, max_total_threads_per_threadgroup(wm * wn * 32)]] void                                                   \
    marmot_gemm_##name##_##iname##_##oname##_bm##bm##_bn##bn##_bk##bk##_wm##wm##_wn##wn(                               \
        const device itype *A [[buffer(0)]], const device itype *B [[buffer(1)]], device otype *D [[buffer(2)]],       \
        const constant marmot_gemm_params_t &params [[buffer(3)]], uint simd_lane_id [[thread_index_in_simdgroup]],    \
        uint simd_group_id [[simdgroup_index_in_threadgroup]], uint3 tid [[threadgroup_position_in_grid]]              \
    ) {                                                                                                                \
        constexpr int pad_a = 16 / sizeof(itype);                                                                      \
        constexpr int pad_b = 16 / sizeof(itype);                                                                      \
        threadgroup itype As[(ta ? (bk * (bm + pad_a)) : (bm * (bk + pad_a)))];                                        \
        threadgroup itype Bs[(tb ? (bn * (bk + pad_b)) : (bk * (bn + pad_b)))];                                        \
        marmot_gemm_run<itype, otype, bm, bn, bk, wm, wn, ta, tb>(                                                     \
            A, B, D, params, simd_lane_id, simd_group_id, tid, As, Bs                                                  \
        );                                                                                                             \
    }

#define MARMOT_INSTANTIATE_GEMM_TRANSPOSE(iname, itype, oname, otype, bm, bn, bk, wm, wn)                              \
    MARMOT_GEMM_KERNEL(nn, iname, itype, oname, otype, bm, bn, bk, wm, wn, false, false)                               \
    MARMOT_GEMM_KERNEL(nt, iname, itype, oname, otype, bm, bn, bk, wm, wn, false, true)

#define MARMOT_INSTANTIATE_GEMM_SHAPES(iname, itype, oname, otype)                                                     \
    MARMOT_INSTANTIATE_GEMM_TRANSPOSE(iname, itype, oname, otype, 64, 64, 16, 2, 2)                                    \
    MARMOT_INSTANTIATE_GEMM_TRANSPOSE(iname, itype, oname, otype, 64, 64, 16, 1, 2)                                    \
    MARMOT_INSTANTIATE_GEMM_TRANSPOSE(iname, itype, oname, otype, 64, 32, 32, 2, 2)                                    \
    MARMOT_INSTANTIATE_GEMM_TRANSPOSE(iname, itype, oname, otype, 32, 64, 16, 1, 2)                                    \
    MARMOT_INSTANTIATE_GEMM_TRANSPOSE(iname, itype, oname, otype, 32, 32, 16, 2, 2)                                    \
    MARMOT_INSTANTIATE_GEMM_TRANSPOSE(iname, itype, oname, otype, 64, 32, 8, 4, 1)

MARMOT_INSTANTIATE_GEMM_SHAPES(f32, float, f32, float);
MARMOT_INSTANTIATE_GEMM_SHAPES(f16, half, f16, half);
MARMOT_INSTANTIATE_GEMM_SHAPES(bf16, ushort, bf16, ushort);

#undef MARMOT_INSTANTIATE_GEMM_SHAPES
#undef MARMOT_INSTANTIATE_GEMM_TRANSPOSE
#undef MARMOT_GEMM_KERNEL
