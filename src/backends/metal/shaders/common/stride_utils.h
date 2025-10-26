#ifndef MARMOT_STRIDE_UTILS_H
#define MARMOT_STRIDE_UTILS_H

#include <metal_stdlib>
using namespace metal;

template <typename IdxT = uint>
METAL_FUNC IdxT elem_to_loc(IdxT elem, constant const uint *shape, constant const size_t *strides, uint ndim) {
    IdxT loc = 0;
    for (uint i = ndim; i > 0 && elem > 0; --i) {
        uint dim_idx = i - 1;
        IdxT idx = elem % shape[dim_idx];
        loc += idx * IdxT(strides[dim_idx]);
        elem /= shape[dim_idx];
    }
    return loc;
}

template <typename IdxT = uint>
METAL_FUNC IdxT elem_to_loc_1(IdxT elem, constant const size_t *strides) {
    return elem * IdxT(strides[0]);
}

template <typename IdxT = uint>
METAL_FUNC IdxT elem_to_loc_2(IdxT elem, constant const uint *shape, constant const size_t *strides) {
    IdxT loc = (elem % shape[1]) * IdxT(strides[1]);
    elem /= shape[1];
    loc += elem * IdxT(strides[0]);
    return loc;
}

template <typename IdxT = uint>
METAL_FUNC IdxT elem_to_loc_3(IdxT elem, constant const uint *shape, constant const size_t *strides) {
    IdxT loc = (elem % shape[2]) * IdxT(strides[2]);
    elem /= shape[2];
    loc += (elem % shape[1]) * IdxT(strides[1]);
    elem /= shape[1];
    loc += elem * IdxT(strides[0]);
    return loc;
}

template <typename IdxT = uint>
METAL_FUNC IdxT elem_to_loc_row_strided(IdxT row, IdxT col, IdxT row_stride) {
    return row * row_stride + col;
}

#endif // MARMOT_STRIDE_UTILS_H
