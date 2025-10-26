#!/usr/bin/env python3
"""
Generate small, text-based golden fixtures for embedding tests.

Format (UTF-8 text):

quant_kind: Q4_0|Q4_1|Q8_0
vocab: <int>
dim: <int>
ids: <space-separated int32>
weights_hex: <hex of concatenated row-major block bytes>
expected: <space-separated float32 values for ids_count*dim>

Notes:
- This script creates synthetic blocks and computes expected by decoding the
  blocks in Python (independent of C code) to keep the golden external.
- For Q4_1 blocks: bytes = [scale:f16][min:f16][qs:16]
  For Q4_0 blocks: bytes = [scale:f16][qs:16]
  For Q8_0 blocks: bytes = [scale:f16][qs:32]

Usage: run from repo root
  python3 scripts/generate_golden_embeddings.py --out tests/golden --kinds Q4_0 Q4_1 Q8_0
"""
from __future__ import annotations

import argparse
import os
import struct
from typing import List


def f16_bits_from_f32(x: float) -> int:
    # roundtrip via half precision using struct (Python 3.11 supports 'e')
    b = struct.pack('<e', x)
    return struct.unpack('<H', b)[0]


def decode_q4_0_block(scale_bits: int, qs: bytes) -> List[float]:
    scale = struct.unpack('<e', struct.pack('<H', scale_bits))[0]
    out = []
    for i in range(32):
        packed = qs[i >> 1]
        q = (packed >> 4) & 0x0F if (i & 1) else (packed & 0x0F)
        q = q - 8
        out.append(scale * float(q))
    return out


def decode_q4_1_block(scale_bits: int, min_bits: int, qs: bytes) -> List[float]:
    scale = struct.unpack('<e', struct.pack('<H', scale_bits))[0]
    mn = struct.unpack('<e', struct.pack('<H', min_bits))[0]
    out = []
    for i in range(32):
        packed = qs[i >> 1]
        q = (packed >> 4) & 0x0F if (i & 1) else (packed & 0x0F)
        out.append(scale * float(q) + mn)
    return out


def decode_q8_0_block(scale_bits: int, qs: bytes) -> List[float]:
    scale = struct.unpack('<e', struct.pack('<H', scale_bits))[0]
    return [scale * float(struct.unpack('<b', bytes([qs[i]]))[0]) for i in range(32)]


def make_blocks(kind: str, vocab: int, dim: int) -> bytes:
    blocks_per_row = (dim + 31) // 32
    out = bytearray()
    for r in range(vocab):
        for b in range(blocks_per_row):
            if kind == 'Q4_0':
                scale = 0.125 + 0.03125 * ((r + b) % 3)
                scale_bits = f16_bits_from_f32(scale)
                out += struct.pack('<H', scale_bits)
                qs = bytearray(16)
                for i in range(16):
                    lo = ((i + r) % 16) & 0xF
                    hi = ((i + b) % 16) & 0xF
                    qs[i] = (hi << 4) | lo
                out += qs
            elif kind == 'Q4_1':
                scale = 0.0625 + 0.015625 * ((r + b) % 4)
                mn = -1.0 + 0.25 * ((r + b) % 5)
                out += struct.pack('<H', f16_bits_from_f32(scale))
                out += struct.pack('<H', f16_bits_from_f32(mn))
                qs = bytearray(16)
                for i in range(16):
                    lo = ((2 * i + r) % 16) & 0xF
                    hi = ((3 * i + b) % 16) & 0xF
                    qs[i] = (hi << 4) | lo
                out += qs
            elif kind == 'Q8_0':
                scale = 0.05 + 0.01 * ((r + b) % 6)
                out += struct.pack('<H', f16_bits_from_f32(scale))
                qs = bytearray(32)
                for i in range(32):
                    val = ((i + r + b) % 127) - 63
                    qs[i] = struct.pack('<b', val)[0]
                out += qs
            else:
                raise ValueError('unsupported kind')
    return bytes(out)


def decode_rows(kind: str, vocab: int, dim: int, rows: List[int], weights: bytes) -> List[float]:
    blocks_per_row = (dim + 31) // 32
    out: List[float] = []
    ptr = 0
    row_stride = 0
    if kind == 'Q4_0':
        row_stride = blocks_per_row * (2 + 16)
    elif kind == 'Q4_1':
        row_stride = blocks_per_row * (2 + 2 + 16)
    elif kind == 'Q8_0':
        row_stride = blocks_per_row * (2 + 32)
    else:
        raise ValueError('unsupported')

    def row_bytes(row: int) -> bytes:
        return weights[row * row_stride:(row + 1) * row_stride]

    for row in rows:
        rb = row_bytes(row)
        off = 0
        vals: List[float] = []
        for _ in range(blocks_per_row):
            if kind == 'Q4_0':
                scale_bits = struct.unpack('<H', rb[off:off + 2])[0]
                off += 2
                qs = rb[off:off + 16]
                off += 16
                vals.extend(decode_q4_0_block(scale_bits, qs))
            elif kind == 'Q4_1':
                scale_bits = struct.unpack('<H', rb[off:off + 2])[0]
                off += 2
                min_bits = struct.unpack('<H', rb[off:off + 2])[0]
                off += 2
                qs = rb[off:off + 16]
                off += 16
                vals.extend(decode_q4_1_block(scale_bits, min_bits, qs))
            elif kind == 'Q8_0':
                scale_bits = struct.unpack('<H', rb[off:off + 2])[0]
                off += 2
                qs = rb[off:off + 32]
                off += 32
                vals.extend(decode_q8_0_block(scale_bits, qs))
        out.extend(vals[:dim])
    return out


def to_hex(b: bytes) -> str:
    return ''.join(f'{x:02x}' for x in b)


def write_fixture(path: str, kind: str, vocab: int, dim: int, ids: List[int]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    w = make_blocks(kind, vocab, dim)
    expected = decode_rows(kind, vocab, dim, ids, w)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"quant_kind: {kind}\n")
        f.write(f"vocab: {vocab}\n")
        f.write(f"dim: {dim}\n")
        f.write("ids: " + ' '.join(str(i) for i in ids) + "\n")
        f.write("weights_hex: " + to_hex(w) + "\n")
        f.write("expected: " + ' '.join(f"{x:.8f}" for x in expected) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='tests/golden', help='output directory')
    ap.add_argument('--vocab', type=int, default=4)
    ap.add_argument('--dim', type=int, default=64)
    ap.add_argument('--ids', type=int, nargs='+', default=[0, 2, 3])
    ap.add_argument('--kinds', nargs='+', default=['Q4_0', 'Q4_1', 'Q8_0'])
    args = ap.parse_args()

    for kind in args.kinds:
        fname = f'embedding_{kind.lower()}.txt'
        write_fixture(os.path.join(args.out, fname), kind, args.vocab, args.dim, args.ids)
        print('wrote', os.path.join(args.out, fname))


if __name__ == '__main__':
    main()

