#!/usr/bin/env python3
"""Create a truncated MoE GGUF fixture from a full model.

Reads a real Qwen3-MoE (or other MoE) GGUF file and writes a new GGUF
with only the first N layers, preserving all metadata and real tensor data.
The truncated model produces garbage text but exercises the full
load -> build -> forward path with real quantized expert weights.

Usage:
    python scripts/make_moe_fixture.py <input.gguf> [--layers N] [-o output.gguf]

Example:
    python scripts/make_moe_fixture.py \
        Qwen3-30B-A3B-Q4_K_M.gguf \
        --layers 1 \
        -o tests/fixtures/gguf/multiarch/qwen3moe-30b-a3b-1layer-q4km.gguf
"""

import argparse
import struct
import sys
from pathlib import Path

# GGUF constants
GGUF_MAGIC = 0x46554747
GGUF_DEFAULT_ALIGNMENT = 32

# GGUF value types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

VALUE_SIZES = {
    GGUF_TYPE_UINT8: 1,
    GGUF_TYPE_INT8: 1,
    GGUF_TYPE_UINT16: 2,
    GGUF_TYPE_INT16: 2,
    GGUF_TYPE_UINT32: 4,
    GGUF_TYPE_INT32: 4,
    GGUF_TYPE_FLOAT32: 4,
    GGUF_TYPE_BOOL: 1,
    GGUF_TYPE_UINT64: 8,
    GGUF_TYPE_INT64: 8,
    GGUF_TYPE_FLOAT64: 8,
}

VALUE_FORMATS = {
    GGUF_TYPE_UINT8: "<B",
    GGUF_TYPE_INT8: "<b",
    GGUF_TYPE_UINT16: "<H",
    GGUF_TYPE_INT16: "<h",
    GGUF_TYPE_UINT32: "<I",
    GGUF_TYPE_INT32: "<i",
    GGUF_TYPE_FLOAT32: "<f",
    GGUF_TYPE_BOOL: "<B",
    GGUF_TYPE_UINT64: "<Q",
    GGUF_TYPE_INT64: "<q",
    GGUF_TYPE_FLOAT64: "<d",
}


class GGUFReader:
    """Minimal GGUF parser that reads header, KV pairs, and tensor info."""

    def __init__(self, path: str):
        self.path = path
        self.f = open(path, "rb")
        self.kv_pairs: list[tuple[str, int, bytes]] = []  # (key, type, raw_value_bytes)
        self.tensors: list[dict] = []
        self.tensor_data_offset = 0
        self.alignment = GGUF_DEFAULT_ALIGNMENT
        self._parse()

    def _read(self, fmt: str) -> tuple:
        size = struct.calcsize(fmt)
        data = self.f.read(size)
        if len(data) < size:
            raise EOFError(f"Expected {size} bytes, got {len(data)}")
        return struct.unpack(fmt, data)

    def _read_string(self) -> str:
        (length,) = self._read("<Q")
        data = self.f.read(length)
        return data.decode("utf-8")

    def _read_value_raw(self, vtype: int) -> bytes:
        """Read a KV value and return the raw bytes (including the type tag)."""
        buf = struct.pack("<I", vtype)
        if vtype == GGUF_TYPE_STRING:
            (length,) = self._read("<Q")
            data = self.f.read(length)
            buf += struct.pack("<Q", length) + data
        elif vtype == GGUF_TYPE_ARRAY:
            (elem_type,) = self._read("<I")
            (count,) = self._read("<Q")
            buf += struct.pack("<I", elem_type) + struct.pack("<Q", count)
            if elem_type == GGUF_TYPE_STRING:
                for _ in range(count):
                    (slen,) = self._read("<Q")
                    sdata = self.f.read(slen)
                    buf += struct.pack("<Q", slen) + sdata
            else:
                elem_size = VALUE_SIZES[elem_type]
                data = self.f.read(elem_size * count)
                buf += data
        else:
            size = VALUE_SIZES[vtype]
            data = self.f.read(size)
            buf += data
        return buf

    def _parse(self):
        (magic,) = self._read("<I")
        if magic != GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file: magic=0x{magic:08X}")
        (self.version,) = self._read("<I")
        (tensor_count,) = self._read("<Q")
        (kv_count,) = self._read("<Q")

        # Read KV pairs
        for _ in range(kv_count):
            key = self._read_string()
            (vtype,) = self._read("<I")
            raw = self._read_value_raw(vtype)
            self.kv_pairs.append((key, vtype, raw))

            # Extract alignment if present
            if key == "general.alignment" and vtype == GGUF_TYPE_UINT32:
                self.alignment = struct.unpack("<I", raw[4:])[0]

        # Read tensor info
        for _ in range(tensor_count):
            name = self._read_string()
            (ndim,) = self._read("<I")
            dims = []
            for _ in range(ndim):
                (d,) = self._read("<Q")
                dims.append(d)
            (ggml_type,) = self._read("<I")
            (data_offset,) = self._read("<Q")
            self.tensors.append({
                "name": name,
                "ndim": ndim,
                "dims": dims,
                "ggml_type": ggml_type,
                "data_offset": data_offset,
            })

        # Compute tensor data start
        header_end = self.f.tell()
        self.tensor_data_offset = align(header_end, self.alignment)

    def read_tensor_data(self, tensor: dict, next_offset: int) -> bytes:
        """Read the raw bytes for a tensor."""
        start = self.tensor_data_offset + tensor["data_offset"]
        size = next_offset - tensor["data_offset"] if next_offset > 0 else 0
        if size <= 0:
            # Last tensor or single tensor — read to end or estimate
            size = self._estimate_tensor_bytes(tensor)
        self.f.seek(start)
        return self.f.read(size)

    def _estimate_tensor_bytes(self, tensor: dict) -> int:
        """Estimate byte size from tensor dimensions and type."""
        from math import prod
        n_elements = prod(tensor["dims"]) if tensor["dims"] else 0
        ggml_type = tensor["ggml_type"]
        # Common types: F32=0(4B), F16=1(2B), Q4_0=2, Q4_1=3, Q5_0=6, Q5_1=7, Q8_0=8, Q8_1=9
        # Q2_K=10, Q3_K=11, Q4_K=12, Q5_K=13, Q6_K=14, Q8_K=15, BF16=30
        bpe = {0: 4, 1: 2, 30: 2, 24: 1, 25: 2, 26: 4, 27: 8}
        if ggml_type in bpe:
            return n_elements * bpe[ggml_type]

        # Block quantized types: compute from block structure
        block_info = {
            # type: (block_values, block_bytes)
            2: (32, 18),     # Q4_0: 32 values, 2+16=18 bytes
            3: (32, 20),     # Q4_1: 32 values, 2+2+16=20 bytes
            6: (32, 22),     # Q5_0: 32 values, 2+4+16=22 bytes
            7: (32, 24),     # Q5_1: 32 values, 2+2+4+16=24 bytes
            8: (32, 34),     # Q8_0: 32 values, 2+32=34 bytes
            9: (32, 36),     # Q8_1: 32 values, 4+32=36 bytes
            10: (256, 82),   # Q2_K
            11: (256, 110),  # Q3_K
            12: (256, 144),  # Q4_K
            13: (256, 176),  # Q5_K
            14: (256, 210),  # Q6_K
            15: (256, 292),  # Q8_K
        }
        if ggml_type in block_info:
            bv, bb = block_info[ggml_type]
            # For multi-dim tensors, blocks are along the innermost dimension (dims[0])
            inner_dim = tensor["dims"][0] if tensor["dims"] else 0
            blocks_per_row = (inner_dim + bv - 1) // bv
            n_rows = n_elements // inner_dim if inner_dim > 0 else 0
            return n_rows * blocks_per_row * bb

        raise ValueError(f"Unknown GGML type {ggml_type} for tensor {tensor['name']}")

    def close(self):
        self.f.close()


def align(offset: int, alignment: int) -> int:
    return (offset + alignment - 1) // alignment * alignment


def tensor_belongs_to_layer(name: str) -> int | None:
    """Extract layer index from tensor name like 'blk.42.attn_q.weight'."""
    if not name.startswith("blk."):
        return None
    parts = name.split(".")
    if len(parts) < 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def write_gguf(
    out_path: str,
    version: int,
    kv_pairs: list[tuple[str, int, bytes]],
    tensors: list[dict],
    tensor_data: list[bytes],
    alignment: int,
):
    """Write a complete GGUF file."""
    with open(out_path, "wb") as f:
        # Header
        f.write(struct.pack("<I", GGUF_MAGIC))
        f.write(struct.pack("<I", version))
        f.write(struct.pack("<Q", len(tensors)))
        f.write(struct.pack("<Q", len(kv_pairs)))

        # KV pairs
        for key, _vtype, raw_value in kv_pairs:
            key_bytes = key.encode("utf-8")
            f.write(struct.pack("<Q", len(key_bytes)))
            f.write(key_bytes)
            f.write(raw_value)

        # Tensor info
        for i, t in enumerate(tensors):
            name_bytes = t["name"].encode("utf-8")
            f.write(struct.pack("<Q", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<I", t["ndim"]))
            for d in t["dims"]:
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", t["ggml_type"]))
            f.write(struct.pack("<Q", t["new_offset"]))

        # Padding to alignment
        pos = f.tell()
        aligned_pos = align(pos, alignment)
        f.write(b"\x00" * (aligned_pos - pos))

        # Tensor data
        for data in tensor_data:
            f.write(data)
            # Pad each tensor to alignment
            pos = f.tell()
            aligned_pos = align(pos, alignment)
            f.write(b"\x00" * (aligned_pos - pos))


def get_kv_value(kv_pairs, key, vtype_expected=None):
    """Extract a scalar value from KV pairs."""
    for k, vtype, raw in kv_pairs:
        if k != key:
            continue
        data = raw[4:]  # skip type tag
        if vtype == GGUF_TYPE_UINT32:
            return struct.unpack("<I", data)[0]
        if vtype == GGUF_TYPE_INT32:
            return struct.unpack("<i", data)[0]
        if vtype == GGUF_TYPE_UINT64:
            return struct.unpack("<Q", data)[0]
        if vtype == GGUF_TYPE_FLOAT32:
            return struct.unpack("<f", data)[0]
        if vtype == GGUF_TYPE_STRING:
            slen = struct.unpack("<Q", data[:8])[0]
            return data[8 : 8 + slen].decode("utf-8")
    return None


def set_kv_u32(kv_pairs, key, value):
    """Replace a u32 KV pair value."""
    new_raw = struct.pack("<I", GGUF_TYPE_UINT32) + struct.pack("<I", value)
    for i, (k, vtype, _raw) in enumerate(kv_pairs):
        if k == key:
            kv_pairs[i] = (key, GGUF_TYPE_UINT32, new_raw)
            return
    kv_pairs.append((key, GGUF_TYPE_UINT32, new_raw))


def set_block_count(kv_pairs, arch, value):
    """Update block_count in every metadata location this script reads from."""
    updated = False
    keys = []
    if arch:
        keys.append(f"{arch}.block_count")
    keys.append("general.block_count")
    for key in keys:
        for i, (k, _vtype, _raw) in enumerate(kv_pairs):
            if k != key:
                continue
            kv_pairs[i] = (key, GGUF_TYPE_UINT32, struct.pack("<I", GGUF_TYPE_UINT32) + struct.pack("<I", value))
            updated = True
    if not updated and arch:
        set_kv_u32(kv_pairs, f"{arch}.block_count", value)


def main():
    parser = argparse.ArgumentParser(description="Create truncated MoE GGUF fixture")
    parser.add_argument("input", help="Input GGUF file path")
    parser.add_argument("--layers", type=int, default=1, help="Number of layers to keep (default: 1)")
    parser.add_argument("-o", "--output", help="Output GGUF file path")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    if args.output is None:
        stem = Path(args.input).stem
        args.output = f"{stem}-{args.layers}layer.gguf"

    print(f"Reading {args.input}...")
    reader = GGUFReader(args.input)

    arch = get_kv_value(reader.kv_pairs, "general.architecture")
    block_count = None
    for prefix in [p for p in [arch, "general"] if p]:
        block_count = get_kv_value(reader.kv_pairs, f"{prefix}.block_count")
        if block_count is not None:
            break

    if block_count is None:
        print("Error: Could not find block_count in metadata", file=sys.stderr)
        sys.exit(1)

    n_experts = get_kv_value(reader.kv_pairs, f"{arch}.expert_count")
    n_experts_used = get_kv_value(reader.kv_pairs, f"{arch}.expert_used_count")

    print(f"  Architecture: {arch}")
    print(f"  Layers: {block_count}")
    print(f"  Experts: {n_experts} (top-{n_experts_used})")
    print(f"  Total tensors: {len(reader.tensors)}")
    print(f"  Keeping first {args.layers} layer(s)")

    # Filter tensors: keep non-layer tensors + layers 0..N-1
    kept_tensors = []
    for t in reader.tensors:
        layer = tensor_belongs_to_layer(t["name"])
        if layer is None or layer < args.layers:
            kept_tensors.append(t)

    dropped = len(reader.tensors) - len(kept_tensors)
    print(f"  Keeping {len(kept_tensors)} tensors (dropping {dropped})")

    # Sort kept tensors by original data offset for sequential reading
    kept_tensors.sort(key=lambda t: t["data_offset"])

    # Build sorted offset list for computing tensor sizes
    all_offsets = sorted(set(t["data_offset"] for t in reader.tensors))
    offset_to_next = {}
    for i, off in enumerate(all_offsets):
        if i + 1 < len(all_offsets):
            offset_to_next[off] = all_offsets[i + 1]
        else:
            offset_to_next[off] = -1

    # Read tensor data
    print("  Reading tensor data...")
    tensor_data_list = []
    current_data_offset = 0
    for t in kept_tensors:
        next_off = offset_to_next.get(t["data_offset"], -1)
        if next_off < 0:
            data = reader.read_tensor_data(t, 0)
        else:
            size = next_off - t["data_offset"]
            reader.f.seek(reader.tensor_data_offset + t["data_offset"])
            data = reader.f.read(size)

        # Strip trailing padding from data (will be re-added during write)
        tensor_data_list.append(data)
        t["new_offset"] = current_data_offset
        current_data_offset = align(current_data_offset + len(data), reader.alignment)

    reader.close()

    # Patch block_count in metadata
    kv_pairs = list(reader.kv_pairs)
    set_block_count(kv_pairs, arch, args.layers)

    # Write output
    print(f"Writing {args.output}...")
    write_gguf(
        args.output,
        reader.version,
        kv_pairs,
        kept_tensors,
        tensor_data_list,
        reader.alignment,
    )

    out_size = Path(args.output).stat().st_size
    print(f"  Output size: {out_size / 1024 / 1024:.1f} MB")
    print("Done.")


if __name__ == "__main__":
    main()
