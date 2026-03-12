#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_fixture_module():
    script_path = repo_root() / "scripts" / "make_moe_fixture.py"
    spec = importlib.util.spec_from_file_location("make_moe_fixture", script_path)
    if spec is None or spec.loader is None:
        raise SystemExit("failed to load make_moe_fixture.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pack_u32_raw(mod, value: int) -> bytes:
    return struct.pack("<I", mod.GGUF_TYPE_UINT32) + struct.pack("<I", value)


def pack_string_raw(mod, value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<I", mod.GGUF_TYPE_STRING) + struct.pack("<Q", len(encoded)) + encoded


def pack_string_array_raw(mod, values: list[str]) -> bytes:
    raw = struct.pack("<I", mod.GGUF_TYPE_ARRAY)
    raw += struct.pack("<I", mod.GGUF_TYPE_STRING)
    raw += struct.pack("<Q", len(values))
    for value in values:
        encoded = value.encode("utf-8")
        raw += struct.pack("<Q", len(encoded)) + encoded
    return raw


def pack_u32_array_raw(mod, values: list[int]) -> bytes:
    raw = struct.pack("<I", mod.GGUF_TYPE_ARRAY)
    raw += struct.pack("<I", mod.GGUF_TYPE_UINT32)
    raw += struct.pack("<Q", len(values))
    for value in values:
        raw += struct.pack("<I", value)
    return raw


def build_input_fixture(mod, path: Path) -> None:
    kv_pairs = [
        ("general.architecture", mod.GGUF_TYPE_STRING, pack_string_raw(mod, "toyarch")),
        ("general.block_count", mod.GGUF_TYPE_UINT32, pack_u32_raw(mod, 2)),
        ("toyarch.block_count", mod.GGUF_TYPE_UINT32, pack_u32_raw(mod, 2)),
        ("toyarch.quant_schemes", mod.GGUF_TYPE_ARRAY, pack_string_array_raw(mod, ["q4_k", "q6_k"])),
        ("general.some_u32s", mod.GGUF_TYPE_ARRAY, pack_u32_array_raw(mod, [4, 8, 15, 16, 23, 42])),
    ]
    tensor_data = [
        struct.pack("<f", 1.0),
        struct.pack("<f", 2.0),
        struct.pack("<f", 3.0),
    ]
    tensors = [
        {"name": "tok_embeddings.weight", "ndim": 1, "dims": [1], "ggml_type": 0},
        {"name": "blk.0.ffn_gate_exps.weight", "ndim": 1, "dims": [1], "ggml_type": 0},
        {"name": "blk.1.ffn_gate_exps.weight", "ndim": 1, "dims": [1], "ggml_type": 0},
    ]

    current_offset = 0
    for tensor, data in zip(tensors, tensor_data):
        tensor["new_offset"] = current_offset
        current_offset = mod.align(current_offset + len(data), mod.GGUF_DEFAULT_ALIGNMENT)

    mod.write_gguf(
        str(path),
        version=3,
        kv_pairs=kv_pairs,
        tensors=tensors,
        tensor_data=tensor_data,
        alignment=mod.GGUF_DEFAULT_ALIGNMENT,
    )


def main() -> None:
    mod = load_fixture_module()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        input_path = tmp / "input.gguf"
        output_path = tmp / "output.gguf"
        build_input_fixture(mod, input_path)

        subprocess.run(
            [
                sys.executable,
                str(repo_root() / "scripts" / "make_moe_fixture.py"),
                str(input_path),
                "--layers",
                "1",
                "-o",
                str(output_path),
            ],
            check=True,
            cwd=repo_root(),
            capture_output=True,
            text=True,
        )

        reader = mod.GGUFReader(str(output_path))
        try:
            kv_raw = {key: raw for key, _vtype, raw in reader.kv_pairs}
            if mod.get_kv_value(reader.kv_pairs, "toyarch.block_count") != 1:
                raise SystemExit("arch block_count was not rewritten")
            if mod.get_kv_value(reader.kv_pairs, "general.block_count") != 1:
                raise SystemExit("general block_count was not rewritten")
            if kv_raw["toyarch.quant_schemes"] != pack_string_array_raw(mod, ["q4_k", "q6_k"]):
                raise SystemExit("string array KV was corrupted during fixture rewrite")
            if kv_raw["general.some_u32s"] != pack_u32_array_raw(mod, [4, 8, 15, 16, 23, 42]):
                raise SystemExit("u32 array KV was corrupted during fixture rewrite")

            tensor_names = [tensor["name"] for tensor in reader.tensors]
            if tensor_names != ["tok_embeddings.weight", "blk.0.ffn_gate_exps.weight"]:
                raise SystemExit(f"unexpected tensor set after truncation: {tensor_names}")
        finally:
            reader.close()


if __name__ == "__main__":
    main()
