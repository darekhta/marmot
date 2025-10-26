#!/usr/bin/env python3

import argparse
from pathlib import Path


def format_byte_array(data: bytes, symbol: str) -> str:
    lines = []
    for i in range(0, len(data), 12):
        chunk = data[i : i + 12]
        formatted = ", ".join(f"0x{byte:02x}" for byte in chunk)
        lines.append(f"    {formatted}")
    body = ",\n".join(lines)
    return (
        "#include <stdint.h>\n"
        "#include <stddef.h>\n\n"
        f"const uint8_t {symbol}[] = {{\n{body}\n}};\n"
        f"const size_t {symbol}_len = sizeof({symbol});\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed binary file as C array")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("symbol", type=str)
    args = parser.parse_args()

    data = args.input.read_bytes()
    c_source = format_byte_array(data, args.symbol)
    args.output.write_text(c_source)


if __name__ == "__main__":
    main()
