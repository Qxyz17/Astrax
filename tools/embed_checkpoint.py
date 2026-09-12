from pathlib import Path
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: embed_checkpoint.py <checkpoint> <header>", file=sys.stderr)
        return 2
    source = Path(sys.argv[1])
    target = Path(sys.argv[2])
    data = source.read_bytes()
    lines = [
        "#pragma once",
        "#include <cstdint>",
        "#include <vector>",
        "namespace astrax::embedded {",
        "inline const std::vector<std::uint8_t> kCheckpoint = {",
    ]
    for index in range(0, len(data), 16):
        lines.append("    " + ", ".join(f"0x{value:02X}" for value in data[index:index + 16]) + ",")
    lines.extend([
        "};",
        "} // namespace astrax::embedded",
        "",
    ])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
