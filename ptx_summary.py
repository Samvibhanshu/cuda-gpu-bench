#!/usr/bin/env python3
"""Count the interesting instructions in each kernel of a .ptx file.

    python3 tools/ptx_summary.py ptx/kernels_sm80.clang.ptx
    nvcc -ptx -arch=sm_80 src/bench.cu -o ptx/kernels_nvcc.ptx && python3 tools/ptx_summary.py ptx/kernels_nvcc.ptx

PTX is NVIDIA's virtual GPU assembly. Reading it shows what a kernel really costs:
global loads vs shared-memory loads, how many barriers (bar.sync), shuffles, atomics.
"""
import re
import sys

COLUMNS = {
    "ld.global": r"\bld\.global",
    "ld.shared": r"\bld\.shared",
    "st.shared": r"\bst\.shared",
    "st.global": r"\bst\.global",
    "fma": r"\bfma\.rn\.f32",
    "bar.sync": r"\bbar\.sync",
    "shfl": r"\bshfl\.sync",
    "atom": r"\batom\.",
    "bra": r"\bbra",
}


def demangle(name: str) -> str:
    """Tiny best-effort demangler: _Z12matmul_tiledPKf... -> matmul_tiled."""
    m = re.match(r"_Z(\d+)", name)
    if not m:
        return name
    n = int(m.group(1))
    start = m.end()
    base = name[start:start + n]
    return base + ("<...>" if name[start + n:start + n + 1] == "I" else "")


def main(path: str) -> int:
    try:
        text = open(path).read()
    except OSError as e:
        print(f"cannot read {path}: {e}", file=sys.stderr)
        return 1
    kernels = re.split(r"(?=^\.visible \.entry )", text, flags=re.M)[1:]
    if not kernels:
        print("no kernels found (is this a PTX file?)", file=sys.stderr)
        return 1
    target = re.search(r"^\.target (\S+)", text, flags=re.M)
    version = re.search(r"^\.version (\S+)", text, flags=re.M)
    print(f"PTX version {version.group(1) if version else '?'}, target {target.group(1) if target else '?'}\n")
    print(f"{'kernel':22}" + "".join(f"{c:>10}" for c in COLUMNS))
    for body in kernels:
        mangled = re.match(r"\.visible \.entry (\S+)\(", body).group(1)
        counts = "".join(f"{len(re.findall(p, body)):>10}" for p in COLUMNS.values())
        print(f"{demangle(mangled):22}{counts}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(64)
    sys.exit(main(sys.argv[1]))
