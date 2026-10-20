#!/usr/bin/env python3
"""Summarize results/results.csv written by `./bench --csv`: speedups per experiment.

    python3 tools/summarize_results.py results/results.csv
"""
import csv
import sys
from collections import defaultdict

BASELINES = {"matmul": "naive", "reduce": "atomic_only", "vec_add": "memcpy_d2d_reference"}


def main(path: str) -> int:
    groups = defaultdict(dict)  # (experiment, size) -> kernel -> row
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            groups[(row["experiment"], int(row["size"]))][row["kernel"]] = row
    if not groups:
        print("no rows found", file=sys.stderr)
        return 1
    for (exp, size), kernels in sorted(groups.items()):
        base = kernels.get(BASELINES.get(exp, ""))
        print(f"\n{exp}  size={size}")
        for name, row in kernels.items():
            ms, metric, unit = float(row["ms"]), float(row["metric"]), row["unit"]
            note = ""
            if base is not None and name != BASELINES[exp]:
                if exp == "vec_add":
                    note = f"  ({100 * metric / float(base['metric']):.0f}% of memcpy bandwidth)"
                else:
                    note = f"  ({float(base['ms']) / ms:.2f}x faster than {BASELINES[exp]})"
            print(f"  {name:24} {ms:10.3f} ms  {metric:12.1f} {unit}{note}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(64)
    sys.exit(main(sys.argv[1]))
