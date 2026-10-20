#!/usr/bin/env python3
"""Profile PyTorch workloads and read the results like a performance engineer.

    python python/profile_pytorch.py                       # everything, auto-detect GPU
    python python/profile_pytorch.py --train               # include backward pass
    python python/profile_pytorch.py --peak-tflops 19.5 --peak-gbps 900   # classify bounds
    python python/profile_pytorch.py --skip matmul         # skip a section

Sections
  1. matmul    TFLOP/s and arithmetic intensity of an N x N x N matmul sweep
  2. memory    an element-wise op, which is limited by memory bandwidth, not FLOPs
  3. model     torch.profiler on a small transformer stack; top ops + a Chrome trace
  4. precision the same model in FP32, TF32 and FP16 autocast (Tensor Cores)

Runs on CPU too, with smaller defaults, so you can test the script anywhere. The
interesting numbers come from a GPU run.
"""
import argparse
import json
import sys
import time

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function, schedule

RESULTS = []  # collected rows, optionally dumped with --json


# --------------------------------------------------------------------------- helpers
def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def timed_ms(fn, device, warmup=3, iters=10):
    """Average milliseconds per call of fn(), measured correctly for async GPU work."""
    for _ in range(warmup):
        fn()
    sync(device)
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) * 1e3 / iters


def record(section, name, **values):
    RESULTS.append({"section": section, "name": name, **values})


def header(title):
    print(f"\n{'=' * 8} {title} {'=' * (60 - len(title))}")


# --------------------------------------------------------------------------- 1. matmul
def bench_matmul(device, args):
    header("matmul sweep (fp32, N x N x N)")
    sizes = [256, 512, 1024, 2048, 4096] if device.type == "cuda" else [128, 256, 512, 1024]
    ridge = None
    if args.peak_tflops and args.peak_gbps:
        ridge = (args.peak_tflops * 1e12) / (args.peak_gbps * 1e9)  # FLOP per byte
        print(f"machine balance (ridge point): {ridge:.1f} FLOP/byte")
    print(f"{'N':>6} {'ms':>9} {'TFLOP/s':>9} {'FLOP/byte':>10}  verdict")
    for n in sizes:
        a = torch.randn(n, n, device=device)
        b = torch.randn(n, n, device=device)
        ms = timed_ms(lambda: a @ b, device)
        flops = 2.0 * n ** 3
        tflops = flops / (ms * 1e-3) / 1e12
        # Ideal traffic: read A and B once, write C once (4 bytes per fp32 element).
        intensity = flops / (3.0 * n * n * 4)
        verdict = "-"
        if ridge is not None:
            verdict = "compute-bound" if intensity > ridge else "memory-bound"
        print(f"{n:>6} {ms:>9.3f} {tflops:>9.2f} {intensity:>10.1f}  {verdict}")
        record("matmul", f"n={n}", ms=ms, tflops=tflops, flop_per_byte=intensity)
    print("Small N has low arithmetic intensity (few FLOPs per byte moved) and under-fills the GPU;")
    print("large N has high intensity, so throughput approaches the hardware's compute peak.")


# --------------------------------------------------------------------------- 2. memory
def bench_memory(device, args):
    header("element-wise ops: memory-bound")
    n = 2 ** 26 if device.type == "cuda" else 2 ** 22
    x = torch.randn(n, device=device)
    y = torch.randn(n, device=device)
    out = torch.empty_like(x)

    def gbps(ms, bytes_moved):
        return bytes_moved / (ms * 1e-3) / 1e9

    cases = [
        ("x + y   (2 reads, 1 write)", lambda: torch.add(x, y, out=out), 3 * n * 4),
        ("x * 2.0 (1 read, 1 write)", lambda: torch.mul(x, 2.0, out=out), 2 * n * 4),
        ("copy    (1 read, 1 write)", lambda: out.copy_(x), 2 * n * 4),
    ]
    print(f"{'op':32} {'ms':>9} {'GB/s':>9}")
    for name, fn, moved in cases:
        ms = timed_ms(fn, device)
        print(f"{name:32} {ms:>9.3f} {gbps(ms, moved):>9.1f}")
        record("memory", name.strip(), ms=ms, gbps=gbps(ms, moved))
    print("All three run at about the same GB/s even though the arithmetic differs: the limit is")
    print("DRAM bandwidth. Fusing such ops into one kernel (torch.compile) removes whole round trips.")


# --------------------------------------------------------------------------- 3. model
class Block(nn.Module):
    """Pre-norm transformer block: attention + MLP, each with a residual connection."""

    def __init__(self, d, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


def build_model(device, args):
    return nn.Sequential(*[Block(args.dim, args.heads) for _ in range(args.layers)]).to(device)


def make_step(model, x, train):
    """Returns a function running one forward (and optionally backward) pass."""
    def step():
        with record_function("forward"):
            out = model(x)
        if train:
            with record_function("backward"):
                out.float().pow(2).mean().backward()
            model.zero_grad(set_to_none=True)
    return step


def print_profile_table(prof, device, row_limit):
    # PyTorch renamed cuda_time_* to device_time_* in newer releases; try both.
    keys = (["device_time_total", "cuda_time_total"] if device.type == "cuda" else ["cpu_time_total"])
    for key in keys:
        try:
            print(prof.key_averages().table(sort_by=key, row_limit=row_limit))
            return
        except (AttributeError, KeyError, ValueError):
            continue
    print(prof.key_averages().table(row_limit=row_limit))


def profile_model(device, args):
    header(f"profile: {args.layers}-layer transformer, d={args.dim}, batch={args.batch}, seq={args.seq}"
           + (" (train)" if args.train else " (inference)"))
    model = build_model(device, args)
    x = torch.randn(args.batch, args.seq, args.dim, device=device)
    step = make_step(model, x, args.train)

    params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {params / 1e6:.1f} M")

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    ctx = torch.no_grad() if not args.train else torch.enable_grad()
    with ctx:
        for _ in range(3):  # warm-up outside the profiler (cuDNN/cuBLAS init, allocator)
            step()
        sync(device)
        with profile(
            activities=activities,
            schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
            record_shapes=True,
        ) as prof:
            for _ in range(5):
                step()
                prof.step()
        sync(device)

    print_profile_table(prof, device, args.rows)
    if args.trace:
        prof.export_chrome_trace(args.trace)
        print(f"Chrome trace written to {args.trace}  (open it at chrome://tracing or ui.perfetto.dev)")

    ms = timed_ms(step, device, warmup=2, iters=5)
    print(f"steady-state: {ms:.2f} ms per step")
    record("model", "step", ms=ms, params_m=params / 1e6)
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 2 ** 20
        print(f"peak GPU memory allocated: {peak:.0f} MiB")
        record("model", "peak_memory_mib", value=peak)
    print("How to read the table: sort by device time, find the few ops that dominate, then ask for")
    print("each one whether it is compute-bound (matmul/GEMM kernels) or memory-bound (layernorm,")
    print("GELU, softmax, adds). Many tiny kernels with gaps in the trace point to launch overhead.")


# --------------------------------------------------------------------------- 4. precision
def bench_precision(device, args):
    header("precision modes: the same model in FP32, TF32 and FP16")
    if device.type != "cuda":
        print("skipped: this section needs a CUDA GPU (Tensor Cores).")
        return
    model = build_model(device, args)
    x = torch.randn(args.batch, args.seq, args.dim, device=device)

    results = {}
    with torch.no_grad():
        torch.set_float32_matmul_precision("highest")
        results["fp32"] = timed_ms(lambda: model(x), device)

        torch.set_float32_matmul_precision("high")  # allows TF32 Tensor Cores for fp32 matmuls
        results["tf32"] = timed_ms(lambda: model(x), device)
        torch.set_float32_matmul_precision("highest")

        def fp16_forward():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return model(x)

        results["fp16 autocast"] = timed_ms(fp16_forward, device)

    base = results["fp32"]
    print(f"{'mode':16} {'ms':>9} {'speedup':>9}")
    for mode, ms in results.items():
        print(f"{mode:16} {ms:>9.3f} {base / ms:>8.2f}x")
        record("precision", mode, ms=ms, speedup=base / ms)
    print("TF32 / FP16 let matmuls run on Tensor Cores and halve memory traffic (FP16). The cost is")
    print("reduced precision, so accuracy must be validated for a real model.")


# --------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dim", type=int, help="model width (default 1024 on GPU, 256 on CPU)")
    p.add_argument("--heads", type=int, help="attention heads (default 16 on GPU, 4 on CPU)")
    p.add_argument("--layers", type=int, help="transformer blocks (default 4 on GPU, 2 on CPU)")
    p.add_argument("--batch", type=int, help="batch size (default 16 on GPU, 4 on CPU)")
    p.add_argument("--seq", type=int, help="sequence length (default 512 on GPU, 128 on CPU)")
    p.add_argument("--train", action="store_true", help="profile forward + backward")
    p.add_argument("--rows", type=int, default=12, help="rows in the profiler table")
    p.add_argument("--trace", default="results/pytorch_trace.json", help="Chrome trace path ('' to skip)")
    p.add_argument("--json", default="", help="also write summary numbers to this JSON file")
    p.add_argument("--peak-tflops", type=float, help="GPU peak FP32 TFLOP/s, to classify bounds")
    p.add_argument("--peak-gbps", type=float, help="GPU peak memory bandwidth in GB/s")
    p.add_argument("--skip", nargs="*", default=[], choices=["matmul", "memory", "model", "precision"])
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = device.type == "cuda"
    args.dim = args.dim or (1024 if gpu else 256)
    args.heads = args.heads or (16 if gpu else 4)
    args.layers = args.layers or (4 if gpu else 2)
    args.batch = args.batch or (16 if gpu else 4)
    args.seq = args.seq or (512 if gpu else 128)
    if args.dim % args.heads != 0:
        sys.exit(f"--dim ({args.dim}) must be divisible by --heads ({args.heads})")

    name = torch.cuda.get_device_name(0) if gpu else "CPU (no CUDA device found; numbers are not GPU numbers)"
    print(f"torch {torch.__version__} | device: {name}")
    if args.trace:
        import os
        os.makedirs(os.path.dirname(args.trace) or ".", exist_ok=True)

    torch.manual_seed(0)
    if "matmul" not in args.skip:
        bench_matmul(device, args)
    if "memory" not in args.skip:
        bench_memory(device, args)
    if "model" not in args.skip:
        profile_model(device, args)
    if "precision" not in args.skip:
        bench_precision(device, args)

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"device": name, "torch": torch.__version__, "results": RESULTS}, f, indent=2)
        print(f"\nsummary written to {args.json}")


if __name__ == "__main__":
    main()
