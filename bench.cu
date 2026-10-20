// bench.cu - correctness check + benchmark of the kernels in kernels.cuh.
//
//   ./bench                          default sizes
//   ./bench --sizes 512,1024,2048    matmul sizes (square N x N x N)
//   ./bench --csv results/run1.csv   also write the results as CSV
//   ./bench --help
//
// Phases:
//   1. device info
//   2. verification of every kernel against a CPU reference (exit code 1 on failure)
//   3. benchmarks: vec_add (bandwidth), matmul (GFLOP/s), reductions (GB/s)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#ifdef USE_CUBLAS
#include <cublas_v2.h>
#endif

#include "common.cuh"
#include "kernels.cuh"

// ------------------------------------------------------------------ options
struct Options {
    std::vector<int> sizes = {512, 1024, 2048};
    size_t vecSize = size_t{1} << 25;     // 32M floats = 128 MB per array
    size_t reduceSize = size_t{1} << 25;
    int iters = 20;
    int warmup = 3;
    bool verify = true;
    std::string csvPath;
};

static std::vector<int> parseSizes(const std::string& s) {
    std::vector<int> out;
    size_t pos = 0;
    while (pos < s.size()) {
        size_t comma = s.find(',', pos);
        if (comma == std::string::npos) comma = s.size();
        int v = std::atoi(s.substr(pos, comma - pos).c_str());
        if (v > 0) out.push_back(v);
        pos = comma + 1;
    }
    return out;
}

static void usage() {
    std::printf(
        "usage: bench [options]\n"
        "  --sizes a,b,c     square matmul sizes (default 512,1024,2048)\n"
        "  --vec-size N      elements for vec_add (default 33554432)\n"
        "  --reduce-size N   elements for reductions (default 33554432)\n"
        "  --iters N         timed iterations per measurement (default 20)\n"
        "  --csv PATH        also write results to a CSV file\n"
        "  --skip-verify     skip the correctness phase\n"
        "  --help\n");
}

static Options parseArgs(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", a.c_str());
                std::exit(64);
            }
            return argv[++i];
        };
        if (a == "--sizes") o.sizes = parseSizes(next());
        else if (a == "--vec-size") o.vecSize = std::strtoull(next(), nullptr, 10);
        else if (a == "--reduce-size") o.reduceSize = std::strtoull(next(), nullptr, 10);
        else if (a == "--iters") o.iters = std::max(1, std::atoi(next()));
        else if (a == "--csv") o.csvPath = next();
        else if (a == "--skip-verify") o.verify = false;
        else if (a == "--help" || a == "-h") { usage(); std::exit(0); }
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); usage(); std::exit(64); }
    }
    if (o.sizes.empty()) { std::fprintf(stderr, "no valid --sizes given\n"); std::exit(64); }
    return o;
}

// ------------------------------------------------------------------ results
struct Row {
    std::string experiment, kernel;
    size_t size;
    double ms;
    double metric;
    std::string unit;
};
static std::vector<Row> g_rows;

static void record(const std::string& exp, const std::string& kernel, size_t size, double ms,
                   double metric, const std::string& unit) {
    g_rows.push_back({exp, kernel, size, ms, metric, unit});
}

static void writeCsv(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (!f) { std::fprintf(stderr, "cannot write %s\n", path.c_str()); return; }
    std::fprintf(f, "experiment,kernel,size,ms,metric,unit\n");
    for (const Row& r : g_rows)
        std::fprintf(f, "%s,%s,%zu,%.6f,%.4f,%s\n", r.experiment.c_str(), r.kernel.c_str(), r.size,
                     r.ms, r.metric, r.unit.c_str());
    std::fclose(f);
    std::printf("\nwrote %s\n", path.c_str());
}

// ------------------------------------------------------------------ timing
// Average time of fn() in milliseconds, measured with CUDA events on the GPU.
static double timeMs(const std::function<void()>& fn, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) fn();  // warm caches, clocks and the driver
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.begin();
    for (int i = 0; i < iters; ++i) fn();
    const float totalMs = timer.end();
    CUDA_CHECK(cudaGetLastError());  // catches launch-configuration errors
    return totalMs / iters;
}

// ------------------------------------------------------------------ matmul
#ifdef USE_CUBLAS
static cublasHandle_t g_cublas = nullptr;
#endif

using MatmulFn = std::function<void(const float*, const float*, float*, int, int, int)>;

static void launchNaive(const float* A, const float* B, float* C, int M, int K, int N) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_naive<<<grid, block>>>(A, B, C, M, K, N);
}

static void launchTiled(const float* A, const float* B, float* C, int M, int K, int N) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_tiled<<<grid, block>>>(A, B, C, M, K, N);
}

#ifdef USE_CUBLAS
// cuBLAS is column-major. Our matrices are row-major, and a row-major matrix is
// the column-major layout of its transpose. So C = A*B is computed as
// C^T = B^T * A^T, i.e. swap the operands and the m/n dimensions.
static void launchCublas(const float* A, const float* B, float* C, int M, int K, int N) {
    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N);
}
#endif

static std::vector<float> randomFloats(size_t n, unsigned seed, float lo, float hi) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(gen);
    return v;
}

// Compare a handful of entries of C against a double-precision dot product.
// Returns the largest error, normalised by the magnitude of the terms summed.
static double spotCheck(const std::vector<float>& A, const std::vector<float>& B,
                        const std::vector<float>& C, int M, int K, int N, int samples) {
    std::mt19937 gen(99);
    double worst = 0.0;
    for (int s = 0; s < samples; ++s) {
        const int i = static_cast<int>(gen() % static_cast<unsigned>(M));
        const int j = static_cast<int>(gen() % static_cast<unsigned>(N));
        double ref = 0.0, scale = 1.0;
        for (int k = 0; k < K; ++k) {
            const double term = static_cast<double>(A[static_cast<size_t>(i) * K + k]) *
                                B[static_cast<size_t>(k) * N + j];
            ref += term;
            scale += std::fabs(term);
        }
        const double err = std::fabs(C[static_cast<size_t>(i) * N + j] - ref) / scale;
        // The buffers are pre-filled with NaN, and std::max silently ignores NaN, so a
        // kernel that never wrote its output would otherwise look correct.
        if (std::isnan(err)) return INFINITY;
        worst = std::max(worst, err);
    }
    return worst;
}

struct NamedMatmul {
    const char* name;
    MatmulFn fn;
};

static std::vector<NamedMatmul> matmulKernels() {
    std::vector<NamedMatmul> v;
    v.push_back({"naive", launchNaive});
    v.push_back({"tiled", launchTiled});
#ifdef USE_CUBLAS
    v.push_back({"cublas", launchCublas});
#endif
    return v;
}

// Run each matmul kernel on M x K x N and spot-check the result. Returns true if all pass.
static bool verifyMatmul(int M, int K, int N) {
    auto hA = randomFloats(static_cast<size_t>(M) * K, 1, -1.0f, 1.0f);
    auto hB = randomFloats(static_cast<size_t>(K) * N, 2, -1.0f, 1.0f);
    DeviceBuffer<float> dA(hA.size()), dB(hB.size()), dC(static_cast<size_t>(M) * N);
    CUDA_CHECK(cudaMemcpy(dA.get(), hA.data(), dA.bytes(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB.get(), hB.data(), dB.bytes(), cudaMemcpyHostToDevice));

    bool allOk = true;
    for (const auto& k : matmulKernels()) {
        CUDA_CHECK(cudaMemset(dC.get(), 0xFF, dC.bytes()));  // NaN sentinel: must be overwritten
        k.fn(dA.get(), dB.get(), dC.get(), M, K, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> hC(dC.count());
        CUDA_CHECK(cudaMemcpy(hC.data(), dC.get(), dC.bytes(), cudaMemcpyDeviceToHost));
        const double err = spotCheck(hA, hB, hC, M, K, N, 200);
        const bool ok = err < 1e-4;  // also false for NaN
        std::printf("  %s  matmul_%-6s %dx%dx%d  max normalized error %.2e\n", ok ? "PASS" : "FAIL",
                    k.name, M, K, N, err);
        allOk = allOk && ok;
    }
    return allOk;
}

static void benchMatmul(const Options& opt) {
    std::printf("\n== matmul: C = A*B, square N x N x N, fp32 ==\n");
    std::printf("%-8s %-8s %10s %12s %10s\n", "N", "kernel", "ms", "GFLOP/s", "vs naive");
    for (int n : opt.sizes) {
        auto hA = randomFloats(static_cast<size_t>(n) * n, 1, -1.0f, 1.0f);
        auto hB = randomFloats(static_cast<size_t>(n) * n, 2, -1.0f, 1.0f);
        DeviceBuffer<float> dA(hA.size()), dB(hB.size()), dC(hA.size());
        CUDA_CHECK(cudaMemcpy(dA.get(), hA.data(), dA.bytes(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB.get(), hB.data(), dB.bytes(), cudaMemcpyHostToDevice));

        const double flops = 2.0 * n * n * static_cast<double>(n);
        double naiveMs = 0.0;
        for (const auto& k : matmulKernels()) {
            const double ms = timeMs(
                [&] { k.fn(dA.get(), dB.get(), dC.get(), n, n, n); }, opt.warmup, opt.iters);
            const double gflops = flops / (ms * 1e-3) / 1e9;
            if (std::strcmp(k.name, "naive") == 0) naiveMs = ms;
            std::printf("%-8d %-8s %10.3f %12.1f %9.2fx\n", n, k.name, ms, gflops, naiveMs / ms);
            record("matmul", k.name, static_cast<size_t>(n), ms, gflops, "GFLOP/s");

            // Also confirm the big result is right (cheap: 64 sampled entries).
            std::vector<float> hC(dC.count());
            CUDA_CHECK(cudaMemcpy(hC.data(), dC.get(), dC.bytes(), cudaMemcpyDeviceToHost));
            if (spotCheck(hA, hB, hC, n, n, n, 64) > 1e-4) {
                std::fprintf(stderr, "WRONG RESULT from matmul_%s at N=%d\n", k.name, n);
                std::exit(1);
            }
        }
    }
}

// ------------------------------------------------------------------ vec_add
static bool verifyVecAdd(size_t n) {
    auto hA = randomFloats(n, 1, -1.0f, 1.0f), hB = randomFloats(n, 2, -1.0f, 1.0f);
    DeviceBuffer<float> dA(n), dB(n), dC(n);
    CUDA_CHECK(cudaMemcpy(dA.get(), hA.data(), dA.bytes(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB.get(), hB.data(), dB.bytes(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC.get(), 0xFF, dC.bytes()));
    vec_add<<<static_cast<unsigned>((n + 255) / 256), 256>>>(dA.get(), dB.get(), dC.get(), n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> hC(n);
    CUDA_CHECK(cudaMemcpy(hC.data(), dC.get(), dC.bytes(), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (size_t i = 0; i < n && ok; ++i) ok = (hC[i] == hA[i] + hB[i]);
    std::printf("  %s  vec_add n=%zu\n", ok ? "PASS" : "FAIL", n);
    return ok;
}

static void benchVecAdd(const Options& opt) {
    const size_t n = opt.vecSize;
    std::printf("\n== vec_add: memory-bound, n = %zu floats ==\n", n);
    DeviceBuffer<float> dA(n), dB(n), dC(n);
    CUDA_CHECK(cudaMemset(dA.get(), 0, dA.bytes()));
    CUDA_CHECK(cudaMemset(dB.get(), 0, dB.bytes()));

    const double trafficBytes = 3.0 * n * sizeof(float);  // read a, read b, write c
    const double ms = timeMs(
        [&] {
            vec_add<<<static_cast<unsigned>((n + 255) / 256), 256>>>(dA.get(), dB.get(), dC.get(), n);
        },
        opt.warmup, opt.iters);
    const double gbps = trafficBytes / (ms * 1e-3) / 1e9;

    // Practical bandwidth ceiling: what the driver's own device-to-device copy achieves.
    const double copyMs = timeMs(
        [&] { CUDA_CHECK(cudaMemcpy(dC.get(), dA.get(), dA.bytes(), cudaMemcpyDeviceToDevice)); },
        opt.warmup, opt.iters);
    const double copyGbps = 2.0 * n * sizeof(float) / (copyMs * 1e-3) / 1e9;  // read + write

    std::printf("%-22s %10s %12s\n", "kernel", "ms", "GB/s");
    std::printf("%-22s %10.3f %12.1f\n", "vec_add", ms, gbps);
    std::printf("%-22s %10.3f %12.1f\n", "cudaMemcpy D2D (ref)", copyMs, copyGbps);
    std::printf("vec_add reaches %.0f%% of the memcpy bandwidth reference\n", 100.0 * gbps / copyGbps);
    record("vec_add", "vec_add", n, ms, gbps, "GB/s");
    record("vec_add", "memcpy_d2d_reference", n, copyMs, copyGbps, "GB/s");
}

// ------------------------------------------------------------------ reductions
using ReduceFn = std::function<void(const float*, float*, size_t, int)>;

struct NamedReduce {
    const char* name;
    ReduceFn fn;
};

static std::vector<NamedReduce> reduceKernels() {
    return {
        {"atomic_only",
         [](const float* in, float* out, size_t n, int blocks) {
             reduce_atomic<REDUCE_BLOCK><<<blocks, REDUCE_BLOCK>>>(in, out, n);
         }},
        {"shared_tree",
         [](const float* in, float* out, size_t n, int blocks) {
             reduce_shared<REDUCE_BLOCK><<<blocks, REDUCE_BLOCK>>>(in, out, n);
         }},
        {"warp_shuffle",
         [](const float* in, float* out, size_t n, int blocks) {
             reduce_warp<REDUCE_BLOCK><<<blocks, REDUCE_BLOCK>>>(in, out, n);
         }},
    };
}

static int reduceBlocks() {
    int dev = 0, sms = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    return 8 * sms;  // a few resident blocks per SM; grid-stride loops cover the rest
}

static bool verifyReduce(size_t n) {
    auto h = randomFloats(n, 7, 0.0f, 1.0f);
    double want = 0.0;
    for (float x : h) want += x;
    DeviceBuffer<float> dIn(n), dOut(1);
    CUDA_CHECK(cudaMemcpy(dIn.get(), h.data(), dIn.bytes(), cudaMemcpyHostToDevice));

    bool allOk = true;
    for (const auto& k : reduceKernels()) {
        CUDA_CHECK(cudaMemset(dOut.get(), 0, sizeof(float)));
        k.fn(dIn.get(), dOut.get(), n, reduceBlocks());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        float got = 0.0f;
        CUDA_CHECK(cudaMemcpy(&got, dOut.get(), sizeof(float), cudaMemcpyDeviceToHost));
        const double rel = std::fabs(got - want) / want;
        const bool ok = rel < 1e-4;
        std::printf("  %s  reduce_%-12s n=%zu  relative error %.2e\n", ok ? "PASS" : "FAIL", k.name, n, rel);
        allOk = allOk && ok;
    }
    return allOk;
}

static void benchReduce(const Options& opt) {
    const size_t n = opt.reduceSize;
    const int blocks = reduceBlocks();
    std::printf("\n== sum reduction: n = %zu floats, %d blocks x %d threads ==\n", n, blocks, REDUCE_BLOCK);
    std::printf("%-14s %10s %12s %12s\n", "kernel", "ms", "GB/s", "vs atomic");

    DeviceBuffer<float> dIn(n), dOut(1);
    {
        auto h = randomFloats(std::min<size_t>(n, 1 << 20), 7, 0.0f, 1.0f);
        CUDA_CHECK(cudaMemset(dIn.get(), 0, dIn.bytes()));
        CUDA_CHECK(cudaMemcpy(dIn.get(), h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    double atomicMs = 0.0;
    for (const auto& k : reduceKernels()) {
        const double ms = timeMs(
            [&] {
                CUDA_CHECK(cudaMemsetAsync(dOut.get(), 0, sizeof(float)));  // tiny, same for all kernels
                k.fn(dIn.get(), dOut.get(), n, blocks);
            },
            opt.warmup, opt.iters);
        const double gbps = static_cast<double>(n) * sizeof(float) / (ms * 1e-3) / 1e9;
        if (std::strcmp(k.name, "atomic_only") == 0) atomicMs = ms;
        std::printf("%-14s %10.3f %12.1f %11.2fx\n", k.name, ms, gbps, atomicMs / ms);
        record("reduce", k.name, n, ms, gbps, "GB/s");
    }
}

// ------------------------------------------------------------------ main
int main(int argc, char** argv) {
    const Options opt = parseArgs(argc, argv);

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device: %s  (compute capability %d.%d, %d SMs, %.1f GB, %zu KB shared mem/block)\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                prop.totalGlobalMem / 1e9, prop.sharedMemPerBlock / 1024);
#ifdef USE_CUBLAS
    if (cublasCreate(&g_cublas) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cublasCreate failed\n");
        return 1;
    }
#else
    std::printf("(built without cuBLAS; rebuild with `make cublas` to add the library baseline)\n");
#endif

    if (opt.verify) {
        std::printf("\n== verification against CPU reference ==\n");
        bool ok = true;
        ok = verifyVecAdd((size_t{1} << 20) + 3) && ok;      // not a multiple of the block size
        ok = verifyMatmul(250, 190, 210) && ok;              // not multiples of TILE
        ok = verifyMatmul(16, 16, 16) && ok;
        ok = verifyMatmul(1, 33, 1) && ok;
        ok = verifyReduce((size_t{1} << 20) + 5) && ok;
        if (!ok) {
            std::fprintf(stderr, "\nverification FAILED - not benchmarking wrong kernels\n");
            return 1;
        }
    }

    benchVecAdd(opt);
    benchMatmul(opt);
    benchReduce(opt);

    if (!opt.csvPath.empty()) writeCsv(opt.csvPath);
#ifdef USE_CUBLAS
    cublasDestroy(g_cublas);
#endif
    return 0;
}
