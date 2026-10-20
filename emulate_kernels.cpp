// emulate_kernels.cpp - run the REAL kernels from src/kernels.cuh on the CPU
// (through cuda_shim.hpp) and compare against straightforward reference code.
//
//   g++ -std=c++20 -pthread -Isrc -Itest test/emulate_kernels.cpp -o emulate && ./emulate
//
// This checks indexing, bounds handling, tiling and synchronization logic. It
// cannot say anything about GPU performance.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "cuda_shim.hpp"   // must come before kernels.cuh: defines __global__ etc.
#include "kernels.cuh"

static int g_failures = 0;
static int g_checks = 0;

static void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) ++g_failures;
    std::printf("  %s  %s\n", ok ? "PASS" : "FAIL", what.c_str());
}

// Small deterministic generator so failures are reproducible.
struct Lcg {
    unsigned long long s = 12345;
    float next() {  // uniform in [-1, 1)
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<float>((s >> 40) & 0xFFFFFF) / 8388608.0f - 1.0f;
    }
};

static std::vector<float> randomVec(size_t n, unsigned long long seed, bool positive = false) {
    Lcg g;
    g.s = seed;
    std::vector<float> v(n);
    for (auto& x : v) x = positive ? (g.next() + 1.0f) * 0.5f : g.next();
    return v;
}

static std::vector<double> cpuMatmul(const std::vector<float>& A, const std::vector<float>& B,
                                     int M, int K, int N) {
    std::vector<double> C(static_cast<size_t>(M) * N, 0.0);
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k) {
            double a = A[static_cast<size_t>(i) * K + k];
            for (int j = 0; j < N; ++j)
                C[static_cast<size_t>(i) * N + j] += a * B[static_cast<size_t>(k) * N + j];
        }
    return C;
}

static double maxError(const std::vector<float>& got, const std::vector<double>& want) {
    double worst = 0;
    for (size_t i = 0; i < got.size(); ++i)
        worst = std::fmax(worst, std::fabs(got[i] - want[i]) / (1.0 + std::fabs(want[i])));
    return worst;
}

static void testVecAdd(size_t n, unsigned gridX, unsigned blockX) {
    auto a = randomVec(n, 1), b = randomVec(n, 2);
    std::vector<float> c(n, -999.0f);
    cudasim::launch(dim3(gridX), dim3(blockX),
                    [&] { vec_add(a.data(), b.data(), c.data(), n); });
    bool ok = true;
    for (size_t i = 0; i < n; ++i) ok = ok && (c[i] == a[i] + b[i]);
    check(ok, "vec_add n=" + std::to_string(n) + " grid=" + std::to_string(gridX) +
                  " block=" + std::to_string(blockX) + " (grid-stride covers every element)");
}

template <class KernelFn>
static void testMatmul(const char* name, KernelFn kernel, int M, int K, int N) {
    auto A = randomVec(static_cast<size_t>(M) * K, 3);
    auto B = randomVec(static_cast<size_t>(K) * N, 4);
    std::vector<float> C(static_cast<size_t>(M) * N, -999.0f);  // sentinel: must be overwritten
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    cudasim::launch(grid, block, [&] { kernel(A.data(), B.data(), C.data(), M, K, N); });
    double err = maxError(C, cpuMatmul(A, B, M, K, N));
    char buf[160];
    std::snprintf(buf, sizeof buf, "%s %dx%dx%d  max normalized error %.2e", name, M, K, N, err);
    check(err < 1e-5, buf);
}

template <class KernelFn>
static void testReduce(const char* name, KernelFn kernel, size_t n, unsigned blocks) {
    auto in = randomVec(n, 5, /*positive=*/true);
    double want = 0;
    for (float x : in) want += x;
    float out = 0.0f;
    cudasim::launch(dim3(blocks), dim3(REDUCE_BLOCK), [&] { kernel(in.data(), &out, n); });
    double rel = std::fabs(out - want) / want;
    char buf[160];
    std::snprintf(buf, sizeof buf, "%s n=%zu blocks=%u  relative error %.2e", name, n, blocks, rel);
    check(rel < 1e-5, buf);
}

int main() {
    std::printf("== vec_add ==\n");
    testVecAdd(1000, 3, 64);      // fewer threads than elements: needs the stride loop
    testVecAdd(7, 4, 32);         // more threads than elements: bounds check
    testVecAdd(4096, 16, 256);

    std::printf("== matmul (sizes chosen to NOT be multiples of the tile) ==\n");
    const int shapes[][3] = {{1, 1, 1},   {16, 16, 16}, {17, 1, 33},  {37, 29, 41},
                             {50, 70, 30}, {64, 64, 64}, {33, 100, 17}, {5, 200, 5}};
    for (auto& s : shapes) {
        testMatmul("matmul_naive", matmul_naive, s[0], s[1], s[2]);
        testMatmul("matmul_tiled", matmul_tiled, s[0], s[1], s[2]);
    }

    std::printf("== reductions ==\n");
    const size_t sizes[] = {1, 100, 5000, 100000};
    for (size_t n : sizes) {
        testReduce("reduce_atomic", reduce_atomic<REDUCE_BLOCK>, n, 4);
        testReduce("reduce_shared", reduce_shared<REDUCE_BLOCK>, n, 4);
        testReduce("reduce_warp  ", reduce_warp<REDUCE_BLOCK>, n, 4);
    }
    testReduce("reduce_shared", reduce_shared<REDUCE_BLOCK>, 100000, 1);  // single block
    testReduce("reduce_warp  ", reduce_warp<REDUCE_BLOCK>, 100000, 1);

    std::printf("\nchecks: %d   failures: %d\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
