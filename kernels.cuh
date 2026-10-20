// kernels.cuh - every GPU kernel in the project.
//
// This header contains ONLY kernel code (no CUDA runtime API calls), so it can be
// compiled two ways:
//   * by nvcc, as part of bench.cu (the real thing), and
//   * by a plain C++ compiler together with test/cuda_shim.hpp, which emulates
//     threads, blocks, shared memory and __syncthreads on the CPU so the kernel
//     logic can be unit-tested on a machine with no GPU (see `make emulate`).
#pragma once

#include <cstddef>

constexpr int TILE = 16;            // matmul tile edge: a TILE x TILE thread block
constexpr int REDUCE_BLOCK = 256;   // threads per block for the reduction kernels

// ============================================================================
// 1. Vector add:  c = a + b
//    Memory-bound: 2 loads + 1 store per element, 1 add. The GPU spends its time
//    waiting for DRAM, so the right metric is GB/s, not FLOP/s.
//    A grid-stride loop lets any launch size cover any n.
// ============================================================================
__global__ void vec_add(const float* __restrict__ a, const float* __restrict__ b,
                        float* __restrict__ c, size_t n) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += stride)
        c[i] = a[i] + b[i];
}

// ============================================================================
// 2. Matrix multiply:  C(MxN) = A(MxK) * B(KxN), all row-major float.
// ============================================================================

// 2a. Naive: one thread per output element, everything read from global memory.
//     Each thread does K multiply-adds and 2K global loads. Neighbouring threads
//     (consecutive col) read consecutive B elements, so B loads coalesce, but the
//     same A row and B column data is re-fetched by many threads again and again.
__global__ void matmul_naive(const float* __restrict__ A, const float* __restrict__ B,
                             float* __restrict__ C, int M, int K, int N) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
        acc += A[static_cast<size_t>(row) * K + k] * B[static_cast<size_t>(k) * N + col];
    C[static_cast<size_t>(row) * N + col] = acc;
}

// 2b. Tiled with shared memory. A block cooperatively loads a TILE x TILE tile of A
//     and of B into fast on-chip shared memory, syncs, then every thread does TILE
//     multiply-adds out of shared memory. Each element fetched from global memory
//     is now reused TILE times, cutting global traffic by a factor of TILE.
//
//     Two rules this kernel must obey:
//       * every thread of the block must reach every __syncthreads(), so there is
//         NO early return before the loop (out-of-range threads load zeros instead),
//       * the second __syncthreads() stops a fast thread from overwriting the tile
//         while a slow thread is still reading it.
__global__ void matmul_tiled(const float* __restrict__ A, const float* __restrict__ B,
                             float* __restrict__ C, int M, int K, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;
    const int numTiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; ++t) {
        const int aCol = t * TILE + threadIdx.x;
        const int bRow = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < M && aCol < K) ? A[static_cast<size_t>(row) * K + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (bRow < K && col < N) ? B[static_cast<size_t>(bRow) * N + col] : 0.0f;
        __syncthreads();  // tile fully loaded

        for (int k = 0; k < TILE; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();  // everyone done reading before the next tile overwrites it
    }

    if (row < M && col < N) C[static_cast<size_t>(row) * N + col] = acc;
}

// ============================================================================
// 3. Sum reduction: out += sum(in[0..n)). `out` must be zeroed before the launch.
//    Reductions are the classic synchronization benchmark: many threads must
//    combine into one value.
// ============================================================================

// 3a. Contended atomics: each thread sums a grid-stride slice, then every one of
//     the gridDim*blockDim threads atomically adds into the SAME address. The
//     atomics serialize at that one memory location.
template <int BLOCK>
__global__ void reduce_atomic(const float* __restrict__ in, float* out, size_t n) {
    float sum = 0.0f;
    const size_t stride = static_cast<size_t>(gridDim.x) * BLOCK;
    for (size_t i = static_cast<size_t>(blockIdx.x) * BLOCK + threadIdx.x; i < n; i += stride)
        sum += in[i];
    atomicAdd(out, sum);
}

// 3b. Shared-memory tree: reduce within the block in shared memory (log2(BLOCK)
//     steps, each separated by __syncthreads), then ONE atomic per block.
template <int BLOCK>
__global__ void reduce_shared(const float* __restrict__ in, float* out, size_t n) {
    __shared__ float s[BLOCK];
    const int tid = threadIdx.x;

    float sum = 0.0f;
    const size_t stride = static_cast<size_t>(gridDim.x) * BLOCK;
    for (size_t i = static_cast<size_t>(blockIdx.x) * BLOCK + tid; i < n; i += stride)
        sum += in[i];
    s[tid] = sum;
    __syncthreads();

    for (int half = BLOCK / 2; half > 0; half >>= 1) {
        if (tid < half) s[tid] += s[tid + half];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, s[0]);
}

// 3c. Warp shuffle: threads of a warp exchange registers directly, with no shared
//     memory and no block-wide barrier. Reduce each warp with shuffles, let lane 0
//     of each warp park its result in shared memory, then warp 0 reduces those.
template <int BLOCK>
__global__ void reduce_warp(const float* __restrict__ in, float* out, size_t n) {
    static_assert(BLOCK % 32 == 0 && BLOCK <= 1024, "BLOCK must be a multiple of 32, <= 1024");
    __shared__ float warpSums[BLOCK / 32];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    float sum = 0.0f;
    const size_t stride = static_cast<size_t>(gridDim.x) * BLOCK;
    for (size_t i = static_cast<size_t>(blockIdx.x) * BLOCK + tid; i < n; i += stride)
        sum += in[i];

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0) warpSums[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        float v = (lane < BLOCK / 32) ? warpSums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, offset);
        if (lane == 0) atomicAdd(out, v);
    }
}
