// cuda_shim.hpp - a tiny CPU emulation of the CUDA execution model.
//
// Purpose: let the kernels in src/kernels.cuh run (and be unit-tested) on a
// machine with no GPU and no nvcc. It is NOT a performance tool.
//
// What is emulated:
//   * threadIdx / blockIdx / blockDim / gridDim
//   * one std::thread per CUDA thread inside a block; blocks run one after another
//   * __shared__  (one copy per kernel, shared by the threads of the running block)
//   * __syncthreads()  (a real barrier across the block's threads)
//   * atomicAdd(float*, float)
//   * __shfl_down_sync  (per-warp barrier + exchange buffer; 1-D blocks only)
//
// Limits: z dimension unsupported, blockDim must be a multiple of 32 for shuffles,
// and unlike a real GPU, blocks never run concurrently, so inter-block races are
// not exercised.
#pragma once

#include <atomic>
#include <barrier>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#define __global__
#define __device__
#define __host__
#define __shared__ static

struct dim3 {
    unsigned x, y, z;
    dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
};

namespace cudasim {
inline thread_local dim3 tIdx;
inline thread_local dim3 bIdx;
inline dim3 bDim;
inline dim3 gDim;
inline std::barrier<>* blockBarrier = nullptr;
inline std::vector<std::unique_ptr<std::barrier<>>> warpBarriers;
inline unsigned char exchange[1024][8];
}  // namespace cudasim

#define threadIdx cudasim::tIdx
#define blockIdx cudasim::bIdx
#define blockDim cudasim::bDim
#define gridDim cudasim::gDim

inline void __syncthreads() { cudasim::blockBarrier->arrive_and_wait(); }

inline float atomicAdd(float* address, float value) {
    std::atomic_ref<float> ref(*address);
    return ref.fetch_add(value);
}

template <class T>
inline T __shfl_down_sync(unsigned /*mask*/, T value, int delta) {
    static_assert(sizeof(T) <= 8, "shim supports values up to 8 bytes");
    const unsigned tid = cudasim::tIdx.x;
    const unsigned warp = tid / 32;
    const unsigned lane = tid % 32;
    std::memcpy(cudasim::exchange[tid], &value, sizeof(T));
    cudasim::warpBarriers[warp]->arrive_and_wait();  // everyone has published
    T result = value;                                // out-of-range lanes keep their own value
    if (lane + static_cast<unsigned>(delta) < 32)
        std::memcpy(&result, cudasim::exchange[tid + static_cast<unsigned>(delta)], sizeof(T));
    cudasim::warpBarriers[warp]->arrive_and_wait();  // everyone has read
    return result;
}

namespace cudasim {

// Run `kernelCall()` once per CUDA thread of every block in the grid.
template <class F>
void launch(dim3 grid, dim3 block, F kernelCall) {
    bDim = block;
    gDim = grid;
    const unsigned nthreads = block.x * block.y * block.z;
    const unsigned nwarps = (nthreads + 31) / 32;

    for (unsigned by = 0; by < grid.y; ++by) {
        for (unsigned bx = 0; bx < grid.x; ++bx) {
            std::barrier<> blockBar(static_cast<std::ptrdiff_t>(nthreads));
            blockBarrier = &blockBar;
            warpBarriers.clear();
            for (unsigned w = 0; w < nwarps; ++w) {
                unsigned inWarp = (w + 1) * 32 <= nthreads ? 32 : nthreads - w * 32;
                warpBarriers.push_back(
                    std::make_unique<std::barrier<>>(static_cast<std::ptrdiff_t>(inWarp)));
            }

            std::vector<std::thread> threads;
            threads.reserve(nthreads);
            for (unsigned ty = 0; ty < block.y; ++ty) {
                for (unsigned tx = 0; tx < block.x; ++tx) {
                    threads.emplace_back([=, &kernelCall] {
                        tIdx = dim3(tx, ty, 0);
                        bIdx = dim3(bx, by, 0);
                        kernelCall();
                    });
                }
            }
            for (auto& t : threads) t.join();
        }
    }
    blockBarrier = nullptr;
}

}  // namespace cudasim
