// common.cuh - error checking and GPU timing helpers (host side).
#pragma once

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

// Every CUDA runtime call can fail silently unless you look at its return code.
#define CUDA_CHECK(call)                                                               \
    do {                                                                               \
        cudaError_t err__ = (call);                                                    \
        if (err__ != cudaSuccess) {                                                    \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err__), \
                         __FILE__, __LINE__, cudaGetErrorString(err__));               \
            std::exit(EXIT_FAILURE);                                                   \
        }                                                                              \
    } while (0)

// Kernel launches are asynchronous, so wall-clock timing on the host measures
// nothing useful. CUDA events are timestamps recorded *on the GPU's stream*.
class GpuTimer {
public:
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    void begin() { CUDA_CHECK(cudaEventRecord(start_)); }

    // Returns elapsed milliseconds between begin() and now.
    float end() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_{}, stop_{};
};

// RAII device buffer so early exits cannot leak GPU memory.
template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : count_(count) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return ptr_; }
    size_t count() const { return count_; }
    size_t bytes() const { return count_ * sizeof(T); }

private:
    T* ptr_ = nullptr;
    size_t count_;
};
