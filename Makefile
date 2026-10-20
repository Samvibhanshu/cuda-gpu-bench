# CUDA GPU bench - build targets
#
#   make               build ./bench (needs the CUDA toolkit: nvcc)
#   make cublas        same, plus a cuBLAS baseline
#   make run           build and run with default sizes, writing results/results.csv
#   make ptx           emit PTX with nvcc and print an instruction summary (needs nvcc)
#   make emulate       unit-test the kernels on the CPU - NO GPU or nvcc needed (g++ >= 11)
#   make emulate-asan  ...under AddressSanitizer + UBSan (catches out-of-bounds access)
#   make emulate-tsan  ...under ThreadSanitizer (catches a missing __syncthreads as a data race)
#
# If -arch=native is not accepted (CUDA < 11.6), pass your GPU's architecture,
# e.g.  make ARCH=sm_75   (T4)   sm_80 (A100)   sm_86 (RTX 30xx)   sm_89 (RTX 40xx)

NVCC  ?= nvcc
ARCH  ?= native
CXX   ?= g++
NVFLAGS = -O3 -std=c++17 -lineinfo -arch=$(ARCH)
EMUFLAGS = -std=c++20 -O1 -g -pthread -Wall -Wextra -Isrc -Itest

SRC = src/bench.cu src/common.cuh src/kernels.cuh

.PHONY: all cublas run ptx emulate emulate-asan emulate-tsan clean

all: bench

bench: $(SRC)
	$(NVCC) $(NVFLAGS) -o $@ src/bench.cu

bench_cublas: $(SRC)
	$(NVCC) $(NVFLAGS) -DUSE_CUBLAS -o $@ src/bench.cu -lcublas

cublas: bench_cublas

run: bench
	mkdir -p results
	./bench --csv results/results.csv

ptx: $(SRC)
	mkdir -p ptx
	$(NVCC) -std=c++17 -O3 -lineinfo -arch=$(ARCH) -ptx src/bench.cu -o ptx/kernels_nvcc.ptx
	python3 tools/ptx_summary.py ptx/kernels_nvcc.ptx

emulate: test/emulate_kernels.cpp test/cuda_shim.hpp src/kernels.cuh
	$(CXX) $(EMUFLAGS) test/emulate_kernels.cpp -o emulate
	./emulate

emulate-asan: test/emulate_kernels.cpp test/cuda_shim.hpp src/kernels.cuh
	$(CXX) $(EMUFLAGS) -fsanitize=address,undefined test/emulate_kernels.cpp -o emulate_asan
	./emulate_asan

emulate-tsan: test/emulate_kernels.cpp test/cuda_shim.hpp src/kernels.cuh
	$(CXX) $(EMUFLAGS) -fsanitize=thread test/emulate_kernels.cpp -o emulate_tsan
	./emulate_tsan

clean:
	rm -f bench bench_cublas emulate emulate_asan emulate_tsan
