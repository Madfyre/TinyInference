# Tests and benchmarks

Self-contained drivers with no external test framework: each one builds to a single
executable, generates its own data, and prints a table. Device allocations are pitched,
so strides are wider than the logical matrices and the stride handling in each kernel is
exercised on every case.

| Driver | Kernel | Checks / measures |
|---|---|---|
| `test_gemm.cu` | `kernels/04-gemm-v1` | Correctness vs. a CPU reference (small shapes) and vs. cuBLAS (large shapes) |
| `bench_gemm.cu` | `kernels/04-gemm-v1` | TFLOP/s and % of cuBLAS on identical inputs |
| `bench_quantization.cu` | `kernels/05-quantization` | Correctness vs. a CPU reference, plus achieved bandwidth vs. theoretical peak |

## Build

Adjust `-arch` to your GPU (`sm_80` A100, `sm_86` RTX 30xx, `sm_89` RTX 40xx, `sm_90` H100):

```bash
nvcc -std=c++17 -O3 -arch=sm_80 -I../common -I../kernels/04-gemm-v1 \
     test_gemm.cu ../common/cuda_helpers.cpp -lcublas -o test_gemm

nvcc -std=c++17 -O3 -arch=sm_80 -I../common -I../kernels/04-gemm-v1 \
     bench_gemm.cu ../common/cuda_helpers.cpp -lcublas -o bench_gemm

nvcc -std=c++17 -O3 -arch=sm_80 -I../common -I../kernels/05-quantization \
     bench_quantization.cu ../kernels/05-quantization/quantization.cu \
     ../common/cuda_helpers.cpp -o bench_quantization
```

Run `./test_gemm` first; it exits non-zero if any check fails.

## Method

Timing a kernel repeatedly over one set of buffers measures the L2-resident case and
overstates throughput. `bench_gemm` therefore allocates enough independent input sets to
exceed the L2 cache and rotates through them across iterations, so most iterations miss
in L2. The cuBLAS baseline rotates over the same sets, so both sides see the same cache
behaviour. Each measurement is 5 warm-up iterations followed by 50 timed iterations
bracketed by CUDA events.

Correctness uses two independent references: a CPU implementation accumulating in float
for small shapes, and cuBLAS for shapes where a CPU reference would be too slow. Because
inputs are FP16 with float accumulation, results are compared with a relative tolerance
rather than for exact equality.

Quantization is memory bound, so it is reported as achieved bandwidth against the
device's theoretical peak (from memory clock and bus width) rather than as FLOP/s.

## Results

Fill in after running on your hardware:

```
GPU: <name>, L2 <n> MB

    M     N     K |  sets |  mine ms   TFLOP/s | cublas ms   TFLOP/s | of cuBLAS
------------------+-------+--------------------+---------------------+----------
 ...
```
