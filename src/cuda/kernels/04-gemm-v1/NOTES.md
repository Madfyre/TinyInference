# FP16 Tiled GEMM with Register Blocking

Half-precision GEMM (RowMajor x ColMajor -> ColMajor) with FP32 accumulation.

**Techniques:** 32x32 shared-memory tiling with bank-conflict padding, 1D register blocking (8 outputs/thread), full boundary handling for arbitrary shapes.
