# CUTLASS TN HGEMM with Fused HardSwish Epilogue

FP16 TN HGEMM (C = HardSwish(A*B)) built on the CUTLASS device API.

**Techniques:** Tensor-core GEMM via CUTLASS, custom fused element-wise HardSwish epilogue, column-major layouts.
