# INT8 Weight Quantization for Linear Layers

Symmetric INT8 quantization of a linear-layer weight matrix with per-column balancing.

**Techniques:** Per-column bias/scale balancing, round-to-nearest, max-based dynamic scale - a quantize-for-inference primitive.
