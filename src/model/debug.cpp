#pragma once

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "matrix.cpp"

namespace model {

inline void CompareRef(const Matrix& matrix, const std::string& path) {
    size_t rows = matrix.rows();
    size_t cols = matrix.cols();

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cout << path << ": NOT FOUND\n";
        return;
    }

    in.seekg(0, std::ios::end);
    size_t bytes = in.tellg();
    in.seekg(0, std::ios::beg);

    if (bytes != rows * cols * sizeof(float)) {
        std::cout << path << ": SIZE MISMATCH, file=" << bytes
                  << " expected=" << rows * cols * sizeof(float) << "\n";
        return;
    }

    std::vector<float> ref(rows * cols);
    in.read(reinterpret_cast<char*>(ref.data()), bytes);

    double max_abs = 0.0;
    double max_rel = 0.0;
    size_t arg_row = 0;
    size_t arg_col = 0;

    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            double got = matrix[row, col];
            double want = ref[row * cols + col];
            double abs_diff = std::fabs(got - want);
            if (abs_diff > max_abs) {
                max_abs = abs_diff;
                arg_row = row;
                arg_col = col;
            }
            double denom = std::fabs(want);
            if (denom > 1e-6) {
                max_rel = std::max(max_rel, abs_diff / denom);
            }
        }
    }

    std::printf("%-16s max_abs=%.3e max_rel=%.3e  at [%zu,%zu] got=%.6f want=%.6f  %s\n",
                path.c_str(), max_abs, max_rel, arg_row, arg_col,
                matrix[arg_row, arg_col], ref[arg_row * cols + arg_col],
                max_abs < 1e-2 ? "OK" : "<<< DIFF");
}

}  // namespace model
