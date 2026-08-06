#pragma once

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "../parsers/model_parser.cpp"
#include "matrix.cpp"

namespace model {

inline Matrix Words2Embeddings(const std::vector<uint32_t>& ids,
                               parser::Embeddings embed_matrix)
{
    Matrix result{ids.size(), embed_matrix.cols};
    for (size_t i = 0; i < ids.size(); ++i) {
        for (size_t j = 0; j < embed_matrix.cols; ++j) {
            const uint16_t* w = reinterpret_cast<const uint16_t*>(embed_matrix.weights);
            result[i, j] = parser::Bf16ToF32(w[ids[i] * embed_matrix.cols + j]);
        }
    }

    return result;
}

inline void AddEmbedding(Matrix& curr_embeddings,
                         uint32_t id,
                         parser::Embeddings embed_matrix)
{
    curr_embeddings.allocate_rows(1);
    for (size_t j = 0; j < embed_matrix.cols; ++j) {
        const uint16_t* w = reinterpret_cast<const uint16_t*>(embed_matrix.weights);
        curr_embeddings[curr_embeddings.rows() - 1, j] = parser::Bf16ToF32(w[id * embed_matrix.cols + j]);
    }
}

inline Matrix RMSNorm(Matrix& result, const parser::Tensor& input_layernorm, float eps = 1e-5) {
    for (size_t row = 0; row < result.rows(); ++row) {
        double sum = 0.0;
        for (size_t col = 0; col < result.cols(); col++) {
            sum += result[row, col] * result[row, col];
        }
        double rms = std::sqrt((sum / result.cols()) + eps);
        for (size_t col = 0; col < result.cols(); col++) {
            const uint16_t* w = reinterpret_cast<const uint16_t*>(input_layernorm.weights);
            result[row, col] = (result[row, col] / rms * parser::Bf16ToF32(w[col]));
        }
    }

    return result;
}

inline Matrix RMSNorm(Matrix& result, const Matrix& rms_weights, float eps = 1e-5) {
    for (size_t row = 0; row < result.rows(); ++row) {
        double sum = 0.0;
        for (size_t col = 0; col < result.cols(); col++) {
            sum += result[row, col] * result[row, col];
        }
        double rms = std::sqrt((sum / result.cols()) + eps);
        for (size_t col = 0; col < result.cols(); col++) {
            result[row, col] = (result[row, col] / rms * parser::Bf16ToF32(rms_weights[0, col]));
        }
    }

    return result;
}

inline Matrix MatMul(const Matrix& left, const parser::Tensor& weight) {
    Matrix result{left.rows(), weight.rows};
    const uint16_t* weights = reinterpret_cast<const uint16_t*>(weight.weights);
    std::vector<float> wrow(weight.cols);

    for (size_t col = 0; col < weight.rows; ++col) {
        const uint16_t* src = weights + col * weight.cols;
        for (size_t ind = 0; ind < weight.cols; ++ind) {
            wrow[ind] = parser::Bf16ToF32(src[ind]);
        }

        for (size_t row = 0; row < left.rows(); ++row) {
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;
            size_t ind = 0;
            for (; ind + 4 <= left.cols(); ind += 4) {
                acc0 += left[row, ind] * wrow[ind];
                acc1 += left[row, ind + 1] * wrow[ind + 1];
                acc2 += left[row, ind + 2] * wrow[ind + 2];
                acc3 += left[row, ind + 3] * wrow[ind + 3];
            }
            float acc = acc0 + acc1 + acc2 + acc3;
            for (; ind < left.cols(); ++ind) {
                acc += left[row, ind] * wrow[ind];
            }
            result[row, col] = acc;
        }
    }

    return result;
}

inline Matrix Add(const Matrix& left, const Matrix& right) {
    Matrix result = left;
    for (size_t row = 0; row < left.rows(); ++row) {
        for (size_t col = 0; col < left.cols(); col++) {
            result[row, col] = left[row, col] + right[row, col];
        }
    }
    return result;
}

inline std::pair<Matrix, Matrix> MakeRopeTables(int32_t head_dim, float theta, int32_t seq_len, float factor,
                          float low_freq_factor, float high_freq_factor, float original_max)
{
    Matrix result_cos{static_cast<size_t>(seq_len), static_cast<size_t>(head_dim / 2)};
    Matrix result_sin{static_cast<size_t>(seq_len), static_cast<size_t>(head_dim / 2)};

    float low_freq_wavelen = original_max / low_freq_factor;
    float high_freq_wavelen = original_max / high_freq_factor;

    for (size_t i = 0; i < head_dim / 2; ++i) {
        float freq = 1.0f / (std::pow(theta, static_cast<float>(2 * i) / head_dim));
        float wavelength = 2.0f * PI / freq;
        if (wavelength > low_freq_wavelen) {
            freq /= factor;
        } else if (wavelength >= high_freq_wavelen) {
            float smooth = (original_max / wavelength - low_freq_factor) / (high_freq_factor - low_freq_factor);
            freq *= (1.0f - smooth) / factor + smooth;
        }
        for (size_t j = 0; j < seq_len; ++j) {
            result_sin[j, i] = std::sin(freq * j);
            result_cos[j, i] = std::cos(freq * j);
        }
    }

    return {std::move(result_sin), std::move(result_cos)};
}

inline void RoPE(Matrix& result, const Matrix& cos_table,
                 const Matrix& sin_table, int32_t n_heads,
                 int32_t head_dim, int32_t seq_len)
{
    if (seq_len > cos_table.rows()) {
        throw std::runtime_error("RoPE: pos out of size");
    }

    for (size_t pos = 0; pos < seq_len; ++pos) {
        for (size_t head = 0; head < n_heads; ++head) {
            size_t base = head * head_dim;
            for (size_t head_index = 0; head_index < head_dim / 2; ++head_index) {
                float r1 = result[pos, base + head_index];
                float r2 = result[pos, base + head_index + head_dim / 2];

                result[pos, base + head_index] = r1 * cos_table[pos, head_index] -
                                                 r2 * sin_table[pos, head_index];

                result[pos, base + head_index + head_dim / 2] = r1 * sin_table[pos, head_index] +
                                                                r2 * cos_table[pos, head_index];

            }
        }
    }
}

struct SubView {
    size_t row;
    size_t row_length;
    size_t col;
    size_t col_length;
};

inline void MatMulT(const SubView& left_view, const Matrix& left,
                    const SubView& right_view, const Matrix& right,
                    SubView& result_view, Matrix& result) {
    const float inv = 1.0f / std::sqrt(static_cast<float>(right_view.col_length));

    for (size_t row = 0; row < left_view.row_length; ++row) {
        const float* lrow = left.row_ptr(left_view.row + row) + left_view.col;

        for (size_t col = 0; col < right_view.row_length; ++col) {
            const float* rrow = right.row_ptr(right_view.row + col) + right_view.col;

            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;

            size_t ind = 0;
            for (; ind + 4 <= right_view.col_length; ind += 4) {
                acc0 += lrow[ind] * rrow[ind];
                acc1 += lrow[ind + 1] * rrow[ind + 1];
                acc2 += lrow[ind + 2] * rrow[ind + 2];
                acc3 += lrow[ind + 3] * rrow[ind + 3];
            }
            float acc = acc0 + acc1 + acc2 + acc3;
            for (; ind < right_view.col_length; ++ind) {
                acc += lrow[ind] * rrow[ind];
            }
            result[result_view.row + row, result_view.col + col] = acc * inv;
        }
    }
}

inline void MatMul(const SubView& left_view, const Matrix& left,
                   const SubView& right_view, const Matrix& right,
                   SubView& result_view, Matrix& result) {
    for (size_t row = 0; row < left_view.row_length; ++row) {
        float* orow = result.row_ptr(result_view.row + row) + result_view.col;
        for (size_t col = 0; col < right_view.col_length; ++col) {
            orow[col] = 0.0f;
        }

        const float* lrow = left.row_ptr(left_view.row + row) + left_view.col;
        for (size_t ind = 0; ind < right_view.row_length; ++ind) {
            const float scale = lrow[ind];
            const float* rrow = right.row_ptr(right_view.row + ind) + right_view.col;
            for (size_t col = 0; col < right_view.col_length; ++col) {
                orow[col] += scale * rrow[col];
            }
        }
    }
}

inline Matrix MatMulFinal(const Matrix& left, parser::Embeddings embed_matrix) {
    Matrix result{left.rows(), embed_matrix.rows};
    const uint16_t* w = reinterpret_cast<const uint16_t*>(embed_matrix.weights);

    for (size_t row = 0; row < left.rows(); ++row) {
        const float* lrow = left.row_ptr(row);
        for (size_t col = 0; col < embed_matrix.rows; ++col) {
            const uint16_t* src = w + col * embed_matrix.cols;

            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
            size_t ind = 0;
            for (; ind + 4 <= embed_matrix.cols; ind += 4) {
                acc0 += lrow[ind] * parser::Bf16ToF32(src[ind]);
                acc1 += lrow[ind + 1] * parser::Bf16ToF32(src[ind + 1]);
                acc2 += lrow[ind + 2] * parser::Bf16ToF32(src[ind + 2]);
                acc3 += lrow[ind + 3] * parser::Bf16ToF32(src[ind + 3]);
            }
            float acc = acc0 + acc1 + acc2 + acc3;
            for (; ind < embed_matrix.cols; ++ind) {
                acc += lrow[ind] * parser::Bf16ToF32(src[ind]);
            }
            result[row, col] = acc;
        }
    }
    return result;
}

inline Matrix AttentionScores(const Matrix& Q, const Matrix& K,
                              int32_t heads_q, int32_t heads_k, int32_t head_dim)
{
    Matrix result{static_cast<size_t>(heads_q * Q.rows()), Q.rows()};
    SubView q_view{0, Q.rows(), 0, static_cast<size_t>(head_dim)};
    SubView k_view{0, K.rows(), 0, static_cast<size_t>(head_dim)};
    SubView result_view{0, Q.rows(), 0, Q.rows()};
    for (size_t head = 0; head < heads_q; ++head) {
        int32_t head_k = head / (heads_q / heads_k);
        q_view.col = head * head_dim;
        k_view.col = head_k * head_dim;
        result_view.row = head * Q.rows();
        MatMulT(q_view, Q, k_view, K, result_view, result);
    }

    return result;
}

inline Matrix CausalMaskAttn(Matrix& matrix)
{
    for (size_t row = 0; row < matrix.rows(); ++row) {
        size_t i = row % matrix.cols(); 
        for (size_t col = 0; col < matrix.cols(); ++col) {
            if (col > i) {
                matrix[row, col] = -INFINITY;
            }
        }
    }
    return matrix;
}

inline Matrix SoftMax(Matrix& matrix)
{
    for (size_t row = 0; row < matrix.rows(); ++row) {
        float max_value = -INFINITY;
        for (size_t col = 0; col < matrix.cols(); ++col) {
            max_value = std::max(max_value, matrix[row, col]);
        }
        float acc = 0.0;
        for (size_t col = 0; col < matrix.cols(); ++col) {
            acc += std::exp(matrix[row, col] - max_value);
        }
        for (size_t col = 0; col < matrix.cols(); ++col) {
            matrix[row, col] = std::exp(matrix[row, col] - max_value) / acc;
        }
    }
    return matrix;
}

inline Matrix SiLU(Matrix& matrix)
{
    for (size_t row = 0; row < matrix.rows(); ++row) {
        for (size_t col = 0; col < matrix.cols(); ++col) {
            matrix[row, col] = matrix[row, col] / (1 + std::exp(-matrix[row, col]));
        }
    }
    return matrix;
}

inline Matrix Hadamard(Matrix& matrix, const Matrix& mult)
{
    for (size_t row = 0; row < matrix.rows(); ++row) {
        for (size_t col = 0; col < matrix.cols(); ++col) {
            matrix[row, col] *= mult[row, col];
        }
    }
    return matrix;
}

inline Matrix Attention(const Matrix& Q, const Matrix& K, const Matrix& V,
                        int32_t heads_q, int32_t heads_k, int32_t head_dim) {
    auto attn_scores = AttentionScores(Q, K, heads_q, heads_k, head_dim);
    Matrix masked = CausalMaskAttn(attn_scores);
    Matrix softmaxed = SoftMax(masked);

    Matrix result{static_cast<size_t>(Q.rows()), Q.cols()};
    SubView softmax_view{0, Q.rows(), 0, Q.rows()};
    SubView v_view{0, K.rows(), 0, static_cast<size_t>(head_dim)};
    SubView result_view{0, Q.rows(), 0, static_cast<size_t>(head_dim)};

    for (size_t head = 0; head < heads_q; ++head) {
        int32_t head_v = head / (heads_q / heads_k);
        softmax_view.row = head * Q.rows();
        v_view.col = head_v * head_dim;
        result_view.col = head * head_dim;
        MatMul(softmax_view, softmaxed, v_view, V, result_view, result);
    }

    return result;
}

inline Matrix LMHead(Matrix& output, parser::Embeddings embed_matrix) {
    return MatMulFinal(output, embed_matrix);
}

inline size_t ArgMax(const Matrix& lm_head) {
    size_t index = -1;
    float value = -INFINITY;
    for (size_t col = 0; col < lm_head.cols(); ++col) {
        if (value < lm_head[0, col]) {
            index = col;
            value = lm_head[0, col];
        }
    }

    return index;
}

}  // namespace model
