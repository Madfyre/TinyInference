#include <cmath>
#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <ostream>
#include <queue>
#include <stdexcept>
#include <string>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "../parsers/tokenizer_parser.cpp"
#include "../parsers/prompt_parser.cpp"
#include "../parsers/model_parser.cpp"

constexpr double PI = 3.14159265358979323846;


using Matrix = std::vector<std::vector<float>>;

void DebugWords(const std::vector<std::string>& words) {
    for (size_t i = 0; i < words.size(); ++i) {
        std::cout << "\"" << words[i] << "\"" << ", ";
    }
}

void DebugMatrix(const Matrix& matrix, const std::string& name, size_t rows_n = 3, size_t cols_n = 8) {
    std::cout << name << ": [" << matrix.size() << ", " 
              << (matrix.empty() ? 0 : matrix[0].size()) << "]\n";

    size_t r = std::min(rows_n, matrix.size());
    for (size_t i = 0; i < r; ++i) {
        std::cout << "  row " << i << ": ";
        size_t c = std::min(cols_n, matrix[i].size());
        for (size_t j = 0; j < c; ++j) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << (matrix[i].size() > cols_n ? "..." : "") << "\n";
    }
}

std::vector<uint32_t> ParseWords(const std::string& word, 
                                 const std::unordered_map<std::string, uint32_t>& vocab,
                                 const std::unordered_map<uint64_t, parser::MergeInfo>& merges)
{
    if (auto it = vocab.find(word); it != vocab.end()) {
        return {it->second};
    }
    std::vector<uint32_t> parts(word.size(), 0); 
    for (size_t i = 0; i < word.size(); ++i) {
        if (auto it = vocab.find(std::string(1, word[i])); it != vocab.end()) {
            parts[i] = it->second;
        } else {
            throw std::runtime_error("main.cpp::ParseWords: " + std::string(1, word[i]) + " doesn't found!");
        }
    }

    uint32_t best_rank = UINT32_MAX;
    uint32_t new_id = UINT32_MAX;
    uint32_t index_in_parts = UINT32_MAX;
    do {
        for (size_t i = 1; i < parts.size(); ++i) {
            uint64_t key = (static_cast<uint64_t>(parts[i - 1]) << 32) | parts[i];
            if (auto it = merges.find(key); it != merges.end()) {
                if (best_rank > it->second.rank) {
                    best_rank = it->second.rank;
                    new_id = it->second.new_id;
                    index_in_parts = i - 1;
                }
            }
        }

        if (best_rank == UINT32_MAX) {
            break;
        }
        parts[index_in_parts] = new_id;
        parts.erase(parts.begin() + index_in_parts + 1);

        best_rank = UINT32_MAX;
        index_in_parts = UINT32_MAX;
        new_id = UINT32_MAX;

    } while (parts.size() > 1);

    return parts;
}

inline Matrix Words2Embeddings(const std::unordered_map<std::string, uint32_t>& vocab,
                        const std::vector<uint32_t>& ids,
                        parser::Embeddings embed_matrix)
{
    Matrix result(ids.size(), std::vector<float>(embed_matrix.cols));
    for (size_t i = 0; i < ids.size(); ++i) {
        for (size_t j = 0; j < embed_matrix.cols; ++j) {
            uint16_t* w = reinterpret_cast<uint16_t*>(embed_matrix.weights);
            result[i][j] = parser::Bf16ToF32(w[ids[i] * embed_matrix.cols + j]);
        }
    }

    return result;
}

inline void RMSNorm(Matrix& result, const parser::Tensor& input_layernorm, float eps = 1e-5) {
    for (size_t row = 0; row < result.size(); ++row) {
        double sum = 0.0;
        for (size_t col = 0; col < result[0].size(); col++) {
            sum += result[row][col] * result[row][col];
        }
        double rms = std::sqrt((sum / result[0].size()) + eps);
        for (size_t col = 0; col < result[0].size(); col++) {
            uint16_t* w = reinterpret_cast<uint16_t*>(input_layernorm.weights);
            result[row][col] = (result[row][col] / rms * parser::Bf16ToF32(w[col]));
        }
    }
}

inline Matrix MatMul(const Matrix& left, const parser::Tensor& weight) {
    Matrix result{left.size(), std::vector<float>(weight.rows, 0.0)};
    uint16_t* weights = reinterpret_cast<uint16_t*>(weight.weights);

    for (size_t row = 0; row < left.size(); ++row) {
        for (size_t col = 0; col < weight.rows; col++) {
            double acc = 0.0f;
            for (size_t ind = 0; ind < left[0].size(); ++ind) {
                acc += left[row][ind] * parser::Bf16ToF32(weights[col * weight.cols + ind]);
            }
            result[row][col] = acc;
        }
    }

    return result;
}

inline std::pair<Matrix, Matrix> MakeRopeTables(int32_t head_dim, int32_t theta, int32_t seq_len, int32_t factor,
                          int32_t lower_freq_factor, int32_t high_freq_factor, int32_t original_max)
{
    Matrix result_cos{static_cast<size_t>(seq_len), std::vector<float>(head_dim / 2)};
    Matrix result_sin{static_cast<size_t>(seq_len), std::vector<float>(head_dim / 2)};

    float low_freq_wavelen = static_cast<float>(original_max) / lower_freq_factor;
    float high_freq_wavelen = static_cast<float>(original_max) / high_freq_factor;

    for (size_t i = 0; i < head_dim / 2; ++i) {
        float freq = 1.0f / (std::pow(theta, static_cast<float>(2 * i) / head_dim));
        float wavelength = 2 * PI / freq;
        if (wavelength > low_freq_wavelen) {
            freq = freq / factor;
        } else if (wavelength < high_freq_wavelen) {
            freq = freq;
        } else {
            float smooth = (original_max / wavelength - lower_freq_factor) / (high_freq_factor - lower_freq_factor);
            freq = freq * ((1 - smooth) / factor + smooth);
        }
        for (size_t j = 0; j < seq_len; ++j) {
            result_sin[j][i] = std::sin(freq * j);
            result_cos[j][i] = std::cos(freq * j);
        }
    }

    return {result_sin, result_cos};
}

inline void RoPE(Matrix& result,const Matrix& cos_table,
                   const Matrix& sin_table, int32_t n_heads, int32_t head_dim, int32_t seq_len)
{
    for (size_t pos = 0; pos < seq_len; ++pos) {
        for (size_t head = 0; head < n_heads; ++head) {
            size_t base = head * head_dim;
            for (size_t head_index = 0; head_index < head_dim / 2; ++head_index) {
                float r1 = result[pos][base + head_index];
                float r2 = result[pos][base + head_index + head_dim / 2];
                result[pos][base + head_index] = r1 * cos_table[pos][head_index] - r2 * sin_table[pos][head_index];
                result[pos][base + head_index + head_dim / 2] = r1 * sin_table[pos][head_index] +
                                                                r2 * cos_table[pos][head_index];

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
    for (size_t row = 0; row < left_view.row_length; ++row) {
        for (size_t col = 0; col < right_view.row_length; col++) {
            double acc = 0.0f;
            for (size_t ind = 0; ind < right_view.col_length; ++ind) {
                acc += left[left_view.row + row][left_view.col + ind] *
                       right[right_view.row + col][right_view.col + ind];
            }
            result[result_view.row + row][result_view.col + col] = acc / std::sqrt(right_view.col_length);
        }
    }
}

inline void MatMul(const SubView& left_view, const Matrix& left,
                   const SubView& right_view, const Matrix& right,
                   SubView& result_view, Matrix& result) {
    for (size_t row = 0; row < left_view.row_length; ++row) {
        for (size_t col = 0; col < right_view.col_length; col++) {
            double acc = 0.0f;
            for (size_t ind = 0; ind < right_view.row_length; ++ind) {
                acc += left[left_view.row + row][left_view.col + ind] *
                       right[right_view.row + ind][right_view.col + col];
            }
            result[result_view.row + row][result_view.col + col] = acc;
        }
    }
}

inline Matrix AttentionScores(const Matrix& Q, const Matrix& K, int32_t heads_q, int32_t heads_k, int32_t head_dim)
{
    Matrix result{static_cast<size_t>(heads_q * Q.size()), std::vector<float>(Q.size())};
    SubView q_view{0, Q.size(), 0, static_cast<size_t>(head_dim)};
    SubView k_view{0, K.size(), 0, static_cast<size_t>(head_dim)};
    SubView result_view{0, Q.size(), 0, Q.size()};
    for (size_t head = 0; head < heads_q; ++head) {
        int32_t head_k = head / (heads_q / heads_k);
        q_view.col = head * head_dim;
        k_view.col = head_k * head_dim;
        result_view.row = head * Q.size();
        MatMulT(q_view, Q, k_view, K, result_view, result);
    }

    return result;
}

inline Matrix CausalMaskAttn(Matrix& matrix)
{
    for (size_t row = 0; row < matrix.size(); ++row) {
        size_t i = row % matrix[0].size(); 
        for (size_t col = 0; col < matrix[0].size(); ++col) {
            if (col > i) {
                matrix[row][col] = -INFINITY;
            }
        }
    }
    return matrix;
}

inline Matrix SoftMax(Matrix& matrix)
{
    for (size_t row = 0; row < matrix.size(); ++row) {
        float max_value = -INFINITY;
        for (size_t col = 0; col < matrix[0].size(); ++col) {
            max_value = std::max(max_value, matrix[row][col]);
        }
        float acc = 0.0;
        for (size_t col = 0; col < matrix[0].size(); ++col) {
            acc += std::exp(matrix[row][col] - max_value);
        }
        for (size_t col = 0; col < matrix[0].size(); ++col) {
            matrix[row][col] = std::exp(matrix[row][col] - max_value) / acc;
        }
    }
    return matrix;
}

inline Matrix Attention(const Matrix& Q, const Matrix& K, const Matrix& V,
                        int32_t heads_q, int32_t heads_k, int32_t head_dim) {
    auto attn_scores = AttentionScores(Q, K, heads_q, heads_k, head_dim);
    auto masked = CausalMaskAttn(attn_scores);
    auto softmaxed = SoftMax(masked);

    Matrix result{static_cast<size_t>(Q.size()), std::vector<float>(Q[0].size())};
    SubView softmax_view{0, Q.size(), 0, Q.size()};
    SubView v_view{0, K.size(), 0, static_cast<size_t>(head_dim)};
    SubView result_view{0, Q.size(), 0, static_cast<size_t>(head_dim)};

    for (size_t head = 0; head < heads_q; ++head) {
        softmax_view.row = head * Q.size();
        v_view.col = head * head_dim;
        result_view.col = head * head_dim;
        MatMul(softmax_view, softmaxed, v_view, V, result_view, result);
    }

    return result;
}

int main() {
    std::string input = "Hello, I'm GPT-4 and I've got 1234 dollars. It'll be fine!!!\nYes, it's true.";

    auto tokenizer_info = OpenFile("models/llama-3.2-1b/tokenizer.json");
    auto vocab = parser::ImportTokens(tokenizer_info);
    auto merges = parser::ImportMerges(tokenizer_info, vocab);

    auto model_info = OpenFile("models/llama-3.2-1b/model.safetensors");
    parser::Model model = parser::ParseConfig(model_info);

    std::vector<std::string> words = parser::Prompt2Words(input);

    std::cout << "TOKENS" << std::endl;
    std::vector<uint32_t> tokens_idx;
    tokens_idx.reserve(words.size());
    for (int i = 0; i < words.size(); ++i) {
        if (auto it = vocab.find(words[i]); it != vocab.end()) {
            std::cout << it->first << " " << it->second << std::endl;
            tokens_idx.push_back(it->second);
        } else {
            std::vector<uint32_t> tokenized_word = ParseWords(words[i], vocab, merges);
            std::cout << words[i] << ": ";
            for (const auto& token : tokenized_word) {
                tokens_idx.push_back(token);
                std::cout << token << " ";
            }
            std::cout << std::endl;
        }
    }

    DebugTensor(model.embed_tensor, "embed");
    DebugTensor(model.layers[0].q_proj, "layer0.q_proj");
    DebugTensor(model.layers[0].k_proj, "k_proj");

    auto entrance_matrix = Words2Embeddings(vocab, tokens_idx, model.embed_tensor);
    DebugMatrix(entrance_matrix, "embeddings: ");

    RMSNorm(entrance_matrix, model.layers[0].input_layernorm);

    DebugMatrix(entrance_matrix, "after RMSNorm: ");

    auto Q = MatMul(entrance_matrix, model.layers[0].q_proj);
    auto K = MatMul(entrance_matrix, model.layers[0].k_proj);
    auto V = MatMul(entrance_matrix, model.layers[0].v_proj);
    auto O = MatMul(entrance_matrix, model.layers[0].o_proj);

    DebugMatrix(Q, "Q: ");
    DebugMatrix(K, "K: ");
    DebugMatrix(V, "V: ");
    DebugMatrix(O, "O: ");

    int32_t heads = 32;
    int32_t head_dim = 64;
    int32_t theta = 500000;
    int32_t seq_len = 512;
    int32_t factor = 4;
    int32_t lower_freq_factor = 1;
    int32_t high_freq_factor = 4;
    int32_t original_max = 8192;

    auto [sin_table, cos_table] = MakeRopeTables(head_dim, theta, seq_len, factor, lower_freq_factor, high_freq_factor, original_max);

    DebugMatrix(sin_table, "sin_table: ");
    DebugMatrix(cos_table, "cos_table: ");

    std::cout << "QSIZE: " << Q.size() << std::endl;

    RoPE(Q, cos_table, sin_table, heads, head_dim, Q.size());
    RoPE(K, cos_table, sin_table, 8, head_dim, K.size());

    DebugMatrix(Q, "Q after RoPE: ");
    DebugMatrix(K, "K after RoPE: ");

    Matrix attn_scores_matrix = Attention(Q, K, V, heads, K[0].size() / head_dim, head_dim);
    DebugMatrix(attn_scores_matrix, "Attention: ");

    return 0;
}