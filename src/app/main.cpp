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
#include <utility>
#include <vector>

#include "../parsers/tokenizer_parser.cpp"
#include "../parsers/prompt_parser.cpp"
#include "../parsers/model_parser.cpp"

constexpr double PI = 3.14159265358979323846;

class Matrix {
public:
    Matrix() : rows_(0), cols_(0), data_(0) {}

    Matrix(size_t rows, size_t cols, float value = 0.0f)
    : rows_(rows), cols_(cols), data_(rows * cols, value) {}

    Matrix(const Matrix& other) : rows_(other.rows_), cols_(other.cols_), data_(other.data_) {}

    Matrix& operator=(const Matrix& other) {
        rows_ = other.rows_;
        cols_ = other.cols_;
        data_ = other.data_;
        return *this;
    }

    Matrix(Matrix&& other) : rows_(std::exchange(other.rows_, 0)),
                             cols_(std::exchange(other.cols_, 0)),
                             data_(std::move(other.data_)) {}

    Matrix& operator=(Matrix&& other) {
        rows_ = std::exchange(other.rows_, 0);
        cols_ = std::exchange(other.cols_, 0);
        data_ = std::move(other.data_);
        return *this;
    }

    float& operator[](size_t row, size_t col) {
        return data_[row * cols_ + col];
    }

    const float& operator[](size_t row, size_t col) const {
        return data_[row * cols_ + col];
    }

    void allocate_rows(size_t rows) {
        data_.resize((rows_ + rows) * cols_);
        rows_ += rows;
    }

    void add_row(const std::vector<float> row) {
        data_.reserve((rows_ + 1) * cols_);
        rows_++;

        for (const auto& elem : row) {
            data_.push_back(elem);
        }
    }

    size_t rows() const {
        return rows_;
    }

    size_t cols() const {
        return cols_;
    }

    Matrix last_row() const {
        Matrix row{1, cols_};
        for (size_t i = 0; i < cols_; ++i) {
            row[0, i] = data_[(rows_ - 1) * cols_ + i];
        }
        return row;
    }

private:
    size_t rows_;
    size_t cols_;
    std::vector<float> data_;
};


void DebugWords(const std::vector<std::string>& words) {
    for (size_t i = 0; i < words.size(); ++i) {
        std::cout << "\"" << words[i] << "\"" << ", ";
    }
}

// void DebugMatrix(const Matrix& matrix, const std::string& name, size_t rows_n = 3, size_t cols_n = 8) {
//     std::cout << name << ": [" << matrix.rows() << ", " 
//               << (matrix.rows() ? 0 : matrix[0].size()) << "]\n";

//     size_t r = std::min(rows_n, matrix.size());
//     for (size_t i = 0; i < r; ++i) {
//         std::cout << "  row " << i << ": ";
//         size_t c = std::min(cols_n, matrix.cols());
//         for (size_t j = 0; j < c; ++j) {
//             std::cout << matrix[i, j] << " ";
//         }
//         std::cout << (matrix[i].size() > cols_n ? "..." : "") << "\n";
//     }
// }

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
    Matrix result{ids.size(), embed_matrix.cols};
    for (size_t i = 0; i < ids.size(); ++i) {
        for (size_t j = 0; j < embed_matrix.cols; ++j) {
            uint16_t* w = reinterpret_cast<uint16_t*>(embed_matrix.weights);
            result[i, j] = parser::Bf16ToF32(w[ids[i] * embed_matrix.cols + j]);
        }
    }

    return result;
}

inline void AddEmbedding(const std::unordered_map<std::string, uint32_t>& vocab,
                           Matrix& curr_embeddings,
                           uint32_t id,
                           parser::Embeddings embed_matrix)
{
    curr_embeddings.allocate_rows(1);
    for (size_t j = 0; j < embed_matrix.cols; ++j) {
        uint16_t* w = reinterpret_cast<uint16_t*>(embed_matrix.weights);
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
            uint16_t* w = reinterpret_cast<uint16_t*>(input_layernorm.weights);
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
    uint16_t* weights = reinterpret_cast<uint16_t*>(weight.weights);
    std::vector<float> wrow(weight.cols);

    // for (size_t row = 0; row < left.rows(); ++row) {
    //     for (size_t col = 0; col < weight.rows; col++) {
    //         double acc = 0.0f;
    //         for (size_t ind = 0; ind < left.cols(); ++ind) {
    //             acc += left[row, ind] * parser::Bf16ToF32(weights[col * weight.cols + ind]);
    //         }
    //         result[row, col] = acc;
    //     }
    // }

    for (size_t col = 0; col < weight.rows; ++col) {
        uint16_t* src = weights + col * weight.cols;
        for (size_t ind = 0; ind < weight.cols; ++ind) {
            wrow[ind] = parser::Bf16ToF32(src[ind]);
        }

        for (size_t row = 0; row < left.rows(); ++row) {
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            size_t ind = 0;
            for (; ind + 4 <= left.cols(); ind += 4) {
                a0 += left[row, ind]     * wrow[ind];
                a1 += left[row, ind + 1] * wrow[ind + 1];
                a2 += left[row, ind + 2] * wrow[ind + 2];
                a3 += left[row, ind + 3] * wrow[ind + 3];
            }
            float acc = a0 + a1 + a2 + a3;
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

inline std::pair<Matrix, Matrix> MakeRopeTables(int32_t head_dim, int32_t theta, int32_t seq_len, int32_t factor,
                          int32_t lower_freq_factor, int32_t high_freq_factor, int32_t original_max)
{
    Matrix result_cos{static_cast<size_t>(seq_len), static_cast<size_t>(head_dim / 2)};
    Matrix result_sin{static_cast<size_t>(seq_len), static_cast<size_t>(head_dim / 2)};

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
            result_sin[j, i] = std::sin(freq * j);
            result_cos[j, i] = std::cos(freq * j);
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
                float r1 = result[pos, base + head_index];
                float r2 = result[pos, base + head_index + head_dim / 2];

                result[pos, base + head_index] = r1 * cos_table[pos, head_index] - r2 * sin_table[pos, head_index];
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
    for (size_t row = 0; row < left_view.row_length; ++row) {
        for (size_t col = 0; col < right_view.row_length; col++) {
            double acc = 0.0f;
            for (size_t ind = 0; ind < right_view.col_length; ++ind) {
                acc += left[left_view.row + row, left_view.col + ind] *
                       right[right_view.row + col, right_view.col + ind];
            }
            result[result_view.row + row, result_view.col + col] = acc / std::sqrt(right_view.col_length);
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
                acc += left[left_view.row + row, left_view.col + ind] *
                       right[right_view.row + ind, right_view.col + col];
            }
            result[result_view.row + row, result_view.col + col] = acc;
        }
    }
}

inline Matrix AttentionScores(const Matrix& Q, const Matrix& K, int32_t heads_q, int32_t heads_k, int32_t head_dim)
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
    auto masked = CausalMaskAttn(attn_scores);
    auto softmaxed = SoftMax(masked);

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

inline Matrix MatMulFinal(const Matrix& left, parser::Embeddings embed_matrix) {
    // std::cout << "embed_matrix: rows: " << embed_matrix.rows << " cols: " << embed_matrix.cols << std::endl;
    Matrix result{left.rows(), embed_matrix.rows};

    uint16_t* w = reinterpret_cast<uint16_t*>(embed_matrix.weights);
    for (size_t row = 0; row < left.rows(); ++row) {
        for (size_t col = 0; col < embed_matrix.rows; col++) {
            double acc = 0.0f;
            for (size_t ind = 0; ind < embed_matrix.cols; ++ind) {
                acc += left[row, ind] * parser::Bf16ToF32(w[col * embed_matrix.cols + ind]);
            }
            result[row, col] = acc;
        }
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

struct ModelInfo {
    int32_t heads_q;
    int32_t heads_k;
    int32_t head_dim;
    int32_t hidden;
    int32_t n_layers;
    int32_t intermediate;
    int32_t vocab;
    int32_t max_seq_len;

    float rms_eps;
    float theta;

    float factor;
    float low_freq_factor;
    float high_freq_factor;
    float original_max;
};

#include <fstream>
#include <cmath>
#include <cstdio>

// Читает [rows, cols] float32 из плоского бинарника и сравнивает с matrix.
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

inline Matrix DecoderLayer(Matrix& input, const parser::Layer& layer,
                           const Matrix& rope_sin, const Matrix& rope_cos,
                           const ModelInfo& model_info)
{
    Matrix x_duplicated = input;
    
    auto rms_x = RMSNorm(x_duplicated, layer.input_layernorm);

    auto Q = MatMul(rms_x, layer.q_proj);
    auto K = MatMul(rms_x, layer.k_proj);
    auto V = MatMul(rms_x, layer.v_proj);

    RoPE(Q, rope_cos, rope_sin, model_info.heads_q, model_info.head_dim, Q.rows());
    RoPE(K, rope_cos, rope_sin, model_info.heads_k, model_info.head_dim, K.rows());

    Matrix attn = Attention(Q, K, V, model_info.heads_q, model_info.heads_k, model_info.head_dim);
    auto attn_o = MatMul(attn, layer.o_proj);
    auto enriched_x = Add(input, attn_o);

    Matrix xn2 = enriched_x;
    xn2 = RMSNorm(xn2, layer.post_attn_layernorm);

    auto g = MatMul(xn2, layer.gate_proj);
    auto u = MatMul(xn2, layer.up_proj);
    auto h = Hadamard(u, SiLU(g));
    auto mlp = MatMul(h, layer.down_proj);

    auto enriched_final = Add(enriched_x, mlp);
    return enriched_final;
}

int main() {
    std::string input = "2+2";
    std::string model_dir = "models/llama-3.2-1b-instruct";

    auto tensors_info = OpenFile(model_dir + "/model.safetensors");
    parser::Model model = parser::ParseConfig(tensors_info);

    auto tokenizer_info = OpenFile(model_dir + "/tokenizer.json");
    auto vocab = parser::ImportTokens(tokenizer_info);

    std::vector<std::string> id_to_token(model.embed_tensor.rows);
    for (const auto& [token, id] : vocab) {
        id_to_token[id] = token;
    }

    auto merges = parser::ImportMerges(tokenizer_info, vocab);

    std::vector<std::string> words = parser::Prompt2Words(input);

    std::vector<uint32_t> tokens_idx;
    tokens_idx.reserve(words.size());

    auto push_text = [&](const std::string& text) {
        for (const auto& w : parser::Prompt2Words(text)) {
            if (auto it = vocab.find(w); it != vocab.end()) {
                tokens_idx.push_back(it->second);
            } else {
                for (uint32_t t : ParseWords(w, vocab, merges)) {
                    tokens_idx.push_back(t);
                }
            }
        }
    };

    auto push_header = [&](const std::string& role) {
        tokens_idx.push_back(vocab.at("<|start_header_id|>"));
        tokens_idx.push_back(vocab.at(role));
        tokens_idx.push_back(vocab.at("<|end_header_id|>"));
        tokens_idx.push_back(vocab.at("\n\n"));
    };

    tokens_idx.push_back(vocab.at("<|begin_of_text|>"));

    push_header("system");
    push_text("Cutting Knowledge Date: December 2023\nToday Date: 02 Aug 2026\n\n");
    tokens_idx.push_back(vocab.at("<|eot_id|>"));

    push_header("user");
    push_text(input);
    tokens_idx.push_back(vocab.at("<|eot_id|>"));

    push_header("assistant");

    ModelInfo model_info{
        .heads_q = 32,
        .heads_k = 8,
        .head_dim = 64,
        .hidden = 2048,
        .n_layers = 16,
        .intermediate = 8192,
        .vocab = 128256,
        .max_seq_len = 512,
        .rms_eps = 1e-5f,
        .theta = 500000.0f,
        .factor = 32.0f,
        .low_freq_factor = 1.0f,
        .high_freq_factor = 4.0f,
        .original_max = 8192.0f,
    };

    auto [sin_table, cos_table] = MakeRopeTables(model_info.head_dim, model_info.theta, model_info.max_seq_len, model_info.factor, model_info.low_freq_factor, model_info.high_freq_factor, model_info.original_max);
    auto embeddings = Words2Embeddings(vocab, tokens_idx, model.embed_tensor);
    std::string ref_dir = "ref_instruct";
    CompareRef(embeddings, ref_dir + "/h0.bin");

    while (true) {
        Matrix x = embeddings;
        for (size_t layer = 0; layer < model.layers.size(); ++layer) {
            x = DecoderLayer(x, model.layers[layer], sin_table, cos_table, model_info);
            if (layer + 1 < model_info.n_layers) {
                CompareRef(x, ref_dir + "/h" + std::to_string(layer + 1) + ".bin");
            }
        }

        x = RMSNorm(x, model.final_norm);  
        CompareRef(x, ref_dir + "/h16.bin");

        Matrix res = x.last_row();
        auto logits = LMHead(res, model.embed_tensor);

        size_t new_token = ArgMax(logits);

        if (new_token == 128001 || new_token == 128009 || embeddings.rows() >= model_info.max_seq_len) {
            break;
        }

        std::cout << id_to_token[new_token];
        std::flush(std::cout);

        AddEmbedding(vocab, embeddings, new_token, model.embed_tensor);
    }
    std::cout << std::endl;

    return 0;
}