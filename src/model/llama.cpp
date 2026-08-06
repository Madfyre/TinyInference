#pragma once

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <sys/mman.h>

#include "../parsers/config_parser.cpp"
#include "../parsers/mapped_file.cpp"
#include "../parsers/model_parser.cpp"
#include "matrix.cpp"
#include "ops.cpp"
#include "tokenizer.cpp"

namespace model {

class Llama {
public:
    // Fired once per hidden state of a forward pass: 0 is the embeddings,
    // n_layers is the output of the final norm.
    using LayerHook = std::function<void(size_t, const Matrix&)>;

    const parser::ModelConfig& config() const { return config_; }
    const Tokenizer& tokenizer() const { return tokenizer_; }

    void set_layer_hook(LayerHook hook) { layer_hook_ = std::move(hook); }

    template <typename OnToken>
    void Generate(const std::vector<uint32_t>& prompt, const OnToken& on_token) {
        Matrix embeddings = Words2Embeddings(prompt, weights_.embed_tensor);
        if (layer_hook_) {
            layer_hook_(0, embeddings);
        }

        while (true) {
            uint32_t next = Forward(embeddings);
            if (parser::IsEos(config_, next) ||
                embeddings.rows() >= static_cast<size_t>(config_.max_seq_len)) {
                return;
            }

            on_token(next);
            AddEmbedding(embeddings, next, weights_.embed_tensor);
        }
    }

private:
    friend class LlamaBuilder;

    uint32_t Forward(const Matrix& embeddings) {
        Matrix x = embeddings;

        for (size_t layer = 0; layer < weights_.layers.size(); ++layer) {
            x = DecoderLayer(x, weights_.layers[layer]);
            if (layer_hook_ && layer + 1 < static_cast<size_t>(config_.n_layers)) {
                layer_hook_(layer + 1, x);
            }
        }

        x = RMSNorm(x, weights_.final_norm, config_.rms_eps);
        if (layer_hook_) {
            layer_hook_(config_.n_layers, x);
        }

        Matrix last = x.last_row();
        Matrix logits = LMHead(last, weights_.embed_tensor);
        return static_cast<uint32_t>(ArgMax(logits));
    }

    Matrix DecoderLayer(Matrix& input, const parser::Layer& layer) {
        Matrix normed = input;
        RMSNorm(normed, layer.input_layernorm, config_.rms_eps);

        Matrix Q = MatMul(normed, layer.q_proj);
        Matrix K = MatMul(normed, layer.k_proj);
        Matrix V = MatMul(normed, layer.v_proj);

        RoPE(Q, rope_cos_, rope_sin_, config_.heads_q, config_.head_dim, Q.rows());
        RoPE(K, rope_cos_, rope_sin_, config_.heads_k, config_.head_dim, K.rows());

        Matrix attn = Attention(Q, K, V, config_.heads_q, config_.heads_k, config_.head_dim);
        Matrix attn_out = MatMul(attn, layer.o_proj);
        Matrix enriched = Add(input, attn_out);

        Matrix normed2 = enriched;
        RMSNorm(normed2, layer.post_attn_layernorm, config_.rms_eps);

        Matrix gate = MatMul(normed2, layer.gate_proj);
        Matrix up = MatMul(normed2, layer.up_proj);
        Matrix hidden = Hadamard(up, SiLU(gate));
        Matrix mlp = MatMul(hidden, layer.down_proj);

        return Add(enriched, mlp);
    }

    parser::ModelConfig config_;
    parser::Model weights_;
    Tokenizer tokenizer_;

    Matrix rope_sin_;
    Matrix rope_cos_;

    LayerHook layer_hook_;
};

class LlamaBuilder {
public:
    LlamaBuilder& Dir(std::string dir) {
        dir_ = std::move(dir);
        return *this;
    }

    LlamaBuilder& MaxSeqLen(int32_t max_seq_len) {
        max_seq_len_ = max_seq_len;
        return *this;
    }

    Llama Build() {
        if (dir_.empty()) {
            throw std::runtime_error("LlamaBuilder: model directory is not set");
        }

        Llama llama;

        parser::MappedFile config_file{dir_ + "/config.json"};
        llama.config_ = parser::ParseModelConfig(config_file.view());
        if (max_seq_len_ > 0) {
            llama.config_.max_seq_len = max_seq_len_;
        }

        parser::MappedFile weights_file{dir_ + "/model.safetensors"};
        llama.weights_ = parser::ParseWeights(std::move(weights_file));

        llama.tokenizer_ = Tokenizer::Load(dir_ + "/tokenizer.json");

        Check(llama);

        auto [sin_table, cos_table] =
            MakeRopeTables(llama.config_.head_dim, llama.config_.theta, llama.config_.max_seq_len,
                           llama.config_.factor, llama.config_.low_freq_factor,
                           llama.config_.high_freq_factor, llama.config_.original_max);
                           
        llama.rope_sin_ = std::move(sin_table);
        llama.rope_cos_ = std::move(cos_table);

        return llama;
    }

private:
    static void Check(const Llama& llama) {
        const parser::ModelConfig& config = llama.config_;

        if (llama.weights_.layers.size() != static_cast<size_t>(config.n_layers)) {
            throw std::runtime_error("model: config says " + std::to_string(config.n_layers) +
                                     " layers, safetensors has " +
                                     std::to_string(llama.weights_.layers.size()));
        }
        if (llama.weights_.embed_tensor.rows != static_cast<size_t>(config.vocab)) {
            throw std::runtime_error("model: config says vocab " + std::to_string(config.vocab) +
                                     ", embeddings have " +
                                     std::to_string(llama.weights_.embed_tensor.rows));
        }
        if (llama.weights_.embed_tensor.cols != static_cast<size_t>(config.hidden)) {
            throw std::runtime_error("model: config says hidden " + std::to_string(config.hidden) +
                                     ", embeddings have " +
                                     std::to_string(llama.weights_.embed_tensor.cols));
        }
    }

    std::string dir_;
    int32_t max_seq_len_ = 0;
};

}  // namespace model
