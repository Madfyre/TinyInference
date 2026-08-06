#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "json_lexer.cpp"

namespace parser {

struct ModelConfig {
    int32_t heads_q;
    int32_t heads_k;
    int32_t head_dim;
    int32_t hidden;
    int32_t n_layers;
    int32_t intermediate;
    int32_t vocab;

    float rms_eps;
    float theta;

    float factor;
    float low_freq_factor;
    float high_freq_factor;
    float original_max;

    uint32_t bos_token_id;
    std::vector<uint32_t> eos_token_ids;

    int32_t max_seq_len = 512;
};

void ReadRopeScaling(JsonLexer& lexer, ModelConfig& config) {
    lexer.Expect(JsonKind::kObjectBegin);
    while (lexer.NextMember()) {
        std::string_view key = lexer.key();
        if (key == "factor") {
            lexer.Expect(JsonKind::kNumber);
            config.factor = lexer.Number<float>();
        } else if (key == "low_freq_factor") {
            lexer.Expect(JsonKind::kNumber);
            config.low_freq_factor = lexer.Number<float>();
        } else if (key == "high_freq_factor") {
            lexer.Expect(JsonKind::kNumber);
            config.high_freq_factor = lexer.Number<float>();
        } else if (key == "original_max_position_embeddings") {
            lexer.Expect(JsonKind::kNumber);
            config.original_max = lexer.Number<float>();
        } else {
            lexer.SkipValue();
        }
    }
}

// eos_token_id is a scalar in some checkpoints and a list in others.
void ReadEosTokenIds(JsonLexer& lexer, ModelConfig& config) {
    if (lexer.Peek() == JsonKind::kArrayBegin) {
        lexer.Next();
        while (lexer.NextElement()) {
            lexer.Expect(JsonKind::kNumber);
            config.eos_token_ids.push_back(lexer.Number<uint32_t>());
        }
        return;
    }
    lexer.Expect(JsonKind::kNumber);
    config.eos_token_ids.push_back(lexer.Number<uint32_t>());
}

ModelConfig ParseModelConfig(std::string_view content) {
    ModelConfig config{};
    uint32_t seen = 0;
    constexpr uint32_t kRequired = 0x3FF;

    JsonLexer lexer{content};
    lexer.Expect(JsonKind::kObjectBegin);
    while (lexer.NextMember()) {
        std::string_view key = lexer.key();

        if (key == "num_attention_heads") {
            lexer.Expect(JsonKind::kNumber);
            config.heads_q = lexer.Number<int32_t>();
            seen |= 1 << 0;
        } else if (key == "num_key_value_heads") {
            lexer.Expect(JsonKind::kNumber);
            config.heads_k = lexer.Number<int32_t>();
            seen |= 1 << 1;
        } else if (key == "head_dim") {
            lexer.Expect(JsonKind::kNumber);
            config.head_dim = lexer.Number<int32_t>();
            seen |= 1 << 2;
        } else if (key == "hidden_size") {
            lexer.Expect(JsonKind::kNumber);
            config.hidden = lexer.Number<int32_t>();
            seen |= 1 << 3;
        } else if (key == "num_hidden_layers") {
            lexer.Expect(JsonKind::kNumber);
            config.n_layers = lexer.Number<int32_t>();
            seen |= 1 << 4;
        } else if (key == "intermediate_size") {
            lexer.Expect(JsonKind::kNumber);
            config.intermediate = lexer.Number<int32_t>();
            seen |= 1 << 5;
        } else if (key == "vocab_size") {
            lexer.Expect(JsonKind::kNumber);
            config.vocab = lexer.Number<int32_t>();
            seen |= 1 << 6;
        } else if (key == "rms_norm_eps") {
            lexer.Expect(JsonKind::kNumber);
            config.rms_eps = lexer.Number<float>();
            seen |= 1 << 7;
        } else if (key == "rope_theta") {
            lexer.Expect(JsonKind::kNumber);
            config.theta = lexer.Number<float>();
            seen |= 1 << 8;
        } else if (key == "rope_scaling") {
            ReadRopeScaling(lexer, config);
            seen |= 1 << 9;
        } else if (key == "bos_token_id") {
            lexer.Expect(JsonKind::kNumber);
            config.bos_token_id = lexer.Number<uint32_t>();
        } else if (key == "eos_token_id") {
            ReadEosTokenIds(lexer, config);
        } else {
            lexer.SkipValue();
        }
    }

    if (seen != kRequired) {
        throw std::runtime_error("config.json: missing required fields (mask " + std::to_string(seen) + ")");
    }
    if (config.eos_token_ids.empty()) {
        throw std::runtime_error("config.json: no eos_token_id");
    }
    if (config.heads_q % config.heads_k != 0) {
        throw std::runtime_error("config.json: num_attention_heads is not a multiple of num_key_value_heads");
    }

    return config;
}

bool IsEos(const ModelConfig& config, uint32_t token_id) {
    for (uint32_t eos : config.eos_token_ids) {
        if (eos == token_id) {
            return true;
        }
    }
    return false;
}

}  // namespace parser
