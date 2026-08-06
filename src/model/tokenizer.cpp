#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "../parsers/mapped_file.cpp"
#include "../parsers/prompt_parser.cpp"
#include "../parsers/tokenizer_parser.cpp"

namespace model {

class Tokenizer {
public:
    static Tokenizer Load(const std::string& path) {
        parser::MappedFile file{path};
        file.Advise(MADV_SEQUENTIAL);

        parser::TokenizerData data = parser::ImportTokenizer(file.view());

        Tokenizer tokenizer;
        tokenizer.vocab_ = std::move(data.vocab);
        tokenizer.merges_ = std::move(data.merges);

        uint32_t max_id = 0;
        for (const auto& [token, id] : tokenizer.vocab_) {
            max_id = std::max(max_id, id);
        }
        tokenizer.id_to_token_.resize(max_id + 1);
        for (const auto& [token, id] : tokenizer.vocab_) {
            tokenizer.id_to_token_[id] = token;
        }

        return tokenizer;
    }

    uint32_t Id(const std::string& token) const {
        auto it = vocab_.find(token);
        if (it == vocab_.end()) {
            throw std::runtime_error("tokenizer: no token '" + token + "' in vocab");
        }
        return it->second;
    }

    std::string_view Token(uint32_t id) const {
        if (id >= id_to_token_.size()) {
            throw std::runtime_error("tokenizer: id out of range: " + std::to_string(id));
        }
        return id_to_token_[id];
    }

    size_t size() const { return id_to_token_.size(); }

    void EncodeInto(const std::string& text, std::vector<uint32_t>& out) const {
        for (const std::string& word : parser::Prompt2Words(text)) {
            if (auto it = vocab_.find(word); it != vocab_.end()) {
                out.push_back(it->second);
            } else {
                MergeWord(word, out);
            }
        }
    }

    std::vector<uint32_t> Encode(const std::string& text) const {
        std::vector<uint32_t> out;
        EncodeInto(text, out);
        return out;
    }

private:
    void MergeWord(const std::string& word, std::vector<uint32_t>& out) const {
        std::vector<uint32_t> parts(word.size(), 0);
        for (size_t i = 0; i < word.size(); ++i) {
            auto it = vocab_.find(std::string(1, word[i]));
            if (it == vocab_.end()) {
                throw std::runtime_error("tokenizer: byte '" + std::string(1, word[i]) + "' not in vocab");
            }
            parts[i] = it->second;
        }

        while (parts.size() > 1) {
            uint32_t best_rank = UINT32_MAX;
            uint32_t new_id = 0;
            size_t at = 0;

            for (size_t i = 1; i < parts.size(); ++i) {
                uint64_t key = (static_cast<uint64_t>(parts[i - 1]) << 32) | parts[i];
                auto it = merges_.find(key);
                if (it != merges_.end() && it->second.rank < best_rank) {
                    best_rank = it->second.rank;
                    new_id = it->second.new_id;
                    at = i - 1;
                }
            }

            if (best_rank == UINT32_MAX) {
                break;
            }
            parts[at] = new_id;
            parts.erase(parts.begin() + at + 1);
        }

        out.insert(out.end(), parts.begin(), parts.end());
    }

    std::unordered_map<std::string, uint32_t> vocab_;
    std::unordered_map<uint64_t, parser::MergeInfo> merges_;
    std::vector<std::string> id_to_token_;
};

// <|begin_of_text|><|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>
class ChatPrompt {
public:
    explicit ChatPrompt(const Tokenizer& tokenizer) : tokenizer_(tokenizer) {
        tokens_.push_back(tokenizer_.Id("<|begin_of_text|>"));
    }

    ChatPrompt& Message(const std::string& role, const std::string& content) {
        Header(role);
        tokenizer_.EncodeInto(content, tokens_);
        tokens_.push_back(tokenizer_.Id("<|eot_id|>"));
        return *this;
    }

    ChatPrompt& AssistantHeader() {
        Header("assistant");
        return *this;
    }

    const std::vector<uint32_t>& tokens() const { return tokens_; }

private:
    void Header(const std::string& role) {
        tokens_.push_back(tokenizer_.Id("<|start_header_id|>"));
        tokens_.push_back(tokenizer_.Id(role));
        tokens_.push_back(tokenizer_.Id("<|end_header_id|>"));
        tokens_.push_back(tokenizer_.Id("\n\n"));
    }

    const Tokenizer& tokenizer_;
    std::vector<uint32_t> tokens_;
};

}  // namespace model
