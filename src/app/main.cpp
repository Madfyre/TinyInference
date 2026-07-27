#include <cstdint>
#include <list>
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

void DebugWords(const std::vector<std::string>& words) {
    for (size_t i = 0; i < words.size(); ++i) {
        std::cout << "\"" << words[i] << "\"" << ", ";
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

int main() {
    std::string input = "Hello, I'm GPT-4 and I've got 1234 dollars. It'll be fine!!!\nYes, it's true.";

    auto tokenizer_info = OpenFile("models/llama-3.2-1b/tokenizer.json");
    auto vocab = parser::ImportTokens(tokenizer_info);
    auto merges = parser::ImportMerges(tokenizer_info, vocab);

    auto model_info = OpenFile("models/llama-3.2-1b/model.safetensors");
    parser::Model model = parser::ParseConfig(model_info);

    std::vector<std::string> words = parser::Prompt2Words(input);

    std::cout << "TOKENS" << std::endl;
    for (int i = 0; i < words.size(); ++i) {
        if (auto it = vocab.find(words[i]); it != vocab.end()) {
            std::cout << it->first << " " << it->second << std::endl;
        } else {
            std::vector<uint32_t> tokenized_word = ParseWords(words[i], vocab, merges);
            std::cout << words[i] << ": ";
            for (const auto& token : tokenized_word) {
                std::cout << token << " ";
            }
            std::cout << std::endl;
        }
    }

    DebugTensor(model.embed_tensor, "embed");
    DebugTensor(model.layers[0].q_proj, "layer0.q_proj");
    
    // DebugWords(words);
}