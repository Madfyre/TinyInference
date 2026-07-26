#include <string>
#include <iostream>
#include <unordered_map>

#include "../parsers/tokenizer_parser.cpp"
#include "../parsers/prompt_parser.cpp"
#include "../parsers/model_parser.cpp"

void DebugWords(const std::vector<std::string>& words) {
    for (size_t i = 0; i < words.size(); ++i) {
        std::cout << "\"" << words[i] << "\"" << ", ";
    }
}

int main() {
    std::string input = "Hello, I'm GPT-4 and I've got 1234 dollars. It'll be fine!!!\nYes, it's true.";

    auto x = parser::ImportTokens("models/llama-3.2-1b/tokenizer.json");
    std::vector<std::string> words = parser::Prompt2Words(input);
    for (int i = 0; i < words.size(); ++i) {
        if (auto it = x.find(words[i]); it != x.end()) {
            std::cout << it->first << " " << it->second << std::endl;
        }
    }
    // DebugWords(words);
}