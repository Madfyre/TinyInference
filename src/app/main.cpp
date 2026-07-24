#include <string>
#include <iostream>

#include "../parsers/prompt_parser.cpp"
#include "../parsers/model_parser.cpp"

void DebugWords(const std::vector<std::string>& words) {
    for (size_t i = 0; i < words.size(); ++i) {
        std::cout << "\"" << words[i] << "\"" << ", ";
    }
}

int main() {
    std::string input = "Hello, I'm GPT-4 and I've got 1234 dollars. It'll be fine!!!\nYes, it's true.";

    std::vector<std::string> words = parser::Prompt2Words(input);
    DebugWords(words);
}