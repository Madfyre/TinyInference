#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <cctype>


namespace parser {

std::string punctuation = {'!', '.', ',',
                           '?', '/', '\\',
                           '*', '-', '#'
                          };

std::string spaces = {'\n', '\r', '\t'};

std::vector<std::string> Prompt2Words(const std::string& prompt) {
    size_t left = 0;
    size_t right = 0;

    std::vector<std::string> result;
    while (right < prompt.size()) {
        if (prompt[right] == '\'') {
            right++;
            if (right < prompt.size() &&
                (prompt[right] == 's' || prompt[right] == 't' ||
                prompt[right] == 'm' || prompt[right] == 'd')) {
                right++;
            } else if (
                (right + 1 < prompt.size()) &&
                ((prompt[right] == 'r' && prompt[right + 1] == 'e') ||
                (prompt[right] == 'v' && prompt[right + 1] == 'e') ||
                (prompt[right] == 'l' && prompt[right + 1] == 'l'))) {
                right += 2;
            }
        } else if (std::isalpha(prompt[right])) {
            while (std::isalpha(prompt[right])) {
                right++;
            }
        } else if (std::isdigit(prompt[right])) {
            if (prompt[left] == ' ') {
                result.push_back(" ");
                left++;
            }
            while (std::isdigit(prompt[right])) {
                if (right == left + 3) {
                    result.push_back(prompt.substr(left, 3));
                    left += 3;
                }
                right++;
            }
        } else if (spaces.find(prompt[right]) != std::string::npos) {
            if (prompt[left] == ' ') {
                result.push_back(" ");
                left++;
            }
            while (spaces.find(prompt[right]) != std::string::npos) {
                right++;
            }
        } else if (punctuation.find(prompt[right]) != std::string::npos) {
            if (prompt[left] == ' ') {
                result.push_back(" ");
                left++;
            }
            while (punctuation.find(prompt[right]) != std::string::npos) {
                right++;
            }
        } else {
            right++;
            continue;
        }

        if (right > left) {
            result.push_back(prompt.substr(left, right - left));
        }
        left = right;
    }

    return result;
}

std::vector<std::u32string> Words2Unicode(const std::vector<std::string>& words) {
    std::vector<std::u32string> result(words.size());
    for (size_t i = 0; i < words.size(); ++i) {
        result[i].resize(words[i].size());
        for (size_t j = 0; j < words[i].size(); ++j) {
            if ((static_cast<uint32_t>(words[i][j]) > 32 && static_cast<uint32_t>(words[i][j]) < 127) ||
                (static_cast<uint32_t>(words[i][j]) > 160 && static_cast<uint32_t>(words[i][j]) < 173) ||
                (static_cast<uint32_t>(words[i][j]) > 173 && static_cast<uint32_t>(words[i][j]) < 256)) {
                result[i][j] = words[i][j];
            } else {
                result[i][j] = static_cast<uint32_t>(words[i][j]) + 256;
            }
        }
    }

    return result;
}

std::vector<size_t> Unicode2Token(const std::vector<std::u32string>& unicode) {
    for (size_t i = 0; i < words.size(); ++i) {
        
    }

    return result;
}

}