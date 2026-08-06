#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <cctype>


namespace parser {

std::string punctuation = {'!', '.', ',',
                           '?', '/', '\\',
                           '*', '-', '#', 
                           '\n', '\r', '\t',
                           ':', ';', '=',
                           '%', 
                          };
                          
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

}