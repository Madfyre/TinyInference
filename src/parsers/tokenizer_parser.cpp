#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <cctype>
#include <charconv>
#include <iostream>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct FileInfo {
    size_t file_size;
    int fd;
};

FileInfo OpenFile(const std::string& filepath) {
    int fd = open(filepath.c_str(), O_RDONLY);
    if (fd < 0) {
        perror(("open failed: " + filepath).c_str());
        throw std::runtime_error("tokenizer not opened!");
    }

    struct stat st;
    fstat(fd, &st);

    return {static_cast<size_t>(st.st_size), fd};
}

namespace parser {

std::array<uint8_t, 512> MakeUnicodeToByte() {
    std::array<uint8_t, 512> table{};
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
            table[b] = b;
        } else {
            table[256 + n] = b;
            ++n;
        }
    }

    return table;
}

std::u32string Utf8ToU32(std::string_view string) {
    size_t i = 0;
    std::u32string result;

    while (i < string.size()) {
        uint8_t uchar = static_cast<uint8_t>(string[i]);
        uint32_t codepoint;
        if (uchar < 0x80) {
            codepoint = uchar;
            i++;
        } else {
            if (i + 1 >= string.size()) {
                throw std::runtime_error("bad token in tokenizer.json! check it!");
            }
            codepoint = ((uchar & 0x1F) << 6) | (static_cast<uint8_t>(string[i + 1]) & 0x3F);
            i += 2;
        }
        result.push_back(codepoint);
    }

    return result;
}

std::string U32ToChar(const std::u32string& u32_string) {
    static const std::array<uint8_t, 512> unicode_to_byte = MakeUnicodeToByte();

    std::string result;
    for (const auto& u32ch : u32_string) {
        result.push_back(static_cast<char>(unicode_to_byte[u32ch]));
    }

    return result;
}

uint32_t CodepointsToHex(std::u32string_view strview) {
    uint32_t result = 0;
    for (int i = 0; i < 4; ++i) {
        if (strview[i] >= '0' && strview[i] <= '9') {
            result = result * 16 + strview[i] - '0';
        } else if (strview[i] >= 'a' && strview[i] <= 'f') {
            result = result * 16 + strview[i] - 'a' + 10;
        }
    }
    return result;
}

std::u32string UnescapeCodepoints(const std::u32string& u32string) {
    std::u32string result;
    size_t i = 0;
    while (i < u32string.size()) {
        if (u32string[i] == '\\') {
            uint32_t charik;
            if (u32string[i + 1] == '"') {
                charik = 34;
                i += 2;
            } else if (u32string[i + 1] == '\\') {
                charik = 92;
                i += 2;
            } else if (u32string[i + 1] == 'u') {
                charik = CodepointsToHex({u32string.data() + i + 2  , 4});
                i += 6;
            } else {
                i += 1;
            }
            result.push_back(charik);
        } else {
            result.push_back(u32string[i]);
            i += 1;
        }
    }

    return result;
}

std::string Utf8ToString(std::string_view string) {
    return U32ToChar(UnescapeCodepoints(Utf8ToU32(string)));
}

size_t FindUnescapedQuote(const std::string_view& content, size_t init_pos) {
    while (true) {
        size_t candidate = content.find("\"", init_pos);
        if (candidate == std::string::npos) {
            return std::string::npos;
        }
        size_t backslashes = 0;
        size_t k = candidate;
        while (k > 0 && content[k - 1] == '\\') {
            backslashes++;
            k--;
        }

        if (backslashes % 2 == 0) {
            return candidate;
        }
        init_pos = candidate + 1;
    }
}

std::unordered_map<std::string, uint32_t> ImportTokens(const FileInfo& file_info) {
    
    char* ptr = static_cast<char*>(mmap(nullptr, file_info.file_size, PROT_READ, MAP_PRIVATE, file_info.fd, 0));
    madvise(ptr, file_info.file_size, MADV_SEQUENTIAL);

    std::string_view content{ptr, static_cast<size_t>(file_info.file_size)};

    size_t vocab_pos = content.find("\"vocab\": {") + 10;
    size_t vocab_end = content.find("\"merges\":", vocab_pos);

    std::unordered_map<std::string, uint32_t> result;
    result.reserve(file_info.file_size / 8);

    while (vocab_end > vocab_pos) {
        size_t token_start = FindUnescapedQuote(content, vocab_pos) + 1;
        size_t token_end = FindUnescapedQuote(content, token_start);

        size_t number_start = token_end + 3;
        size_t number_end = content.find(",", number_start);

        uint32_t id;
        std::from_chars(content.data() + number_start, content.data() + number_end, id);

        std::string key = Utf8ToString({ptr + token_start, token_end - token_start});
        result[key] = id;

        vocab_pos = number_end;
    }

    munmap(ptr, static_cast<size_t>(file_info.file_size));

    return result;
}

struct MergeInfo {
    uint32_t rank;
    uint32_t new_id;
};

std::unordered_map<uint64_t, MergeInfo> ImportMerges(const FileInfo& file_info,
                                                     const std::unordered_map<std::string, uint32_t>& vocab) {

    char* ptr = static_cast<char*>(mmap(nullptr, file_info.file_size, PROT_READ, MAP_PRIVATE, file_info.fd, 0));
    madvise(ptr, file_info.file_size, MADV_SEQUENTIAL);

    std::string_view content{ptr, static_cast<size_t>(file_info.file_size)};
    size_t merges_pos = content.find("\"merges\": [") + 11;

    std::unordered_map<uint64_t, MergeInfo> result;
    result.reserve(file_info.file_size / 8);

    uint32_t rank = 0;
    while (true) {
        size_t next_open = content.find("[", merges_pos);
        if (next_open == std::string::npos) {
            break;
        }

        size_t l_start = FindUnescapedQuote(content, merges_pos) + 1;
        size_t l_end = FindUnescapedQuote(content, l_start);
        size_t r_start =  FindUnescapedQuote(content, l_end + 1) + 1;
        size_t r_end = FindUnescapedQuote(content, r_start);

        std::string left_key = Utf8ToString({ptr + l_start, l_end - l_start});
        std::string right_key = Utf8ToString({ptr + r_start, r_end - r_start});

        uint32_t left_id, right_id;
        if (auto it = vocab.find(left_key); it != vocab.end()) {
            left_id = it->second;
        }
        if (auto it = vocab.find(right_key); it != vocab.end()) {
            right_id = it->second;
        }
        uint64_t key = (static_cast<uint64_t>(left_id) << 32) | right_id;

        uint32_t new_id;
        if (auto it = vocab.find(left_key + right_key); it != vocab.end()) {
            new_id = it->second;
        }

        result[key] = {.rank = rank, .new_id = new_id};

        merges_pos = r_end + 1;
        rank++;
    }

    munmap(ptr, static_cast<size_t>(file_info.file_size));

    return result;
}


}