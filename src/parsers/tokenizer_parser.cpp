#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <cctype>
#include <charconv>
#include <iostream>

#include "json_lexer.cpp"
#include "mapped_file.cpp"

namespace parser {

constexpr size_t kAlphabetSize = 512;

constexpr std::array<int16_t, kAlphabetSize> MakeUnicodeToByte() {
    std::array<int16_t, kAlphabetSize> table;
    table.fill(-1);
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

constexpr std::array<int16_t, kAlphabetSize> kUnicodeToByte = MakeUnicodeToByte();

uint32_t ReadHex4(std::string_view raw, size_t& pos) {
    if (pos + 4 > raw.size()) {
        throw std::runtime_error("tokenizer.json: truncated \\u escape");
    }

    uint32_t value = 0;
    for (size_t k = 0; k < 4; ++k) {
        char c = raw[pos + k];
        uint32_t digit;
        if (c >= '0' && c <= '9') {
            digit = c - '0';
        } else if (c >= 'a' && c <= 'f') {
            digit = c - 'a' + 10;
        } else if (c >= 'A' && c <= 'F') {
            digit = c - 'A' + 10;
        } else {
            throw std::runtime_error("tokenizer.json: bad hex digit in \\u escape");
        }
        value = value * 16 + digit;
    }

    pos += 4;
    return value;
}

// One escape or UTF-8 sequence -> codepoint. JSON layer only.
uint32_t NextCodepoint(std::string_view raw, size_t& pos) {
    uint8_t lead = static_cast<uint8_t>(raw[pos]);

    if (lead == '\\') {
        if (pos + 1 >= raw.size()) {
            throw std::runtime_error("tokenizer.json: trailing backslash in string");
        }
        char esc = raw[pos + 1];
        pos += 2;

        switch (esc) {
            case '"': return 0x22;
            case '\\': return 0x5C;
            case '/': return 0x2F;
            case 'b': return 0x08;
            case 'f': return 0x0C;
            case 'n': return 0x0A;
            case 'r': return 0x0D;
            case 't': return 0x09;
            case 'u': {
                uint32_t codepoint = ReadHex4(raw, pos);
                if (codepoint >= 0xD800 && codepoint <= 0xDBFF) {
                    if (pos + 2 > raw.size() || raw[pos] != '\\' || raw[pos + 1] != 'u') {
                        throw std::runtime_error("tokenizer.json: unpaired high surrogate");
                    }
                    pos += 2;
                    uint32_t low = ReadHex4(raw, pos);
                    if (low < 0xDC00 || low > 0xDFFF) {
                        throw std::runtime_error("tokenizer.json: malformed surrogate pair");
                    }
                    codepoint = 0x10000 + ((codepoint - 0xD800) << 10) + (low - 0xDC00);
                } else if (codepoint >= 0xDC00 && codepoint <= 0xDFFF) {
                    throw std::runtime_error("tokenizer.json: unpaired low surrogate");
                }
                return codepoint;
            }
            default:
                throw std::runtime_error(std::string("tokenizer.json: unknown escape \\") + esc);
        }
    }

    if (lead < 0x80) {
        pos += 1;
        return lead;
    }

    uint32_t codepoint;
    size_t extra;
    if ((lead & 0xE0) == 0xC0) {
        codepoint = lead & 0x1F;
        extra = 1;
    } else if ((lead & 0xF0) == 0xE0) {
        codepoint = lead & 0x0F;
        extra = 2;
    } else if ((lead & 0xF8) == 0xF0) {
        codepoint = lead & 0x07;
        extra = 3;
    } else {
        throw std::runtime_error("tokenizer.json: invalid UTF-8 lead byte");
    }

    if (pos + extra >= raw.size()) {
        throw std::runtime_error("tokenizer.json: truncated UTF-8 sequence");
    }
    for (size_t k = 1; k <= extra; ++k) {
        uint8_t cont = static_cast<uint8_t>(raw[pos + k]);
        if ((cont & 0xC0) != 0x80) {
            throw std::runtime_error("tokenizer.json: invalid UTF-8 continuation byte");
        }
        codepoint = (codepoint << 6) | (cont & 0x3F);
    }
    pos += extra + 1;

    return codepoint;
}

void AppendUtf8(std::string& out, uint32_t codepoint) {
    if (codepoint < 0x80) {
        out.push_back(static_cast<char>(codepoint));
    } else if (codepoint < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
        out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else if (codepoint < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
        out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (codepoint >> 18)));
        out.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    }
}

// Literal text, e.g. "<|begin_of_text|>": unescape, keep the UTF-8 as is.
void DecodeUtf8(std::string_view raw, std::string& out) {
    out.clear();

    size_t i = 0;
    while (i < raw.size()) {
        AppendUtf8(out, NextCodepoint(raw, i));
    }
}

// Byte-level BPE text: every codepoint stands for one raw byte (U+0120 is 0x20).
void DecodeByteLevel(std::string_view raw, std::string& out) {
    out.clear();

    size_t i = 0;
    while (i < raw.size()) {
        uint32_t codepoint = NextCodepoint(raw, i);
        if (codepoint >= kAlphabetSize || kUnicodeToByte[codepoint] < 0) {
            throw std::runtime_error("tokenizer.json: codepoint outside byte-level alphabet: " +
                                     std::to_string(codepoint));
        }
        out.push_back(static_cast<char>(kUnicodeToByte[codepoint]));
    }
}

void ReadAddedTokens(JsonLexer& lexer, std::unordered_map<std::string, uint32_t>& out) {
    std::string token;

    lexer.Expect(JsonKind::kArrayBegin);
    while (lexer.NextElement()) {
        lexer.Expect(JsonKind::kObjectBegin);

        uint32_t id = 0;
        bool has_id = false;
        bool has_content = false;

        while (lexer.NextMember()) {
            if (lexer.key() == "id") {
                lexer.Expect(JsonKind::kNumber);
                id = lexer.Number<uint32_t>();
                has_id = true;
            } else if (lexer.key() == "content") {
                lexer.Expect(JsonKind::kString);
                DecodeUtf8(lexer.text(), token);
                has_content = true;
            } else {
                lexer.SkipValue();
            }
        }

        if (!has_id || !has_content) {
            throw std::runtime_error("tokenizer.json: added token without id or content");
        }
        out[token] = id;
    }
}

void ReadVocab(JsonLexer& lexer, std::unordered_map<std::string, uint32_t>& out) {
    std::string key;

    lexer.Expect(JsonKind::kObjectBegin);
    while (lexer.NextMember()) {
        DecodeByteLevel(lexer.key(), key);
        lexer.Expect(JsonKind::kNumber);
        out[key] = lexer.Number<uint32_t>();
    }
}

struct MergeInfo {
    uint32_t rank;
    uint32_t new_id;
};

struct TokenizerData {
    std::unordered_map<std::string, uint32_t> vocab;
    std::unordered_map<uint64_t, MergeInfo> merges;
};

// [ ["\u0120", "\u0120"], ["\u0120", "t"], ... ]
void ReadMerges(JsonLexer& lexer, const std::unordered_map<std::string, uint32_t>& vocab,
                std::unordered_map<uint64_t, MergeInfo>& out) {
    std::string left_key;
    std::string right_key;
    std::string pair_key;

    uint32_t rank = 0;
    lexer.Expect(JsonKind::kArrayBegin);
    while (lexer.NextElement()) {
        lexer.Expect(JsonKind::kArrayBegin);

        lexer.Expect(JsonKind::kString);
        DecodeByteLevel(lexer.text(), left_key);
        lexer.Expect(JsonKind::kComma);
        lexer.Expect(JsonKind::kString);
        DecodeByteLevel(lexer.text(), right_key);

        lexer.Expect(JsonKind::kArrayEnd);

        auto left = vocab.find(left_key);
        auto right = vocab.find(right_key);
        if (left == vocab.end() || right == vocab.end()) {
            throw std::runtime_error("tokenizer.json: merge rank " + std::to_string(rank) +
                                     " references a token missing from vocab");
        }

        pair_key.assign(left_key);
        pair_key.append(right_key);

        auto merged = vocab.find(pair_key);
        if (merged == vocab.end()) {
            throw std::runtime_error("tokenizer.json: merge rank " + std::to_string(rank) +
                                     " produces a token missing from vocab");
        }

        uint64_t key = (static_cast<uint64_t>(left->second) << 32) | right->second;
        out[key] = {.rank = rank, .new_id = merged->second};

        rank++;
    }
}

// Vocab and merges in a single walk. Merges need a complete vocab, so if the file
// happens to list them first they are parked and replayed from their offset.
TokenizerData ImportTokenizer(std::string_view content) {
    TokenizerData data;
    data.vocab.reserve(content.size() / 8);
    data.merges.reserve(content.size() / 8);

    JsonLexer lexer{content};
    size_t parked_merges = 0;
    bool vocab_read = false;

    lexer.Expect(JsonKind::kObjectBegin);
    while (lexer.NextMember()) {
        if (lexer.key() == "added_tokens") {
            ReadAddedTokens(lexer, data.vocab);
        } else if (lexer.key() == "model") {
            lexer.Expect(JsonKind::kObjectBegin);
            while (lexer.NextMember()) {
                if (lexer.key() == "vocab") {
                    ReadVocab(lexer, data.vocab);
                    vocab_read = true;
                } else if (lexer.key() == "merges") {
                    if (vocab_read) {
                        ReadMerges(lexer, data.vocab, data.merges);
                    } else {
                        parked_merges = lexer.offset();
                        lexer.SkipValue();
                    }
                } else {
                    lexer.SkipValue();
                }
            }
        } else {
            lexer.SkipValue();
        }
    }

    if (!vocab_read) {
        throw std::runtime_error("tokenizer.json: no model.vocab section");
    }
    if (parked_merges != 0) {
        JsonLexer merges_lexer{content, parked_merges};
        ReadMerges(merges_lexer, data.vocab, data.merges);
    }

    return data;
}


}