#pragma once

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace parser {

enum class JsonKind : uint8_t {
    kEnd,
    kObjectBegin,
    kObjectEnd,
    kArrayBegin,
    kArrayEnd,
    kString,
    kNumber,
    kTrue,
    kFalse,
    kNull,
    kColon,
    kComma,
};

const char* JsonKindName(JsonKind kind) {
    switch (kind) {
        case JsonKind::kEnd: return "end of input";
        case JsonKind::kObjectBegin: return "'{'";
        case JsonKind::kObjectEnd: return "'}'";
        case JsonKind::kArrayBegin: return "'['";
        case JsonKind::kArrayEnd: return "']'";
        case JsonKind::kString: return "string";
        case JsonKind::kNumber: return "number";
        case JsonKind::kTrue: return "true";
        case JsonKind::kFalse: return "false";
        case JsonKind::kNull: return "null";
        case JsonKind::kColon: return "':'";
        case JsonKind::kComma: return "','";
    }
    return "?";
}

class JsonLexer {
public:
    explicit JsonLexer(std::string_view text, size_t pos = 0) : text_(text), pos_(pos) {}

    JsonKind Next() {
        Token token = has_peek_ ? peek_ : Scan();
        has_peek_ = false;
        kind_ = token.kind;
        slice_ = token.text;
        return kind_;
    }

    JsonKind Peek() {
        if (!has_peek_) {
            peek_start_ = pos_;
            peek_ = Scan();
            has_peek_ = true;
        }
        return peek_.kind;
    }

    void Expect(JsonKind expected) {
        if (Next() != expected) {
            throw std::runtime_error(std::string("json: expected ") + JsonKindName(expected) +
                                     " but found " + JsonKindName(kind_) + " at offset " +
                                     std::to_string(pos_));
        }
    }

    JsonKind kind() const { return kind_; }

    // String body without quotes, escapes left intact; or a number literal.
    std::string_view text() const { return slice_; }

    // Key of the member opened by the last NextMember(), same encoding as text().
    std::string_view key() const { return key_; }

    template <typename T>
    T Number() const {
        T value{};
        auto [end, ec] = std::from_chars(slice_.data(), slice_.data() + slice_.size(), value);
        if (ec != std::errc{}) {
            throw std::runtime_error("json: malformed number '" + std::string(slice_) + "'");
        }
        return value;
    }

    // False once '}' is consumed; otherwise key() is set and the value is unread.
    bool NextMember() {
        JsonKind ahead = Peek();
        if (ahead == JsonKind::kComma) {
            Next();
            ahead = Peek();
        }
        if (ahead == JsonKind::kObjectEnd) {
            Next();
            return false;
        }

        Expect(JsonKind::kString);
        key_ = slice_;
        Expect(JsonKind::kColon);
        return true;
    }

    // False once ']' is consumed; otherwise the element is left unread.
    bool NextElement() {
        JsonKind ahead = Peek();
        if (ahead == JsonKind::kComma) {
            Next();
            ahead = Peek();
        }
        if (ahead == JsonKind::kArrayEnd) {
            Next();
            return false;
        }
        return true;
    }

    void SkipValue() {
        JsonKind kind = Next();
        if (kind != JsonKind::kObjectBegin && kind != JsonKind::kArrayBegin) {
            return;
        }

        size_t depth = 1;
        while (depth > 0) {
            switch (Next()) {
                case JsonKind::kObjectBegin:
                case JsonKind::kArrayBegin:
                    ++depth;
                    break;
                case JsonKind::kObjectEnd:
                case JsonKind::kArrayEnd:
                    --depth;
                    break;
                case JsonKind::kEnd:
                    throw std::runtime_error("json: unexpected end of input inside a value");
                default:
                    break;
            }
        }
    }

    // Where an independent lexer would have to resume to see the same tokens.
    size_t offset() const { return has_peek_ ? peek_start_ : pos_; }

private:
    struct Token {
        JsonKind kind;
        std::string_view text;
    };

    void SkipWhitespace() {
        while (pos_ < text_.size()) {
            char c = text_[pos_];
            if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
                return;
            }
            ++pos_;
        }
    }

    void ExpectLiteral(std::string_view word) {
        if (text_.substr(pos_, word.size()) != word) {
            throw std::runtime_error("json: bad literal at offset " + std::to_string(pos_));
        }
        pos_ += word.size();
    }

    Token ScanString() {
        ++pos_;  // opening quote
        size_t start = pos_;
        while (pos_ < text_.size() && text_[pos_] != '"') {
            pos_ += (text_[pos_] == '\\') ? 2 : 1;
        }
        if (pos_ >= text_.size()) {
            throw std::runtime_error("json: unterminated string at offset " + std::to_string(start));
        }

        std::string_view body = text_.substr(start, pos_ - start);
        ++pos_;  // closing quote
        return {JsonKind::kString, body};
    }

    Token ScanNumber() {
        size_t start = pos_;
        if (text_[pos_] == '-' || text_[pos_] == '+') {
            ++pos_;
        }
        while (pos_ < text_.size()) {
            char c = text_[pos_];
            bool part = (c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '-' || c == '+';
            if (!part) {
                break;
            }
            ++pos_;
        }
        if (pos_ == start) {
            throw std::runtime_error(std::string("json: unexpected character '") + text_[start] +
                                     "' at offset " + std::to_string(start));
        }

        return {JsonKind::kNumber, text_.substr(start, pos_ - start)};
    }

    Token Scan() {
        SkipWhitespace();
        if (pos_ >= text_.size()) {
            return {JsonKind::kEnd, {}};
        }

        switch (text_[pos_]) {
            case '{': ++pos_; return {JsonKind::kObjectBegin, {}};
            case '}': ++pos_; return {JsonKind::kObjectEnd, {}};
            case '[': ++pos_; return {JsonKind::kArrayBegin, {}};
            case ']': ++pos_; return {JsonKind::kArrayEnd, {}};
            case ':': ++pos_; return {JsonKind::kColon, {}};
            case ',': ++pos_; return {JsonKind::kComma, {}};
            case '"': return ScanString();
            case 't': ExpectLiteral("true"); return {JsonKind::kTrue, {}};
            case 'f': ExpectLiteral("false"); return {JsonKind::kFalse, {}};
            case 'n': ExpectLiteral("null"); return {JsonKind::kNull, {}};
            default: return ScanNumber();
        }
    }

    std::string_view text_;
    size_t pos_ = 0;

    JsonKind kind_ = JsonKind::kEnd;
    std::string_view slice_;
    std::string_view key_;

    Token peek_{};
    size_t peek_start_ = 0;
    bool has_peek_ = false;
};

}  // namespace parser
