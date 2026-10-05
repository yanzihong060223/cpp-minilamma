#include "include/tonizer.h"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <unordered_set>

namespace yan_lamma {

static void SkipWhitespace(const std::string& text, std::size_t& pos) {
    while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t' ||
           text[pos] == '\r' || text[pos] == '\n')) ++pos;
}

static std::string ParseString(const std::string& text, std::size_t& pos) {
    SkipWhitespace(text, pos);
    if (pos == text.size() || text[pos++] != '"') {
        throw std::runtime_error("Expected a JSON string");
    }
    std::string result;
    while (pos < text.size()) {
        const auto byte = static_cast<unsigned char>(text[pos++]);
        if (byte == '"') return result;
        if (byte < 0x20) throw std::runtime_error("Unescaped JSON control character");
        if (byte != '\\') {
            const auto start = pos - 1;
            std::size_t length = 1;
            if (byte >= 0x80) {
                if (byte >= 0xC2 && byte <= 0xDF) length = 2;
                else if (byte >= 0xE0 && byte <= 0xEF) length = 3;
                else if (byte >= 0xF0 && byte <= 0xF4) length = 4;
                else throw std::runtime_error("Invalid UTF-8 string");
                if (length > text.size() - start) {
                    throw std::runtime_error("Truncated UTF-8 string");
                }
                for (std::size_t i = 1; i < length; ++i) {
                    const auto next = static_cast<unsigned char>(text[start + i]);
                    if (next < 0x80 || next > 0xBF) {
                        throw std::runtime_error("Invalid UTF-8 continuation byte");
                    }
                }
                const auto second = static_cast<unsigned char>(text[start + 1]);
                if ((byte == 0xE0 && second < 0xA0) ||
                    (byte == 0xED && second >= 0xA0) ||
                    (byte == 0xF0 && second < 0x90) ||
                    (byte == 0xF4 && second >= 0x90)) {
                    throw std::runtime_error("Invalid UTF-8 codepoint");
                }
            }
            result.append(text, start, length);
            pos = start + length;
            continue;
        }
        if (pos == text.size()) throw std::runtime_error("Truncated JSON escape");
        switch (text[pos++]) {
            case '"': result += '"'; break;
            case '\\': result += '\\'; break;
            case '/': result += '/'; break;
            case 'b': result += '\b'; break;
            case 'f': result += '\f'; break;
            case 'n': result += '\n'; break;
            case 'r': result += '\r'; break;
            case 't': result += '\t'; break;
            case 'u': {
                std::uint32_t cp = 0;
                for (int part = 0; part < 2; ++part) {
                    if (text.size() - pos < 4) {
                        throw std::runtime_error("Truncated Unicode escape");
                    }
                    std::uint32_t value = 0;
                    for (int i = 0; i < 4; ++i) {
                        const char c = text[pos++];
                        value <<= 4;
                        if (c >= '0' && c <= '9') value += c - '0';
                        else if (c >= 'a' && c <= 'f') value += c - 'a' + 10;
                        else if (c >= 'A' && c <= 'F') value += c - 'A' + 10;
                        else throw std::runtime_error("Invalid Unicode escape");
                    }
                    if (part == 0) {
                        cp = value;
                        if (cp >= 0xD800 && cp <= 0xDBFF) {
                            if (text.compare(pos, 2, "\\u") != 0) {
                                throw std::runtime_error("Missing low Unicode surrogate");
                            }
                            pos += 2;
                            continue;
                        }
                        if (cp >= 0xDC00 && cp <= 0xDFFF) {
                            throw std::runtime_error("Unpaired low Unicode surrogate");
                        }
                    } else {
                        if (value < 0xDC00 || value > 0xDFFF) {
                            throw std::runtime_error("Invalid low Unicode surrogate");
                        }
                        cp = 0x10000 + ((cp - 0xD800) << 10) + value - 0xDC00;
                    }
                    break;
                }
                if (cp <= 0x7F) {
                    result += static_cast<char>(cp);
                } else if (cp <= 0x7FF) {
                    result += static_cast<char>(0xC0 | (cp >> 6));
                    result += static_cast<char>(0x80 | (cp & 0x3F));
                } else if (cp <= 0xFFFF) {
                    result += static_cast<char>(0xE0 | (cp >> 12));
                    result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                    result += static_cast<char>(0x80 | (cp & 0x3F));
                } else {
                    result += static_cast<char>(0xF0 | (cp >> 18));
                    result += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
                    result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                    result += static_cast<char>(0x80 | (cp & 0x3F));
                }
                break;
            }
            default: throw std::runtime_error("Invalid JSON string escape");
        }
    }
    throw std::runtime_error("Unterminated JSON string");
}

static int ParseInt(const std::string& text, std::size_t& pos) {
    SkipWhitespace(text, pos);
    const auto start = pos;
    if (pos < text.size() && text[pos] == '-') ++pos;
    if (pos == text.size() || text[pos] < '0' || text[pos] > '9') {
        throw std::runtime_error("Expected an integer token ID");
    }
    if (text[pos] == '0') ++pos;
    else while (pos < text.size() && text[pos] >= '0' && text[pos] <= '9') ++pos;
    int id = -1;
    const auto parsed = std::from_chars(text.data() + start, text.data() + pos, id);
    if (parsed.ec != std::errc{} || id < 0 || id == std::numeric_limits<int>::max()) {
        throw std::runtime_error("Token ID must be in [0, INT_MAX - 1]");
    }
    return id;
}

static bool ParseBool(const std::string& text, std::size_t& pos) {
    SkipWhitespace(text, pos);
    if (text.compare(pos, 4, "true") == 0) { pos += 4; return true; }
    if (text.compare(pos, 5, "false") == 0) { pos += 5; return false; }
    throw std::runtime_error("Expected a JSON boolean");
}

JsonVocabTokenizer::JsonVocabTokenizer(const std::string& path_name) {
    std::ifstream file(path_name, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open vocabulary: " + path_name);
    std::ostringstream buffer;
    buffer << file.rdbuf();
    if (file.bad()) throw std::runtime_error("Cannot read vocabulary: " + path_name);
    const auto text = buffer.str();
    std::size_t pos = 0;
    SkipWhitespace(text, pos);
    if (pos == text.size() || text[pos++] != '[') {
        throw std::runtime_error("JSON vocabulary must be an array");
    }
    std::unordered_set<int> seen_ids;
    std::unordered_set<std::string> seen_tokens;
    bool found_bos = false, found_eos = false, found_unk = false;
    SkipWhitespace(text, pos);
    while (pos < text.size() && text[pos] != ']') {
        if (text[pos++] != '{') throw std::runtime_error("Expected a vocabulary entry");
        int id = -1;
        std::string token;
        bool special = false, found_context = false;
        std::unordered_set<std::string> fields;
        while (true) {
            const auto key = ParseString(text, pos);
            if (!fields.insert(key).second) throw std::runtime_error("Duplicate field: " + key);
            SkipWhitespace(text, pos);
            if (pos == text.size() || text[pos++] != ':') {
                throw std::runtime_error("Missing vocabulary field colon");
            }
            if (key == "id") id = ParseInt(text, pos);
            else if (key == "context") { token = ParseString(text, pos); found_context = true; }
            else if (key == "special") special = ParseBool(text, pos);
            else throw std::runtime_error("Unknown vocabulary field: " + key);
            SkipWhitespace(text, pos);
            if (pos == text.size()) throw std::runtime_error("Truncated vocabulary entry");
            const char separator = text[pos++];
            if (separator == '}') break;
            if (separator != ',') throw std::runtime_error("Missing vocabulary field comma");
        }
        if (id < 0 || !found_context || token.empty()) {
            throw std::runtime_error("Vocabulary entry requires an ID and nonempty context");
        }
        if (!seen_ids.insert(id).second || !seen_tokens.insert(token).second) {
            throw std::runtime_error("Duplicate vocabulary ID or token");
        }
        if (token == "<BOS>") { BOS_id_ = id; found_bos = true; special = true; }
        else if (token == "<EOS>") { EOS_id_ = id; found_eos = true; special = true; }
        else if (token == "<UNK>") { UNK_id_ = id; found_unk = true; special = true; }
        entry_.push_back(VocabEntry{id, token, special});
        context_id_.emplace_back(token, id);
        vocab_size_ = std::max(vocab_size_, id + 1);
        SkipWhitespace(text, pos);
        if (pos == text.size()) throw std::runtime_error("Truncated vocabulary array");
        if (text[pos] == ']') break;
        if (text[pos++] != ',') throw std::runtime_error("Missing vocabulary entry comma");
        SkipWhitespace(text, pos);
        if (pos == text.size() || text[pos] == ']') {
            throw std::runtime_error("Missing vocabulary entry after comma");
        }
    }
    if (pos == text.size() || text[pos++] != ']') {
        throw std::runtime_error("Missing vocabulary closing bracket");
    }
    SkipWhitespace(text, pos);
    if (pos != text.size()) throw std::runtime_error("Unexpected trailing vocabulary data");
    if (!found_bos || !found_eos || !found_unk) {
        throw std::runtime_error("JSON vocabulary requires <BOS>, <EOS>, and <UNK>");
    }
    std::sort(context_id_.begin(), context_id_.end(), [](const auto& a, const auto& b) {
        if (a.first.size() != b.first.size()) return a.first.size() > b.first.size();
        return a.first < b.first;
    });
}

std::vector<int> JsonVocabTokenizer::Encode(const std::string& text) const {
    std::vector<int> result{BOS_id_};
    std::size_t pos = 0;
    while (pos < text.size()) {
        bool matched = false;
        for (const auto& token : context_id_) {
            if (text.compare(pos, token.first.size(), token.first) == 0) {
                result.push_back(token.second);
                pos += token.first.size();
                matched = true;
                break;
            }
        }
        if (!matched) {
            result.push_back(UNK_id_);
            const auto first = static_cast<unsigned char>(text[pos]);
            std::size_t length = 1;
            if (first >= 0xC2 && first <= 0xDF) length = 2;
            else if (first >= 0xE0 && first <= 0xEF) length = 3;
            else if (first >= 0xF0 && first <= 0xF4) length = 4;
            if (length > text.size() - pos) length = 1;
            for (std::size_t i = 1; i < length; ++i) {
                const auto byte = static_cast<unsigned char>(text[pos + i]);
                if (byte < 0x80 || byte > 0xBF) { length = 1; break; }
            }
            if (length > 1) {
                const auto second = static_cast<unsigned char>(text[pos + 1]);
                if ((first == 0xE0 && second < 0xA0) || (first == 0xED && second >= 0xA0) ||
                    (first == 0xF0 && second < 0x90) || (first == 0xF4 && second >= 0x90)) length = 1;
            }
            pos += length;
        }
    }
    result.push_back(EOS_id_);
    return result;
}

std::string JsonVocabTokenizer::DecodeToken(int token) const {
    const auto it = std::find_if(entry_.begin(), entry_.end(), [token](const auto& entry) {
        return entry.id == token;
    });
    if (it == entry_.end()) throw std::out_of_range("Unknown JSON vocabulary ID");
    return it->context;
}

std::string JsonVocabTokenizer::Decode(std::vector<int> tokens) const {
    std::string result;
    for (int token : tokens) result += DecodeToken(token);
    return result;
}

}  // namespace yan_lamma
