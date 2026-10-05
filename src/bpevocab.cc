#include "include/tonizer.h"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <unordered_set>
#include <utility>

namespace yan_lamma {

static std::string CodepointToUtf8(char32_t cp) {
    if (cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) {
        throw std::runtime_error("Invalid Unicode codepoint");
    }
    std::string result;
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
    return result;
}

static std::string NextUtf8Codepoint(const std::string& text, std::size_t& pos) {
    if (pos >= text.size()) return "";
    const auto first = static_cast<unsigned char>(text[pos]);
    std::size_t length = 0;
    if (first < 0x80) length = 1;
    else if (first >= 0xC2 && first <= 0xDF) length = 2;
    else if (first >= 0xE0 && first <= 0xEF) length = 3;
    else if (first >= 0xF0 && first <= 0xF4) length = 4;
    if (length == 0 || length > text.size() - pos) { ++pos; return ""; }
    for (std::size_t i = 1; i < length; ++i) {
        const auto byte = static_cast<unsigned char>(text[pos + i]);
        if (byte < 0x80 || byte > 0xBF) { ++pos; return ""; }
    }
    if (length > 1) {
        const auto second = static_cast<unsigned char>(text[pos + 1]);
        if ((first == 0xE0 && second < 0xA0) || (first == 0xED && second >= 0xA0) ||
            (first == 0xF0 && second < 0x90) || (first == 0xF4 && second >= 0x90)) {
            ++pos;
            return "";
        }
    }
    const auto result = text.substr(pos, length);
    pos += length;
    return result;
}

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
                result += CodepointToUtf8(static_cast<char32_t>(cp));
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

void BpeTokenizer::BuildByteMap() {
    b2u_.resize(256);
    u2b_.clear();
    char32_t next = 256;
    for (int byte = 0; byte < 256; ++byte) {
        const bool direct = (byte >= 33 && byte <= 126) ||
                            (byte >= 0xA1 && byte <= 0xAC) || byte >= 0xAE;
        b2u_[byte] = CodepointToUtf8(direct ? static_cast<char32_t>(byte) : next++);
        u2b_[b2u_[byte]] = static_cast<std::uint8_t>(byte);
    }
}

bool BpeTokenizer::Load(const std::string& json_path, const std::string& merge_path,
                        const std::string& special_path) {
    BpeTokenizer pending;
    pending.vocab_size_ = 0;
    pending.BOS_id_ = pending.EOS_id_ = pending.UNK_id_ = -1;
    pending.BuildByteMap();
    std::ifstream vocab_file(json_path, std::ios::binary);
    if (!vocab_file) throw std::runtime_error("Cannot open BPE vocabulary: " + json_path);
    std::ostringstream vocab_buffer;
    vocab_buffer << vocab_file.rdbuf();
    if (vocab_file.bad()) throw std::runtime_error("Cannot read BPE vocabulary: " + json_path);
    const auto text = vocab_buffer.str();
    std::size_t pos = 0;
    SkipWhitespace(text, pos);
    if (pos == text.size() || (text[pos] != '{' && text[pos] != '[')) {
        throw std::runtime_error("BPE vocabulary must be an object or entry array");
    }
    const bool array = text[pos++] == '[';
    const char closing = array ? ']' : '}';
    std::unordered_set<int> seen_ids, special_ids;
    SkipWhitespace(text, pos);
    while (pos < text.size() && text[pos] != closing) {
        int id = -1;
        std::string token;
        bool special = false;
        if (array) {
            if (text[pos++] != '{') throw std::runtime_error("Expected a BPE vocabulary entry");
            std::unordered_set<std::string> fields;
            bool found_context = false;
            while (true) {
                const auto key = ParseString(text, pos);
                if (!fields.insert(key).second) throw std::runtime_error("Duplicate BPE field: " + key);
                SkipWhitespace(text, pos);
                if (pos == text.size() || text[pos++] != ':') {
                    throw std::runtime_error("Missing BPE field colon");
                }
                if (key == "id") id = ParseInt(text, pos);
                else if (key == "context") { token = ParseString(text, pos); found_context = true; }
                else if (key == "special") {
                    SkipWhitespace(text, pos);
                    if (text.compare(pos, 4, "true") == 0) { special = true; pos += 4; }
                    else if (text.compare(pos, 5, "false") == 0) pos += 5;
                    else throw std::runtime_error("BPE special field must be boolean");
                } else throw std::runtime_error("Unknown BPE vocabulary field: " + key);
                SkipWhitespace(text, pos);
                if (pos == text.size()) throw std::runtime_error("Truncated BPE vocabulary entry");
                const char separator = text[pos++];
                if (separator == '}') break;
                if (separator != ',') throw std::runtime_error("Missing BPE field comma");
            }
            if (id < 0 || !found_context) throw std::runtime_error("Missing BPE ID or context");
        } else {
            token = ParseString(text, pos);
            SkipWhitespace(text, pos);
            if (pos == text.size() || text[pos++] != ':') {
                throw std::runtime_error("Missing BPE token colon");
            }
            id = ParseInt(text, pos);
        }
        if (token.empty()) throw std::runtime_error("Empty BPE token");
        if (!pending.vocab_.emplace(token, id).second || !seen_ids.insert(id).second) {
            throw std::runtime_error("Duplicate BPE vocabulary token or ID");
        }
        pending.vocab_size_ = std::max(pending.vocab_size_, id + 1);
        if (special || (token.size() >= 4 && token.compare(0, 2, "<|") == 0 &&
                        token.compare(token.size() - 2, 2, "|>") == 0)) special_ids.insert(id);
        SkipWhitespace(text, pos);
        if (pos == text.size()) throw std::runtime_error("Truncated BPE vocabulary");
        if (text[pos] == closing) break;
        if (text[pos++] != ',') throw std::runtime_error("Missing BPE token comma");
        SkipWhitespace(text, pos);
        if (pos == text.size() || text[pos] == closing) {
            throw std::runtime_error("Missing BPE token after comma");
        }
    }
    if (pos == text.size() || text[pos++] != closing || pending.vocab_.empty()) {
        throw std::runtime_error("Incomplete or empty BPE vocabulary");
    }
    SkipWhitespace(text, pos);
    if (pos != text.size()) throw std::runtime_error("Unexpected trailing BPE vocabulary data");
    pending.id_to_token_.resize(static_cast<std::size_t>(pending.vocab_size_));
    for (const auto& item : pending.vocab_) {
        pending.id_to_token_[static_cast<std::size_t>(item.second)] = item.first;
    }

    std::ifstream special_file(special_path, std::ios::binary);
    if (!special_file) throw std::runtime_error("Cannot open special token file: " + special_path);
    std::ostringstream special_buffer;
    special_buffer << special_file.rdbuf();
    if (special_file.bad()) throw std::runtime_error("Cannot read special token file: " + special_path);
    const auto config = special_buffer.str();
    pos = 0;
    SkipWhitespace(config, pos);
    if (pos == config.size() || config[pos++] != '{') {
        throw std::runtime_error("Special token configuration must be an object");
    }
    std::unordered_set<std::string> fields;
    SkipWhitespace(config, pos);
    while (pos < config.size() && config[pos] != '}') {
        const auto key = ParseString(config, pos);
        if (!fields.insert(key).second) throw std::runtime_error("Duplicate special token field");
        SkipWhitespace(config, pos);
        if (pos == config.size() || config[pos++] != ':') {
            throw std::runtime_error("Missing special token field colon");
        }
        if (key == "bos_id") pending.BOS_id_ = ParseInt(config, pos);
        else if (key == "eos_id") pending.EOS_id_ = ParseInt(config, pos);
        else if (key == "unk_id") pending.UNK_id_ = ParseInt(config, pos);
        else if (key == "special_tokens") {
            SkipWhitespace(config, pos);
            if (pos == config.size() || config[pos++] != '[') {
                throw std::runtime_error("special_tokens must be an array of strings");
            }
            SkipWhitespace(config, pos);
            while (pos < config.size() && config[pos] != ']') {
                const auto token = ParseString(config, pos);
                const auto it = pending.vocab_.find(token);
                if (it == pending.vocab_.end()) throw std::runtime_error("Unknown special token: " + token);
                special_ids.insert(it->second);
                SkipWhitespace(config, pos);
                if (pos == config.size()) throw std::runtime_error("Truncated special_tokens array");
                if (config[pos] == ']') break;
                if (config[pos++] != ',') throw std::runtime_error("Missing special token comma");
                SkipWhitespace(config, pos);
                if (pos == config.size() || config[pos] == ']') {
                    throw std::runtime_error("Missing special token after comma");
                }
            }
            if (pos == config.size() || config[pos++] != ']') {
                throw std::runtime_error("Missing special_tokens closing bracket");
            }
        } else throw std::runtime_error("Unknown special token field: " + key);
        SkipWhitespace(config, pos);
        if (pos == config.size()) throw std::runtime_error("Truncated special token configuration");
        if (config[pos] == '}') break;
        if (config[pos++] != ',') throw std::runtime_error("Missing special token field comma");
        SkipWhitespace(config, pos);
        if (pos == config.size() || config[pos] == '}') {
            throw std::runtime_error("Missing special token field after comma");
        }
    }
    if (pos == config.size() || config[pos++] != '}') {
        throw std::runtime_error("Missing special token configuration closing brace");
    }
    SkipWhitespace(config, pos);
    if (pos != config.size()) throw std::runtime_error("Unexpected trailing special token data");
    if (pending.BOS_id_ == pending.EOS_id_ || pending.BOS_id_ == pending.UNK_id_ ||
        pending.EOS_id_ == pending.UNK_id_) throw std::runtime_error("Special IDs must be distinct");
    for (int id : {pending.BOS_id_, pending.EOS_id_, pending.UNK_id_}) {
        if (seen_ids.count(id) == 0) throw std::runtime_error("Missing special token ID in BPE vocabulary");
        special_ids.insert(id);
    }
    for (const auto& item : pending.vocab_) {
        if (special_ids.count(item.second) != 0) {
            pending.special_tokens.push_back(item);
        } else {
            std::size_t offset = 0;
            while (offset < item.first.size()) {
                const auto symbol = NextUtf8Codepoint(item.first, offset);
                if (symbol.empty() || pending.u2b_.find(symbol) == pending.u2b_.end()) {
                    throw std::runtime_error("BPE token contains an invalid byte symbol");
                }
            }
        }
    }
    std::sort(pending.special_tokens.begin(), pending.special_tokens.end(),
              [](const auto& a, const auto& b) {
                  if (a.first.size() != b.first.size()) return a.first.size() > b.first.size();
                  return a.first < b.first;
              });

    std::ifstream merge_file(merge_path);
    if (!merge_file) throw std::runtime_error("Cannot open BPE merge file: " + merge_path);
    std::string line;
    while (std::getline(merge_file, line)) {
        std::istringstream fields_in_line(line);
        std::string first, second, trailing;
        if (!(fields_in_line >> first) || first == "#version:") continue;
        if (!(fields_in_line >> second) || (fields_in_line >> trailing)) {
            throw std::runtime_error("Invalid BPE merge rule");
        }
        for (const auto& token : {first, second, first + second}) {
            const auto it = pending.vocab_.find(token);
            if (it == pending.vocab_.end() || special_ids.count(it->second) != 0) {
                throw std::runtime_error("BPE merge must reference ordinary vocabulary tokens");
            }
        }
        if (pending.merge_rank_.size() >= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("Too many BPE merge rules");
        }
        const int rank = static_cast<int>(pending.merge_rank_.size());
        if (!pending.merge_rank_.emplace(std::make_pair(first, second), rank).second) {
            throw std::runtime_error("Duplicate BPE merge rule");
        }
    }
    if (merge_file.bad()) throw std::runtime_error("Cannot read BPE merge file: " + merge_path);
    *this = std::move(pending);
    return true;
}

std::vector<int> BpeTokenizer::Encode(const std::string& text) const {
    std::vector<int> result;
    result.push_back(BOS_id_);
    std::size_t pos = 0;
    while (pos < text.size()) {
        bool matched = false;
        for (const auto& token : special_tokens) {
            if (text.compare(pos, token.first.size(), token.first) == 0) {
                result.push_back(token.second);
                pos += token.first.size();
                matched = true;
                break;
            }
        }
        if (matched) continue;

        std::size_t end = text.size();
        for (const auto& token : special_tokens) {
            const auto next = text.find(token.first, pos);
            if (next != std::string::npos) end = std::min(end, next);
        }
        std::vector<std::string> pieces;
        for (; pos < end; ++pos) {
            pieces.push_back(b2u_[static_cast<unsigned char>(text[pos])]);
        }
        while (pieces.size() > 1) {
            int best_rank = std::numeric_limits<int>::max();
            std::size_t best_index = pieces.size();
            for (std::size_t i = 0; i + 1 < pieces.size(); ++i) {
                const auto it = merge_rank_.find({pieces[i], pieces[i + 1]});
                if (it != merge_rank_.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_index = i;
                }
            }
            if (best_index == pieces.size()) break;
            pieces[best_index] += pieces[best_index + 1];
            pieces.erase(pieces.begin() + static_cast<std::ptrdiff_t>(best_index + 1));
        }
        for (const auto& piece : pieces) {
            const auto it = vocab_.find(piece);
            result.push_back(it == vocab_.end() ? UNK_id_ : it->second);
        }
    }
    result.push_back(EOS_id_);
    return result;
}

std::string BpeTokenizer::DecodeToken(int token) const {
    if (token < 0 || static_cast<std::size_t>(token) >= id_to_token_.size() ||
        id_to_token_[static_cast<std::size_t>(token)].empty()) {
        throw std::out_of_range("Unknown BPE vocabulary ID");
    }
    const auto& value = id_to_token_[static_cast<std::size_t>(token)];
    const bool special = std::any_of(special_tokens.begin(), special_tokens.end(),
                                     [token](const auto& item) { return item.second == token; });
    if (special) return value;
    std::string result;
    std::size_t pos = 0;
    while (pos < value.size()) {
        const auto symbol = NextUtf8Codepoint(value, pos);
        const auto it = u2b_.find(symbol);
        if (symbol.empty() || it == u2b_.end()) {
            throw std::runtime_error("BPE token contains an invalid byte symbol");
        }
        result += static_cast<char>(it->second);
    }
    return result;
}

std::string BpeTokenizer::Decode(std::vector<int> tokens) const {
    std::string result;
    for (int token : tokens) result += DecodeToken(token);
    return result;
}

}  // namespace yan_lamma
