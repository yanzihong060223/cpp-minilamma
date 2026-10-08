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

static std::string ParseString(const std::string& text, std::size_t& pos) {
    while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t' ||
           text[pos] == '\r' || text[pos] == '\n')) ++pos;
    if (pos == text.size() || text[pos++] != '"') {
        throw std::runtime_error("Expected a JSON string");
    }
    std::string result;
    while (pos < text.size()) {
        const auto byte = static_cast<unsigned char>(text[pos++]);
        if (byte == '"') return result;
        if (byte < 0x20) throw std::runtime_error("Unescaped JSON control character");
        if (byte != '\\') {
            result += static_cast<char>(byte);
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
    while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t' ||
           text[pos] == '\r' || text[pos] == '\n')) ++pos;
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

static std::vector<std::string> BuildBytesToUnicode() {
    std::vector<std::string> map(256);
    std::vector<int> bs;
    for (int c = '!'; c <= '~'; ++c) bs.push_back(c);
    for (int c = 0xA1; c <= 0xAC; ++c) bs.push_back(c);
    for (int c = 0xAE; c <= 0xFF; ++c) bs.push_back(c);

    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }
    for (std::size_t i = 0; i < bs.size(); ++i) {
        map[bs[i]] = CodepointToUtf8(static_cast<char32_t>(cs[i]));
    }
    return map;
}

void BpeTokenizer::BuildByteMap() {
    b2u_ = BuildBytesToUnicode();
    u2b_.clear();
    for (int b = 0; b < 256; ++b) {
        u2b_[b2u_[b]] = static_cast<std::uint8_t>(b);
    }
}

bool BpeTokenizer::Load(const std::string& json_path, const std::string& merge_path,
                        const std::string& special_path) {
    BpeTokenizer pending;
    pending.vocab_size_ = 0;
    pending.BOS_id_ = pending.EOS_id_ = pending.UNK_id_ = -1;
    pending.BuildByteMap();
    std::unordered_set<int> seen_ids, special_ids;

    // vocab.json: {"token": ID}
    {
        std::ifstream file(json_path, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open BPE vocabulary: " + json_path);
        std::stringstream buffer;
        buffer << file.rdbuf();
        if (file.bad()) throw std::runtime_error("Cannot read BPE vocabulary: " + json_path);
        const auto content = buffer.str();
        std::size_t pos = 0;
        while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t' ||
               content[pos] == '\r' || content[pos] == '\n')) ++pos;
        if (pos == content.size() || content[pos++] != '{') {
            throw std::runtime_error("BPE vocabulary must be an object");
        }
        while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t' ||
               content[pos] == '\r' || content[pos] == '\n')) ++pos;
        while (pos < content.size() && content[pos] != '}') {
            const auto token = ParseString(content, pos);
            while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t' ||
                   content[pos] == '\r' || content[pos] == '\n')) ++pos;
            if (pos == content.size() || content[pos++] != ':') {
                throw std::runtime_error("Missing BPE token colon");
            }
            const int id = ParseInt(content, pos);
            if (token.empty() || !pending.vocab_.emplace(token, id).second ||
                !seen_ids.insert(id).second) {
                throw std::runtime_error("Empty or duplicate BPE token or ID");
            }
            pending.vocab_size_ = std::max(pending.vocab_size_, id + 1);
            while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t' ||
                   content[pos] == '\r' || content[pos] == '\n')) ++pos;
            if (pos == content.size()) throw std::runtime_error("Truncated BPE vocabulary");
            if (content[pos] == '}') break;
            if (content[pos++] != ',') throw std::runtime_error("Missing BPE token comma");
            while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t' ||
                   content[pos] == '\r' || content[pos] == '\n')) ++pos;
            if (pos == content.size() || content[pos] == '}') {
                throw std::runtime_error("Missing BPE token after comma");
            }
        }
        if (pos == content.size() || content[pos++] != '}' || pending.vocab_.empty()) {
            throw std::runtime_error("Incomplete or empty BPE vocabulary");
        }
        while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t' ||
               content[pos] == '\r' || content[pos] == '\n')) ++pos;
        if (pos != content.size()) throw std::runtime_error("Unexpected trailing vocabulary data");

        pending.id_to_token_.resize(static_cast<std::size_t>(pending.vocab_size_));
        for (const auto& item : pending.vocab_) {
            pending.id_to_token_[static_cast<std::size_t>(item.second)] = item.first;
            if (item.first.size() >= 4 && item.first.compare(0, 2, "<|") == 0 &&
                item.first.compare(item.first.size() - 2, 2, "|>") == 0) {
                special_ids.insert(item.second);
                pending.special_tokens.push_back(item);
            }
        }
    }

    // merges.txt: token1 token2; the line order is the rank.
    {
        std::ifstream file(merge_path);
        if (!file) throw std::runtime_error("Cannot open BPE merge file: " + merge_path);
        std::string line;
        int rank = 0;
        while (std::getline(file, line)) {
            std::stringstream stream(line);
            std::string first, second, trailing;
            if (!(stream >> first) || first == "#version:") continue;
            if (!(stream >> second) || (stream >> trailing)) {
                throw std::runtime_error("Invalid BPE merge rule");
            }
            for (const auto& token : {first, second, first + second}) {
                if (pending.vocab_.find(token) == pending.vocab_.end()) {
                    throw std::runtime_error("BPE merge references a missing token");
                }
            }
            if (rank == std::numeric_limits<int>::max() ||
                !pending.merge_rank_.emplace(std::make_pair(first, second), rank).second) {
                throw std::runtime_error("Duplicate or too many BPE merge rules");
            }
            ++rank;
        }
        if (file.bad()) throw std::runtime_error("Cannot read BPE merge file: " + merge_path);
    }

    // special_tokens.json: bos_id, eos_id, unk_id.
    {
        std::ifstream file(special_path, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open special token file: " + special_path);
        std::stringstream buffer;
        buffer << file.rdbuf();
        if (file.bad()) throw std::runtime_error("Cannot read special token file: " + special_path);
        const auto content = buffer.str();
        std::size_t pos = 0;
        while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t' ||
               content[pos] == '\r' || content[pos] == '\n')) ++pos;
        if (pos == content.size() || content[pos++] != '{') {
            throw std::runtime_error("Special token configuration must be an object");
        }
        std::unordered_set<std::string> fields;
        while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t' ||
               content[pos] == '\r' || content[pos] == '\n')) ++pos;
        while (pos < content.size() && content[pos] != '}') {
            const auto key = ParseString(content, pos);
            if (!fields.insert(key).second) throw std::runtime_error("Duplicate special ID field");
            while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t' ||
                   content[pos] == '\r' || content[pos] == '\n')) ++pos;
            if (pos == content.size() || content[pos++] != ':') {
                throw std::runtime_error("Missing special ID colon");
            }
            const int id = ParseInt(content, pos);
            if (key == "bos_id") pending.BOS_id_ = id;
            else if (key == "eos_id") pending.EOS_id_ = id;
            else if (key == "unk_id") pending.UNK_id_ = id;
            else throw std::runtime_error("Unknown special ID field: " + key);
            while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t' ||
                   content[pos] == '\r' || content[pos] == '\n')) ++pos;
            if (pos == content.size()) throw std::runtime_error("Truncated special configuration");
            if (content[pos] == '}') break;
            if (content[pos++] != ',') throw std::runtime_error("Missing special ID comma");
            while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t' ||
                   content[pos] == '\r' || content[pos] == '\n')) ++pos;
            if (pos == content.size() || content[pos] == '}') {
                throw std::runtime_error("Missing special ID after comma");
            }
        }
        if (pos == content.size() || content[pos++] != '}') {
            throw std::runtime_error("Missing special configuration closing brace");
        }
        while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t' ||
               content[pos] == '\r' || content[pos] == '\n')) ++pos;
        if (pos != content.size()) throw std::runtime_error("Unexpected trailing special ID data");
    }

    if (pending.BOS_id_ == pending.EOS_id_ || pending.BOS_id_ == pending.UNK_id_ ||
        pending.EOS_id_ == pending.UNK_id_) throw std::runtime_error("Special IDs must be distinct");
    for (int id : {pending.BOS_id_, pending.EOS_id_, pending.UNK_id_}) {
        if (seen_ids.count(id) == 0) throw std::runtime_error("Special ID is missing from vocabulary");
        if (special_ids.insert(id).second) {
            pending.special_tokens.emplace_back(pending.id_to_token_[id], id);
        }
    }
    std::sort(pending.special_tokens.begin(), pending.special_tokens.end(),
              [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
    for (const auto& item : pending.vocab_) {
        if (special_ids.count(item.second) != 0) continue;
        std::size_t pos = 0;
        while (pos < item.first.size()) {
            const auto symbol = NextUtf8Codepoint(item.first, pos);
            if (symbol.empty() || pending.u2b_.find(symbol) == pending.u2b_.end()) {
                throw std::runtime_error("BPE token contains an invalid byte symbol");
            }
        }
    }
    for (const auto& item : pending.merge_rank_) {
        if (special_ids.count(pending.vocab_.at(item.first.first)) != 0 ||
            special_ids.count(pending.vocab_.at(item.first.second)) != 0 ||
            special_ids.count(pending.vocab_.at(item.first.first + item.first.second)) != 0) {
            throw std::runtime_error("BPE merge must reference ordinary vocabulary tokens");
        }
    }
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
