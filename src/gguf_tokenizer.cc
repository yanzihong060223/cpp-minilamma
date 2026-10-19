#include "include/gguf_tokenizer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <fstream>
#include <unordered_set>
#include <utility>

namespace yan_lamma {
// 将 Unicode 标量值编码为 UTF-8；拒绝代理码点和超出 Unicode 范围的值。
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

// 返回下一个码点的 UTF-8 字节串；遇到非法序列时前进一个字节并返回空串。
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

// GPT-2 byte-level BPE 映射：索引是原始字节，值是映射字符的 UTF-8 字节串。
// 保留 0x21..0x7E、0xA1..0xAC、0xAE..0xFF，其余字节按升序映射到 U+0100 起。
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
// 建立 int -> unicode

void GgufTokenizer:: BuildByteMap() {
    b2u_ =  BuildBytesToUnicode();
    u2b_.clear();
    for (int i = 0; i < 256; i++) {
        u2b_[b2u_[i]]  = i;
    } 
}
bool GgufTokenizer::Load(const std::string& gguf_path) {
    std::ifstream file(gguf_path, std::ios::binary);
    unsigned char header[8]{};
    if (!file.read(reinterpret_cast<char*>(header), sizeof(header)) ||
        std::string(reinterpret_cast<char*>(header), 4) != "GGUF" ||
        (header[4] != 2 && header[4] != 3) || header[5] || header[6] || header[7]) {
        throw std::runtime_error("Expected a little-endian GGUF v2/v3 file");
    }
    auto has_metadata = [&](const char* key) {
        GgufValueType type;
        return GetGuffMetadaType(gguf_path, key, type);
    };
    std::string model;
    if (has_metadata("tokenizer.ggml.model") &&
        (!GetGuffMetadaString(gguf_path, "tokenizer.ggml.model", model) || model != "gpt2")) {
        throw std::runtime_error("GGUF tokenizer requires GPT-2 byte BPE metadata");
    }
    GgufTokenizer pending;
    pending.BuildByteMap();
    auto& tokens = pending.id_to_token_;
    if (!GetGuffMetadaStringArry(gguf_path, "tokenizer.ggml.tokens", tokens) ||
        tokens.empty() || tokens.size() >= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Missing, empty or oversized GGUF vocabulary");
    }
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i].empty() || !pending.vocab_.emplace(tokens[i], static_cast<int>(i)).second) {
            throw std::runtime_error("Empty or duplicate GGUF token");
        }
    }
    pending.vocab_size_ = static_cast<int>(tokens.size());
    std::unordered_set<int> special_ids;
    std::vector<int> types;
    if (has_metadata("tokenizer.ggml.token_type")) {
        if (!GetGuffMetadaIntArry(gguf_path, "tokenizer.ggml.token_type", types) ||
            types.size() != tokens.size()) throw std::runtime_error("Invalid GGUF token_type array");
        for (std::size_t i = 0; i < types.size(); ++i) {
            if (types[i] == 3) special_ids.insert(static_cast<int>(i));
        }
    }
    auto read_id = [&](const char* key, int& target) {
        if (!has_metadata(key)) return;
        int64_t value = -1;
        if (!GetGuffMetadaInt(gguf_path, key, value) || value < 0 || value >= pending.vocab_size_) {
            throw std::runtime_error("Invalid GGUF special token ID");
        }
        target = static_cast<int>(value);
        special_ids.insert(target);
    };
    read_id("tokenizer.ggml.bos_token_id", pending.BOS_id_);
    read_id("tokenizer.ggml.eos_token_id", pending.EOS_id_);
    read_id("tokenizer.ggml.unknown_token_id", pending.UNK_id_);
    int pad_id = -1;
    read_id("tokenizer.ggml.padding_token_id", pad_id);
    for (int id : special_ids) pending.special_tokens.emplace_back(tokens[id], id);
    std::sort(pending.special_tokens.begin(), pending.special_tokens.end(),
              [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        if (special_ids.count(static_cast<int>(i))) continue;
        std::size_t pos = 0;
        while (pos < tokens[i].size()) {
            const auto symbol = NextUtf8Codepoint(tokens[i], pos);
            if (symbol.empty() || !pending.u2b_.count(symbol)) {
                throw std::runtime_error("GGUF token contains an invalid byte symbol");
            }
        }
    }
    std::vector<std::string> merges;
    if (!GetGuffMetadaStringArry(gguf_path, "tokenizer.ggml.merges", merges)) {
        throw std::runtime_error("Missing GGUF merges array");
    }
    for (std::size_t rank = 0; rank < merges.size(); ++rank) {
        const auto& merge = merges[rank];
        const auto space = merge.find(' ');
        if (space == std::string::npos || space == 0 || space + 1 == merge.size()) {
            throw std::runtime_error("Invalid GGUF merge rule");
        }
        const auto first = merge.substr(0, space), second = merge.substr(space + 1);
        for (const auto& token : {first, second, first + second}) {
            const auto it = pending.vocab_.find(token);
            if (it == pending.vocab_.end() || special_ids.count(it->second)) {
                throw std::runtime_error("GGUF merge references a missing or special token");
            }
        }
        if (rank >= static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            !pending.merge_rank_.emplace(std::make_pair(first, second), static_cast<int>(rank)).second) {
            throw std::runtime_error("Duplicate or too many GGUF merge rules");
        }
    }
    *this = std::move(pending);
    return true;
}
std::vector<int> GgufTokenizer::Encode(const std::string& text) const {
    if (b2u_.size() != 256) throw std::runtime_error("GGUF tokenizer is not loaded");
    std::vector<int> result;
    if (BOS_id_ >= 0) result.push_back(BOS_id_);
    std::size_t pos = 0;
    while (pos < text.size()) {
        bool matched = false;
        for (const auto& token : special_tokens) {
            if (!token.first.empty() && text.compare(pos, token.first.size(), token.first) == 0) {
                result.push_back(token.second);
                pos += token.first.size();
                matched = true;
                break;
            }
        }
        if (matched) continue;

        // 普通文本只处理到下一个特殊 token，避免跨越特殊 token 合并。
        std::size_t end = text.size();
        for (const auto& token : special_tokens) {
            if (token.first.empty()) continue;
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
            if (it == vocab_.end() && UNK_id_ < 0) {
                throw std::runtime_error("Cannot encode GGUF piece without an UNK token");
            }
            result.push_back(it == vocab_.end() ? UNK_id_ : it->second);
        }
    }
    if (EOS_id_ >= 0) result.push_back(EOS_id_);
    return result;
}

std::string GgufTokenizer::DecodeToken(int token) const {
    if (token < 0 || static_cast<std::size_t>(token) >= id_to_token_.size() ||
        id_to_token_[static_cast<std::size_t>(token)].empty()) {
        throw std::out_of_range("Unknown GGUF vocabulary ID");
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
            throw std::runtime_error("GGUF token contains an invalid byte symbol");
        }
        result += static_cast<char>(it->second);
    }
    return result;
}

std::string GgufTokenizer::Decode(std::vector<int> tokens) const {
    std::string result;
    for (int token : tokens) result += DecodeToken(token);
    return result;
}
std::unique_ptr<GgufTokenizer> GetGgufTokenizer(const std::string& gguf_path) {
    std::unique_ptr<GgufTokenizer> ptr = std::make_unique<GgufTokenizer>();
    if (ptr->Load(gguf_path)) {
        return ptr;
    }
    return nullptr;

}
}//namespace yan_lamma



