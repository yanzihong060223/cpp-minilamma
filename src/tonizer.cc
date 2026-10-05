#include "include/tonizer.h"

#include <stdexcept>

namespace yan_lamma {

std::vector<int> AsciiTokenizer::Encode(const std::string& text) const {
    std::vector<int> result;
    result.push_back(GetBOSId());
    for (unsigned char byte : text) {
        result.push_back(byte < 128 ? static_cast<int>(byte) : GetUNKId());
    }
    result.push_back(GetEOSId());
    return result;
}

std::string AsciiTokenizer::DecodeToken(int token) const {
    if (token == GetBOSId()) return "<BOS>";
    if (token == GetEOSId()) return "<EOS>";
    if (token == GetUNKId()) return "<UNK>";
    if (token < 0 || token >= 128) {
        throw std::out_of_range("ASCII token ID out of range");
    }
    return std::string(1, static_cast<char>(token));
}

std::string AsciiTokenizer::Decode(std::vector<int> tokens) const {
    std::string result;
    for (int token : tokens) {
        result += DecodeToken(token);
    }
    return result;
}

std::unique_ptr<Tonizer> GetTonizer(const std::string& path_name) {
    if (path_name.empty()) return std::make_unique<AsciiTokenizer>();
    return std::make_unique<JsonVocabTokenizer>(path_name);
}

std::unique_ptr<Tonizer> GetTonizer(const std::string& json_path,
                                  const std::string& merge_path,
                                  const std::string& special_path) {
    if (json_path.empty() || merge_path.empty() || special_path.empty()) {
        throw std::invalid_argument("BPE tokenizer requires all three file paths");
    }
    auto tokenizer = std::make_unique<BpeTokenizer>();
    tokenizer->Load(json_path, merge_path, special_path);
    return tokenizer;
}

}  // namespace yan_lamma
