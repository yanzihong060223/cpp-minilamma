#pragma once
#include "include/tonizer.h"
#include "include/gguf.h"
#include "include/gguf_load.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
// ---------------------------------------------------------------------------
// GgufTokenizer: loads tokenizer data directly from GGUF metadata.
//
// Reads:
//   - tokenizer.ggml.tokens      (vocab pieces)
//   - tokenizer.ggml.merges      (BPE merge rules)
//   - tokenizer.ggml.token_type  (to identify special/control tokens)
//   - tokenizer.ggml.bos_token_id
//   - tokenizer.ggml.eos_token_id
//
// Implements GPT-2 style BPE with byte-level fallback.
// ---------------------------------------------------------------------------
namespace yan_lamma {
class GgufTokenizer : public Tonizer {
public:
 GgufTokenizer() = default;
 ~GgufTokenizer() noexcept override = default;
 std::vector<int>Encode(const std::string& text) const override;
 std::string DecodeToken(int token) const override;
 std::string Decode (std::vector<int> tokens) const override;
 int GetVocabSize() const override {
    return vocab_size_;
 }
 int GetBOSId() const override {
    return BOS_id_;
  }
 int GetEOSId() const override {
    return EOS_id_;
 }
 int GetUNKId() const override {
    return UNK_id_;
 }
 //加载初始化
 bool Load(const std::string& gguf_path);
 private:
 int BOS_id_ = -1;
 int EOS_id_ = -1;
 int UNK_id_ = -1;
 int vocab_size_ = 0;
// token字符串 -> tokenid
std::unordered_map<std::string, int> vocab_;
//token id -> token字符串
std::vector<std::string> id_to_token_;
std::map<std::pair<std::string, std::string>, int> merge_rank_;
std::vector<std::string> b2u_;
std::unordered_map<std::string, uint8_t> u2b_;
std::vector<std::pair<std::string, int>> special_tokens;
void BuildByteMap();
};
std::unique_ptr<GgufTokenizer> GetGgufTokenizer(const std::string& gguf_path);
};