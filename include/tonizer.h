#pragma once
#include "include/tensor.h"

#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <cstdint>
#include <memory>
namespace yan_lamma {
class Tonizer {
public:
 virtual ~Tonizer() noexcept= default;
 virtual std::vector<int> Encode(const std::string& text) const  = 0;
 virtual std::string DecodeToken(int token) const  = 0;
 virtual std::string Decode(std::vector<int> tokens) const = 0;
 //词表大小
 virtual int GetVocabSize() const = 0; 
 //序列开始标记
 virtual int GetBOSId() const  = 0;
 //结束标记
 virtual int GetEOSId() const  = 0; 
 //未知标记
 virtual int GetUNKId() const = 0;
};
class AsciiTokenizer : public Tonizer {
public:
 static constexpr int kVocabSize = 131;
 std::vector<int> Encode(const std::string& text) const override;
 std::string DecodeToken(int token) const override;
 std::string Decode(std::vector<int> tokens) const override;
 int GetVocabSize() const override {
    return kVocabSize;
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
private:
int BOS_id_ = 128;
int EOS_id_ = 129;
int UNK_id_ = 130;
};
class JsonVocabTokenizer : public Tonizer {
public:
 JsonVocabTokenizer(const std::string& path_name);
 std::vector<int> Encode(const std::string& text) const override;
 std::string DecodeToken(int token) const override;
 std::string Decode(std::vector<int> tokens) const override;
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
private:
struct VocabEntry{
    int id;
    std::string context;
    bool special = false;
};
 // id-> VocabEntry
 std::vector<VocabEntry> entry_;
 // name, id;
 std::vector<std::pair<std::string, int>> context_id_;
 int vocab_size_ = 0;
 int BOS_id_ = 1;
 int EOS_id_ = 2;
 int UNK_id_ = 0;

};
class BpeTokenizer : public Tonizer {
public:
 BpeTokenizer() = default;
 ~BpeTokenizer() noexcept override = default;
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
 bool Load(const std::string& json_path, const std::string& merge_path, const std::string& special_path);
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
std::unique_ptr<Tonizer> GetTonizer(const std::string& path_name);
std::unique_ptr<Tonizer> GetTonizer(const std::string& json_path, const std::string& merge_path, const std::string& special_path);

}// namespace yan_lamma
