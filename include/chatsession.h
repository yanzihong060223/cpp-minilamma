#pragma once
#include "include/tensor.h"
#include "include/context.h"
#include "include/forward.h"
#include "include/kvcache.h"
#include "include/radix_tree.h"
#include "include/sampler.h"

#include <string>
#include <vector>

namespace yan_lamma {
struct ChatMessage {
std::string role; // system user,assistant
std::string context;
};
//会话状态
class ChatSession {
public:
 ChatSession() = default;
 std::vector<ChatMessage> messages_;
 std::vector<int> token_history_;
 // /stat
 float total_time_ = 0.0f;
 // prompt_token 数量
 int prompt_token_ = 0;
 int generate_token_ = 0;
 // 保存前缀token
 RadixTree ra_cache_;
 //采样数据
 SamplerParmas parma_;
 void Clear();
 void AddMessage(const ChatMessage& message);
 void SetTokenHistory(const std::vector<int>& token_history);
 void AddToken(int token_id);
 // 更新前缀token_id
 void RecordRaCahce(std::vector<int>& tokens);
 size_t LongestPrefix(std::vector<int>& tokens);
 // 更新统计信息
 void RecordTurn(float time, int prompt_token, int generate_token);
    
};





}// namespace yan_lamma