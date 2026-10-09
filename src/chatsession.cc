#include "include/chatsession.h"
#include <utility>
#include <iostream>
namespace yan_lamma {
void ChatSession::Clear() {
    messages_.clear();
    token_history_.clear();
    ra_cache_.Clear();
    total_time_ = 0.0f;
    prompt_token_ = 0;
    generate_token_ = 0;  
}
void ChatSession::AddMessage(const ChatMessage& message) {
    messages_.push_back(message);
 }
void ChatSession::SetTokenHistory(const std::vector<int>& token_history) {
    token_history_ = token_history;
}
void ChatSession::AddToken(int token_id) {
    token_history_.push_back(token_id);
 }
void ChatSession::RecordRaCahce(std::vector<int>& tokens) {
    if (tokens.empty()) {
        std::cerr << "Token Arry Is Empty" << "\n";
        return;
    }
    ra_cache_.Insert(tokens);

}
size_t ChatSession::LongestPrefix(std::vector<int>& tokens) {
    if (tokens.empty()) {
        std::cerr << "Token Is Empty Can Not Get Longest Prefix" << "\n";
        return 0;
    }
    return ra_cache_.LongestPrefis(tokens);

}
void ChatSession::RecordTurn(float time, int prompt_token, int generate_token) {
    total_time_ += time;
    prompt_token_ += prompt_token;
    generate_token_ += generate_token;
}
}//namespace yan_lamma