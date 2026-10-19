#pragma once
#include "include/chatsession.h"
#include "include/gguf.h"
#include <vector>
#include <string>

namespace yan_lamma {
class PromptBuild {
public:
 PromptBuild() = default;
 void SetChatTemplate(const std::string& t);
 std::string Build(const std::vector<ChatMessage> message);
private:
 std::string chat_template;
 std::string BuildPlain(const std::vector<ChatMessage>& message);
 std::string BuildQwen2(const std::vector<ChatMessage>& message); 
 
};
std::string GetChatTemplate(const std::string& gguf_path);



}//namespace yan_lamma