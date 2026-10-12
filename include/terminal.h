#pragma once
#include "include/model.h"
#include "include/chatsession.h"
#include "include/tonizer.h"

#include <string>
namespace yan_lamma {
class Terminal {
public:
    Terminal() = default;
    // mini -llama
    void PrintUserPrompt();
    std::string ReadLine();
    //assistant >
    void PrintAssistantPrefix();
    // 不自动换行
    void PrintTokenText(std::string text);
    void flush();
    void NewLine();
    void PrintHelp();
    void PrintStats(const ChatSession& session);
    void PrintParams(const SamplerParmas& params);
    void PrintMessage(const std::string& message);


};
}