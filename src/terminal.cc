#include "include/terminal.h"

#include <string>
#include <iostream>

namespace yan_lamma {
void Terminal::PrintUserPrompt() {
    std::cout << "mini - llama > ";
    std::cout.flush();

}

std::string Terminal::ReadLine() {
    std::string res;
    if (std::getline(std::cin, res)) {
        return res;
    }
    return "";
}
void Terminal::PrintAssistantPrefix() {
    std::cout << "assistant  > ";
    std::cout.flush();
 }
void Terminal::PrintTokenText(std::string text) {
    std::cout << text;
}
void Terminal::flush() {
    std::cout.flush();
}
void Terminal::NewLine() {
    std::cout << "\n";
}
void Terminal::PrintHelp() {
    std::cout << "Command : \n" 
              << " /help         Show this help message\n"
              << " /clear        Clear chat history and context\n"
              << " /stats        Show session stats\n"
              << " /parmas       Show sampler parma\n"
              << " /exit         Exit chat\n";
}
void Terminal::PrintStats(const ChatSession& session) {
    std::cout << " Session stats : \n"
              << " messages :        " << session.messages_.size() << "\n"
              << " context tokens    " << session.token_history_.size() << "\n"
              << " total prompt tokens  " << session.prompt_token_ << "\n"
              << " generate tokens"     << session.generate_token_ << "\n";
}
void Terminal::PrintParams(const SamplerParmas& params) {
    std::cout << "Sampler parmas : \n"
              << "temperature:     " << params.temperature << "\n"
              << "Top k :          " << params.top_k << "\n"
              << "Seed :           " << params.seed << "\n";

}
 void Terminal::PrintMessage(const std::string& message) {
    std::cout << message << "\n";
 }
}//namespace yan_lamma