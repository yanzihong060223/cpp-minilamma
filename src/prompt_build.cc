#include "include/prompt_build.h"

#include <cctype>
#include <sstream>
#include <stack>
#include <string>
#include <vector>
#include <stdexcept>

namespace yan_lamma {
namespace {
// 模板执行时的数据：所有消息、当前消息，以及是否添加 assistant 开头。
struct EvalContext {
    const std::vector<ChatMessage>* messages = nullptr;
    const ChatMessage* current_message = nullptr;
    size_t message_idex = 0;
    bool add_generation_prompt = false;
};

std::string Trim(const std::string& s) {
    size_t a = 0;
    while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) {
        ++a;
    }
    size_t b = s.size();
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) {
        --b;
    }
    return s.substr(a, b - a);
}

// 去掉标签内的空白和 - 标记；相邻文本的空白由 CompileTemplate 处理。
std::string StripJiniaTrim(const std::string& s) {
    std::string result = Trim(s);
    if (!result.empty() && result.front() == '-') {
        result = Trim(result.substr(1));
    }
    if (!result.empty() && result.back() == '-') {
        result.pop_back();
        result = Trim(result);
    }
    return result;
}

// 只在引号和 [] 外分割，避免把 'a+b' 或 message['role'] 拆坏。
std::vector<std::string> SplitExpression(const std::string& s,
                                         const std::string& op) {
    std::vector<std::string> parts;
    size_t start = 0;
    size_t brackets = 0;
    char quote = 0;
    const bool word_op = op == "and" || op == "or";
    for (size_t i = 0; i < s.size(); ++i) {
        const char c = s[i];
        if (quote != 0) {
            if (c == '\\') {
                if (++i >= s.size()) {
                    throw std::invalid_argument("Unfinished string escape");
                }
            } else if (c == quote) {
                quote = 0;
            }
            continue;
        }
        if (c == '\'' || c == '"') {
            quote = c;
        } else if (c == '[') {
            ++brackets;
        } else if (c == ']') {
            if (brackets == 0) {
                throw std::invalid_argument("Unexpected ] in expression");
            }
            --brackets;
        } else if (brackets == 0 && s.compare(i, op.size(), op) == 0) {
            if (word_op &&
                (i == 0 || i + op.size() == s.size() ||
                 !std::isspace(static_cast<unsigned char>(s[i - 1])) ||
                 !std::isspace(static_cast<unsigned char>(s[i + op.size()])))) {
                continue;
            }
            parts.push_back(Trim(s.substr(start, i - start)));
            i += op.size() - 1;
            start = i + 1;
        }
    }
    if (quote != 0 || brackets != 0) {
        throw std::invalid_argument("Unclosed quote or [] in expression");
    }
    parts.push_back(Trim(s.substr(start)));
    return parts;
}

// 把模板里的 '\n' 转成真正的换行，并支持单双引号和常用转义。
std::string ReadStringLiteral(const std::string& token) {
    if (token.empty() || (token.front() != '\'' && token.front() != '"')) {
        throw std::invalid_argument("Expected a quoted string");
    }
    const char quote = token.front();
    std::string value;
    for (size_t i = 1; i < token.size(); ++i) {
        char c = token[i];
        if (c == quote) {
            if (i + 1 != token.size()) {
                throw std::invalid_argument("Unexpected text after string");
            }
            return value;
        }
        if (c == '\\') {
            if (++i == token.size()) {
                throw std::invalid_argument("Unfinished string escape");
            }
            switch (token[i]) {
                case 'n': c = '\n'; break;
                case 'r': c = '\r'; break;
                case 't': c = '\t'; break;
                case 'b': c = '\b'; break;
                case 'f': c = '\f'; break;
                case '\\': c = '\\'; break;
                case '\'': c = '\''; break;
                case '"': c = '"'; break;
                default:
                    throw std::invalid_argument("Unsupported string escape");
            }
        }
        value += c;
    }
    throw std::invalid_argument("Unclosed string literal");
}

// suffix 可以是 ['role']、["content"]、.role 或 .content。
std::string ReadMessageField(const ChatMessage& message,
                             const std::string& suffix_raw) {
    const std::string suffix = Trim(suffix_raw);
    std::string field;
    if (!suffix.empty() && suffix.front() == '.') {
        field = Trim(suffix.substr(1));
    } else if (suffix.size() >= 2 &&
               suffix.front() == '[' && suffix.back() == ']') {
        field = ReadStringLiteral(Trim(suffix.substr(1, suffix.size() - 2)));
    } else {
        throw std::invalid_argument("Expected a message field");
    }
    if (field == "role") {
        return message.role;
    }
    if (field == "content") {
        return message.context;  
    }
    throw std::invalid_argument("Unsupported message field: " + field);
}

// 从 ctx 中读取变量的实际值，不再对消息正文进行模板解析。
std::string ResolveVariable(const std::string& token, const EvalContext& ctx) {
    if (token == "true" || token == "True") {
        return "true";
    }
    if (token == "false" || token == "False") {
        return "false";
    }
    if (token == "add_generation_prompt") {
        return ctx.add_generation_prompt ? "true" : "false";
    }
    if (token == "loop.first") {
        if (ctx.current_message == nullptr) {
            throw std::invalid_argument("loop.first requires a message loop");
        }
        return ctx.message_idex == 0 ? "true" : "false";
    }
    if (token.rfind("message", 0) == 0 && token.rfind("messages", 0) != 0) {
        const std::string suffix = Trim(token.substr(7));
        if (suffix.empty() || (suffix.front() != '[' && suffix.front() != '.')) {
            throw std::invalid_argument("Unsupported variable: " + token);
        }
        if (ctx.current_message == nullptr) {
            throw std::invalid_argument("message requires a message loop");
        }
        return ReadMessageField(*ctx.current_message, suffix);
    }
    if (token.rfind("messages", 0) == 0) {
        const std::string access = Trim(token.substr(8));
        if (access.empty() || access.front() != '[') {
            throw std::invalid_argument("Expected messages[index]");
        }
        const size_t end = access.find(']');
        if (end == std::string::npos) {
            throw std::invalid_argument("Unclosed messages index");
        }
        const std::string index_text = Trim(access.substr(1, end - 1));
        if (index_text.empty()) {
            throw std::invalid_argument("Empty messages index");
        }
        for (char c : index_text) {
            if (!std::isdigit(static_cast<unsigned char>(c))) {
                throw std::invalid_argument("Message index must be nonnegative");
            }
        }
        const unsigned long long index = std::stoull(index_text);
        if (ctx.messages == nullptr || index >= ctx.messages->size()) {
            throw std::out_of_range("Message index out of range");
        }
        return ReadMessageField((*ctx.messages)[static_cast<size_t>(index)],
                                access.substr(end + 1));
    }
    throw std::invalid_argument("Unsupported variable or expression: " + token);
}

// 表达式 -> 字符串，例如 message['role'] + ': ' + message['content']。
std::string EvalExpr(const std::string& expr_raw, const EvalContext& ctx) {
    const std::string expr = Trim(expr_raw);
    if (expr.empty()) {
        return "";
    }
    std::string result;
    for (const auto& token : SplitExpression(expr, "+")) {
        if (token.empty()) {
            throw std::invalid_argument("Missing operand around +");
        }
        if (token.front() == '\'' || token.front() == '"') {
            result += ReadStringLiteral(token);
        } else {
            result += ResolveVariable(token, ctx);
        }
    }
    return result;
}

// 条件比较需要保留类型：布尔 false 和字符串 'false' 不相同。
struct ConditionValue {
    std::string text;
    bool is_bool = false;
};

ConditionValue EvalConditionValue(const std::string& expr_raw,
                                  const EvalContext& ctx) {
    const std::string expr = Trim(expr_raw);
    const bool is_bool = expr == "true" || expr == "True" ||
                         expr == "false" || expr == "False" ||
                         expr == "add_generation_prompt" || expr == "loop.first";
    return {EvalExpr(expr, ctx), is_bool};
}

// 条件 -> bool；先拆 or，再拆 and，保证 and 的优先级更高。
bool EvalCondition(const std::string& cond_raw, const EvalContext& ctx) {
    const std::string cond = Trim(cond_raw);
    if (cond.empty()) {
        throw std::invalid_argument("Empty if condition");
    }
    const auto or_parts = SplitExpression(cond, "or");
    if (or_parts.size() > 1) {
        for (const auto& part : or_parts) {
            if (part.empty()) {
                throw std::invalid_argument("Missing operand around or");
            }
        }
        for (const auto& part : or_parts) {
            if (EvalCondition(part, ctx)) {
                return true;
            }
        }
        return false;
    }
    const auto and_parts = SplitExpression(cond, "and");
    if (and_parts.size() > 1) {
        for (const auto& part : and_parts) {
            if (part.empty()) {
                throw std::invalid_argument("Missing operand around and");
            }
        }
        for (const auto& part : and_parts) {
            if (!EvalCondition(part, ctx)) {
                return false;
            }
        }
        return true;
    }
    for (const std::string op : {"==", "!="}) {
        const auto sides = SplitExpression(cond, op);
        if (sides.size() == 1) {
            continue;
        }
        if (sides.size() != 2 || sides[0].empty() || sides[1].empty()) {
            throw std::invalid_argument("Invalid comparison");
        }
        const ConditionValue left = EvalConditionValue(sides[0], ctx);
        const ConditionValue right = EvalConditionValue(sides[1], ctx);
        const bool equal = left.is_bool == right.is_bool && left.text == right.text;
        return op == "==" ? equal : !equal;
    }
    const ConditionValue value = EvalConditionValue(cond, ctx);
    return value.is_bool ? value.text == "true" : !value.text.empty();
}

// 编译后的六种指令：文本、输出、判断及循环。
struct Instr {
    enum Type { kText, kOutput, kIf, kEndIf, kFor, kEndFor } type;
    std::string text;
    size_t jump = 0;
};

// 循环运行状态；previous_* 用于退出嵌套循环时恢复外层消息。
struct ForFrame {
    size_t pc = 0;
    size_t index = 0;
    const ChatMessage* previous_message = nullptr;
    size_t previous_index = 0;
};

// 找标签结尾时忽略字符串内的 }} 和 %}。
size_t FindTagEnd(const std::string& tmpl, size_t start,
                  const std::string& closing) {
    char quote = 0;
    for (size_t i = start; i < tmpl.size(); ++i) {
        const char c = tmpl[i];
        if (quote != 0) {
            if (c == '\\') {
                ++i;
            } else if (c == quote) {
                quote = 0;
            }
        } else if (c == '\'' || c == '"') {
            quote = c;
        } else if (tmpl.compare(i, closing.size(), closing) == 0) {
            return i;
        }
    }
    throw std::invalid_argument("Unclosed template tag");
}

// 模板字符串 -> 指令数组。jump 在遇到 endif/endfor 时回填。
std::vector<Instr> CompileTemplate(const std::string& tmpl) {
    std::vector<Instr> instrs;
    std::stack<size_t> blocks;
    size_t pos = 0;
    while (pos < tmpl.size()) {
        const bool output = tmpl.compare(pos, 2, "{{") == 0;
        const bool statement = tmpl.compare(pos, 2, "{%") == 0;
        if (!output && !statement) {
            const size_t start = pos;
            while (pos < tmpl.size() && tmpl.compare(pos, 2, "{{") != 0 &&
                   tmpl.compare(pos, 2, "{%") != 0) {
                if (tmpl.compare(pos, 2, "{#") == 0) {
                    throw std::invalid_argument("Template comments are unsupported");
                }
                ++pos;
            }
            instrs.push_back({Instr::kText, tmpl.substr(start, pos - start), 0});
            continue;
        }
        const size_t end = FindTagEnd(tmpl, pos + 2, output ? "}}" : "%}");
        const std::string body = tmpl.substr(pos + 2, end - pos - 2);
        const bool trim_left = !body.empty() && body.front() == '-';
        const bool trim_right = !body.empty() && body.back() == '-';
        if (trim_left && !instrs.empty() && instrs.back().type == Instr::kText) {
            std::string& text = instrs.back().text;
            while (!text.empty() &&
                   std::isspace(static_cast<unsigned char>(text.back()))) {
                text.pop_back();
            }
        }
        const std::string tag = StripJiniaTrim(body);
        pos = end + 2;
        if (trim_right) {
            while (pos < tmpl.size() &&
                   std::isspace(static_cast<unsigned char>(tmpl[pos]))) {
                ++pos;
            }
        }
        if (output) {
            if (tag.empty()) {
                throw std::invalid_argument("Empty output expression");
            }
            instrs.push_back({Instr::kOutput, tag, 0});
            continue;
        }
        std::istringstream words(tag);
        std::string command;
        words >> command;
        if (command == "for") {
            std::string variable, in, collection, extra;
            if (!(words >> variable >> in >> collection) ||
                variable != "message" || in != "in" || collection != "messages" ||
                (words >> extra)) {
                throw std::invalid_argument(
                    "Only for message in messages is supported");
            }
            blocks.push(instrs.size());
            instrs.push_back({Instr::kFor, tag, 0});
        } else if (command == "if") {
            const std::string condition = Trim(tag.substr(2));
            if (condition.empty()) {
                throw std::invalid_argument("Empty if condition");
            }
            blocks.push(instrs.size());
            instrs.push_back({Instr::kIf, condition, 0});
        } else if (tag == "endif" || tag == "endfor") {
            const Instr::Type expected =
                tag == "endif" ? Instr::kIf : Instr::kFor;
            if (blocks.empty() || instrs[blocks.top()].type != expected) {
                throw std::invalid_argument("Mismatched closing tag: " + tag);
            }
            const size_t begin = blocks.top();
            blocks.pop();
            instrs.push_back({tag == "endif" ? Instr::kEndIf : Instr::kEndFor,
                              "", begin});
            instrs[begin].jump = instrs.size();
        } else {
            throw std::invalid_argument("Unsupported template statement: " + tag);
        }
    }
    if (!blocks.empty()) {
        throw std::invalid_argument("Unclosed if or for block");
    }
    return instrs;
}

// 指令数组 + 消息数组 -> 最终 prompt；这里不分词、不调用 forward。
std::string ExecuteTemplate(const std::vector<Instr>& instrs,
                                            const std::vector<ChatMessage>& messages,
                                            bool add_gen) {
    std::string output;
    EvalContext ctx;
    ctx.messages = &messages;
    ctx.add_generation_prompt = add_gen;
    std::stack<ForFrame> loops;
    size_t pc = 0;
    while (pc < instrs.size()) {
        const Instr& instr = instrs[pc];
        switch (instr.type) {
            case Instr::kText:
                output += instr.text;
                ++pc;
                break;
            case Instr::kOutput:
                output += EvalExpr(instr.text, ctx);
                ++pc;
                break;
            case Instr::kIf:
                pc = EvalCondition(instr.text, ctx) ? pc + 1 : instr.jump;
                break;
            case Instr::kEndIf:
                ++pc;
                break;
            case Instr::kFor:
                if (loops.empty() || loops.top().pc != pc) {
                    if (messages.empty()) {
                        pc = instr.jump;
                        break;
                    }
                    loops.push({pc, 0, ctx.current_message, ctx.message_idex});
                } else {
                    ++loops.top().index;
                    if (loops.top().index >= messages.size()) {
                        ctx.current_message = loops.top().previous_message;
                        ctx.message_idex = loops.top().previous_index;
                        loops.pop();
                        pc = instr.jump;
                        break;
                    }
                }
                ctx.message_idex = loops.top().index;
                ctx.current_message = &messages[ctx.message_idex];
                ++pc;
                break;
            case Instr::kEndFor:
                pc = instr.jump;
                break;
        }
    }
    return output;
}
} // namespace 
void PromptBuild::SetChatTemplate(const std::string& t) {
    chat_template = t;
}
std::string PromptBuild::BuildPlain(const std::vector<ChatMessage>& message) {
    int n = message.size();
    std::string res;
    for (int i = 0; i < n; i++) {
        if (message[i].role == "user") {
            res += "User: " + message[i].context + "\n";
        } else if(message[i].role == "system") {
            res += "System: " + message[i].context + "\n";
        } else if (message[i].role == "assistant") {
            res += "Assistant: " + message[i].context + "\n";
        }
    }
    res += "Assistant: ";
    return res;
}
std::string PromptBuild::BuildQwen2(const std::vector<ChatMessage>& message) {
    // 若信息没有system头 补齐一个
    bool has_system = false;
    int n = message.size();
    for (int i = 0; i < n; i++) {
        if (message[i].role == "system") {
            has_system = true;
            break;
        }
    }
    std::string res;
    if (! has_system) {
        res += "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
    }
    for (int i = 0; i < n; i++) {
        if(message[i].role != "user" && message[i].role != "assistant" && 
            message[i].role != "system") {
                continue;
            }
        res += "<|im_start|>" + message[i].role + "\n" + message[i].context + "<|im_end|>" + "\n";
    }
    res += "<|im_start|>assistant\n";
    return res;
}
std::string PromptBuild ::Build(const std::vector<ChatMessage> message) {
    if (chat_template.empty()) {
        return BuildPlain(message);
    }
    if (chat_template == "qwen2") {
        return BuildQwen2(message);
    }
    if (chat_template.size() >= 2 &&
        chat_template[0] == '{' &&
        (chat_template[1] == '{' || chat_template[1] == '%')) {
        auto it = CompileTemplate(chat_template);
        return ExecuteTemplate(it, message, true);
    }
    return BuildPlain(message);
}



}// namespace yan_lamma