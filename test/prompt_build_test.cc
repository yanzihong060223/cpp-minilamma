#include "include/prompt_build.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace yan_lamma {
namespace {

TEST(PromptBuildTest, PlainFormatsMultipleTurns) {
    PromptBuild builder;
    const std::vector<ChatMessage> messages{
        {"system", "Be concise."},
        {"user", "你好"},
        {"assistant", "你好！"},
        {"user", "继续"}
    };

    EXPECT_EQ(builder.Build(messages),
              "System: Be concise.\n"
              "User: 你好\n"
              "Assistant: 你好！\n"
              "User: 继续\n"
              "Assistant: ");
}

TEST(PromptBuildTest, PlainEmptyHistoryStillHasAssistantPrefix) {
    PromptBuild builder;

    EXPECT_EQ(builder.Build({}), "Assistant: ");
}

TEST(PromptBuildTest, UnrecognizedPrefixesFallBackToPlain) {
    PromptBuild builder;
    const std::vector<ChatMessage> messages{{"user", "你好"}};
    const std::vector<std::string> templates{
        "unknown", "a%", "x%anything", "{", "%", "x",
        "Hi {{ messages[0]['content'] }}",
        " {{ messages[0]['content'] }}"
    };

    for (const auto& chat_template : templates) {
        SCOPED_TRACE(chat_template);
        builder.SetChatTemplate(chat_template);
        EXPECT_EQ(builder.Build(messages), "User: 你好\nAssistant: ");
    }
}

TEST(PromptBuildTest, Qwen2AddsDefaultSystemAndCompleteAssistantPrefix) {
    PromptBuild builder;
    builder.SetChatTemplate("qwen2");

    EXPECT_EQ(builder.Build({{"user", "你好"}}),
              "<|im_start|>system\n"
              "You are a helpful assistant.<|im_end|>\n"
              "<|im_start|>user\n你好<|im_end|>\n"
              "<|im_start|>assistant\n");
}

TEST(PromptBuildTest, Qwen2PreservesExistingSystemAndMultipleTurns) {
    PromptBuild builder;
    builder.SetChatTemplate("qwen2");
    const std::vector<ChatMessage> messages{
        {"system", "Be concise."},
        {"user", "Hi"},
        {"assistant", "Hello"},
        {"user", "Continue"}
    };

    EXPECT_EQ(builder.Build(messages),
              "<|im_start|>system\nBe concise.<|im_end|>\n"
              "<|im_start|>user\nHi<|im_end|>\n"
              "<|im_start|>assistant\nHello<|im_end|>\n"
              "<|im_start|>user\nContinue<|im_end|>\n"
              "<|im_start|>assistant\n");
}

TEST(PromptBuildTest, Qwen2SkipsUnsupportedRoles) {
    PromptBuild builder;
    builder.SetChatTemplate("qwen2");

    EXPECT_EQ(builder.Build({{"system", "S"}, {"tool", "ignored"}, {"user", "Hi"}}),
              "<|im_start|>system\nS<|im_end|>\n"
              "<|im_start|>user\nHi<|im_end|>\n"
              "<|im_start|>assistant\n");
}

TEST(PromptBuildTest, Qwen2EmptyHistoryKeepsDefaultSystemAndAssistantPrefix) {
    PromptBuild builder;
    builder.SetChatTemplate("qwen2");

    EXPECT_EQ(builder.Build({}),
              "<|im_start|>system\n"
              "You are a helpful assistant.<|im_end|>\n"
              "<|im_start|>assistant\n");
}

TEST(PromptBuildTest, JinjaExpressionReadsContextAndDecodesStringEscapes) {
    PromptBuild builder;
    builder.SetChatTemplate(R"({{ 'a+b\n' + messages[0]["content"] }})");

    EXPECT_EQ(builder.Build({{"user", "你好"}}), "a+b\n你好");
}

TEST(PromptBuildTest, JinjaLoopUsesEachMessagesFields) {
    PromptBuild builder;
    builder.SetChatTemplate(
        R"({% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\n' }}{% endfor %})");

    EXPECT_EQ(builder.Build({{"user", "你好"}, {"assistant", "你好！"}}),
              "user: 你好\nassistant: 你好！\n");
}

TEST(PromptBuildTest, JinjaEmptyLoopDoesNotEvaluateMessageFields) {
    PromptBuild builder;
    builder.SetChatTemplate(
        "{% for message in messages %}{{ message.content }}{% endfor %}");

    EXPECT_EQ(builder.Build({}), "");
}

TEST(PromptBuildTest, JinjaConditionsFilterLoopMessages) {
    PromptBuild builder;
    builder.SetChatTemplate(
        "{% for message in messages %}"
        "{% if message.role == 'user' and loop.first %}"
        "{{ message.content }}"
        "{% endif %}{% endfor %}");

    EXPECT_EQ(builder.Build({{"user", "first"}, {"user", "second"}}), "first");
}

TEST(PromptBuildTest, JinjaReceivesGenerationPromptFlag) {
    PromptBuild builder;
    builder.SetChatTemplate(
        "{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}");

    EXPECT_EQ(builder.Build({}), "assistant: ");
}

TEST(PromptBuildTest, JinjaTrimMarkersRemoveOnlyAdjacentWhitespace) {
    PromptBuild builder;
    builder.SetChatTemplate("{{ 'A' }}  \n {{- 'B' -}}  \n {{ 'C' }}");
    EXPECT_EQ(builder.Build({}), "ABC");

    builder.SetChatTemplate("{{ 'A' }}  \n {{ 'B' }}  \n {{ 'C' }}");
    EXPECT_EQ(builder.Build({}), "A  \n B  \n C");
}

TEST(PromptBuildTest, JinjaDistinguishesBooleansFromNonemptyStrings) {
    PromptBuild builder;
    builder.SetChatTemplate(
        "{% if False %}wrong{% endif %}"
        "{% if 'false' %}text{% endif %}"
        "{% if false == 'false' %}wrong{% endif %}");

    EXPECT_EQ(builder.Build({}), "text");
}

TEST(PromptBuildTest, JinjaNestedLoopRestoresOuterMessage) {
    PromptBuild builder;
    builder.SetChatTemplate(
        "{% for message in messages %}"
        "{{ message.role }}["
        "{% for message in messages %}{{ message.role }}{% endfor %}"
        "]{{ message.role }};"
        "{% endfor %}");

    EXPECT_EQ(builder.Build({{"user", "Hi"}, {"assistant", "Hello"}}),
              "user[userassistant]user;assistant[userassistant]assistant;");
}

TEST(PromptBuildTest, InvalidJinjaTemplatesReportErrors) {
    PromptBuild builder;
    const std::vector<std::string> templates{
        "{{ }}", "{{ unknown }}", "{{ 'unfinished }}",
        "{% if True %}", "{% endfor %}",
        "{% if True %}{% for message in messages %}{% endif %}{% endfor %}",
        "{% for item in messages %}{% endfor %}"
    };

    for (const auto& chat_template : templates) {
        SCOPED_TRACE(chat_template);
        builder.SetChatTemplate(chat_template);
        EXPECT_THROW(builder.Build({{"user", "Hi"}}), std::invalid_argument);
    }
}

}  // namespace
}  // namespace yan_lamma
