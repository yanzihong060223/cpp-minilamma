#include "include/prompt_build.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace yan_lamma {
namespace {

class GgufPromptBuildTest : public ::testing::Test {
protected:
    void SetUp() override {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        test_dir_ = std::filesystem::temp_directory_path() /
            ("yan-gguf-prompt-test-" + std::to_string(stamp));
        ASSERT_TRUE(std::filesystem::create_directories(test_dir_));
    }

    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(test_dir_, ec);
    }

    // 只写元数据的 GGUF v3，无需下载模型或加载权重。
    void WriteGguf(const std::vector<std::pair<std::string, std::string>>& metadata) {
        std::ofstream file(GgufPath(), std::ios::binary);
        ASSERT_TRUE(file.is_open());
        const auto write_integer = [&file](uint64_t value, size_t bytes) {
            for (size_t i = 0; i < bytes; ++i) {
                file.put(static_cast<char>((value >> (8 * i)) & 0xff));
            }
        };
        const auto write_string = [&file, &write_integer](const std::string& value) {
            write_integer(value.size(), 8);
            file.write(value.data(), static_cast<std::streamsize>(value.size()));
        };

        file.write("GGUF", 4);
        write_integer(3, 4);
        write_integer(0, 8);  // tensor_count
        write_integer(metadata.size(), 8);
        for (const auto& entry : metadata) {
            write_string(entry.first);
            write_integer(static_cast<uint32_t>(GgufValueType::kGgufTypeString), 4);
            write_string(entry.second);
        }
        while (file.tellp() % 32 != 0 && file.good()) {
            file.put('\0');
        }
        file.close();
        ASSERT_TRUE(file.good());
    }

    std::string GgufPath() const { return (test_dir_ / "prompt.gguf").string(); }

    std::filesystem::path test_dir_;
};

TEST_F(GgufPromptBuildTest, ReadsTemplateAndBuildsMultipleTurns) {
    const std::string chat_template =
        "{% for message in messages %}"
        "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";
    ASSERT_NO_FATAL_FAILURE(WriteGguf({
        {"general.name", "Prompt test fixture"},
        {"general.architecture", "qwen2"},
        {"tokenizer.chat_template", chat_template}
    }));
    GgufRead reader;
    ASSERT_TRUE(reader.load(GgufPath()));

    const auto loaded_template = GetChatTemplate(GgufPath());
    ASSERT_EQ(loaded_template, chat_template);
    PromptBuild builder;
    builder.SetChatTemplate(loaded_template);

    EXPECT_EQ(builder.Build({{"system", "简洁回答。"}, {"user", "你好"},
                             {"assistant", "你好！"}, {"user", "保留 {{ 原文 }}"}}),
              "<|im_start|>system\n简洁回答。<|im_end|>\n"
              "<|im_start|>user\n你好<|im_end|>\n"
              "<|im_start|>assistant\n你好！<|im_end|>\n"
              "<|im_start|>user\n保留 {{ 原文 }}<|im_end|>\n"
              "<|im_start|>assistant\n");
    EXPECT_EQ(builder.Build({}), "<|im_start|>assistant\n");
}

TEST_F(GgufPromptBuildTest, ExplicitTemplateTakesPrecedenceOverQwen2Architecture) {
    ASSERT_NO_FATAL_FAILURE(WriteGguf({
        {"general.architecture", "qwen2"},
        {"tokenizer.chat_template", "{{ messages[0].content + ' -> reply: ' }}"}
    }));
    PromptBuild builder;
    builder.SetChatTemplate(GetChatTemplate(GgufPath()));

    EXPECT_EQ(builder.Build({{"user", "你好"}}), "你好 -> reply: ");
}

TEST_F(GgufPromptBuildTest, MissingTemplateFallsBackToQwen2Architecture) {
    ASSERT_NO_FATAL_FAILURE(WriteGguf({{"general.architecture", "qwen2"}}));
    const auto chat_template = GetChatTemplate(GgufPath());
    EXPECT_EQ(chat_template, "qwen2");
    PromptBuild builder;
    builder.SetChatTemplate(chat_template);

    EXPECT_EQ(builder.Build({{"user", "你好"}}),
              "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              "<|im_start|>user\n你好<|im_end|>\n"
              "<|im_start|>assistant\n");
    EXPECT_EQ(builder.Build({{"system", "简洁回答。"}, {"user", "你好"}}),
              "<|im_start|>system\n简洁回答。<|im_end|>\n"
              "<|im_start|>user\n你好<|im_end|>\n"
              "<|im_start|>assistant\n");
}

TEST_F(GgufPromptBuildTest, MissingTemplateAndUnknownOrMissingArchitectureUsePlain) {
    for (const std::string architecture : {"llama", ""}) {
        SCOPED_TRACE(architecture);
        std::vector<std::pair<std::string, std::string>> metadata;
        if (!architecture.empty()) {
            metadata.emplace_back("general.architecture", architecture);
        }
        ASSERT_NO_FATAL_FAILURE(WriteGguf(metadata));
        const auto chat_template = GetChatTemplate(GgufPath());
        EXPECT_TRUE(chat_template.empty());
        PromptBuild builder;
        builder.SetChatTemplate(chat_template);

        EXPECT_EQ(builder.Build({{"user", "你好"}}), "User: 你好\nAssistant: ");
    }
}

TEST_F(GgufPromptBuildTest, EmptyTemplateUsesPlain) {
    ASSERT_NO_FATAL_FAILURE(WriteGguf({
        {"general.architecture", "qwen2"}, {"tokenizer.chat_template", ""}
    }));
    const auto chat_template = GetChatTemplate(GgufPath());
    EXPECT_TRUE(chat_template.empty());
    PromptBuild builder;
    builder.SetChatTemplate(chat_template);

    EXPECT_EQ(builder.Build({{"user", "你好"}}), "User: 你好\nAssistant: ");
}

TEST_F(GgufPromptBuildTest, EmptyAndNonexistentPathsReturnEmptyTemplate) {
    EXPECT_TRUE(GetChatTemplate("").empty());
    EXPECT_TRUE(GetChatTemplate(GgufPath()).empty());
}

TEST_F(GgufPromptBuildTest, InvalidTemplateFromFileReportsErrorWhenBuilding) {
    const std::string chat_template = "{% for message in messages %}{{ message.content }}";
    ASSERT_NO_FATAL_FAILURE(WriteGguf({{"tokenizer.chat_template", chat_template}}));
    const auto loaded_template = GetChatTemplate(GgufPath());
    ASSERT_EQ(loaded_template, chat_template);
    PromptBuild builder;
    builder.SetChatTemplate(loaded_template);

    EXPECT_THROW(builder.Build({{"user", "你好"}}), std::invalid_argument);
}

}  // namespace
}  // namespace yan_lamma
