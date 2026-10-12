#include "include/terminal.h"

#include <gtest/gtest.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace yan_lamma {
namespace {

class ScopedStreamRedirect {
public:
    ScopedStreamRedirect(std::ios& stream, std::streambuf* replacement)
        : stream_(stream), original_buffer_(stream.rdbuf()),
          original_state_(stream.rdstate()), original_exceptions_(stream.exceptions()) {
        stream_.exceptions(std::ios::goodbit);
        stream_.rdbuf(replacement);
    }

    ~ScopedStreamRedirect() noexcept {
        stream_.exceptions(std::ios::goodbit);
        stream_.rdbuf(original_buffer_);
        stream_.clear(original_state_);
        try {
            stream_.exceptions(original_exceptions_);
        } catch (const std::ios_base::failure&) {
            // exceptions() installs the saved mask before reporting a saved error state.
        }
    }

    ScopedStreamRedirect(const ScopedStreamRedirect&) = delete;
    ScopedStreamRedirect& operator=(const ScopedStreamRedirect&) = delete;

private:
    std::ios& stream_;
    std::streambuf* original_buffer_;
    std::ios::iostate original_state_;
    std::ios::iostate original_exceptions_;
};

template <typename Action>
std::string CaptureStdout(Action action) {
    std::ostringstream output;
    {
        ScopedStreamRedirect redirect(std::cout, output.rdbuf());
        action();
    }
    return output.str();
}

class FlushRecordingBuffer : public std::stringbuf {
public:
    std::vector<std::string> flush_snapshots;

protected:
    int sync() override {
        flush_snapshots.push_back(str());
        return std::stringbuf::sync();
    }
};

double NumberAfterLabel(const std::string& output, const std::string& label) {
    std::istringstream lines(output);
    std::string line;
    while (std::getline(lines, line)) {
        const auto label_position = line.find(label);
        if (label_position == std::string::npos) {
            continue;
        }
        const auto number_position = line.find_first_of("-0123456789", label_position + label.size());
        if (number_position != std::string::npos) {
            return std::stod(line.substr(number_position));
        }
    }
    throw std::runtime_error("Missing numeric output for label: " + label);
}

TEST(TerminalTest, TokenTextAppendsChunksWithoutAddingNewlines) {
    Terminal terminal;
    const auto output = CaptureStdout([&] {
        terminal.PrintTokenText("hel");
        terminal.PrintTokenText("");
        terminal.PrintTokenText("lo!");
    });

    EXPECT_EQ(output, "hello!");
}

TEST(TerminalTest, MemberPrintMessageAddsOneNewlinePerMessage) {
    Terminal terminal;
    const auto output = CaptureStdout([&] {
        terminal.PrintMessage("first");
        terminal.PrintMessage("second");
        terminal.PrintMessage("");
    });

    EXPECT_EQ(output, "first\nsecond\n\n");
}

TEST(TerminalTest, FlushAndNewLinePreserveOutputOrder) {
    Terminal terminal;
    FlushRecordingBuffer buffer;
    {
        ScopedStreamRedirect redirect(std::cout, &buffer);
        terminal.PrintTokenText("first");
        terminal.flush();
        terminal.NewLine();
        terminal.PrintTokenText("second");
        terminal.flush();
    }

    EXPECT_EQ(buffer.str(), "first\nsecond");
    EXPECT_EQ(buffer.flush_snapshots, (std::vector<std::string>{"first", "first\nsecond"}));
}

TEST(TerminalTest, ReadLinePreservesWholeLinesAndAcceptsEmptyLines) {
    Terminal terminal;
    std::istringstream input("  hello world\t!\n\nlast line");
    std::vector<std::string> lines;
    {
        ScopedStreamRedirect redirect(std::cin, input.rdbuf());
        lines.push_back(terminal.ReadLine());
        lines.push_back(terminal.ReadLine());
        lines.push_back(terminal.ReadLine());
    }

    EXPECT_EQ(lines, (std::vector<std::string>{"  hello world\t!", "", "last line"}));
}

TEST(TerminalTest, ReadLineReturnsEmptyAtEofAndRestoresInputState) {
    Terminal terminal;
    std::istringstream input;
    const auto original_state = std::cin.rdstate();
    const auto original_buffer = std::cin.rdbuf();
    const auto original_exceptions = std::cin.exceptions();
    std::string first;
    std::string second;
    bool reached_eof = false;
    {
        ScopedStreamRedirect redirect(std::cin, input.rdbuf());
        first = terminal.ReadLine();
        reached_eof = std::cin.eof();
        second = terminal.ReadLine();
    }

    EXPECT_TRUE(first.empty());
    EXPECT_TRUE(second.empty());
    EXPECT_TRUE(reached_eof);
    EXPECT_EQ(std::cin.rdstate(), original_state);
    EXPECT_EQ(std::cin.rdbuf(), original_buffer);
    EXPECT_EQ(std::cin.exceptions(), original_exceptions);
}

TEST(TerminalTest, PrintStatsReflectsCurrentSessionCounts) {
    Terminal terminal;
    ChatSession session;
    session.AddMessage({"user", "hello"});
    session.AddMessage({"assistant", "reply"});
    session.SetTokenHistory({10, 20, 30});
    session.RecordTurn(17.5f, 24, 13);

    const auto output = CaptureStdout([&] { terminal.PrintStats(session); });

    EXPECT_DOUBLE_EQ(NumberAfterLabel(output, "messages"), 2.0);
    EXPECT_DOUBLE_EQ(NumberAfterLabel(output, "context tokens"), 3.0);
    EXPECT_DOUBLE_EQ(NumberAfterLabel(output, "total prompt tokens"), 24.0);
    EXPECT_DOUBLE_EQ(NumberAfterLabel(output, "generate tokens"), 13.0);
}

TEST(TerminalTest, PrintParamsReflectsProvidedSamplingValues) {
    Terminal terminal;
    SamplerParmas params;
    params.temperature = 0.75f;
    params.top_k = 9;
    params.seed = 1234;

    const auto output = CaptureStdout([&] { terminal.PrintParams(params); });

    EXPECT_DOUBLE_EQ(NumberAfterLabel(output, "temperature"), 0.75);
    EXPECT_DOUBLE_EQ(NumberAfterLabel(output, "Top k"), 9.0);
    EXPECT_DOUBLE_EQ(NumberAfterLabel(output, "Seed"), 1234.0);
}

}  // namespace
}  // namespace yan_lamma
