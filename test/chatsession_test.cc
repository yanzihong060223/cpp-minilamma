#include "include/chatsession.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace yan_lamma {
namespace {

TEST(ChatSessionTest, DefaultConstructionInitializesAllSessionState) {
    ChatSession session;

    EXPECT_TRUE(session.messages_.empty());
    EXPECT_TRUE(session.token_history_.empty());
    EXPECT_TRUE(session.ra_cache_.Empty());
    EXPECT_FLOAT_EQ(session.total_time_, 0.0f);
    EXPECT_EQ(session.prompt_token_, 0);
    EXPECT_EQ(session.generate_token_, 0);
    EXPECT_FLOAT_EQ(session.parma_.temperature, 0.0f);
    EXPECT_EQ(session.parma_.top_k, 0);
    EXPECT_EQ(session.parma_.seed, 0);
}

TEST(ChatSessionTest, AddMessagePreservesOrderAndCopiesCallerMessage) {
    ChatSession session;
    ChatMessage user{"user", "hello"};
    session.AddMessage({"system", "Be concise."});
    session.AddMessage(user);
    session.AddMessage({"assistant", "Hello!"});
    user.role = "assistant";
    user.context = "changed";

    ASSERT_EQ(session.messages_.size(), 3u);
    EXPECT_EQ(session.messages_[0].role, "system");
    EXPECT_EQ(session.messages_[0].context, "Be concise.");
    EXPECT_EQ(session.messages_[1].role, "user");
    EXPECT_EQ(session.messages_[1].context, "hello");
    EXPECT_EQ(session.messages_[2].role, "assistant");
    EXPECT_EQ(session.messages_[2].context, "Hello!");
    EXPECT_TRUE(session.token_history_.empty());
    EXPECT_TRUE(session.ra_cache_.Empty());
}

TEST(ChatSessionTest, SetTokenHistoryCopiesReplacesAndAcceptsEmptyHistory) {
    ChatSession session;
    std::vector<int> tokens{11, 22, 33};
    session.SetTokenHistory(tokens);

    EXPECT_EQ(tokens, (std::vector<int>{11, 22, 33}));
    EXPECT_EQ(session.token_history_, tokens);
    tokens[0] = 99;
    tokens.push_back(44);
    EXPECT_EQ(session.token_history_, (std::vector<int>{11, 22, 33}));

    const std::vector<int> replacement{55, 66};
    session.SetTokenHistory(replacement);
    EXPECT_EQ(session.token_history_, replacement);
    EXPECT_EQ(replacement, (std::vector<int>{55, 66}));

    session.SetTokenHistory({});
    EXPECT_TRUE(session.token_history_.empty());
    EXPECT_TRUE(session.messages_.empty());
    EXPECT_TRUE(session.ra_cache_.Empty());
}

TEST(ChatSessionTest, AddTokenAppendsToEmptyAndExistingHistory) {
    ChatSession session;
    session.AddToken(11);
    session.AddToken(22);
    session.AddToken(22);

    EXPECT_EQ(session.token_history_, (std::vector<int>{11, 22, 22}));
    session.SetTokenHistory({33, 44});
    session.AddToken(55);
    EXPECT_EQ(session.token_history_, (std::vector<int>{33, 44, 55}));
    EXPECT_TRUE(session.messages_.empty());
    EXPECT_TRUE(session.ra_cache_.Empty());
    EXPECT_EQ(session.generate_token_, 0);
}

TEST(ChatSessionTest, RecordTurnAccumulatesTimeAndTokenCounts) {
    ChatSession session;
    session.RecordTurn(1.25f, 10, 3);
    session.RecordTurn(2.5f, 4, 7);

    EXPECT_FLOAT_EQ(session.total_time_, 3.75f);
    EXPECT_EQ(session.prompt_token_, 14);
    EXPECT_EQ(session.generate_token_, 10);
    EXPECT_TRUE(session.messages_.empty());
    EXPECT_TRUE(session.token_history_.empty());
    EXPECT_TRUE(session.ra_cache_.Empty());
}

TEST(ChatSessionTest, PrefixCacheFindsBranchesWithoutChangingTokenHistory) {
    ChatSession session;
    session.SetTokenHistory({99, 88});
    std::vector<int> first{11, 22, 33};
    std::vector<int> branch{11, 22, 44};
    session.RecordRaCahce(first);
    session.RecordRaCahce(branch);
    std::vector<int> extension{11, 22, 33, 55};
    std::vector<int> partial{11, 22, 55};
    std::vector<int> missing{77, 88};

    EXPECT_FALSE(session.ra_cache_.Empty());
    EXPECT_EQ(session.LongestPrefix(first), 3u);
    EXPECT_EQ(session.LongestPrefix(branch), 3u);
    EXPECT_EQ(session.LongestPrefix(extension), 3u);
    EXPECT_EQ(session.LongestPrefix(partial), 2u);
    EXPECT_EQ(session.LongestPrefix(missing), 0u);
    EXPECT_EQ(first, (std::vector<int>{11, 22, 33}));
    EXPECT_EQ(branch, (std::vector<int>{11, 22, 44}));
    EXPECT_EQ(extension, (std::vector<int>{11, 22, 33, 55}));
    EXPECT_EQ(session.token_history_, (std::vector<int>{99, 88}));
    EXPECT_TRUE(session.messages_.empty());
}

TEST(ChatSessionTest, ClearResetsHistoryCacheAndStatsButKeepsSamplingParameters) {
    ChatSession session;
    session.parma_.temperature = 0.7f;
    session.parma_.top_k = 40;
    session.parma_.seed = 42;
    session.AddMessage({"user", "hello"});
    session.AddMessage({"assistant", "Hello!"});
    session.SetTokenHistory({11, 22});
    session.AddToken(33);
    std::vector<int> first{11, 22, 33};
    std::vector<int> branch{11, 22, 44};
    session.RecordRaCahce(first);
    session.RecordRaCahce(branch);
    session.RecordTurn(1.25f, 10, 3);
    session.RecordTurn(2.5f, 4, 7);

    session.Clear();

    EXPECT_TRUE(session.messages_.empty());
    EXPECT_TRUE(session.token_history_.empty());
    EXPECT_TRUE(session.ra_cache_.Empty());
    EXPECT_EQ(session.LongestPrefix(first), 0u);
    EXPECT_EQ(session.LongestPrefix(branch), 0u);
    EXPECT_FLOAT_EQ(session.total_time_, 0.0f);
    EXPECT_EQ(session.prompt_token_, 0);
    EXPECT_EQ(session.generate_token_, 0);
    EXPECT_FLOAT_EQ(session.parma_.temperature, 0.7f);
    EXPECT_EQ(session.parma_.top_k, 40);
    EXPECT_EQ(session.parma_.seed, 42);
}

TEST(ChatSessionTest, RepeatedClearAllowsFreshHistoryAndPrefixCacheReuse) {
    ChatSession session;
    session.parma_.temperature = 0.5f;
    session.parma_.top_k = 8;
    session.parma_.seed = 123;
    std::vector<int> old_tokens{11, 22};
    session.RecordRaCahce(old_tokens);
    session.SetTokenHistory(old_tokens);
    session.AddMessage({"user", "old"});
    session.RecordTurn(1.0f, 2, 1);

    session.Clear();
    session.Clear();

    EXPECT_TRUE(session.messages_.empty());
    EXPECT_TRUE(session.token_history_.empty());
    EXPECT_TRUE(session.ra_cache_.Empty());
    EXPECT_EQ(session.LongestPrefix(old_tokens), 0u);
    EXPECT_FLOAT_EQ(session.total_time_, 0.0f);
    EXPECT_EQ(session.prompt_token_, 0);
    EXPECT_EQ(session.generate_token_, 0);

    std::vector<int> new_tokens{77, 88};
    std::vector<int> extended_query{77, 88, 99};
    session.AddMessage({"user", "fresh"});
    session.SetTokenHistory(new_tokens);
    session.AddToken(99);
    session.RecordRaCahce(new_tokens);
    session.RecordTurn(0.5f, 2, 1);

    ASSERT_EQ(session.messages_.size(), 1u);
    EXPECT_EQ(session.messages_[0].role, "user");
    EXPECT_EQ(session.messages_[0].context, "fresh");
    EXPECT_EQ(session.token_history_, extended_query);
    EXPECT_FALSE(session.ra_cache_.Empty());
    EXPECT_EQ(session.LongestPrefix(extended_query), 2u);
    EXPECT_EQ(session.LongestPrefix(old_tokens), 0u);
    EXPECT_FLOAT_EQ(session.total_time_, 0.5f);
    EXPECT_EQ(session.prompt_token_, 2);
    EXPECT_EQ(session.generate_token_, 1);
    EXPECT_FLOAT_EQ(session.parma_.temperature, 0.5f);
    EXPECT_EQ(session.parma_.top_k, 8);
    EXPECT_EQ(session.parma_.seed, 123);
}

}  // namespace
}  // namespace yan_lamma
