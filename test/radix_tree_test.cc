#include "include/radix_tree.h"

#include <gtest/gtest.h>

#include <utility>
#include <vector>

namespace yan_lamma {
namespace {

TEST(RadixTreeTest, EmptyTreeHasNoMatchingPrefix) {
    RadixTree tree;

    EXPECT_TRUE(tree.Empty());
    EXPECT_EQ(tree.LongestPrefis({}), 0u);
    EXPECT_EQ(tree.LongestPrefis({11, 22, 33}), 0u);
}

TEST(RadixTreeTest, EmptySequenceIsIgnored) {
    RadixTree tree;
    std::vector<int> empty;
    tree.Insert(empty);
    tree.Insert(empty);

    EXPECT_TRUE(tree.Empty());
    EXPECT_TRUE(empty.empty());
    EXPECT_EQ(tree.LongestPrefis({}), 0u);
    EXPECT_EQ(tree.LongestPrefis({11}), 0u);

    std::vector<int> tokens{11, 22};
    tree.Insert(tokens);
    tree.Insert(empty);

    EXPECT_FALSE(tree.Empty());
    EXPECT_EQ(tree.LongestPrefis({11, 22, 33}), 2u);
}

TEST(RadixTreeTest, InsertionPreservesCallerTokenVectors) {
    RadixTree tree;
    std::vector<int> first{11, 22, 33, 44};
    std::vector<int> branch{11, 22, 55};
    std::vector<int> prefix{11, 22};
    std::vector<int> extension{11, 22, 33, 44, 66};
    const auto original_first = first;
    const auto original_branch = branch;
    const auto original_prefix = prefix;
    const auto original_extension = extension;

    tree.Insert(first);
    tree.Insert(branch);
    tree.Insert(prefix);
    tree.Insert(extension);
    tree.Insert(first);

    EXPECT_EQ(first, original_first);
    EXPECT_EQ(branch, original_branch);
    EXPECT_EQ(prefix, original_prefix);
    EXPECT_EQ(extension, original_extension);
    EXPECT_EQ(tree.LongestPrefis(first), 4u);
    EXPECT_EQ(tree.LongestPrefis(branch), 3u);
    EXPECT_EQ(tree.LongestPrefis(extension), 5u);
}

TEST(RadixTreeTest, FindsExactAndPartialPrefixes) {
    RadixTree tree;
    std::vector<int> tokens{11, 22, 33, 44};
    tree.Insert(tokens);

    EXPECT_FALSE(tree.Empty());
    EXPECT_EQ(tree.LongestPrefis({11, 22, 33, 44}), 4u);
    EXPECT_EQ(tree.LongestPrefis({11, 22}), 2u);
    EXPECT_EQ(tree.LongestPrefis({11, 22, 33, 44, 55}), 4u);
    EXPECT_EQ(tree.LongestPrefis({11, 22, 55}), 2u);
    EXPECT_EQ(tree.LongestPrefis({99, 22}), 0u);
}

TEST(RadixTreeTest, SplitsAtDivergingBranches) {
    RadixTree tree;
    std::vector<int> first{11, 22, 33, 44};
    std::vector<int> branch{11, 22, 55};
    std::vector<int> other_root{77, 88};
    tree.Insert(first);
    tree.Insert(branch);
    tree.Insert(other_root);

    EXPECT_FALSE(tree.Empty());
    EXPECT_EQ(tree.LongestPrefis({11, 99}), 1u);
    EXPECT_EQ(tree.LongestPrefis({11, 22, 33}), 3u);
    EXPECT_EQ(tree.LongestPrefis({11, 22, 33, 44}), 4u);
    EXPECT_EQ(tree.LongestPrefis({11, 22, 55, 66}), 3u);
    EXPECT_EQ(tree.LongestPrefis({11, 22, 99}), 2u);
    EXPECT_EQ(tree.LongestPrefis({77, 88, 99}), 2u);
}

TEST(RadixTreeTest, SplitsWhenShorterPrefixIsInserted) {
    RadixTree tree;
    std::vector<int> first{11, 22, 33, 44, 55};
    std::vector<int> prefix{11, 22, 33};
    std::vector<int> branch{11, 22, 33, 66};
    tree.Insert(first);
    tree.Insert(prefix);
    tree.Insert(branch);

    EXPECT_FALSE(tree.Empty());
    EXPECT_EQ(tree.LongestPrefis({11, 22, 33}), 3u);
    EXPECT_EQ(tree.LongestPrefis({11, 22, 33, 44, 55}), 5u);
    EXPECT_EQ(tree.LongestPrefis({11, 22, 33, 66, 77}), 4u);
    EXPECT_EQ(tree.LongestPrefis({11, 22, 33, 99}), 3u);
    EXPECT_EQ(tree.LongestPrefis({11, 22, 99}), 2u);
}

TEST(RadixTreeTest, AppendsLongerSuffixBelowExistingPrefix) {
    RadixTree tree;
    std::vector<int> prefix{11, 22};
    std::vector<int> extension{11, 22, 33, 44};
    std::vector<int> branch{11, 22, 33, 55};
    tree.Insert(prefix);
    tree.Insert(extension);
    tree.Insert(branch);

    EXPECT_EQ(tree.LongestPrefis({11, 22}), 2u);
    EXPECT_EQ(tree.LongestPrefis({11, 22, 33, 44, 66}), 4u);
    EXPECT_EQ(tree.LongestPrefis({11, 22, 33, 55}), 4u);
    EXPECT_EQ(tree.LongestPrefis({11, 22, 33, 99}), 3u);
}

TEST(RadixTreeTest, DuplicateInsertionKeepsExistingPaths) {
    RadixTree tree;
    std::vector<int> first{11, 22, 33};
    std::vector<int> branch{11, 22, 55};
    std::vector<int> prefix{11, 22};
    tree.Insert(first);
    tree.Insert(first);
    tree.Insert(branch);
    tree.Insert(prefix);
    tree.Insert(prefix);
    tree.Insert(first);
    tree.Insert(branch);

    EXPECT_FALSE(tree.Empty());
    EXPECT_EQ(tree.LongestPrefis({11, 22, 33, 44}), 3u);
    EXPECT_EQ(tree.LongestPrefis({11, 22, 55, 66}), 3u);
    EXPECT_EQ(tree.LongestPrefis({11, 22, 99}), 2u);
}

TEST(RadixTreeTest, ClearRemovesAllPathsAndAllowsReuse) {
    RadixTree tree;
    std::vector<int> first{11, 22, 33, 44};
    std::vector<int> branch{11, 22, 55};
    tree.Insert(first);
    tree.Insert(branch);

    tree.Clear();
    tree.Clear();

    EXPECT_TRUE(tree.Empty());
    EXPECT_EQ(tree.LongestPrefis({11, 22, 33, 44}), 0u);
    EXPECT_EQ(tree.LongestPrefis({11, 22, 55}), 0u);

    std::vector<int> empty;
    tree.Insert(empty);
    EXPECT_TRUE(tree.Empty());

    std::vector<int> replacement{77, 88};
    tree.Insert(replacement);
    EXPECT_FALSE(tree.Empty());
    EXPECT_EQ(tree.LongestPrefis({77, 88, 99}), 2u);
    EXPECT_EQ(tree.LongestPrefis({11, 22}), 0u);
}

TEST(RadixTreeTest, MoveConstructionTransfersPathsAndResetsSource) {
    RadixTree source;
    std::vector<int> empty;
    std::vector<int> first{11, 22, 33, 44};
    std::vector<int> branch{11, 22, 55};
    std::vector<int> prefix{11, 22};
    std::vector<int> extension{11, 22, 33, 44, 66};
    source.Insert(empty);
    source.Insert(first);
    source.Insert(branch);
    source.Insert(prefix);
    source.Insert(extension);

    RadixTree destination(std::move(source));

    EXPECT_FALSE(destination.Empty());
    EXPECT_EQ(destination.LongestPrefis({11, 22}), 2u);
    EXPECT_EQ(destination.LongestPrefis({11, 22, 33, 44}), 4u);
    EXPECT_EQ(destination.LongestPrefis({11, 22, 55}), 3u);
    EXPECT_EQ(destination.LongestPrefis({11, 22, 33, 44, 66, 77}), 5u);
    EXPECT_EQ(destination.LongestPrefis({11, 22, 99}), 2u);
    EXPECT_TRUE(source.Empty());
    EXPECT_EQ(source.LongestPrefis({11, 22, 33, 44}), 0u);
    EXPECT_EQ(source.LongestPrefis({11, 22, 55}), 0u);

    source.Insert(empty);
    EXPECT_TRUE(source.Empty());
    std::vector<int> replacement{77, 88};
    source.Insert(replacement);
    EXPECT_FALSE(source.Empty());
    EXPECT_EQ(source.LongestPrefis({77, 88, 99}), 2u);
    EXPECT_EQ(source.LongestPrefis({11, 22}), 0u);
    EXPECT_EQ(destination.LongestPrefis({77, 88}), 0u);
    EXPECT_EQ(destination.LongestPrefis({11, 22, 33, 44, 66}), 5u);
    EXPECT_EQ(destination.LongestPrefis({11, 22, 55}), 3u);
}

}  // namespace
}  // namespace yan_lamma
