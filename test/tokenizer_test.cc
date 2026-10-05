#include "include/tonizer.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace yan_lamma {
namespace {

const char* kJsonVocabulary = R"json([
    {"id":0,"context":"<BOS>","special":true},
    {"id":1,"context":"<EOS>","special":true},
    {"id":2,"context":"<UNK>","special":true},
    {"id":3,"context":"ab","special":false},
    {"id":4,"context":"a"},
    {"id":5,"context":"b"},
    {"id":6,"context":"abc"},
    {"id":7,"context":"\n"},
    {"id":8,"context":"\t"},
    {"id":9,"context":"\""},
    {"id":10,"context":"\\"},
    {"id":11,"context":"\/"},
    {"id":12,"context":"\b"},
    {"id":13,"context":"\f"},
    {"id":14,"context":"\r"},
    {"id":15,"context":"\u4E2D"},
    {"id":16,"context":"\uD83D\uDE00"},
    {"id":17,"context":"\u0000"},
    {"id":18,"context":"<|special|>","special":true}
])json";

// Fixed GPT-2 byte symbols: space=U+0120, LF=U+010A, TAB=U+0109.
// U+00E4/U+00B8/U+0143 represent the three UTF-8 bytes of U+4E2D;
// U+00F0/U+0141/U+013A/U+0122 represent the four bytes of U+1F600.
const char* kBpeVocabulary = R"json({
    "<bos>":0,"<eos>":1,"<unk>":2,
    "h":3,"e":4,"l":5,"o":6,"he":7,"hel":8,"hell":9,"hello":10,
    "a":11,"b":12,"c":13,"ab":14,"bc":15,"abc":16,
    "\u0120":19,"\u010A":20,"\u0109":21,"\u0100":22,
    "\u00E4":23,"\u00B8":24,"\u0143":25,
    "\u00F0":26,"\u0141":27,"\u013A":28,"\u0122":29,
    "\u0121":30,"\u00FF":31,
    "<tag>":32,"<tag>long":33,"<|im_start|>":34
})json";

const char* kBpeSpecials = R"json({
    "bos_id":0,"eos_id":1,"unk_id":2,
    "special_tokens":["<tag>","<tag>long"]
})json";

const char* kBpeMerges = "#version: 0.2\nh e\nhe l\nhel l\nhell o\nb c\na b\n";

std::vector<int> EncodeContent(const Tonizer& tokenizer, const std::string& text) {
    const auto encoded = tokenizer.Encode(text);
    if (encoded.size() < 2 || encoded.front() != tokenizer.GetBOSId() ||
        encoded.back() != tokenizer.GetEOSId()) {
        throw std::runtime_error("Tokenizer did not add BOS and EOS");
    }
    return {encoded.begin() + 1, encoded.end() - 1};
}

class TokenizerFiles : public ::testing::Test {
protected:
    void SetUp() override {
        std::random_device random;
        for (int attempt = 0; attempt < 100; ++attempt) {
            auto candidate = std::filesystem::temp_directory_path() /
                ("yan-minillama-tokenizer-" + std::to_string(random()) +
                 "-" + std::to_string(attempt));
            std::error_code error;
            if (std::filesystem::create_directory(candidate, error)) {
                directory_ = std::move(candidate);
                return;
            }
            if (error) throw std::runtime_error(error.message());
        }
        throw std::runtime_error("Cannot create tokenizer test directory");
    }

    void TearDown() override {
        std::error_code error;
        std::filesystem::remove_all(directory_, error);
        EXPECT_FALSE(error) << error.message();
    }

    std::string Path(const std::string& name) const {
        return (directory_ / name).string();
    }

    std::string Write(const std::string& name, const std::string& contents) const {
        const auto path = Path(name);
        std::ofstream file(path, std::ios::binary);
        file.write(contents.data(), static_cast<std::streamsize>(contents.size()));
        file.close();
        if (!file) throw std::runtime_error("Cannot write tokenizer fixture");
        return path;
    }

    std::string JsonPath() const { return Write("json.json", kJsonVocabulary); }

    void WriteBpe(const std::string& vocab = kBpeVocabulary,
                  const std::string& merges = kBpeMerges,
                  const std::string& specials = kBpeSpecials) const {
        Write("vocab.json", vocab);
        Write("merges.txt", merges);
        Write("special.json", specials);
    }

    void LoadBpe(BpeTokenizer& tokenizer) const {
        ASSERT_TRUE(tokenizer.Load(Path("vocab.json"), Path("merges.txt"),
                                   Path("special.json")));
    }

private:
    std::filesystem::path directory_;
};

TEST(AsciiTokenizerTest, LongInputChecksByteValueRatherThanPosition) {
    const AsciiTokenizer tokenizer;
    EXPECT_EQ(EncodeContent(tokenizer, std::string(512, 'a')),
              std::vector<int>(512, 'a'));
}

TEST(AsciiTokenizerTest, AllAsciiBytesRoundTripIncludingControlsAndNul) {
    const AsciiTokenizer tokenizer;
    std::string input;
    std::vector<int> expected;
    for (int byte = 0; byte < 128; ++byte) {
        input += static_cast<char>(byte);
        expected.push_back(byte);
    }
    EXPECT_EQ(EncodeContent(tokenizer, input), expected);
    EXPECT_EQ(tokenizer.Decode(expected), input);
    EXPECT_EQ(tokenizer.DecodeToken(127), std::string(1, '\x7F'));
    EXPECT_GE(tokenizer.GetBOSId(), 128);
    EXPECT_GE(tokenizer.GetEOSId(), 128);
    EXPECT_GE(tokenizer.GetUNKId(), 128);
}

TEST(AsciiTokenizerTest, NonAsciiBytesUseUnkWithoutSignedCharIds) {
    const AsciiTokenizer tokenizer;
    EXPECT_EQ(EncodeContent(tokenizer, std::string("\x80\xFF", 2)),
              (std::vector<int>{tokenizer.GetUNKId(), tokenizer.GetUNKId()}));
    EXPECT_EQ(tokenizer.Decode({tokenizer.GetUNKId()}), "<UNK>");
    EXPECT_THROW(tokenizer.DecodeToken(-1), std::out_of_range);
    EXPECT_THROW(tokenizer.Decode({tokenizer.GetVocabSize()}), std::out_of_range);
}

TEST(AsciiTokenizerTest, EncodeAlwaysAddsMarkersAndDecodePreservesTheirText) {
    const AsciiTokenizer tokenizer;
    EXPECT_EQ(tokenizer.Encode("a"), (std::vector<int>{128, 97, 129}));
    EXPECT_EQ(tokenizer.Encode(""), (std::vector<int>{128, 129}));
    EXPECT_EQ(tokenizer.Decode(tokenizer.Encode("a")), "<BOS>a<EOS>");
}

TEST_F(TokenizerFiles, JsonLongestMatchRestartsAtEachNewPosition) {
    const JsonVocabTokenizer tokenizer(JsonPath());
    EXPECT_EQ(EncodeContent(tokenizer, "abab"), (std::vector<int>{3, 3}));
    EXPECT_EQ(EncodeContent(tokenizer, "abcab"), (std::vector<int>{6, 3}));
    EXPECT_EQ(tokenizer.Decode({6, 3}), "abcab");
}

TEST_F(TokenizerFiles, JsonHandlesEveryEscapeChineseEmojiAndEmbeddedNul) {
    const JsonVocabTokenizer tokenizer(JsonPath());
    std::string text = "\n\t\"\\/\b\f\r";
    text += u8"\u4E2D\U0001F600";
    text += '\0';
    const std::vector<int> ids{7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
    EXPECT_EQ(EncodeContent(tokenizer, text), ids);
    EXPECT_EQ(tokenizer.Decode(ids), text);
    EXPECT_EQ(tokenizer.DecodeToken(15), u8"\u4E2D");
    EXPECT_EQ(tokenizer.DecodeToken(16), u8"\U0001F600");
}

TEST_F(TokenizerFiles, JsonUnkIsOnePerUnicodeScalarOrMalformedByte) {
    const JsonVocabTokenizer tokenizer(JsonPath());
    EXPECT_EQ(EncodeContent(tokenizer, u8"\u4E2D\u6587\U0001F642"),
              (std::vector<int>{15, 2, 2}));
    EXPECT_EQ(EncodeContent(tokenizer, std::string("\xC0\xAF\xF0\x9F", 4)),
              (std::vector<int>{2, 2, 2, 2}));
}

TEST_F(TokenizerFiles, JsonEncodeAlwaysAddsMarkersAndDecodePreservesSpecialText) {
    const JsonVocabTokenizer tokenizer(JsonPath());
    EXPECT_EQ(tokenizer.Encode("ab"), (std::vector<int>{0, 3, 1}));
    EXPECT_EQ(tokenizer.Encode(""), (std::vector<int>{0, 1}));
    EXPECT_EQ(EncodeContent(tokenizer, "a<|special|>b"),
              (std::vector<int>{4, 18, 5}));
    EXPECT_EQ(tokenizer.Decode({0, 3, 18, 1}), "<BOS>ab<|special|><EOS>");
}

TEST_F(TokenizerFiles, JsonSparseIdsAndDecodeBoundsAreChecked) {
    const auto path = Write("sparse.json", R"json([
        {"id":0,"context":"<BOS>"},{"id":1,"context":"<EOS>"},
        {"id":2,"context":"<UNK>"},{"id":1000,"context":"abc"}
    ])json");
    const JsonVocabTokenizer tokenizer(path);
    EXPECT_EQ(tokenizer.GetVocabSize(), 1001);
    EXPECT_EQ(EncodeContent(tokenizer, "abc"), (std::vector<int>{1000}));
    EXPECT_EQ(tokenizer.DecodeToken(0), "<BOS>");
    EXPECT_EQ(tokenizer.DecodeToken(1000), "abc");
    EXPECT_THROW(tokenizer.DecodeToken(-1), std::out_of_range);
    EXPECT_THROW(tokenizer.DecodeToken(3), std::out_of_range);
    EXPECT_THROW(tokenizer.Decode({1001}), std::out_of_range);
}

class InvalidJsonVocabulary : public TokenizerFiles,
                              public ::testing::WithParamInterface<std::string> {};

TEST_P(InvalidJsonVocabulary, RejectsInvalidDocumentOrEntry) {
    const auto path = Write("invalid.json", GetParam());
    EXPECT_THROW((void)JsonVocabTokenizer{path}, std::runtime_error);
}

std::vector<std::string> InvalidJsonDocuments() {
    const std::string prefix = R"json([
        {"id":0,"context":"<BOS>"},{"id":1,"context":"<EOS>"},
        {"id":2,"context":"<UNK>"},)json";
    std::vector<std::string> documents{
        "", "[", "[]", "{}", kJsonVocabulary + std::string(" trailing"),
        R"json([{"id":0,"context":"<BOS>"},{"id":1,"context":"<EOS>"}])json"
    };
    for (const auto* entry : {
        R"({"id":-1,"context":"a"})", R"({"id":2147483647,"context":"a"})",
        R"({"id":2147483648,"context":"a"})", R"({"id":3.5,"context":"a"})",
        R"({"id":3e0,"context":"a"})", R"({"id":03,"context":"a"})",
        R"({"id":"3","context":"a"})", R"({"id":true,"context":"a"})",
        R"({"id":0,"context":"a"})", R"({"id":3,"context":"<UNK>"})",
        R"({"id":3,"context":""})", R"({"id":3})", R"({"context":"a"})",
        R"({"id":3,"id":4,"context":"a"})",
        R"({"id":3,"context":"a","special":"false"})",
        R"({"id":3,"context":"\q"})", R"({"id":3,"context":"\u12"})",
        R"({"id":3,"context":"\uD83D"})", R"({"id":3,"context":"\uDE00"})",
        R"({"id":3,"context":"\uD83D\u0041"})",
        R"({"id":3,"context":"a",})"
    }) {
        documents.push_back(prefix + entry + "]");
    }
    documents.push_back(prefix + "{\"id\":3,\"context\":\"a\nb\"}]");
    documents.push_back(prefix + "{\"id\":3,\"context\":\"" +
                        std::string(1, '\xFF') + "\"}]");
    return documents;
}

INSTANTIATE_TEST_SUITE_P(TokenizerValidation, InvalidJsonVocabulary,
                        ::testing::ValuesIn(InvalidJsonDocuments()));

TEST_F(TokenizerFiles, BpeRecomputesPairsAfterEachMergeAndHandlesRepeatedWords) {
    WriteBpe();
    BpeTokenizer tokenizer;
    LoadBpe(tokenizer);
    EXPECT_EQ(EncodeContent(tokenizer, "hello"), (std::vector<int>{10}));
    EXPECT_EQ(EncodeContent(tokenizer, "hellohello"), (std::vector<int>{10, 10}));
    EXPECT_EQ(tokenizer.Decode({10, 10}), "hellohello");
}

TEST_F(TokenizerFiles, BpeChoosesLowestRankRatherThanLeftmostOrWholeVocabToken) {
    WriteBpe();
    BpeTokenizer tokenizer;
    LoadBpe(tokenizer);
    // "b c" precedes "a b"; "abc" being in the vocabulary is insufficient.
    EXPECT_EQ(EncodeContent(tokenizer, "abc"), (std::vector<int>{11, 15}));
    Write("merges.txt", std::string(kBpeMerges) + "a bc\n");
    LoadBpe(tokenizer);
    EXPECT_EQ(EncodeContent(tokenizer, "abc"), (std::vector<int>{16}));
}

TEST_F(TokenizerFiles, BpeMapsWhitespaceUnicodeAndSignedByteBoundaries) {
    WriteBpe();
    BpeTokenizer tokenizer;
    LoadBpe(tokenizer);
    EXPECT_EQ(EncodeContent(tokenizer, " \n\t"), (std::vector<int>{19, 20, 21}));
    const auto text = std::string(u8"\u4E2D\U0001F600");
    const std::vector<int> ids{23, 24, 25, 26, 27, 28, 29};
    EXPECT_EQ(EncodeContent(tokenizer, text), ids);
    EXPECT_EQ(tokenizer.Decode(ids), text);
    EXPECT_EQ(tokenizer.DecodeToken(23), std::string(1, '\xE4'));
    const std::string bytes("\x00\x7F\x80\xAD\xFF", 5);
    const std::vector<int> byte_ids{22, 30, 29, 25, 31};
    EXPECT_EQ(EncodeContent(tokenizer, bytes), byte_ids);
    EXPECT_EQ(tokenizer.Decode(byte_ids), bytes);
}

TEST_F(TokenizerFiles, BpeSpecialsUseLongestMatchAndDoNotMergeAcrossMarkers) {
    WriteBpe();
    BpeTokenizer tokenizer;
    LoadBpe(tokenizer);
    EXPECT_EQ(EncodeContent(tokenizer, "<tag>long<|im_start|><tag>h"),
              (std::vector<int>{33, 34, 32, 3}));
    EXPECT_EQ(EncodeContent(tokenizer, "a<tag>b"),
              (std::vector<int>{11, 32, 12}));
    EXPECT_EQ(EncodeContent(tokenizer, "hello<|im_start|>hello"),
              (std::vector<int>{10, 34, 10}));
    EXPECT_EQ(tokenizer.Decode({33, 34, 32, 3}), "<tag>long<|im_start|><tag>h");
}

TEST_F(TokenizerFiles, BpeEncodeAlwaysAddsMarkersAndDecodePreservesSpecialText) {
    WriteBpe();
    BpeTokenizer tokenizer;
    LoadBpe(tokenizer);
    EXPECT_EQ(tokenizer.Encode("hello"), (std::vector<int>{0, 10, 1}));
    EXPECT_EQ(tokenizer.Encode(""), (std::vector<int>{0, 1}));
    EXPECT_EQ(EncodeContent(tokenizer, "Q"), (std::vector<int>{2}));
    EXPECT_EQ(tokenizer.Decode({0, 10, 2, 1}), "<bos>hello<unk><eos>");
    EXPECT_THROW(tokenizer.DecodeToken(-1), std::out_of_range);
    EXPECT_THROW(tokenizer.DecodeToken(17), std::out_of_range);
    EXPECT_THROW(tokenizer.Decode({tokenizer.GetVocabSize()}), std::out_of_range);
}

TEST_F(TokenizerFiles, BpeAcceptsEntryArrayAndHonorsExplicitSpecialFlags) {
    WriteBpe(R"json([
        {"id":0,"context":"<bos>"},{"id":1,"context":"<eos>"},
        {"id":2,"context":"<unk>"},{"id":3,"context":"a"},
        {"id":4,"context":"\u4E2D","special":true}
    ])json", "", R"({"bos_id":0,"eos_id":1,"unk_id":2})");
    BpeTokenizer tokenizer;
    LoadBpe(tokenizer);
    EXPECT_EQ(EncodeContent(tokenizer, u8"\u4E2Da"), (std::vector<int>{4, 3}));
    EXPECT_EQ(tokenizer.Decode({4, 3}), u8"\u4E2Da");
}

TEST_F(TokenizerFiles, BpeFailedReloadPreservesPreviouslyUsableState) {
    WriteBpe();
    BpeTokenizer tokenizer;
    LoadBpe(tokenizer);
    for (const auto* file : {"vocab.json", "merges.txt", "special.json"}) {
        SCOPED_TRACE(file);
        WriteBpe();
        Write(file, "invalid");
        EXPECT_THROW(tokenizer.Load(Path("vocab.json"), Path("merges.txt"),
                                    Path("special.json")), std::runtime_error);
        EXPECT_EQ(EncodeContent(tokenizer, "hello<tag>long"),
                  (std::vector<int>{10, 33}));
        EXPECT_EQ(tokenizer.Decode({0, 10, 1}), "<bos>hello<eos>");
        EXPECT_EQ(tokenizer.GetVocabSize(), 35);
    }
}

TEST_F(TokenizerFiles, BpeSuccessfulReloadReplacesOldVocabularyMergesAndSpecials) {
    WriteBpe();
    BpeTokenizer tokenizer;
    LoadBpe(tokenizer);
    WriteBpe(R"({"<b2>":100,"<e2>":101,"<u2>":102,"a":10,"b":11,"ab":12,"<|new|>":13})",
             "", R"({"bos_id":100,"eos_id":101,"unk_id":102})");
    LoadBpe(tokenizer);
    EXPECT_EQ(tokenizer.GetVocabSize(), 103);
    EXPECT_EQ(tokenizer.GetBOSId(), 100);
    EXPECT_EQ(tokenizer.GetEOSId(), 101);
    EXPECT_EQ(tokenizer.GetUNKId(), 102);
    EXPECT_EQ(EncodeContent(tokenizer, "ab"), (std::vector<int>{10, 11}));
    EXPECT_EQ(EncodeContent(tokenizer, "h"), (std::vector<int>{102}));
    EXPECT_EQ(EncodeContent(tokenizer, "<|new|>"), (std::vector<int>{13}));
    EXPECT_THROW(tokenizer.DecodeToken(34), std::out_of_range);
    EXPECT_THROW(tokenizer.DecodeToken(0), std::out_of_range);
    EXPECT_EQ(tokenizer.Decode({100, 10, 11, 101}), "<b2>ab<e2>");
}

TEST_F(TokenizerFiles, BpeRejectsInvalidVocabularyBeforeUsingIdsOrByteSymbols) {
    for (const auto* vocab : {
        R"({})", R"([])", R"({"<bos>":0,"<eos>":1,"<unk>":2,"a":-1})",
        R"({"<bos>":0,"<eos>":1,"<unk>":2,"a":2147483647})",
        R"({"<bos>":0,"<eos>":1,"<unk>":2,"a":3.5})",
        R"({"<bos>":0,"<eos>":1,"<unk>":2,"a":3,"b":3})",
        R"({"<bos>":0,"<eos>":1,"<unk>":2,"a":3,"a":4})",
        R"({"<bos>":0,"<eos>":1,"<unk>":2,"":3})",
        R"({"<bos>":0,"<eos>":1,"<unk>":2,"\u4E2D":3})",
        R"({"<bos>":0,"<eos>":1,"<unk>":2,"\uD83D":3})"
    }) {
        SCOPED_TRACE(vocab);
        WriteBpe(vocab);
        BpeTokenizer tokenizer;
        EXPECT_THROW(tokenizer.Load(Path("vocab.json"), Path("merges.txt"),
                                    Path("special.json")), std::runtime_error);
    }
}

TEST_F(TokenizerFiles, BpeRejectsMissingInvalidOrConflictingSpecialIds) {
    for (const auto* specials : {
        R"({})", R"({"bos_id":0,"eos_id":1})",
        R"({"bos_id":0,"eos_id":1,"unk_id":999})",
        R"({"bos_id":0,"eos_id":0,"unk_id":2})",
        R"({"bos_id":0,"eos_id":1,"unk_id":-1})",
        R"({"bos_id":true,"eos_id":1,"unk_id":2})",
        R"({"bos_id":0,"eos_id":1,"unk_id":2,"special_tokens":{}})",
        R"({"bos_id":0,"eos_id":1,"unk_id":2,"special_tokens":[3]})",
        R"({"bos_id":0,"eos_id":1,"unk_id":2,"special_tokens":["missing"]})"
    }) {
        SCOPED_TRACE(specials);
        WriteBpe(kBpeVocabulary, kBpeMerges, specials);
        BpeTokenizer tokenizer;
        EXPECT_THROW(tokenizer.Load(Path("vocab.json"), Path("merges.txt"),
                                    Path("special.json")), std::runtime_error);
    }
}

TEST_F(TokenizerFiles, BpeRejectsMalformedDuplicateOrImpossibleMergeRules) {
    for (const auto* merges : {
        "h", "h e extra", "h e\nh e\n", "h missing", "h o", "<bos> h"
    }) {
        SCOPED_TRACE(merges);
        WriteBpe(kBpeVocabulary, merges);
        BpeTokenizer tokenizer;
        EXPECT_THROW(tokenizer.Load(Path("vocab.json"), Path("merges.txt"),
                                    Path("special.json")), std::runtime_error);
    }
}

TEST_F(TokenizerFiles, FactorySelectsRequestedTokenizerAndNeverHidesLoadFailure) {
    EXPECT_NE(dynamic_cast<AsciiTokenizer*>(GetTonizer("").get()), nullptr);
    const auto json = GetTonizer(JsonPath());
    EXPECT_NE(dynamic_cast<JsonVocabTokenizer*>(json.get()), nullptr);
    EXPECT_EQ(EncodeContent(*json, "ab"), (std::vector<int>{3}));
    WriteBpe();
    const auto bpe = GetTonizer(Path("vocab.json"), Path("merges.txt"), Path("special.json"));
    EXPECT_NE(dynamic_cast<BpeTokenizer*>(bpe.get()), nullptr);
    EXPECT_EQ(EncodeContent(*bpe, "hello"), (std::vector<int>{10}));
    EXPECT_THROW(GetTonizer(Path("missing.json")), std::runtime_error);
    EXPECT_THROW(GetTonizer("", Path("merges.txt"), Path("special.json")),
                 std::invalid_argument);
    for (const auto* file : {"vocab.json", "merges.txt", "special.json"}) {
        SCOPED_TRACE(file);
        WriteBpe();
        Write(file, "invalid");
        EXPECT_THROW(GetTonizer(Path("vocab.json"), Path("merges.txt"),
                                Path("special.json")), std::runtime_error);
    }
}

}  // namespace
}  // namespace yan_lamma
