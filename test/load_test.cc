#include "include/load.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace yan_lamma {
namespace {

struct FixtureTensor {
    std::vector<int> shape;
    std::vector<float> values;
};

void ReplaceFirst(std::string& text, const std::string& from,
                  const std::string& to) {
    const size_t pos = text.find(from);
    if (pos == std::string::npos) {
        throw std::logic_error("test fixture replacement not found: " + from);
    }
    text.replace(pos, from.size(), to);
}

class LoadTest : public ::testing::Test {
protected:
    void SetUp() override {
        static unsigned sequence = 0;
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        for (unsigned attempt = 0; attempt < 32; ++attempt) {
            const auto candidate = std::filesystem::temp_directory_path() /
                ("yan-minillama-load-test-" + std::to_string(stamp) + "-" +
                 std::to_string(sequence++));
            if (std::filesystem::create_directory(candidate)) {
                directory_ = candidate;
                break;
            }
        }
        ASSERT_FALSE(directory_.empty());
        BuildWeights();
    }

    void TearDown() override {
        if (!directory_.empty()) {
            std::error_code error;
            std::filesystem::remove_all(directory_, error);
        }
    }

    void AddTensor(const std::string& name, const std::vector<int>& shape,
                   bool gaps) {
        if (gaps && !tensors_.empty()) {
            bytes_.insert(bytes_.end(), 8, static_cast<char>(0xCA));
        }
        size_t count = 1;
        std::ostringstream dimensions;
        for (size_t i = 0; i < shape.size(); ++i) {
            count *= static_cast<size_t>(shape[i]);
            if (i) dimensions << ',';
            dimensions << shape[i];
        }
        FixtureTensor expected{shape, std::vector<float>(count)};
        for (size_t i = 0; i < count; ++i) {
            expected.values[i] = static_cast<float>(expected_.size() + 1) / 2.0f +
                                 static_cast<float>(i) / 64.0f;
        }
        const size_t offset = bytes_.size();
        const size_t byte_size = count * sizeof(float);
        bytes_.resize(offset + byte_size);
        std::memcpy(bytes_.data() + offset, expected.values.data(), byte_size);
        tensors_.push_back("{\"name\":\"" + name + "\",\"shape\":[" +
            dimensions.str() + "],\"dtype\":\"float32\",\"offset\":" +
            std::to_string(offset) + ",\"byte_size\":" +
            std::to_string(byte_size) + "}");
        expected_.push_back(std::move(expected));
    }

    void BuildWeights(bool gaps = false) {
        bytes_.assign(gaps ? 16 : 0, static_cast<char>(0xCA));
        tensors_.clear();
        expected_.clear();
        AddTensor("token_embedding", {16, 8}, gaps);
        AddTensor("final_norm", {8}, gaps);
        AddTensor("lm_head", {16, 8}, gaps);
        for (int layer = 0; layer < 2; ++layer) {
            const std::string prefix = "layers." + std::to_string(layer) + ".";
            AddTensor(prefix + "attention_norm", {8}, gaps);
            AddTensor(prefix + "wq", {8, 8}, gaps);
            AddTensor(prefix + "wk", {4, 8}, gaps);
            AddTensor(prefix + "wv", {4, 8}, gaps);
            AddTensor(prefix + "wo", {8, 8}, gaps);
            AddTensor(prefix + "ffn_norm", {8}, gaps);
            AddTensor(prefix + "w_gate", {12, 8}, gaps);
            AddTensor(prefix + "w_up", {12, 8}, gaps);
            AddTensor(prefix + "w_down", {8, 12}, gaps);
        }
    }

    std::string WriteText(const std::string& text) {
        const auto path = directory_ / "model.json";
        std::ofstream file(path);
        file << text;
        if (!file.good()) throw std::runtime_error("cannot write test JSON");
        return path.string();
    }

    std::string WriteManifest(bool nested = true, bool tokenizer = true) {
        std::ostringstream text;
        text << '{';
        if (nested) text << "\"config\":" << config_;
        else text << config_.substr(1, config_.size() - 2);
        if (tokenizer) text << ",\"tokenizer\":" << tokenizer_;
        text << ",\"tensors\":[";
        for (size_t i = 0; i < tensors_.size(); ++i) {
            if (i) text << ',';
            text << tensors_[i];
        }
        text << "]}";
        return WriteText(text.str());
    }

    std::string WriteBinary() {
        const auto path = directory_ / "model.bin";
        std::ofstream file(path, std::ios::binary);
        if (!bytes_.empty()) {
            file.write(bytes_.data(), static_cast<std::streamsize>(bytes_.size()));
        }
        if (!file.good()) throw std::runtime_error("cannot write test BIN");
        return path.string();
    }

    void ExpectManifestError(const std::string& message) {
        const std::string path = WriteManifest();
        try {
            (void)ParseModelJson(path);
            FAIL() << "invalid manifest was accepted";
        } catch (const std::exception& error) {
            EXPECT_NE(std::string(error.what()).find(message), std::string::npos)
                << error.what();
        }
    }

    void ExpectUnloaded(const Model& model) {
        EXPECT_FALSE(model.is_loaded);
        EXPECT_FALSE(model.load_error_.empty());
        EXPECT_TRUE(model.embedding_data_.data.empty());
        EXPECT_TRUE(model.n_layers_.empty());
        EXPECT_TRUE(model.final_norm.data.empty());
        EXPECT_TRUE(model.lm_head_.kF32_data.empty());
    }

    void ExpectPlainWeight(const Tensor& actual, size_t index) {
        EXPECT_EQ(actual.shape, expected_[index].shape);
        EXPECT_EQ(actual.data, expected_[index].values);
    }

    void ExpectQuantizedWeight(const QuanTensor& actual, size_t index) {
        EXPECT_EQ(actual.type, QuantType::kF32);
        EXPECT_EQ(actual.shape, expected_[index].shape);
        EXPECT_EQ(actual.kF32_data, expected_[index].values);
    }

    void ExpectAllWeights(const Model& model) {
        ASSERT_TRUE(model.is_loaded) << model.load_error_;
        EXPECT_TRUE(model.load_error_.empty());
        EXPECT_EQ(model.config.vocab_size, 16);
        EXPECT_EQ(model.config.dim, 8);
        EXPECT_EQ(model.config.hidden_dim, 12);
        EXPECT_EQ(model.config.layers, 2);
        EXPECT_EQ(model.config.n_heads, 2);
        EXPECT_EQ(model.config.n_kv_heads, 1);
        EXPECT_EQ(model.config.head_dim, 4);
        EXPECT_EQ(model.config.max_seq_len, 16);
        ExpectPlainWeight(model.embedding_data_, 0);
        ExpectPlainWeight(model.final_norm, 1);
        ExpectQuantizedWeight(model.lm_head_, 2);
        ASSERT_EQ(model.n_layers_.size(), 2u);
        for (size_t i = 0; i < model.n_layers_.size(); ++i) {
            SCOPED_TRACE("layer " + std::to_string(i));
            const Layer& layer = model.n_layers_[i];
            const size_t first = 3 + i * 9;
            ExpectPlainWeight(layer.attention_norm, first);
            ExpectQuantizedWeight(layer.wq, first + 1);
            ExpectQuantizedWeight(layer.wk, first + 2);
            ExpectQuantizedWeight(layer.wv, first + 3);
            ExpectQuantizedWeight(layer.wo, first + 4);
            ExpectPlainWeight(layer.ffn_norm, first + 5);
            ExpectQuantizedWeight(layer.w_gate, first + 6);
            ExpectQuantizedWeight(layer.w_up, first + 7);
            ExpectQuantizedWeight(layer.w_down, first + 8);
        }
    }

    std::filesystem::path directory_;
    std::string config_ = R"({"vocab_size":16,"dim":8,"hidden_dim":12,"layers":2,"n_heads":2,"n_kv_heads":1,"head_dim":4,"max_seq_len":16,"rope_theta":10000,"rms_norm_eps":0.00001})";
    std::string tokenizer_ = R"({"type":"ASCII","pathname":"","bos_id":3,"eos_id":4,"unk_id":5})";
    std::vector<std::string> tensors_;
    std::vector<FixtureTensor> expected_;
    std::vector<char> bytes_;
};

TEST_F(LoadTest, ParsesNestedConfigTokenizerAndTensorMetadata) {
    const ModelJson manifest = ParseModelJson(WriteManifest());
    EXPECT_EQ(manifest.config.vocab_size, 16);
    EXPECT_EQ(manifest.config.dim, 8);
    EXPECT_EQ(manifest.config.hidden_dim, 12);
    EXPECT_EQ(manifest.config.layers, 2);
    EXPECT_EQ(manifest.config.n_heads, 2);
    EXPECT_EQ(manifest.config.n_kv_heads, 1);
    EXPECT_EQ(manifest.config.head_dim, 4);
    EXPECT_EQ(manifest.config.max_seq_len, 16);
    EXPECT_FLOAT_EQ(manifest.config.rope_theta, 10000.0f);
    EXPECT_FLOAT_EQ(manifest.config.rms_norm_eps, 0.00001f);
    EXPECT_EQ(manifest.toninzer_json.type, "ASCII");
    EXPECT_EQ(manifest.toninzer_json.bos_id, 3);
    EXPECT_EQ(manifest.toninzer_json.eos_id, 4);
    EXPECT_EQ(manifest.toninzer_json.unk_id, 5);
    ASSERT_EQ(manifest.tensor_json.size(), expected_.size());
    size_t offset = 0;
    for (size_t i = 0; i < manifest.tensor_json.size(); ++i) {
        SCOPED_TRACE(i);
        const auto& tensor = manifest.tensor_json[i];
        EXPECT_EQ(tensor.shape, expected_[i].shape);
        EXPECT_EQ(tensor.dtype, "float32");
        EXPECT_EQ(tensor.offset, offset);
        EXPECT_EQ(tensor.byte_size, expected_[i].values.size() * sizeof(float));
        offset += tensor.byte_size;
    }
    EXPECT_EQ(manifest.tensor_json.front().tensor_name, "token_embedding");
    EXPECT_EQ(manifest.tensor_json.back().tensor_name, "layers.1.w_down");
}

TEST_F(LoadTest, AcceptsFlatConfig) {
    const auto manifest = ParseModelJson(WriteManifest(false));
    EXPECT_EQ(manifest.config.layers, 2);
    EXPECT_EQ(manifest.config.dim, 8);
    EXPECT_EQ(manifest.tensor_json.size(), 21u);
}

TEST_F(LoadTest, DefaultsToAsciiWhenTokenizerIsOmitted) {
    const auto manifest = ParseModelJson(WriteManifest(true, false));
    EXPECT_EQ(manifest.toninzer_json.type, "ASCII");
    EXPECT_TRUE(manifest.toninzer_json.path_name.empty());
    EXPECT_EQ(manifest.toninzer_json.bos_id, 128);
    EXPECT_EQ(manifest.toninzer_json.eos_id, 129);
    EXPECT_EQ(manifest.toninzer_json.unk_id, 130);
}

TEST_F(LoadTest, ParsesJsonVocabPathAndDefaultSpecialIdsWithoutLoadingVocab) {
    tokenizer_ = R"({"type":"JsonVocab","pathname":"vocab.json"})";
    const auto manifest = ParseModelJson(WriteManifest());
    EXPECT_EQ(manifest.toninzer_json.type, "JsonVocab");
    EXPECT_EQ(manifest.toninzer_json.path_name, "vocab.json");
    EXPECT_EQ(manifest.toninzer_json.bos_id, 1);
    EXPECT_EQ(manifest.toninzer_json.eos_id, 2);
    EXPECT_EQ(manifest.toninzer_json.unk_id, 0);
    EXPECT_FALSE(std::filesystem::exists(directory_ / "vocab.json"));
}

TEST_F(LoadTest, RejectsFewerThanThreeGlobalTensorEntries) {
    tensors_.resize(2);
    ExpectManifestError("missing required tensors");
}

TEST_F(LoadTest, RejectsGlobalEntriesWithoutLayerWeights) {
    tensors_.resize(3);
    ExpectManifestError("missing required tensors");
}

TEST_F(LoadTest, RejectsOneMissingEntryForConfiguredLayers) {
    tensors_.pop_back();
    ExpectManifestError("missing required tensors");
}

TEST_F(LoadTest, RejectsExtremeLayerCountBeforeGeneratingExpectedShapes) {
    ReplaceFirst(config_, "\"layers\":2", "\"layers\":" +
                 std::to_string(std::numeric_limits<int>::max()));
    ExpectManifestError("missing required tensors");
}

TEST_F(LoadTest, RejectsEmptyTensorList) {
    tensors_.clear();
    EXPECT_THROW(ParseModelJson(WriteManifest()), std::exception);
}

TEST_F(LoadTest, ChecksRequiredNamesEvenWhenEntryCountIsEnough) {
    ReplaceFirst(tensors_.front(), "token_embedding", "unknown_weight");
    EXPECT_THROW(ParseModelJson(WriteManifest()), std::exception);
}

TEST_F(LoadTest, RejectsDuplicateTensorNames) {
    ReplaceFirst(tensors_[1], "final_norm", "token_embedding");
    EXPECT_THROW(ParseModelJson(WriteManifest()), std::exception);
}

TEST_F(LoadTest, RejectsShapeMismatchWithSameElementCount) {
    ReplaceFirst(tensors_.front(), "[16,8]", "[8,16]");
    EXPECT_THROW(ParseModelJson(WriteManifest()), std::exception);
}

TEST_F(LoadTest, RejectsUnsupportedDtype) {
    ReplaceFirst(tensors_.front(), "float32", "float16");
    EXPECT_THROW(ParseModelJson(WriteManifest()), std::exception);
}

TEST_F(LoadTest, RejectsByteSizeDifferentFromShapeTimesFloatSize) {
    ReplaceFirst(tensors_.front(), "\"byte_size\":512", "\"byte_size\":508");
    EXPECT_THROW(ParseModelJson(WriteManifest()), std::exception);
}

TEST_F(LoadTest, RejectsOverlappingTensorByteRanges) {
    ReplaceFirst(tensors_[1], "\"offset\":512", "\"offset\":508");
    EXPECT_THROW(ParseModelJson(WriteManifest()), std::exception);
}

TEST_F(LoadTest, RejectsElementCountOverflowWithoutAllocatingTensor) {
    const std::string dimension = std::to_string(std::numeric_limits<int>::max());
    ReplaceFirst(tensors_.front(), "[16,8]",
                 "[" + dimension + "," + dimension + "," + dimension + "]");
    ExpectManifestError("tensor element count overflow");
}

TEST_F(LoadTest, RejectsFloatByteSizeOverflowWithoutAllocatingTensor) {
    const std::string shape = sizeof(size_t) == 8
        ? "[1073741824,1073741824,4]" : "[1073741824]";
    ReplaceFirst(tensors_.front(), "[16,8]", shape);
    ExpectManifestError("tensor byte_size overflow");
}

TEST_F(LoadTest, RejectsOffsetPlusByteSizeOverflow) {
    ReplaceFirst(tensors_.front(), "\"offset\":0", "\"offset\":" +
                 std::to_string(std::numeric_limits<size_t>::max() - 511));
    ExpectManifestError("byte_size overflow");
}

struct InvalidField {
    std::string name;
    std::string before;
    std::string after;
};

class InvalidConfigTest : public LoadTest,
                          public ::testing::WithParamInterface<InvalidField> {};

TEST_P(InvalidConfigTest, RejectsInvalidConfiguration) {
    ReplaceFirst(config_, GetParam().before, GetParam().after);
    EXPECT_THROW(ParseModelJson(WriteManifest()), std::exception);
}

INSTANTIATE_TEST_SUITE_P(ModelValidation, InvalidConfigTest, ::testing::Values(
    InvalidField{"ZeroVocab", "\"vocab_size\":16", "\"vocab_size\":0"},
    InvalidField{"NegativeDimension", "\"dim\":8", "\"dim\":-8"},
    InvalidField{"ZeroHiddenDimension", "\"hidden_dim\":12", "\"hidden_dim\":0"},
    InvalidField{"ZeroLayers", "\"layers\":2", "\"layers\":0"},
    InvalidField{"ZeroQueryHeads", "\"n_heads\":2", "\"n_heads\":0"},
    InvalidField{"ZeroKvHeads", "\"n_kv_heads\":1", "\"n_kv_heads\":0"},
    InvalidField{"ZeroHeadDimension", "\"head_dim\":4", "\"head_dim\":0"},
    InvalidField{"ZeroSequenceLength", "\"max_seq_len\":16", "\"max_seq_len\":0"},
    InvalidField{"DimensionNotDivisibleByHeads", "\"dim\":8", "\"dim\":9"},
    InvalidField{"QueryHeadsNotDivisibleByKvHeads", "\"n_kv_heads\":1", "\"n_kv_heads\":3"},
    InvalidField{"OddHeadDimension", "\"head_dim\":4", "\"head_dim\":3"},
    InvalidField{"InconsistentHeadDimension", "\"head_dim\":4", "\"head_dim\":6"},
    InvalidField{"ZeroRopeTheta", "\"rope_theta\":10000", "\"rope_theta\":0"},
    InvalidField{"ZeroNormEpsilon", "\"rms_norm_eps\":0.00001", "\"rms_norm_eps\":0"},
    InvalidField{"NonfiniteTheta", "\"rope_theta\":10000", "\"rope_theta\":nan"},
    InvalidField{"NonfiniteEpsilon", "\"rms_norm_eps\":0.00001", "\"rms_norm_eps\":inf"},
    InvalidField{"FractionalLayers", "\"layers\":2", "\"layers\":2.5"},
    InvalidField{"NumericSuffix", "\"max_seq_len\":16", "\"max_seq_len\":16junk"},
    InvalidField{"MissingHeadDimension", "\"head_dim\":4,", ""}
), [](const ::testing::TestParamInfo<InvalidField>& info) { return info.param.name; });

class InvalidShapeTest : public LoadTest,
                         public ::testing::WithParamInterface<InvalidField> {};

TEST_P(InvalidShapeTest, RejectsInvalidShapeDimensionSyntax) {
    ReplaceFirst(tensors_.front(), "[16,8]", GetParam().after);
    EXPECT_THROW(ParseModelJson(WriteManifest()), std::exception);
}

INSTANTIATE_TEST_SUITE_P(TensorValidation, InvalidShapeTest, ::testing::Values(
    InvalidField{"Empty", "", "[]"},
    InvalidField{"Zero", "", "[16,0]"},
    InvalidField{"Negative", "", "[16,-8]"},
    InvalidField{"Fractional", "", "[16,8.5]"},
    InvalidField{"MissingDimension", "", "[16,,8]"},
    InvalidField{"TrailingComma", "", "[16,8,]"},
    InvalidField{"LeadingZero", "", "[016,8]"},
    InvalidField{"NumericSuffix", "", "[16,8x]"},
    InvalidField{"NotArray", "", "16"},
    InvalidField{"UnclosedArray", "", "[16,8"}
), [](const ::testing::TestParamInfo<InvalidField>& info) { return info.param.name; });

class MissingTensorFieldTest : public LoadTest,
                              public ::testing::WithParamInterface<InvalidField> {};

TEST_P(MissingTensorFieldTest, RejectsMissingRequiredTensorMetadata) {
    ReplaceFirst(tensors_.front(), GetParam().before, "");
    EXPECT_THROW(ParseModelJson(WriteManifest()), std::exception);
}

INSTANTIATE_TEST_SUITE_P(TensorValidation, MissingTensorFieldTest, ::testing::Values(
    InvalidField{"Name", "\"name\":\"token_embedding\",", ""},
    InvalidField{"Shape", "\"shape\":[16,8],", ""},
    InvalidField{"Dtype", "\"dtype\":\"float32\",", ""},
    InvalidField{"Offset", "\"offset\":0,", ""},
    InvalidField{"ByteSize", ",\"byte_size\":512", ""}
), [](const ::testing::TestParamInfo<InvalidField>& info) { return info.param.name; });

class InvalidTensorNumberTest : public LoadTest,
                               public ::testing::WithParamInterface<InvalidField> {};

TEST_P(InvalidTensorNumberTest, RejectsInvalidOffsetOrByteSize) {
    ReplaceFirst(tensors_.front(), GetParam().before, GetParam().after);
    EXPECT_THROW(ParseModelJson(WriteManifest()), std::exception);
}

INSTANTIATE_TEST_SUITE_P(TensorValidation, InvalidTensorNumberTest, ::testing::Values(
    InvalidField{"NegativeOffset", "\"offset\":0", "\"offset\":-1"},
    InvalidField{"FractionalOffset", "\"offset\":0", "\"offset\":0.5"},
    InvalidField{"OffsetOutOfRange", "\"offset\":0", "\"offset\":18446744073709551616"},
    InvalidField{"NegativeByteSize", "\"byte_size\":512", "\"byte_size\":-1"},
    InvalidField{"FractionalByteSize", "\"byte_size\":512", "\"byte_size\":512.5"},
    InvalidField{"ByteSizeOutOfRange", "\"byte_size\":512", "\"byte_size\":18446744073709551616"}
), [](const ::testing::TestParamInfo<InvalidField>& info) { return info.param.name; });

class InvalidTokenizerTest : public LoadTest,
                            public ::testing::WithParamInterface<InvalidField> {};

TEST_P(InvalidTokenizerTest, RejectsInvalidTokenizerMetadata) {
    ReplaceFirst(tokenizer_, GetParam().before, GetParam().after);
    EXPECT_THROW(ParseModelJson(WriteManifest()), std::exception);
}

INSTANTIATE_TEST_SUITE_P(TokenizerValidation, InvalidTokenizerTest, ::testing::Values(
    InvalidField{"UnknownType", "ASCII", "Unknown"},
    InvalidField{"MissingJsonVocabPath", "ASCII", "JsonVocab"},
    InvalidField{"NegativeSpecialId", "\"bos_id\":3", "\"bos_id\":-1"},
    InvalidField{"DuplicateSpecialIds", "\"eos_id\":4", "\"eos_id\":3"},
    InvalidField{"FractionalSpecialId", "\"unk_id\":5", "\"unk_id\":5.5"}
), [](const ::testing::TestParamInfo<InvalidField>& info) { return info.param.name; });

TEST_F(LoadTest, LoadsEveryWeightIntoTheCorrespondingModelMember) {
    const auto json = WriteManifest();
    const auto bin = WriteBinary();
    ExpectAllWeights(ParseModel(json, bin));
}

TEST_F(LoadTest, LoadsByNameAndOffsetRegardlessOfManifestOrder) {
    std::reverse(tensors_.begin(), tensors_.end());
    const auto json = WriteManifest();
    const auto bin = WriteBinary();
    ExpectAllWeights(ParseModel(json, bin));
}

TEST_F(LoadTest, AllowsNonOverlappingGapsAndReadsExactTensorRanges) {
    BuildWeights(true);
    const auto json = WriteManifest();
    const auto bin = WriteBinary();
    ExpectAllWeights(ParseModel(json, bin));
}

TEST_F(LoadTest, ReportsMissingBinaryWithoutReturningPartialWeights) {
    ExpectUnloaded(ParseModel(WriteManifest(), (directory_ / "missing.bin").string()));
}

TEST_F(LoadTest, ReportsTruncatedBinaryWithoutReturningPartialWeights) {
    bytes_.pop_back();
    const auto json = WriteManifest();
    const auto bin = WriteBinary();
    ExpectUnloaded(ParseModel(json, bin));
}

TEST_F(LoadTest, ReportsEmptyBinaryWithoutReturningPartialWeights) {
    bytes_.clear();
    const auto json = WriteManifest();
    const auto bin = WriteBinary();
    ExpectUnloaded(ParseModel(json, bin));
}

TEST_F(LoadTest, ReportsTrailingBinaryBytesWithoutReturningPartialWeights) {
    bytes_.push_back('\0');
    const auto json = WriteManifest();
    const auto bin = WriteBinary();
    ExpectUnloaded(ParseModel(json, bin));
}

TEST_F(LoadTest, ConvertsInvalidManifestExceptionToFailedModel) {
    ReplaceFirst(tensors_.front(), "float32", "float16");
    const auto json = WriteManifest();
    const auto bin = WriteBinary();
    EXPECT_NO_THROW(ExpectUnloaded(ParseModel(json, bin)));
}

TEST_F(LoadTest, RejectsEmptyOrMissingConfigPath) {
    EXPECT_THROW(ParseModelJson(""), std::exception);
    const auto missing = (directory_ / "missing.json").string();
    EXPECT_THROW(ParseModelJson(missing), std::exception);
    ExpectUnloaded(ParseModel(missing, WriteBinary()));
}

TEST_F(LoadTest, RejectsEmptyOrNonObjectConfigFile) {
    EXPECT_THROW(ParseModelJson(WriteText(" \n\t")), std::exception);
    EXPECT_THROW(ParseModelJson(WriteText("[]")), std::exception);
}

TEST_F(LoadTest, InspectPrintsMetadataWithoutOpeningBinaryOrVocab) {
    tokenizer_ = R"({"type":"JsonVocab","pathname":"vocab.json"})";
    const auto path = WriteManifest();
    ASSERT_FALSE(std::filesystem::exists(directory_ / "model.bin"));
    ASSERT_FALSE(std::filesystem::exists(directory_ / "vocab.json"));
    ::testing::internal::CaptureStdout();
    const bool success = InspectModel(path);
    const std::string output = ::testing::internal::GetCapturedStdout();
    EXPECT_TRUE(success);
    for (const char* field : {"vocab_size: 16", "layers: 2", "head_dim: 4",
                                    "token_embedding", "shape: [16, 8]",
                                    "offset: 0", "byte_size: 512", "layers.1.w_down",
                                    "type: JsonVocab", "pathname: vocab.json"}) {
        EXPECT_NE(output.find(field), std::string::npos) << field;
    }
}

TEST_F(LoadTest, InspectReturnsFalseForMissingOrInvalidManifest) {
    EXPECT_FALSE(InspectModel((directory_ / "missing.json").string()));
    tensors_.clear();
    EXPECT_FALSE(InspectModel(WriteManifest()));
}

}  // namespace
}  // namespace yan_lamma
