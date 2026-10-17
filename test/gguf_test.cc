#include "include/gguf.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace yan_lamma {
namespace {

class GgufWriteHelper {
public:
    void WriteU32(uint32_t v) {
        char buf[4];
        std::memcpy(buf, &v, 4);
        data_.insert(data_.end(), buf, buf + 4);
    }
    void WriteU64(uint64_t v) {
        char buf[8];
        std::memcpy(buf, &v, 8);
        data_.insert(data_.end(), buf, buf + 8);
    }
    void WriteString(const std::string& s) {
        WriteU64(s.size());
        data_.insert(data_.end(), s.begin(), s.end());
    }
    void WriteFloat(float v) {
        char buf[4];
        std::memcpy(buf, &v, 4);
        data_.insert(data_.end(), buf, buf + 4);
    }
    void WriteRaw(const void* ptr, size_t len) {
        const char* p = static_cast<const char*>(ptr);
        data_.insert(data_.end(), p, p + len);
    }
    const std::vector<char>& data() const { return data_; }
private:
    std::vector<char> data_;
};

class GgufTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        test_dir_ = std::filesystem::temp_directory_path() /
            ("yan-gguf-test-" + std::to_string(stamp));
        std::filesystem::create_directories(test_dir_);
    }

    void TearDown() override {
        if (!test_dir_.empty()) {
            std::error_code ec;
            std::filesystem::remove_all(test_dir_, ec);
        }
    }

    std::string WriteGgufFile(const std::vector<char>& data, const std::string& name = "test.gguf") {
        auto path = test_dir_ / name;
        std::ofstream f(path, std::ios::binary);
        f.write(data.data(), static_cast<std::streamsize>(data.size()));
        return path.string();
    }

    std::vector<char> BuildMinimalGguf(size_t n_metadata = 1, size_t n_tensors = 1) {
        GgufWriteHelper w;
        w.WriteRaw("GGUF", 4);
        w.WriteU32(3);
        w.WriteU64(n_tensors);
        w.WriteU64(n_metadata);

        w.WriteString("test.key");
        w.WriteU32(static_cast<uint32_t>(GgufValueType::kGgufTypeUint32));
        w.WriteU32(12345);

        w.WriteString("test_tensor");
        w.WriteU32(1);
        w.WriteU64(4);
        w.WriteU32(static_cast<uint32_t>(Ggmltype::GgmltypeF32));
        w.WriteU64(0);

        size_t header_size = w.data().size();
        size_t alignment = 32;
        size_t padding = (alignment - (header_size % alignment)) % alignment;
        for (size_t i = 0; i < padding; i++) {
            w.WriteRaw("\0", 1);
        }

        float tensor_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        w.WriteRaw(tensor_data, sizeof(tensor_data));

        return w.data();
    }

    std::filesystem::path test_dir_;
};

TEST_F(GgufTest, RejectsNonExistentFile) {
    GgufRead reader;
    EXPECT_FALSE(reader.load("/nonexistent/path/to/file.gguf"));
}

TEST_F(GgufTest, RejectsEmptyFile) {
    auto path = WriteGgufFile({});
    GgufRead reader;
    EXPECT_FALSE(reader.load(path));
}

TEST_F(GgufTest, RejectsInvalidMagic) {
    std::vector<char> bad_magic = {'B', 'A', 'D', '!'};
    auto path = WriteGgufFile(bad_magic);
    GgufRead reader;
    EXPECT_FALSE(reader.load(path));
}

TEST_F(GgufTest, RejectsWrongVersion) {
    GgufWriteHelper w;
    w.WriteRaw("GGUF", 4);
    w.WriteU32(2);
    w.WriteU64(0);
    w.WriteU64(0);
    auto path = WriteGgufFile(w.data());
    GgufRead reader;
    EXPECT_FALSE(reader.load(path));
}

TEST_F(GgufTest, LoadsMinimalGgufFile) {
    auto data = BuildMinimalGguf();
    auto path = WriteGgufFile(data);
    GgufRead reader;
    bool ok = reader.load(path);
    EXPECT_TRUE(ok) << "load failed";
    EXPECT_EQ(reader.version_, 3u);
    EXPECT_EQ(reader.n_metadata_, 1u);
    EXPECT_EQ(reader.n_tensors_, 1u);
}

TEST_F(GgufTest, InspectPrintsInfo) {
    auto data = BuildMinimalGguf();
    auto path = WriteGgufFile(data);
    GgufRead reader;
    ASSERT_TRUE(reader.load(path));

    ::testing::internal::CaptureStdout();
    bool ok = Inspect(reader);
    std::string output = ::testing::internal::GetCapturedStdout();

    EXPECT_TRUE(ok);
    EXPECT_NE(output.find("GGUF Header"), std::string::npos);
    EXPECT_NE(output.find("version"), std::string::npos);
}

}  // namespace
}  // namespace yan_lamma
