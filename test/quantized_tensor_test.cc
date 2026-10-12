#include "include/quantized_tensor.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace yan_lamma {
namespace {

TEST(QuantizedTensorTest, ToQuanTensorPreservesF32ShapeAndValues) {
    Tensor source({2, 3});
    source.data = {-7.25f, 0.0f, 0.125f, 17.0f, 2.5f, -0.5f};

    const QuanTensor converted = ToQuanTensor(source);

    EXPECT_EQ(converted.type, QuantType::kF32);
    EXPECT_EQ(converted.shape, source.shape);
    EXPECT_EQ(converted.kF32_data, source.data);
}

TEST(QuantizedTensorTest, ToTensorPreservesF32ShapeAndValues) {
    QuanTensor source;
    source.shape = {2, 3};
    source.kF32_data = {-7.25f, 0.0f, 0.125f, 17.0f, 2.5f, -0.5f};

    const Tensor converted = ToTensor(source);

    EXPECT_EQ(converted.shape, source.shape);
    EXPECT_EQ(converted.data, source.kF32_data);
}

TEST(QuantizedTensorTest, F32RoundTripsPreserveMultidimensionalTensor) {
    Tensor original({2, 2, 2});
    original.data = {1.0f, -2.0f, 3.25f, 0.0f, -0.125f, 6.5f, 7.0f, -8.0f};

    const QuanTensor wrapped = ToQuanTensor(original);
    const Tensor restored = ToTensor(wrapped);
    const QuanTensor wrapped_again = ToQuanTensor(restored);

    EXPECT_EQ(restored.shape, original.shape);
    EXPECT_EQ(restored.data, original.data);
    EXPECT_EQ(wrapped_again.type, wrapped.type);
    EXPECT_EQ(wrapped_again.shape, wrapped.shape);
    EXPECT_EQ(wrapped_again.kF32_data, wrapped.kF32_data);
}

TEST(QuantizedTensorTest, ConversionsOwnIndependentCopies) {
    Tensor source({2, 2});
    source.data = {1.0f, 2.0f, 3.0f, 4.0f};
    QuanTensor wrapped = ToQuanTensor(source);

    source.data[0] = 20.0f;
    EXPECT_FLOAT_EQ(wrapped.kF32_data[0], 1.0f);
    wrapped.kF32_data[1] = -4.0f;
    EXPECT_FLOAT_EQ(source.data[1], 2.0f);

    Tensor restored = ToTensor(wrapped);
    wrapped.kF32_data[2] = 30.0f;
    EXPECT_FLOAT_EQ(restored.data[2], 3.0f);
    restored.data[3] = 40.0f;
    EXPECT_FLOAT_EQ(wrapped.kF32_data[3], 4.0f);

    source.shape[0] = 3;
    EXPECT_EQ(wrapped.shape, (std::vector<int>{2, 2}));
    wrapped.shape[1] = 1;
    EXPECT_EQ(restored.shape, (std::vector<int>{2, 2}));
}

TEST(QuantizedTensorTest, DefaultEmptyTensorsRemainEmpty) {
    const Tensor empty;
    const QuanTensor wrapped = ToQuanTensor(empty);
    EXPECT_EQ(wrapped.type, QuantType::kF32);
    EXPECT_TRUE(wrapped.shape.empty());
    EXPECT_TRUE(wrapped.kF32_data.empty());

    const Tensor restored = ToTensor(wrapped);
    EXPECT_TRUE(restored.shape.empty());
    EXPECT_TRUE(restored.data.empty());

    const QuanTensor default_quantized;
    const Tensor from_default = ToTensor(default_quantized);
    EXPECT_TRUE(from_default.shape.empty());
    EXPECT_TRUE(from_default.data.empty());
}

TEST(QuantizedTensorTest, ToTensorRejectsNonF32Formats) {
    for (QuantType type : {QuantType::kQ80, QuantType::kQ40, QuantType::kQ41}) {
        SCOPED_TRACE(static_cast<int>(type));
        QuanTensor source;
        source.type = type;
        source.shape = {2};
        source.kF32_data = {1.0f, 2.0f};
        EXPECT_THROW(ToTensor(source), std::runtime_error);
    }
}

TEST(QuantizedTensorTest, ToTensorRejectsShapeAndStorageMismatch) {
    for (std::size_t storage_size : {0u, 5u, 7u}) {
        SCOPED_TRACE(storage_size);
        QuanTensor source;
        source.shape = {2, 3};
        source.kF32_data.resize(storage_size, 1.0f);
        EXPECT_THROW(ToTensor(source), std::runtime_error);
    }

    QuanTensor malformed_scalar;
    malformed_scalar.kF32_data = {1.0f, 2.0f};
    EXPECT_THROW(ToTensor(malformed_scalar), std::runtime_error);
}

TEST(QuantizedTensorTest, ToQuanTensorRejectsShapeAndStorageMismatch) {
    for (std::size_t storage_size : {0u, 5u, 7u}) {
        SCOPED_TRACE(storage_size);
        Tensor source({2, 3});
        source.data.resize(storage_size, 1.0f);
        EXPECT_THROW(ToQuanTensor(source), std::runtime_error);
    }

    Tensor malformed_scalar;
    malformed_scalar.data = {1.0f, 2.0f};
    EXPECT_THROW(ToQuanTensor(malformed_scalar), std::runtime_error);
}

TEST(QuantizedTensorTest, ConversionsRejectNonPositiveDimensions) {
    const std::vector<std::vector<int>> invalid_shapes = {
        {0}, {-1}, {2, 0}, {2, -3}, {std::numeric_limits<int>::min(), 2}
    };
    for (std::size_t index = 0; index < invalid_shapes.size(); ++index) {
        SCOPED_TRACE(index);
        QuanTensor quantized;
        quantized.shape = invalid_shapes[index];
        quantized.kF32_data = {1.0f};
        EXPECT_THROW(ToTensor(quantized), std::runtime_error);

        Tensor plain;
        plain.shape = invalid_shapes[index];
        plain.data = {1.0f};
        EXPECT_THROW(ToQuanTensor(plain), std::runtime_error);
    }
}

TEST(QuantizedTensorTest, ConversionsRejectElementCountOverflowBeforeAllocating) {
    const int large_dimension = std::numeric_limits<int>::max();
    const std::vector<int> overflowing_shape = {
        large_dimension, large_dimension, large_dimension
    };
    // Assign metadata directly so the setup never allocates the requested shape.
    QuanTensor quantized;
    quantized.shape = overflowing_shape;
    quantized.kF32_data = {1.0f};
    try {
        (void)ToTensor(quantized);
        ADD_FAILURE() << "ToTensor accepted an overflowing shape";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("element count overflow"),
                  std::string::npos) << error.what();
    }

    Tensor plain;
    plain.shape = overflowing_shape;
    plain.data = {1.0f};
    try {
        (void)ToQuanTensor(plain);
        ADD_FAILURE() << "ToQuanTensor accepted an overflowing shape";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("element count overflow"),
                  std::string::npos) << error.what();
    }
}

}  // namespace
}  // namespace yan_lamma
