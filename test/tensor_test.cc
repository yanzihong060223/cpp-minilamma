#include "include/tensor.h"

#include <gtest/gtest.h>

#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace yan_lamma {
namespace {

TEST(TensorTest, DefaultConstructionIsEmpty) {
    const Tensor tensor;
    EXPECT_TRUE(tensor.data.empty());
    EXPECT_TRUE(tensor.shape.empty());
    EXPECT_EQ(tensor.GetDataSize(), 0u);
    EXPECT_EQ(tensor.GetNumDims(), 0u);
    EXPECT_EQ(tensor.ShapeToString(), "[]");
}

TEST(TensorTest, ConstructionPreservesShapeAndFillsEveryElement) {
    const Tensor tensor({2, 3}, 1.25f);
    EXPECT_EQ(tensor.shape, (std::vector<int>{2, 3}));
    EXPECT_EQ(tensor.GetNumDims(), 2u);
    EXPECT_EQ(tensor.GetDataSize(), 6u);
    EXPECT_EQ(tensor.data, std::vector<float>(6, 1.25f));
    EXPECT_EQ(Tensor({3}).data, std::vector<float>(3, 0.0f));
}

TEST(TensorTest, ConstructionRejectsNonPositiveDimensions) {
    EXPECT_THROW(Tensor({0}), std::runtime_error);
    EXPECT_THROW(Tensor({2, 0}), std::runtime_error);
    EXPECT_THROW(Tensor({-1, 3}), std::runtime_error);
}

TEST(TensorTest, ConstructionRejectsElementCountOverflowBeforeAllocating) {
    const int large_dimension = std::numeric_limits<int>::max();
    EXPECT_THROW(Tensor({large_dimension, large_dimension, large_dimension}),
                 std::runtime_error);
}

TEST(TensorTest, OneDimensionalAccessReadsAndWritesStorage) {
    Tensor tensor({3});
    tensor.At1(1) = 7.0f;
    tensor[2] = 9.0f;
    EXPECT_FLOAT_EQ(tensor.data[1], 7.0f);
    EXPECT_FLOAT_EQ(tensor.At1(2), 9.0f);

    const Tensor& constant = tensor;
    EXPECT_FLOAT_EQ(constant.At1(1), 7.0f);
    EXPECT_FLOAT_EQ(constant[2], 9.0f);
}

TEST(TensorTest, TwoDimensionalAccessUsesRowMajorOrder) {
    Tensor tensor({2, 3});
    std::iota(tensor.data.begin(), tensor.data.end(), 0.0f);
    EXPECT_FLOAT_EQ(tensor.At2(0, 2), 2.0f);
    EXPECT_FLOAT_EQ(tensor.At2(1, 0), 3.0f);
    tensor.At2(1, 1) = 42.0f;
    EXPECT_FLOAT_EQ(tensor.data[4], 42.0f);

    const Tensor& constant = tensor;
    EXPECT_FLOAT_EQ(constant.At2(1, 1), 42.0f);
    EXPECT_FLOAT_EQ(constant.At2(1, 2), 5.0f);
}

TEST(TensorTest, ThreeDimensionalAccessUsesRowMajorOrder) {
    Tensor tensor({2, 3, 4});
    std::iota(tensor.data.begin(), tensor.data.end(), 0.0f);
    EXPECT_FLOAT_EQ(tensor.At3(0, 2, 3), 11.0f);
    EXPECT_FLOAT_EQ(tensor.At3(1, 0, 0), 12.0f);
    tensor.At3(1, 1, 2) = 42.0f;
    EXPECT_FLOAT_EQ(tensor.data[18], 42.0f);

    const Tensor& constant = tensor;
    EXPECT_FLOAT_EQ(constant.At3(1, 1, 2), 42.0f);
    EXPECT_FLOAT_EQ(constant.At3(1, 2, 3), 23.0f);
}

TEST(TensorTest, FourDimensionalAccessUsesRowMajorOrder) {
    Tensor tensor({2, 3, 2, 4});
    std::iota(tensor.data.begin(), tensor.data.end(), 0.0f);
    EXPECT_FLOAT_EQ(tensor.At4(0, 2, 1, 3), 23.0f);
    EXPECT_FLOAT_EQ(tensor.At4(1, 0, 0, 0), 24.0f);
    tensor.At4(1, 1, 1, 2) = 42.0f;
    EXPECT_FLOAT_EQ(tensor.data[38], 42.0f);

    const Tensor& constant = tensor;
    EXPECT_FLOAT_EQ(constant.At4(1, 1, 1, 2), 42.0f);
    EXPECT_FLOAT_EQ(constant.At4(1, 2, 1, 3), 47.0f);
}

TEST(TensorTest, CoordinatesMapToOffsetsAndRejectInvalidIndices) {
    const Tensor tensor({2, 3, 4});
    EXPECT_EQ(tensor.SwitchIdex({0, 0, 0}), 0u);
    EXPECT_EQ(tensor.SwitchIdex({0, 1, 0}), 4u);
    EXPECT_EQ(tensor.SwitchIdex({1, 0, 0}), 12u);
    EXPECT_EQ(tensor.SwitchIdex({1, 2, 3}), 23u);

    EXPECT_THROW(tensor.SwitchIdex({}), std::invalid_argument);
    EXPECT_THROW(tensor.SwitchIdex({0, 0}), std::invalid_argument);
    EXPECT_THROW(tensor.SwitchIdex({0, 0, 0, 0}), std::invalid_argument);
    EXPECT_THROW(tensor.SwitchIdex({-1, 0, 0}), std::out_of_range);
    EXPECT_THROW(tensor.SwitchIdex({2, 0, 0}), std::out_of_range);
    EXPECT_THROW(tensor.SwitchIdex({0, 3, 0}), std::out_of_range);
    EXPECT_THROW(tensor.SwitchIdex({0, 0, 4}), std::out_of_range);
}

TEST(TensorTest, AccessorsRejectWrongRankAndOutOfBoundsCoordinates) {
    Tensor one({3});
    Tensor two({2, 3});
    Tensor three({2, 3, 4});
    Tensor four({2, 3, 2, 4});
    const Tensor& constant_one = one;
    const Tensor& constant_two = two;
    const Tensor& constant_three = three;
    const Tensor& constant_four = four;

    EXPECT_THROW(two.At1(0), std::runtime_error);
    EXPECT_THROW(one.At2(0, 0), std::runtime_error);
    EXPECT_THROW(two.At3(0, 0, 0), std::runtime_error);
    EXPECT_THROW(three.At4(0, 0, 0, 0), std::runtime_error);
    EXPECT_THROW(constant_two.At1(0), std::runtime_error);
    EXPECT_THROW(constant_one.At2(0, 0), std::runtime_error);
    EXPECT_THROW(constant_two.At3(0, 0, 0), std::runtime_error);
    EXPECT_THROW(constant_three.At4(0, 0, 0, 0), std::runtime_error);

    EXPECT_THROW(one.At1(-1), std::out_of_range);
    EXPECT_THROW(constant_one.At1(3), std::out_of_range);
    EXPECT_THROW(two.At2(2, 0), std::out_of_range);
    EXPECT_THROW(constant_two.At2(0, 3), std::out_of_range);
    EXPECT_THROW(three.At3(0, -1, 0), std::out_of_range);
    EXPECT_THROW(constant_three.At3(0, 0, 4), std::out_of_range);
    EXPECT_THROW(four.At4(0, 0, 2, 0), std::out_of_range);
    EXPECT_THROW(constant_four.At4(0, 0, 0, 4), std::out_of_range);
}

TEST(TensorTest, RowPointerSharesStorageAndChecksRankAndRow) {
    Tensor tensor({2, 3});
    float* row = tensor.RowPtr(1);
    ASSERT_EQ(row, tensor.data.data() + 3);
    row[2] = 9.0f;
    EXPECT_FLOAT_EQ(tensor.At2(1, 2), 9.0f);

    const Tensor& constant = tensor;
    const float* constant_row = constant.RowPtr(1);
    EXPECT_EQ(constant_row, tensor.data.data() + 3);
    EXPECT_FLOAT_EQ(constant_row[2], 9.0f);
    EXPECT_THROW(tensor.RowPtr(-1), std::out_of_range);
    EXPECT_THROW(constant.RowPtr(2), std::out_of_range);

    Tensor one({3});
    const Tensor& constant_one = one;
    EXPECT_THROW(one.RowPtr(0), std::runtime_error);
    EXPECT_THROW(constant_one.RowPtr(0), std::runtime_error);
}

TEST(TensorTest, ShapeAssertionChecksEveryDimension) {
    Tensor tensor({2, 3});
    EXPECT_NO_THROW(tensor.AssertShape({2, 3}, "tensor test"));
    EXPECT_THROW(tensor.AssertShape({3, 2}, "tensor test"), std::runtime_error);
    EXPECT_THROW(tensor.AssertShape({6}, nullptr), std::runtime_error);
}

TEST(TensorTest, ReshapePreservesElementsAndCreatesIndependentStorage) {
    Tensor source({2, 3});
    source.data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor reshaped = source.Reshape({3, 2}, "tensor test");

    EXPECT_EQ(reshaped.shape, (std::vector<int>{3, 2}));
    EXPECT_EQ(reshaped.data, source.data);
    EXPECT_FLOAT_EQ(reshaped.At2(1, 0), 3.0f);
    EXPECT_EQ(source.shape, (std::vector<int>{2, 3}));

    reshaped.At2(0, 0) = 99.0f;
    EXPECT_FLOAT_EQ(source.At2(0, 0), 1.0f);
    source.At2(1, 2) = 88.0f;
    EXPECT_FLOAT_EQ(reshaped.At2(2, 1), 6.0f);
}

TEST(TensorTest, ReshapeRejectsInvalidShapeAndChangedElementCount) {
    Tensor tensor({2, 3});
    EXPECT_THROW(tensor.Reshape({4, 2}, "tensor test"), std::runtime_error);
    EXPECT_THROW(tensor.Reshape({0, 6}, "tensor test"), std::runtime_error);
    EXPECT_THROW(tensor.Reshape({-1, 6}, nullptr), std::runtime_error);
    EXPECT_EQ(tensor.shape, (std::vector<int>{2, 3}));
    EXPECT_EQ(tensor.GetDataSize(), 6u);
}

TEST(TensorTest, FactoriesCreateRequestedShapeAndFill) {
    Tensor factory;
    const Tensor one = factory.MakeOneD(3, 1.0f);
    const Tensor two = factory.MakeSecondD(2, 3, 2.0f);
    const Tensor three = factory.MakeThirdD(2, 3, 4, 3.0f);
    const Tensor four = factory.MakeFouthD(2, 3, 2, 4, 4.0f);

    EXPECT_EQ(one.shape, (std::vector<int>{3}));
    EXPECT_EQ(one.data, std::vector<float>(3, 1.0f));
    EXPECT_EQ(two.ShapeToString(), "[2, 3]");
    EXPECT_EQ(two.data, std::vector<float>(6, 2.0f));
    EXPECT_EQ(three.shape, (std::vector<int>{2, 3, 4}));
    EXPECT_EQ(three.data, std::vector<float>(24, 3.0f));
    EXPECT_EQ(four.shape, (std::vector<int>{2, 3, 2, 4}));
    EXPECT_EQ(four.data, std::vector<float>(48, 4.0f));
    EXPECT_TRUE(factory.data.empty());
}

}  // namespace
}  // namespace yan_lamma
