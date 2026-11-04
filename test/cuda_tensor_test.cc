#include "include/cuda_tensor.h"
#include "include/cuda_runtime.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <exception>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace yan_lamma {
namespace {

template <typename Operation>
void ExpectCudaError(Operation operation,
                     std::initializer_list<const char*> fragments) {
    try {
        operation();
        ADD_FAILURE() << "Expected std::runtime_error";
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        for (const char* fragment : fragments) {
            EXPECT_NE(message.find(fragment), std::string::npos) << message;
        }
    }
}

void ExpectEmptyTensor(const CudaTensor& tensor) {
    EXPECT_TRUE(tensor.IsEmpty());
    EXPECT_EQ(tensor.GetData(), nullptr);
    EXPECT_EQ(tensor.NumEle(), 0u);
    EXPECT_EQ(tensor.num_dims(), 0);
    EXPECT_EQ(tensor.Bytes(), 0u);
    EXPECT_EQ(tensor.DeviceId(), 0);
    EXPECT_EQ(tensor.ShapeStringShort(), "[]");
}

Tensor HostValues() {
    Tensor tensor({2, 3});
    tensor.data = {1.0f, -2.0f, 3.5f, 0.0f, 5.0f, -6.25f};
    return tensor;
}

void ExpectValues(const Tensor& actual, const Tensor& expected) {
    EXPECT_EQ(actual.shape, expected.shape);
    EXPECT_EQ(actual.data, expected.data);
}

TEST(CudaTensorTest, DefaultStateAndRepeatedResetAreEmpty) {
    CudaTensor tensor;
    ExpectEmptyTensor(tensor);
    EXPECT_EQ(tensor.GetData(), nullptr);
    EXPECT_NO_THROW(tensor.Reset());
    EXPECT_NO_THROW(tensor.Reset());
    ExpectEmptyTensor(tensor);
}

TEST(CudaTensorTest, RejectsEmptyShapeBeforeUsingCudaRuntime) {
    ExpectCudaError([] { CudaTensor tensor(std::vector<int>{}); },
                    {"CudaTensor::Reset", "shape must not be empty"});
}

TEST(CudaTensorTest, RejectsNonPositiveDimensionsBeforeUsingCudaRuntime) {
    const std::vector<std::vector<int>> shapes = {
        {0}, {-1}, {2, 0}, {2, -3}, {std::numeric_limits<int>::min(), 2}
    };
    for (const auto& shape : shapes) {
        SCOPED_TRACE(::testing::PrintToString(shape));
        ExpectCudaError([&] { CudaTensor tensor(shape); },
                        {"CudaTensor::Reset", "must be positive"});
    }
}

TEST(CudaTensorTest, RejectsElementCountOverflowBeforeUsingCudaRuntime) {
    const int large = std::numeric_limits<int>::max();
    ExpectCudaError([&] { CudaTensor tensor({large, large, large}); },
                    {"CudaTensor::Reset", "element count overflow"});
}

TEST(CudaTensorTest, EmptyTransfersReportOperationContext) {
    CudaTensor tensor;
    Tensor host;
    ExpectCudaError([&] { tensor.UploadFromHost(host); },
                    {"CudaTensor::UploadFromHost", "empty"});
    const CudaTensor& read_only = tensor;
    ExpectCudaError([&] { read_only.DownloadToHost(host); },
                    {"CudaTensor::DownloadToHost", "empty"});
    ExpectCudaError([&] { (void)read_only.Download(); },
                    {"CudaTensor::Download", "empty"});
}

TEST(CudaTensorTest, CpuBuildReportsUnavailableBackendAndKeepsResetTargetEmpty) {
    if (CudaRuntimeBuilt()) {
        GTEST_SKIP() << "This test checks the build without CUDA support";
    }
    ExpectCudaError([] { CudaTensor tensor({2, 3}); },
                    {"not built", "MINI_LLAMA_CUDA=ON"});
    CudaTensor tensor;
    ExpectCudaError([&] { tensor.Reset({2, 3}, 0); },
                    {"not built", "MINI_LLAMA_CUDA=ON"});
    ExpectEmptyTensor(tensor);
}

class CudaTensorGpuTest : public ::testing::Test {
 protected:
    void SetUp() override {
        try {
            (void)GetCudaInfo(0);
        } catch (const std::exception& error) {
            GTEST_SKIP() << "CUDA device 0 is unavailable: " << error.what();
        }
    }
};

TEST_F(CudaTensorGpuTest, HostHelperRoundTripsThroughConstTensor) {
    const Tensor host = HostValues();
    const CudaTensor tensor = TensorHostToDevice(host);
    EXPECT_FALSE(tensor.IsEmpty());
    EXPECT_NE(tensor.GetData(), nullptr);
    EXPECT_EQ(tensor.NumEle(), host.data.size());
    EXPECT_EQ(tensor.num_dims(), 2);
    EXPECT_EQ(tensor.Bytes(), host.data.size() * sizeof(float));
    EXPECT_EQ(tensor.DeviceId(), 0);
    EXPECT_EQ(tensor.ShapeStringShort(), "[2, 3]");
    ExpectValues(tensor.Download(), host);

    Tensor destination(host.shape, 99.0f);
    tensor.DownloadToHost(destination);
    ExpectValues(destination, host);
}

TEST_F(CudaTensorGpuTest, RejectsDifferentShapesWithoutChangingEitherTensor) {
    const Tensor host = HostValues();
    CudaTensor tensor = TensorHostToDevice(host);
    Tensor wrong_shape({3, 2}, 99.0f);
    const Tensor original_destination = wrong_shape;
    ExpectCudaError([&] { tensor.UploadFromHost(wrong_shape); },
                    {"CudaTensor::UploadFromHost", "shape"});
    const CudaTensor& read_only = tensor;
    ExpectCudaError([&] { read_only.DownloadToHost(wrong_shape); },
                    {"CudaTensor::DownloadToHost", "shape"});
    ExpectValues(wrong_shape, original_destination);
    ExpectValues(tensor.Download(), host);
}

TEST_F(CudaTensorGpuTest, InvalidShapeResetPreservesExistingAllocationAndValues) {
    const Tensor host = HostValues();
    CudaTensor tensor = TensorHostToDevice(host);
    const void* original_pointer = tensor.GetData();
    ExpectCudaError([&] { tensor.Reset({2, 0}, 0); },
                    {"CudaTensor::Reset", "must be positive"});
    EXPECT_EQ(tensor.GetData(), original_pointer);
    EXPECT_EQ(tensor.NumEle(), host.data.size());
    EXPECT_EQ(tensor.Bytes(), host.data.size() * sizeof(float));
    EXPECT_EQ(tensor.ShapeStringShort(), "[2, 3]");
    ExpectValues(tensor.Download(), host);
}

TEST_F(CudaTensorGpuTest, InvalidDeviceResetPreservesExistingAllocationAndValues) {
    const Tensor host = HostValues();
    CudaTensor tensor = TensorHostToDevice(host);
    const void* original_pointer = tensor.GetData();
    ExpectCudaError([&] { tensor.Reset({8}, -1); }, {});
    EXPECT_EQ(tensor.GetData(), original_pointer);
    EXPECT_EQ(tensor.DeviceId(), 0);
    EXPECT_EQ(tensor.NumEle(), host.data.size());
    EXPECT_EQ(tensor.ShapeStringShort(), "[2, 3]");
    ExpectValues(tensor.Download(), host);
}

TEST_F(CudaTensorGpuTest, MoveConstructionClearsSourceAndAllowsItsReuse) {
    const Tensor host = HostValues();
    CudaTensor source = TensorHostToDevice(host);
    const void* original_pointer = source.GetData();
    CudaTensor destination(std::move(source));
    ExpectEmptyTensor(source);
    EXPECT_EQ(destination.GetData(), original_pointer);
    ExpectValues(destination.Download(), host);

    source.Reset(host.shape, 0);
    source.UploadFromHost(host);
    ExpectValues(source.Download(), host);
    ExpectValues(destination.Download(), host);
}

TEST_F(CudaTensorGpuTest, MoveAssignmentReplacesAllocationAndClearsSource) {
    const Tensor host = HostValues();
    CudaTensor source = TensorHostToDevice(host);
    const void* original_pointer = source.GetData();
    CudaTensor destination({1});
    destination = std::move(source);
    ExpectEmptyTensor(source);
    EXPECT_EQ(destination.GetData(), original_pointer);
    ExpectValues(destination.Download(), host);
    EXPECT_NO_THROW(source.Reset());
    ExpectEmptyTensor(source);
}

TEST_F(CudaTensorGpuTest, ResetReleasesStorageAndSupportsANewShape) {
    CudaTensor tensor = TensorHostToDevice(HostValues());
    tensor.Reset();
    ExpectEmptyTensor(tensor);
    EXPECT_NO_THROW(tensor.Reset());
    tensor.Reset({3}, 0);
    Tensor host({3});
    host.data = {2.0f, -3.0f, 4.0f};
    tensor.UploadFromHost(host);
    EXPECT_EQ(tensor.ShapeStringShort(), "[3]");
    EXPECT_EQ(tensor.NumEle(), 3u);
    ExpectValues(tensor.Download(), host);
}

}  // namespace
}  // namespace yan_lamma
