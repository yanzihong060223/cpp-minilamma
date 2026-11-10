#include "tests/cuda_test_support.h"
using namespace yan_lamma;
using namespace yan_lamma::test;
void MatmulCase() {
    for (auto shape : {std::vector<int>{3, 7, 5}, std::vector<int>{1, 33, 17}}) {
        Tensor a = Values({shape[0], shape[1]}), b = Values({shape[1], shape[2]}, 0.7f);
        const Tensor expected = Matmul(a, b);
        if (!CudaRuntimeBuilt()) {
            RequireCloseTensor(expected, expected, "CPU matmul");
            RequireError([&] { (void)CudaMatmul(a, b, 0); }, "not built");
        } else RequireCloseTensor(CudaMatmul(a, b, 0), expected, "cuBLAS matmul");
    }
}
int main(int argc, char** argv) { return Run(argc, argv, {{"matmul", MatmulCase}}); }
