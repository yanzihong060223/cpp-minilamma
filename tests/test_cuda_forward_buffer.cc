#include "tests/cuda_test_support.h"
using namespace yan_lamma;
using namespace yan_lamma::test;
int main(int argc, char** argv) {
    return Run(argc, argv, {{"f32", [] { ForwardCase(QuantType::kF32, true); }},
                           {"q8_0", [] { ForwardCase(QuantType::kQ80, true); }}});
}
