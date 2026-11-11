#include "tests/cuda_test_support.h"
using namespace yan_lamma;
using namespace yan_lamma::test;
int main(int argc, char** argv) { return Run(argc, argv, {{"forward", [] { ForwardCase(QuantType::kF32); }}}); }
