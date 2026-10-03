# cpp-minilamma

目前包含 Tensor 和 CPU 基础算子，使用 C++17。Linear/Matmul 默认使用 Native 普通循环；Threaded、SIMD、ThreadedSIMD 留待后续实现，调用时会抛出异常。

## 目录

```text
include/                  Tensor 和算子接口
src/                      Tensor 和基础算子实现
matumal/                  Linear、Matmul 及执行模式分发
test/                     Google Test 单元测试
tests/                    之前的独立回归测试和运行脚本
third_party/googletest/    克隆的 Google Test v1.18.0 源码
build/                    默认构建结果
```

## 构建与测试

需要 CMake 3.20 以上和支持 C++17 的编译器。Google Test 直接使用项目内克隆的源码，无需安装到系统。

如果重新克隆本项目，先获取依赖（已有这个目录时无需再执行）：

```bash
git clone --depth 1 --branch v1.18.0 https://github.com/google/googletest.git third_party/googletest
```

在项目根目录执行：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

CMake 将源码构建为 `yan_minillama` 静态库，Google Test 测试程序为 `build/test/yan_minillama_tests`。CTest 会分别列出每个测试用例。

也可以直接运行或筛选测试：

```bash
./build/test/yan_minillama_tests
./build/test/yan_minillama_tests --gtest_list_tests
./build/test/yan_minillama_tests --gtest_filter='TensorTest.*'
```

Linear 的输入支持 `[in]` 和 `[1, in]`，输出分别为 `[out]` 和 `[1, out]`；多行二维输入会被拒绝。

## ASan / UBSan

使用另一个保留的构建目录，以 Clang 检查内存访问和未定义行为：

```bash
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=clang++ -DYAN_MINILLAMA_ENABLE_SANITIZERS=ON
cmake --build build-asan --parallel 2
ctest --test-dir build-asan --output-on-failure
```

只构建库时可增加 `-DBUILD_TESTING=OFF`，此时不需要 Google Test。

测试源码、Google Test 克隆目录和构建产物都会保留，不执行自动清理。`.gitignore` 仅避免把构建结果和嵌套的 Google Test 仓库提交到本项目。

框架接入可参考 [Google Test 官方 CMake 指南](https://google.github.io/googletest/quickstart-cmake.html)。
