# cpp-minilamma

目前包含 Tensor、CPU 基础算子和 ASCII / JSON 词表 / 字节级 BPE 分词器，使用 C++17。Linear/Matmul 默认使用 Native 普通循环；Threaded、SIMD、ThreadedSIMD 留待后续实现，调用时会抛出异常。

## 目录

```text
include/                  Tensor、算子和分词器接口
src/                      Tensor、基础算子和分词器实现
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

## 分词器

接口位于 `include/tonizer.h`。`GetTonizer("")` 显式选择 ASCII；单个非空路径选择 JSON 词表；三个路径依次是 BPE 词表、合并规则、特殊 token 配置。文件不存在或内容非法会抛出异常，不会自动退回 ASCII。

```cpp
#include "include/tonizer.h"

auto tokenizer = yan_lamma::GetTonizer("vocab.json", "merges.txt", "special.json");
auto ids = tokenizer->Encode("hello");  // 自动添加 BOS 和 EOS
auto text = tokenizer->Decode(ids);
```

三个分词器的 `Encode()` 都会添加 BOS 和 EOS。解码保留特殊 token 的文本，没有跳过特殊 token 的选项；因此编码后直接解码会包含 BOS/EOS 的文本。解码不存在的 ID 会抛出 `std::out_of_range`。

ASCII 保留全部 128 个字节值，包括 NUL、换行和制表符；BOS/EOS/UNK 的 ID 分别为 128/129/130。非 ASCII 的每个输入字节产生一个 UNK。

JSON 词表使用下面的数组结构，`special` 可省略，省略时为 `false`。必须包含 `<BOS>`、`<EOS>`、`<UNK>`，普通文本按最长 token 匹配；未匹配的每个字节产生一个 UNK。

```json
[
  {"id": 0, "context": "<BOS>", "special": true},
  {"id": 1, "context": "<EOS>", "special": true},
  {"id": 2, "context": "<UNK>", "special": true},
  {"id": 3, "context": "hello", "special": false}
]
```

BPE 词表使用 `{"token": id}` 对象。普通 token 字符串使用 GPT-2 的字节到 Unicode 映射，例如空格字节对应 `Ġ`。`merges.txt` 每行是两个 token，行序决定优先级，支持 `#version: 0.2` 头；合并规则的两个输入及合并结果都必须存在于普通词表中。

特殊 token 配置的格式如下，三个 ID 必须存在于词表且互不相同。BOS/EOS/UNK 和词表中 `<|...|>` 形式的项都会被作为整体识别，优先匹配最长的特殊 token。

```json
{
  "bos_id": 0,
  "eos_id": 1,
  "unk_id": 2
}
```

词表 ID 必须是非负整数且小于 `INT_MAX`，不允许重复 ID、重复字符串或空字符串。允许稀疏 ID，`GetVocabSize()` 返回最大 ID 加一，缺失的 ID 不能用于解码。JSON 解析支持标准转义、中文和 emoji 的 Unicode 代理对；普通字符串按原字节保存，JSON 语法错误或未完整解析的文件会被拒绝。

BPE 必须先成功 `Load()` 才能编码、解码；成功重载替换全部状态，失败重载保留原来可用的状态。`DecodeToken()` 已还原字节映射，但单个 BPE token 可能仅包含 UTF-8 字符的一部分，应使用 `Decode()` 拼接后再读取完整文本。当前 BPE 在特殊 token 之间的普通文本片段内合并，未实现 GPT-2 的正则预分词，因此不保证与完整 GPT-2 分词器产生相同的 token 序列。

分词器回归测试位于 `test/tokenizer_test.cc`，与 Tensor、算子测试共用 `yan_minillama_tests` 和 Google Test。只运行分词器用例：

```bash
./build/test/yan_minillama_tests --gtest_filter='*Tokenizer*'
```
