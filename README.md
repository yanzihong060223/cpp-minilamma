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

ASCII 和 JSON 分词器的 `Encode()` 会添加 BOS 和 EOS；BPE 与 GGUF 分词器只添加已配置的 BOS/EOS。解码保留特殊 token 的文本，没有跳过特殊 token 的选项；因此编码后直接解码会包含已添加的 BOS/EOS 文本。解码不存在的 ID 会抛出 `std::out_of_range`。

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

特殊 token 配置的格式如下。BOS/EOS/UNK/PAD 角色可以省略或设为 `-1`，均表示缺失；其他 ID 必须存在于词表，允许多个角色共用同一 ID。缺失 BOS/EOS/UNK 时对应 getter 返回 `-1`，PAD 不新增查询接口，也不会自动补齐序列。未配置 UNK 时，遇到无法编码的片段会抛出异常。有效角色 ID 和词表中 `<|...|>` 形式的项都会被作为整体识别，优先匹配最长的特殊 token。

```json
{
  "bos_id": 0,
  "eos_id": 1,
  "unk_id": 2,
  "pad_id": -1
}
```

可额外提供 `"special_ids": [0, 1, 2, 42]`，显式指定特殊 token；存在此字段时会取代 `<|...|>` 的自动推断，同时保留已配置的 BOS/EOS/UNK/PAD。例如 GGUF 中名为 `<ctrl>` 的 control token 可以通过该列表保留，而不需要改名。词表和 `special_ids` 中仍不允许负数。

词表 ID 必须是非负整数且小于 `INT_MAX`，不允许重复 ID、重复字符串或空字符串。允许稀疏 ID，`GetVocabSize()` 返回最大 ID 加一，缺失的 ID 不能用于解码。JSON 解析支持标准转义、中文和 emoji 的 Unicode 代理对；普通字符串按原字节保存，JSON 语法错误或未完整解析的文件会被拒绝。

BPE 必须先成功 `Load()` 才能编码、解码；成功重载替换全部状态，失败重载保留原来可用的状态。`DecodeToken()` 已还原字节映射，但单个 BPE token 可能仅包含 UTF-8 字符的一部分，应使用 `Decode()` 拼接后再读取完整文本。当前 BPE 在特殊 token 之间的普通文本片段内合并，未实现 GPT-2 的正则预分词，因此不保证与完整 GPT-2 分词器产生相同的 token 序列。

分词器回归测试位于 `test/tokenizer_test.cc`，与 Tensor、算子测试共用 `yan_minillama_tests` 和 Google Test。只运行分词器用例：

```bash
./build/test/yan_minillama_tests --gtest_filter='*Tokenizer*'
```

### 下载演示模型

在项目根目录运行，默认从 ModelScope 下载 Qwen2-0.5B-Instruct Q8_0：

```bash
python3 tools/download_demo_model.py
```

脚本只依赖 Python 标准库，默认保存到当前工作目录下的
`models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf`，可用于后续真实模型端到端测试。
参考自 [mini-llama-cpp 下载脚本](https://github.com/CppTrainingHub/mini-llama-cpp/blob/main/scripts/download_demo_model.py)。

可通过 `--dir`、`--file` 指定本地目录和文件名，通过 `--repo`、`--remote-file`
指定 ModelScope 仓库和远端文件。已有文件以 `GGUF` 开头时跳过下载。
下载先写入 `.part` 临时文件，检查长度（服务端提供时）和 GGUF 魔数后再替换目标文件；
失败返回非零退出码并清理临时文件。魔数检查不代表完整模型校验。
默认目录中的 `.gguf` 和 `.gguf.part` 文件已加入 Git 忽略规则。

### GGUF 文件检查入口

`gguf_inspect` 使用 `GgufRead::load()` 和 `Inspect()` 输出文件头、元数据、
张量形状、类型、偏移及大小，同时校验文件内的张量边界，不加载权重数据。

```bash
cmake -S . -B build-smoke-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-smoke-release --target gguf_inspect --parallel 2
./build-smoke-release/test/gguf_inspect models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf
```

`--help` 查看用法；加载或检查失败返回非零退出码。

### GGUF 生成测试入口

`gguf_generate` 是独立的单轮生成测试程序，使用实际的 GGUF loader、tokenizer、
prompt builder、prefill、decode 和贪心采样。启用 `BUILD_TESTING`（默认开启）后构建：

```bash
cmake -S . -B build
cmake --build build --target gguf_generate --parallel 2
./build/test/gguf_generate models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf "你好" 16
```

第一个参数是模型路径；默认下载位置是 `models/chat/`，请按实际位置替换。
后两个参数可省略，默认 prompt 为 `你好`、最多生成 16 个 token。
传 `0` 只检查 prefill；`--help` 查看用法。生成在 EOS 或 token 上限处停止。
程序校验 token 范围、logits 形状及有限值、上下文长度与推理历史；
生成文本输出到 stdout，阶段诊断和通过/失败状态输出到 stderr，失败返回非零退出码。
该入口不加入默认 CTest，运行时需自行提供真实模型。

loader 在缺少 `output.weight` 时使用 `token_embd.weight` 作为输出权重，并保留其量化格式。
`gguf_generate` 和 `gguf_run` 直接将加载后的模型交给 forward，不再单独预转换权重。
forward 支持 F32、Q8_0、Q4_0、Q4_1 的线性权重，按类型调用已有转换函数，
再使用普通 F32 算子计算；F16 线性权重尚不支持。
目前每个 token 的前向会重复转换权重，仍存在复制和反量化开销。

### GGUF 交互式对话

`gguf_run` 复用 `Terminal` 和 `ChatSession`，每轮用全部已完成消息和当前输入渲染
GGUF 中的 chat template，重新编码并完整 prefill，然后流式输出 assistant 回复。
模型、tokenizer 和模板只在启动时加载一次。推荐使用 Release：

```bash
cmake -S . -B build-smoke-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-smoke-release --target gguf_run --parallel 2
./build-smoke-release/test/gguf_run models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf 16
```

模型路径按实际位置替换；第二个参数为每轮最多生成的 token 数，默认 16，必须为正整数。
输入一行后回车发送；prefill 会显示 `Prefill i/N` 进度，完成后显示 `assistant >` 并输出回复。
支持 `/help`、`/clear`、`/stats`、`/params`（兼容 `/parmas`）、`/exit`，EOF 也会正常退出。
空白行不会发送；正常消息保留原始空格。`/clear` 清除消息、token 记录和统计，不重新加载模型。
生成在 EOS 或本轮上限处停止，成功轮次才写入历史；失败或上下文超限时保留此前历史。
超限后可缩短输入或执行 `/clear`。每轮独立分配 KV，不复用历史缓存，不自动裁剪消息。
`/stats` 的 prompt token 数累计每轮完整 prefill 的工作量，生成 token 数不包括 EOS。
完整聊天路径的测试使用小型 GGUF fixture，不依赖下载模型：

```bash
cmake --build build-smoke-release --target gguf_run gguf_generate --parallel 2
ctest --test-dir build-smoke-release -R GgufInteractive --output-on-failure
```

### 拆解 GGUF tokenizer

Python 工具只使用标准库，支持小端 GGUF v2/v3，只读取 metadata，不加载权重：

```bash
python3 tools/export_gguf_tokenizer.py model.gguf --output-dir out/tokenizer
```

`--out-dir` 是 `--output-dir` 的别名；不指定时默认输出到当前工作目录的 `models/chat`。

输出目录中的 `tokenizer_metadata.json` 保存全部 `tokenizer.*` 字段及类型，`tokens.tsv` 按 ID 展示 token、类型和 score，其中 token 使用 JSON 转义，换行和 NUL 不会破坏行结构。若存在 `tokenizer.chat_template`，会原样写入 `chat_template.txt`（包括空字符串），不解析或执行 Jinja；没有该字段时不生成模板文件。

工具还会为当前支持的 GPT-2 字节 BPE 生成 `vocab.json`、`merges.txt` 和 `special_tokens.json`，保持原始 ID 和 merge 顺序。特殊配置固定包含 `bos_id`、`eos_id`、`unk_id`、`pad_id`，缺失值写为 `-1`，并保留显式 `special_ids` 列表。已有同名输出文件时拒绝覆盖。

不支持的 tokenizer 类型或无法转换的内容会返回非零状态，保留检查文件和已导出的模板，并且不生成 BPE 三文件。工具不会补造 UNK 或重编号。GGUF 的 `add_bos_token` / `add_eos_token` 仅保存在检查文件中，当前两条加载路径都沿用“有对应 ID 就添加”的规则；PAD 只按特殊 token 识别，不自动添加。

路线 A：先导出，再从三个文件加载，适合检查和调试：

```cpp
auto tokenizer = yan_lamma::GetTonizer(
    "out/tokenizer/vocab.json", "out/tokenizer/merges.txt",
    "out/tokenizer/special_tokens.json");
```

路线 B：通过 GGUF 接口直接读取 metadata，不依赖 Python 或导出文件：

```cpp
#include "include/gguf_tokenizer.h"

auto tokenizer = yan_lamma::GetGgufTokenizer("model.gguf");
```

两条路径使用相同的字节 BPE 规则，保留特殊 token；都不实现完整模型原生的正则预分词。上面的 `gguf_generate` 测试入口采用路线 B。

构建测试需要 Python 3。导出与双路径对照测试已接入 CTest：

```bash
ctest --test-dir build -R 'GgufTokenizerExport|TokenizerFiles' --output-on-failure
```
