# cpp-minilamma

目前包含 Tensor、CPU 基础算子和 ASCII / JSON 词表 / 字节级 BPE 分词器，使用 C++17。上层 Linear/Matmul 只接收输入和权重，普通浮点版本内部默认分别使用 Native 和 Threaded。需要显式选择执行模式时，调用底层 LinearDispatch/MatmulDispatch；两者都支持 Native 和 Threaded。Linear 还支持 SIMD 和 ThreadedSIMD，按编译目标使用 ARM NEON 或 AVX2/FMA，未启用对应指令集时回退标量点积。Matmul 的 SIMD 和 ThreadedSIMD 尚未实现，调用时会抛出异常。

量化 `LinearQ80`、`LinearQ40`、`LinearQ41` 分别以同名函数实现 NEON、AVX2/FMA 和普通分支，编译时按指令集宏选择，不增加 CMake 开关。SIMD 分支每次处理 8 个元素，在寄存器中解码并乘加，不创建反量化浮点数组；普通分支保留现有 `LinearQuant`。每个输出独立交给 `ParallelFor` 计算，支持一维输入和批量输入，尾块仅访问有效输入。Q40/Q41 的低四位对应块内前 16 个权重，高四位对应后 16 个权重。

量化 Linear 用例按独立 double 点积验证结果，容差为 `1e-5 × max(1, 乘积绝对值之和)`；单线程和多线程结果仍精确比较。用例覆盖 SIMD 分组边界、半块边界、尾块、零比例，以及 Q41 的非零最小值。

## 目录

```text
app/                      mini-llama 命令入口
scripts/                  NumPy golden-logits 数值参考和回归脚本
include/                  Tensor、算子和分词器接口
src/                      Tensor、基础算子和分词器实现
matumal/                  Linear、Matmul 及执行模式分发
test/                     Google Test 单元测试
tests/                    之前的独立回归测试和运行脚本
third_party/googletest/    克隆的 Google Test v1.18.0 源码
build/                    默认构建结果
```

## 构建与测试

需要 CMake 3.20 以上和支持 C++17 的编译器。Google Test 直接使用项目内克隆的源码，无需安装到系统。CLI 集成测试复用 tiny exporter，需要 Python 3 和 NumPy。

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

CMake 将源码构建为 `yan_minillama` 静态库，命令程序为 `build/mini-llama`，Google Test 测试程序为 `build/test/yan_minillama_tests`。CTest 会分别列出每个测试用例，项目测试名统一以 `mini-llama-tests.` 开头。

也可以直接运行或筛选测试：

```bash
./build/test/yan_minillama_tests
./build/test/yan_minillama_tests --gtest_list_tests
./build/test/yan_minillama_tests --gtest_filter='TensorTest.*'
```

Linear 的输入支持 `[in]` 和 `[1, in]`，输出分别为 `[out]` 和 `[1, out]`；多行二维输入会被拒绝。

## SIMD 章节验证与 benchmark

普通浮点 `Linear()` 默认保持 Native；显式 `LinearDispatch` 可选择 Native、Threaded、SIMD、ThreadedSIMD。`linear_all_modes_match` 在同一份正负混合输入上验证四种模式，覆盖向量边界、尾部、两种输入形状以及 1/4 线程。Native/Threaded 和 SIMD/ThreadedSIMD 分别精确比较，各模式对独立 double 点积使用 `1e-5 × max(1, 乘积绝对值之和)` 容差。65 个输出使四线程分发实际进入并行区间。

先复用现有 exporter 准备随机 tiny 模型及 JSON 词表，默认 seed 为 42：

```bash
python3 model/tinymodel/export_model.py --output-dir models/tiny --json-vocab
```

输出包含 `model.json`、`model.bin` 和 `vocab.json`。JSON 词表与已有 ASCII tokenizer 使用相同的 131 个 ID：ASCII 为 0–127，BOS/EOS/UNK 为 128/129/130。不加 `--json-vocab` 时仍使用原来的 ASCII 导出流程。生成文件已加入 Git 忽略规则。

在支持 AVX2 和 FMA 的 x86 主机上执行：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-mavx2 -mfma"
cmake --build build -j4
ctest --test-dir build -R mini-llama-tests --output-on-failure --no-tests=error

./build/mini-llama bench models/tiny \
  --tokenizer models/tiny/vocab.json \
  --prompt hello \
  --n-predict 4 \
  --threads 4 \
  --verbose
```

`bench` 加载模型目录里的 JSON/BIN，调用现有 loader、tokenizer、线程控制和 `RunBetchMrk`；原有 GGUF benchmark 入口继续保留。参数默认值为 `--prompt hello --n-predict 4 --threads 0`；线程数 0 自动选择，`-threads` 是 `--threads` 的别名。`--linear-mode native|threaded|simd|threaded-simd` 可显式选择本次进程的 Linear 模式，未指定时仍为 Native。显式 tokenizer 路径相对于工作目录；省略该参数时使用模型元数据，其中的相对路径按模型目录解析。

输出包含 `threads:4`、默认的 `linear_mode:Native`、`compiled_simd`、`tokens/s(total)` 和 `tokens/s(Decode)`。线程数表示配置，小任务仍可能按现有阈值串行执行；`compiled_simd` 表示编译进来的后端，实际模式以 `linear_mode` 为准。固定长度生成不因 EOS 提前结束，只受剩余上下文容量限制；`hello` 去掉自动追加的 EOS 后有 6 个 prompt token，生成 4 个 token 对应 3 次 decode forward。`--n-predict 0` 只测 prefill，0/1 个生成 token 的 decode 吞吐量为零。总吞吐量为生成数除以 prefill+decode 耗时，decode 吞吐量为 decode forward 数除以 decode 耗时。verbose 的 decode 张量打印计入 decode 耗时，tiny 模型的短跑结果只用于验证流程和输出。

三条点积路径按编译条件选择，没有运行时 CPU 探测：

| 路径 | 条件 |
|---|---|
| NEON | 定义 `__ARM_NEON` 或 `__ARM_NEON__`；AArch64 通常默认启用 |
| AVX2/FMA | 同时定义 `__AVX2__` 和 `__FMA__` |
| 标量回退 | 两组条件都不满足；仅启用 AVX2 而没有 FMA 也回退 |

在 x86 上可用独立目录验证标量回退，避免复用不同指令集的构建产物：

```bash
cmake -S . -B build-scalar -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-mno-avx2 -mno-fma"
cmake --build build-scalar -j4
ctest --test-dir build-scalar -R mini-llama-tests --output-on-failure --no-tests=error
./build-scalar/mini-llama bench models/tiny --tokenizer models/tiny/vocab.json --prompt hello --n-predict 4 --threads 4 --verbose
```

NEON 的运行验证需要 ARM 环境。Linear 的输入和每行权重连续，计算是重复、独立的点积，适合向量乘加；当前通用 Matmul 内循环按列读取右矩阵，存在跨步访问，因此公开 `Matmul()` 继续使用 Threaded。Matmul 也能进行 SIMD 优化，但需要相应的循环组织、打包、转置或分块；目前显式请求其 SIMD 模式会报未实现。

## Python golden-logits 数值参考

`scripts/reference_forward.py` 独立读取 JSON manifest 和 little-endian F32 权重，用 NumPy 实现 embedding、RMSNorm、Q/K/V、Normal RoPE、持久 K/V cache、GQA attention、SwiGLU、final norm 和 lm_head。`forward_token(model, pos, token_id)` 只处理一个 token，显式按 pos 写入同一个缓存；不调用 C++ 或读取 C++ logits 计算参考。权重方向为 `[out,in]`，RMSNorm 规约和 RoPE 旋转使用 double 中间计算，其他激活和缓存保持 float32。

基准复用本地 tiny 的 **131 词表、BOS=128**。golden 入口只接受 ASCII prompt，编码为 BOS 加字符 ID，不追加 EOS，也不进行词表字符串合并。`hello` 对应 `[128,104,101,108,108,111]`。普通 `Linear()` 默认保持 Native；`SetDefaultMatulMode` 和 `SetThreadNum` 仅在推理开始前设置，不能在推理期间修改。

一键回归自动在临时目录导出 seed=42 的模型，再串起 C++ 导出、Python 生成和比较，无需提前准备 `models/tiny`：

```bash
cmake --build build -j4
ctest --test-dir build -R golden-logits --output-on-failure --no-tests=error

# 也可单独运行
python3 scripts/test_golden.py --binary ./build/mini-llama
python3 scripts/test_threaded_cli.py --binary ./build/mini-llama
```

CTest 包含主链路、线程对拍和比较器故障注入三个测试，依赖 Python 3 与 NumPy。成功时清理临时产物，失败时保留并打印路径。主链路覆盖四种模式；另有 4 个 query head、2 个 KV head、非默认 eps 和 rope_theta 的 GQA 模型。Threaded/ThreadedSIMD 的 1/4 线程对拍要求每步 logits 字节一致。C++ 用例额外验证多 token ForwardBatch 与逐 token 执行的 logits、K/V cache、pos 和 token history 一致。

手动闭环先准备模型，再使用新的空输出目录。两端可以共用目录，但已有本端结果会被拒绝覆盖；重复手动运行时应换一个空目录：

```bash
python3 model/tinymodel/export_model.py --output-dir models/tiny --json-vocab

./build/mini-llama \
  --config models/tiny/model.json --model models/tiny/model.bin \
  --prompt hello --n-predict 16 --linear-mode native --threads 1 \
  --dump-logits artifacts/golden

python3 scripts/make_golden.py \
  --config models/tiny/model.json --weights models/tiny/model.bin \
  --prompt hello -n 16 --output-dir artifacts/golden

python3 scripts/compare_logits.py \
  --cpp-dir artifacts/golden --py-dir artifacts/golden --vocab-size 131
```

产物为 `logits_stepN.bin`、`python_logits_stepN.bin`、`generation_tokens.txt`、`python_generation_tokens.txt`、`cpp_run.json` 和 `python_run.json`。每份 logits 包含 131 个 little-endian float32，共 524 字节；token 文件使用空白分隔的十进制 ID。元数据最后写入，记录 prompt token、请求生成数、原始上下文容量、实际生成数和 step 数。

令 P 为含 BOS 的 prompt 长度，G 为 `min(n_predict, max_seq_len - P)`。第一个生成 token 来自最后一次 prefill，只对前 G−1 个生成 token继续 forward，总 step 数为 `P + max(G−1,0)`。`hello -n 16` 有 **21 份 logits（step0–20）和 16 个生成 ID**；`-n 0/1` 都只有 6 份 logits，生成文件分别为空或只有一个 ID。生成不因 EOS 提前结束。

比较器分三类检查，只有全部通过才输出 `RESULT: ALL PASSED`，任何失败返回非零：

| 检查 | 判定标准 |
|---|---|
| 文件集合 | 从运行参数推导完整编号，缺失、额外、断档、非规范编号、错误长度或缺元数据均失败；双方同时缺末步也失败 |
| logits | 拒绝 NaN/Inf；用 float64 计算差值，最大绝对和最大相对误差分别都必须 ≤ `1e-4`；相对误差仅统计 `abs(py) >= 1e-3`，空集合取 0；每步 top-1 一致 |
| 采样 | 两端序列一致，长度和 ID 范围正确；每枚 ID 还须等于本端对应 logits 的 greedy argmax，平局选最小 ID |

不使用相加容差的 `allclose` 代替两个独立阈值。一键脚本从输入配置独立计算预期 step 数并传给比较器，防止两边共同截断而误通过。失败报告包含 step、prefill/decode、pos、输入 token、最大误差所在 token、两侧值和 top-1。故障注入覆盖文件缺失、错误长度、NaN/Inf、阈值边界、top-1 和两端相同错误生成序列。

改动触及 **ForwardBatch/ForwardToken、RMSNorm、RoPE、KV cache、sampler、matmul dispatch、多线程或 SIMD** 时，必须运行 `golden-logits`。所有 step 发散时先查权重读取、eps 和矩阵方向；仅 decode 发散时先查 pos、缓存写入和 GQA 分组；仅 top-1 分叉时查候选接近程度和平局规则；Native 通过而优化模式失败时查向量尾部、累加顺序、线程切片和写回。不要通过放宽阈值掩盖实现错误。

本回归只判定 JSON/BIN tiny 的 CPU F32、Normal RoPE 路径；量化、NeoX/GGUF 和其他后端保留各自测试。SIMD 实际执行 NEON、AVX2/FMA 还是标量回退，由前文的编译条件决定。

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
