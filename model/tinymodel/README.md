# Tiny model（JSON + BIN）

本目录用于检验 yan-minillama 的实际加载和推理流程。模型使用 seed=42 的随机权重，
没有训练过，生成文本只用于检查代码运行与数值有效性。

- `model.json`：匹配 `include/model.h` 和 `src/load.cc` 的配置与 21 个 tensor 清单。
- `model.bin`：无文件头的 little-endian float32 权重，按行主序连续存储；共 33,248 个参数、132,992 字节（约 130 KiB）。
- `export_model.py`：重新生成 JSON/BIN，依赖 Python 3 和 NumPy。
- `smoke_test.cc`、`run_smoke.sh`：使用当前项目源码进行加载与推理检查。

配置为 vocab_size=131、dim=32、hidden_dim=86、layers=2、n_heads=4、
n_kv_heads=4、head_dim=8、max_seq_len=128，Norm 权重为 1，其余权重为
标准差 0.02 的随机数。

ASCII 分词器保留全部 128 个 ASCII ID，以及 BOS=128、EOS=129、UNK=130；
因此模型词表必须为 131，不能直接使用参考项目的 vocab_size=128。
它不需要外部 vocab.json。`Encode("hello")` 的实际结果是
`[128, 104, 101, 108, 108, 111, 129]`，包括项目分词器自动附加的 EOS。

## 运行验证

在 WSL 项目根目录执行：

```bash
bash mdel/tinymodel/run_smoke.sh
# 可选：自定义 prompt 和生成步数
bash mdel/tinymodel/run_smoke.sh "hello world" 8
# 空 prompt、特殊 token 以及最长序列也可以检查
bash mdel/tinymodel/run_smoke.sh "" 0
```

程序使用实际的 `ParseModelJson`、`ParseModel`、`GetTonizer`、
`Context`、`ForwardBatch`、`ForwardToken` 和 `Sampler`，
检查 JSON/BIN 加载、词表与特殊 ID、KV cache、logits shape 与有限值、
batch prefill / single decode 和逐 token 调用的一致性，以及 greedy 采样和历史记录。
输出以 `PASS:` 结尾，失败返回非零退出码。随机输出中的控制字符会显示为 `\\xNN`。

编译结果保存在根目录下的 `build-tiny-model/tiny-model-smoke`。
当前根 CMake 的静态库尚未收录全部推理源码，因此验证脚本直接编译这些现有源码。
运行脚本只读取模型文件；不会自动重新生成权重。

## 重新生成

```bash
python3 mdel/tinymodel/export_model.py --seed 42
# 可选：在另一个目录生成，不覆盖本目录
python3 mdel/tinymodel/export_model.py --seed 42 --output-dir /tmp/yan-tiny-model
```

重复使用同一 seed 会得到相同 JSON/BIN。此命令会替换目标目录中的这两个文件。

## 在你的代码中加载

```cpp
#include "include/load.h"

auto model = yan_lamma::ParseModel(
    "mdel/tinymodel/model.json", "mdel/tinymodel/model.bin");
if (!model.is_loaded) {
    throw std::runtime_error(model.load_error_);
}
```

JSON 使用 WSL loader 的 `layers`、`head_dim`、`tokenizer.type="ASCII"`
和 `tokenizer.pathname` 字段。每个 tensor 的 `offset` 和 `byte_size` 都以字节计，
BIN 总长度恰好等于最后一个 tensor 的结尾。不包含当前 loader 不加载的可选 QKV bias。
`ForwardToken` 不推进位置，`ForwardBatch` 用 batch 的位置更新 context；
decode 的第一位置是 prompt token 数量，默认 hello 为 7。
