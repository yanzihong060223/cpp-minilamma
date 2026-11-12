## 环境表

| 字段 | 值 |
| --- | --- |
| timestamp_utc | 2026-11-12T03:28:18.782059+00:00 |
| repo | /home/yan06/yan-minillama |
| build_dir | /home/yan06/yan-minillama/build |
| binary | /home/yan06/yan-minillama/build/mini-llama |
| os | Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.43 |
| python | 3.14.4 (main, Aug 20 2026, 10:41:58) [GCC 15.2.0] |
| errors | {"cuda_info": "unavailable: Backend setup failed: CUDA backend was not built. Reconfigure with -DMINI_LLAMA_CUDA=ON on a NVIDIA CUDA machine.\ntrace summary: trace_id=3315718344724-1 status=error", "gpu": "unavailable: Failed to initialize NVML: GPU access blocked by the operating system\nFailed to properly shut down NVML: GPU access blocked by the operating system", "gpu_clocks_after": "unavailable: Failed to initialize NVML: GPU access blocked by the operating system\nFailed to properly shut down NVML: GPU access blocked by the operating system", "gpu_clocks_before": "unavailable: Failed to initialize NVML: GPU access blocked by the operating system\nFailed to properly shut down NVML: GPU access blocked by the operating system", "nvcc": "unavailable: cannot execute ['nvcc', '--version']: [Errno 2] No such file or directory: 'nvcc'", "power_draw_after": "unavailable: Failed to initialize NVML: GPU access blocked by the operating system\nFailed to properly shut down NVML: GPU access blocked by the operating system", "power_draw_before": "unavailable: Failed to initialize NVML: GPU access blocked by the operating system\nFailed to properly shut down NVML: GPU access blocked by the operating system", "temperature_after": "unavailable: Failed to initialize NVML: GPU access blocked by the operating system\nFailed to properly shut down NVML: GPU access blocked by the operating system", "temperature_before": "unavailable: Failed to initialize NVML: GPU access blocked by the operating system\nFailed to properly shut down NVML: GPU access blocked by the operating system"} |
| commit | 7967381 |
| dirty | True |
| branch | feature/cuda_check |
| cmake | cmake version 4.2.3 |
| cxx | c++ (Ubuntu 15.2.0-16ubuntu1) 15.2.0 |
| gpu | unavailable |
| nvcc | unavailable |
| cuda_info | unavailable |
| cuda_built | False |
| cuda_available | False |
| gpu_clocks | {"after": "unavailable", "before": "unavailable"} |
| temperature | {"after": "unavailable", "before": "unavailable"} |
| power_draw | {"after": "unavailable", "before": "unavailable"} |

## CTest 结论

| suite | status | passed | total | failed | skipped |
| --- | --- | --- | --- | --- | --- |
| targeted | not_run | unavailable | unavailable | unavailable | unavailable |
| full | not_run | unavailable | unavailable | unavailable | unavailable |

## 性能对比表

前向吞吐（不含加载与采样）；ratio = 当前值 / 基线值。

| case | prefill_ms median | prefill_ms min | prefill_ms max | prefill_ms ratio | decode_ms median | decode_ms min | decode_ms max | decode_ms ratio | total_ms median | total_ms min | total_ms max | total_ms ratio | tokens_s_total median | tokens_s_total min | tokens_s_total max | tokens_s_total ratio | tokens_s_decode median | tokens_s_decode min | tokens_s_decode max | tokens_s_decode ratio | load_ms median | load_ms min | load_ms max | load_ms ratio | sampling_ms median | sampling_ms min | sampling_ms max | sampling_ms ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tiny-cpu | 0.09 | 0.09 | 0.09 | unavailable | 0.05 | 0.05 | 0.05 | unavailable | 0.13 | 0.13 | 0.13 | unavailable | 29955.82 | 29955.82 | 29955.82 | unavailable | 64905.56 | 64905.56 | 64905.56 | unavailable | 0.749 | 0.749 | 0.749 | unavailable | 0.001 | 0.001 | 0.001 | unavailable |
| chat-cpu | 581.42 | 581.42 | 581.42 | unavailable | 952.23 | 952.23 | 952.23 | unavailable | 1533.64 | 1533.64 | 1533.64 | unavailable | 2.61 | 2.61 | 2.61 | unavailable | 3.15 | 3.15 | 3.15 | unavailable | 2390.131 | 2390.131 | 2390.131 | unavailable | 0.513 | 0.513 | 0.513 | unavailable |
| tiny-q4_0-cpu | 0.79 | 0.79 | 0.79 | unavailable | 0.42 | 0.42 | 0.42 | unavailable | 1.21 | 1.21 | 1.21 | unavailable | 3301.72 | 3301.72 | 3301.72 | unavailable | 7187.54 | 7187.54 | 7187.54 | unavailable | 0.594 | 0.594 | 0.594 | unavailable | 0.001 | 0.001 | 0.001 | unavailable |

## counters 表

| case | uploaded_weights | gpu_mem_mb | cuda_f32_linear_weights | cuda_q8_0_linear_weights | cuda_q4_0_linear_weights | cuda_q4_1_linear_weights | cuda_linear_calls | cuda_activation_calls | cuda_attention_calls | cpu_attention_fallback | cuda_kv_write_bytes | cuda_kv_read_bytes | h2d_copies | d2h_copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tiny-cpu | {"max": "unavailable", "median": "unavailable", "min": "unavailable"} | {"max": "unavailable", "median": "unavailable", "min": "unavailable"} | {"total": {"max": "unavailable", "median": "unavailable", "min": "unavailable"}, "uploaded": {"max": "unavailable", "median": "unavailable", "min": "unavailable"}} | {"total": {"max": "unavailable", "median": "unavailable", "min": "unavailable"}, "uploaded": {"max": "unavailable", "median": "unavailable", "min": "unavailable"}} | {"total": {"max": "unavailable", "median": "unavailable", "min": "unavailable"}, "uploaded": {"max": "unavailable", "median": "unavailable", "min": "unavailable"}} | {"total": {"max": "unavailable", "median": "unavailable", "min": "unavailable"}, "uploaded": {"max": "unavailable", "median": "unavailable", "min": "unavailable"}} | {"max": "unavailable", "median": "unavailable", "min": "unavailable"} | {"max": "unavailable", "median": "unavailable", "min": "unavailable"} | {"max": "unavailable", "median": "unavailable", "min": "unavailable"} | {"max": "unavailable", "median": "unavailable", "min": "unavailable"} | {"max": "unavailable", "median": "unavailable", "min": "unavailable"} | {"max": "unavailable", "median": "unavailable", "min": "unavailable"} | {"max": "unavailable", "median": "unavailable", "min": "unavailable"} | {"max": "unavailable", "median": "unavailable", "min": "unavailable"} |
| chat-cpu | {"max": "unavailable", "median": "unavailable", "min": "unavailable"} | {"max": "unavailable", "median": "unavailable", "min": "unavailable"} | {"total": {"max": "unavailable", "median": "unavailable", "min": "unavailable"}, "uploaded": {"max": "unavailable", "median": "unavailable", "min": "unavailable"}} | {"total": {"max": "unavailable", "median": "unavailable", "min": "unavailable"}, "uploaded": {"max": "unavailable", "median": "unavailable", "min": "unavailable"}} | {"total": {"max": "unavailable", "median": "unavailable", "min": "unavailable"}, "uploaded": {"max": "unavailable", "median": "unavailable", "min": "unavailable"}} | {"total": {"max": "unavailable", "median": "unavailable", "min": "unavailable"}, "uploaded": {"max": "unavailable", "median": "unavailable", "min": "unavailable"}} | {"max": "unavailable", "median": "unavailable", "min": "unavailable"} | {"max": "unavailable", "median": "unavailable", "min": "unavailable"} | {"max": "unavailable", "median": "unavailable", "min": "unavailable"} | {"max": "unavailable", "median": "unavailable", "min": "unavailable"} | {"max": "unavailable", "median": "unavailable", "min": "unavailable"} | {"max": "unavailable", "median": "unavailable", "min": "unavailable"} | {"max": "unavailable", "median": "unavailable", "min": "unavailable"} | {"max": "unavailable", "median": "unavailable", "min": "unavailable"} |
| tiny-q4_0-cpu | {"max": "unavailable", "median": "unavailable", "min": "unavailable"} | {"max": "unavailable", "median": "unavailable", "min": "unavailable"} | {"total": {"max": "unavailable", "median": "unavailable", "min": "unavailable"}, "uploaded": {"max": "unavailable", "median": "unavailable", "min": "unavailable"}} | {"total": {"max": "unavailable", "median": "unavailable", "min": "unavailable"}, "uploaded": {"max": "unavailable", "median": "unavailable", "min": "unavailable"}} | {"total": {"max": "unavailable", "median": "unavailable", "min": "unavailable"}, "uploaded": {"max": "unavailable", "median": "unavailable", "min": "unavailable"}} | {"total": {"max": "unavailable", "median": "unavailable", "min": "unavailable"}, "uploaded": {"max": "unavailable", "median": "unavailable", "min": "unavailable"}} | {"max": "unavailable", "median": "unavailable", "min": "unavailable"} | {"max": "unavailable", "median": "unavailable", "min": "unavailable"} | {"max": "unavailable", "median": "unavailable", "min": "unavailable"} | {"max": "unavailable", "median": "unavailable", "min": "unavailable"} | {"max": "unavailable", "median": "unavailable", "min": "unavailable"} | {"max": "unavailable", "median": "unavailable", "min": "unavailable"} | {"max": "unavailable", "median": "unavailable", "min": "unavailable"} | {"max": "unavailable", "median": "unavailable", "min": "unavailable"} |

KV 为逻辑流量；显存为本次 CUDA buffer 持有量峰值，MB = 1,000,000 bytes。

GPU 验收：未验证（无 GPU 实测记录）。

## 原始 stdout 附录

tiny-cpu repeat 1

```text
prompt tokens: 6
generated tokens: 4
prefill ms: 0.09
decode ms: 0.05
total ms: 0.13
tokens/s (total): 29955.82
tokens/s (decode): 64905.56
uploaded weights: unavailable
gpu memory used: unavailable
cuda f32 linear weights: unavailable
cuda q8_0 linear weights: unavailable
cuda q4_0 linear weights: unavailable
cuda q4_1 linear weights: unavailable
cuda linear calls: unavailable
cuda activation calls: unavailable
cuda attention calls: unavailable
cpu attention fallback calls: unavailable
cuda kv write bytes: unavailable
cuda kv read bytes: unavailable
host->device copies: unavailable
device->host copies: unavailable
load ms: 0.749
sampling ms: 0.001
```

chat-cpu repeat 1

```text
prompt tokens: 2
generated tokens: 4
prefill ms: 581.42
decode ms: 952.23
total ms: 1533.64
tokens/s (total): 2.61
tokens/s (decode): 3.15
uploaded weights: unavailable
gpu memory used: unavailable
cuda f32 linear weights: unavailable
cuda q8_0 linear weights: unavailable
cuda q4_0 linear weights: unavailable
cuda q4_1 linear weights: unavailable
cuda linear calls: unavailable
cuda activation calls: unavailable
cuda attention calls: unavailable
cpu attention fallback calls: unavailable
cuda kv write bytes: unavailable
cuda kv read bytes: unavailable
host->device copies: unavailable
device->host copies: unavailable
load ms: 2390.131
sampling ms: 0.513
```

tiny-q4_0-cpu repeat 1

```text
prompt tokens: 6
generated tokens: 4
prefill ms: 0.79
decode ms: 0.42
total ms: 1.21
tokens/s (total): 3301.72
tokens/s (decode): 7187.54
uploaded weights: unavailable
gpu memory used: unavailable
cuda f32 linear weights: unavailable
cuda q8_0 linear weights: unavailable
cuda q4_0 linear weights: unavailable
cuda q4_1 linear weights: unavailable
cuda linear calls: unavailable
cuda activation calls: unavailable
cuda attention calls: unavailable
cpu attention fallback calls: unavailable
cuda kv write bytes: unavailable
cuda kv read bytes: unavailable
host->device copies: unavailable
device->host copies: unavailable
load ms: 0.594
sampling ms: 0.001
```
