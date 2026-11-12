## 环境表

| 字段 | 值 |
| --- | --- |
| timestamp_utc | 2026-11-12T03:27:17.973767+00:00 |
| repo | /home/yan06/yan-minillama |
| build_dir | /home/yan06/yan-minillama/build |
| binary | /home/yan06/yan-minillama/build/mini-llama |
| os | Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.43 |
| python | 3.14.4 (main, Aug 20 2026, 10:41:58) [GCC 15.2.0] |
| errors | {"cuda_info": "unavailable: Backend setup failed: CUDA backend was not built. Reconfigure with -DMINI_LLAMA_CUDA=ON on a NVIDIA CUDA machine.\ntrace summary: trace_id=3254893720304-1 status=error", "gpu": "unavailable: Failed to initialize NVML: GPU access blocked by the operating system\nFailed to properly shut down NVML: GPU access blocked by the operating system", "gpu_clocks_after": "unavailable: Failed to initialize NVML: GPU access blocked by the operating system\nFailed to properly shut down NVML: GPU access blocked by the operating system", "gpu_clocks_before": "unavailable: Failed to initialize NVML: GPU access blocked by the operating system\nFailed to properly shut down NVML: GPU access blocked by the operating system", "nvcc": "unavailable: cannot execute ['nvcc', '--version']: [Errno 2] No such file or directory: 'nvcc'", "power_draw_after": "unavailable: Failed to initialize NVML: GPU access blocked by the operating system\nFailed to properly shut down NVML: GPU access blocked by the operating system", "power_draw_before": "unavailable: Failed to initialize NVML: GPU access blocked by the operating system\nFailed to properly shut down NVML: GPU access blocked by the operating system", "temperature_after": "unavailable: Failed to initialize NVML: GPU access blocked by the operating system\nFailed to properly shut down NVML: GPU access blocked by the operating system", "temperature_before": "unavailable: Failed to initialize NVML: GPU access blocked by the operating system\nFailed to properly shut down NVML: GPU access blocked by the operating system"} |
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
| targeted | passed | 6 | 6 | 0 | 0 |
| full | passed | 20 | 20 | 0 | 0 |

## 性能对比表

前向吞吐（不含加载与采样）；ratio = 当前值 / 基线值。

| case | prefill_ms median | prefill_ms min | prefill_ms max | prefill_ms ratio | decode_ms median | decode_ms min | decode_ms max | decode_ms ratio | total_ms median | total_ms min | total_ms max | total_ms ratio | tokens_s_total median | tokens_s_total min | tokens_s_total max | tokens_s_total ratio | tokens_s_decode median | tokens_s_decode min | tokens_s_decode max | tokens_s_decode ratio | load_ms median | load_ms min | load_ms max | load_ms ratio | sampling_ms median | sampling_ms min | sampling_ms max | sampling_ms ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tiny-cpu | 0.08 | 0.08 | 0.08 | unavailable | 0.04 | 0.04 | 0.04 | unavailable | 0.13 | 0.13 | 0.13 | unavailable | 30840.4 | 30840.4 | 30840.4 | unavailable | 66940.38 | 66940.38 | 66940.38 | unavailable | 0.556 | 0.556 | 0.556 | unavailable | 0.001 | 0.001 | 0.001 | unavailable |
| chat-cpu | 564.26 | 564.26 | 564.26 | unavailable | 932.88 | 932.88 | 932.88 | unavailable | 1497.14 | 1497.14 | 1497.14 | unavailable | 2.67 | 2.67 | 2.67 | unavailable | 3.22 | 3.22 | 3.22 | unavailable | 2527.443 | 2527.443 | 2527.443 | unavailable | 0.463 | 0.463 | 0.463 | unavailable |
| tiny-q4_0-cpu | 0.79 | 0.79 | 0.79 | unavailable | 0.38 | 0.38 | 0.38 | unavailable | 1.17 | 1.17 | 1.17 | unavailable | 3423.32 | 3423.32 | 3423.32 | unavailable | 7906.05 | 7906.05 | 7906.05 | unavailable | 0.568 | 0.568 | 0.568 | unavailable | 0.001 | 0.001 | 0.001 | unavailable |

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
prefill ms: 0.08
decode ms: 0.04
total ms: 0.13
tokens/s (total): 30840.40
tokens/s (decode): 66940.38
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
load ms: 0.556
sampling ms: 0.001
```

chat-cpu repeat 1

```text
prompt tokens: 2
generated tokens: 4
prefill ms: 564.26
decode ms: 932.88
total ms: 1497.14
tokens/s (total): 2.67
tokens/s (decode): 3.22
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
load ms: 2527.443
sampling ms: 0.463
```

tiny-q4_0-cpu repeat 1

```text
prompt tokens: 6
generated tokens: 4
prefill ms: 0.79
decode ms: 0.38
total ms: 1.17
tokens/s (total): 3423.32
tokens/s (decode): 7906.05
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
load ms: 0.568
sampling ms: 0.001
```
