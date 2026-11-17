# ROCm 云端部署学习笔记：vLLM 与 llama.cpp 对比

日期：2026-11-17
环境：AMD Radeon Cloud（算力云），单卡 Radeon PRO W7900D
模型：Gemma 4 E4B-it

---

## 1. 这次做了什么

1. 把 hello-rocm 仓库里的 Skill 装到 `.claude/skills/hello-rocm`。
2. 在算力云上启动一台 GPU 实例，验证 ROCm 与 PyTorch 可用。
3. 用 vLLM 部署 Gemma 4 E4B-it（BF16 原始权重）。
4. 用 llama.cpp 部署同一个模型（Q4_K_M 与 BF16 两种 GGUF）。
5. 用同一个脚本测速，比较两个引擎。

## 2. 环境与平台事实

| 项目 | 值 |
|---|---|
| GPU | AMD Radeon PRO W7900D，gfx1100，显存 48 GB（49136 MB） |
| 规格 | 1 GPU，16 核，55 GB 内存，1 积分/小时 |
| ROCm | 10.0.0，amdgpu 驱动 6.16.13，amd-smi 27.0.0 |
| PyTorch | 2.13.0+rocm10.0.0 |
| vLLM | 0.33.0（来自平台的 vLLM 镜像） |
| 积分兑换 | 1 积分 = 1 小时，100 积分起兑，券约 30 天有效 |

选镜像的结论：
- 想自己装环境：选 **PyTorch · Stable**。
- 想直接学 vLLM：选 **vLLM · Stable / Latest**（本次最终用了带 vLLM 的镜像）。
- 微调：选 **Unsloth · Fine-tuning**。

## 3. 验证环境的三条命令

```bash
rocminfo | grep -E "Marketing Name|gfx"
amd-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

期望：看到 gfx1100，`torch.cuda.is_available()` 返回 True。

## 4. 遇到的问题与解决办法

| 问题 | 原因 | 解决 |
|---|---|---|
| Notebook 里运行 `python -c "..."` 报 SyntaxError | 把 shell 命令粘进了 Python 单元格 | 行首加 `!`，或单元格首行写 `%%bash`，或改用终端 |
| `amd-smi` 提示 "Process Name may require elevated permissions" | 容器里查不到进程名，只影响显示 | 可忽略；用 `ps -p <PID>` 查进程 |
| `rocminfo \| head -30` 只看到 CPU | 输出被截断，GPU 在后面 | 用 `grep` 过滤 |
| 日志警告 `queue-intercept ... falling back` | 性能分析队列退回系统内存 | 可忽略 |
| GitHub 下载 llama.cpp 包只有约 59 KB/s | 实例出网走代理，限速 | 在 Mac 下载后上传，或换源 |
| `uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/` 报 tunnel error | 出网代理（Squid）白名单拒绝该域名，返回 403 | 改用平台自带的 vLLM 镜像 |
| 不能直接用 PyPI 装 vllm | PyPI 上是 CUDA 版，在 AMD 卡上会报 `libcudart.so` | 用 ROCm 专用 wheel 或平台镜像 |
| `unzip: command not found`，`apt` 找不到包 | 容器里没有包列表 | `python -m zipfile -e xxx.zip 目录`，再 `chmod +x` |
| `jq: command not found` | 未安装 | 用 Python 解析 JSON，或 `python -m json.tool` |
| llama.cpp 回答是在"续写试卷" | 用了 `/v1/completions`，没有套聊天模板 | 改用 `/v1/chat/completions` |
| 回答在中途被截断 | `finish_reason` 是 `length`，撞到 `max_tokens=512` | 调大到 2048；回答断掉时先看 `finish_reason` |
| 端口不是教程里的 8080 | llama-server 实际监听 9931 | 以启动日志 `listening on` 那一行为准 |

## 5. GGUF 与原始权重

| | 原始权重（safetensors） | GGUF |
|---|---|---|
| 组成 | 目录：config.json、tokenizer、`.safetensors` 分片 | 单个 `.gguf` 文件 |
| 精度 | 通常 BF16 或 FP16 | 常量化（如 Q4_K_M、Q8_0），也可以是 BF16 / F16 |
| 使用者 | Transformers、vLLM、SGLang、微调 | llama.cpp、Ollama、LM Studio |

要点：
- 与 vLLM 精度一致的是 **BF16 的 GGUF**，不是 Q8_0。
- Q4_K_M 的 "Q4" 约每权重 4 位，"K" 是 K-quant 分组方法，"M" 是中等档位。
- LM Studio 与 Ollama 底层都是 llama.cpp；vLLM 是独立引擎。

本次文件大小：Q4_K_M 5.1 GB，BF16 15 GB。

## 6. 部署命令

vLLM：
```bash
vllm serve ~/models/gemma-4-E4B-it \
  --served-model-name gemma-4-E4B-it \
  --max-model-len 4096
```

llama.cpp（预构建的 gfx110X Linux 包，覆盖 gfx1100）：
```bash
./llama-server -m <GGUF 路径> -ngl 99 -c 16384 -np 8
```
`-np 8` 表示 8 个并发槽位；`-c` 是所有槽位共享的上下文，每个槽位分到 `-c / -np`。

## 7. 测速方法

- 同一个客户端脚本（见附录），同一个提示词，`temperature=0`，`max_tokens=512`，流式输出。
- 一次只开一个服务，避免抢 GPU。
- 每个并发数先预热一轮（脚本会把它打印出来，读数时要跳过），再取三次有效结果的平均。
- llama.cpp 用 `-np 8` 启动，保证并发 8 时没有请求排队。

## 8. 结果

### 8.1 三方对比（去掉预热后的三次平均）

| 并发 | 指标 | vLLM（BF16） | llama.cpp（BF16） | llama.cpp（Q4_K_M） |
|---|---|---|---|---|
| 1 | 单请求解码 tok/s | **65.6** | 40.3 | 69.5 |
| 1 | 首 token 延迟 | 约 0.02 s | 约 0.16 s | 约 0.14 s |
| 4 | 总吞吐 tok/s | **241.9** | 141.7 | 180.1 |
| 8 | 总吞吐 tok/s | **374.2** | 159 到 272（不稳定） | 194.8 |
| 8 | 单请求解码 tok/s | 47.0 | 20 到 35（不稳定） | 24.7 |

### 8.2 显存

| | 占用 |
|---|---|
| vLLM | 约 41.7 GB（预先占住约九成显存，主要给 KV 缓存） |
| llama.cpp BF16 | 约 10.0 GB（文件 15 GB，显存比文件小，原因未查明） |
| llama.cpp Q4_K_M | 约 4.1 GB |

### 8.3 结论

1. 同为 BF16 时，vLLM 在所有并发数上都更快：单请求约 1.6 倍，并发 4 约 1.7 倍，并发 8 约 1.4 到 2.4 倍。
2. llama.cpp 用 Q4_K_M 后单请求速度提高约 1.7 倍，显存从 10 GB 降到 4.1 GB。所以单请求时，Q4 的 llama.cpp 与 BF16 的 vLLM 看起来持平，这是量化带来的，不是引擎本身一样快。
3. llama.cpp 的总吞吐在并发 4 以后基本不再增长；vLLM 到并发 8 仍在增长。
4. 只有一个人在本地用：llama.cpp 够用，显存占用小。给多个请求提供服务：用 vLLM。

### 8.4 数据可信度与局限

- vLLM 与 llama.cpp 并发 1 的结果很稳定。
- **llama.cpp BF16 并发 8 的结果分成两档（约 159 与约 240 到 272），原因未查明，不要取平均。**
- 并发 8 测试时的 `amd-smi` 快照：GFX 利用率约 59% 到 66%，功耗约 140 到 149 W（上限 241 W），说明 GPU 没有被压满。瓶颈可能在别处，这只是推测。vLLM 没有拍同样的快照，无法对比。
- 每个条件只测了三次，只用一种提示词和一种输出长度（512 token）。
- 提示词很短且每次相同，首 token 延迟不能代表长提示词的表现，vLLM 的前缀缓存可能也有影响。
- llama.cpp 启动日志有 "Flash Attention not supported, set to disabled"，可能拖慢并发表现，影响大小未量化。

## 9. 没有查明的问题（不要当作结论）

1. llama.cpp 第一次聊天请求在 `max_tokens=512` 下只显示了很短的内容就被截断。是否有隐藏内容占用了 token，没有检查 `usage` 来证实。
2. llama.cpp BF16 的 15 GB 文件只占 10 GB 显存，原因未查明（可能部分权重留在主机内存）。
3. llama.cpp BF16 并发 8 为什么出现两档吞吐。
4. Flash Attention 被禁用对结果的影响有多大。
5. vLLM 上 `--gpu-memory-utilization` 调低后，吞吐是否变化。

## 10. 下一步

- 进入微调：`docs/zh/02-fine-tune/gemma4`，需要 Unsloth 镜像或在当前镜像里装依赖。
- 补充实验：Q8_0 对比；vLLM 调低 `--gpu-memory-utilization`；长提示词的首 token 延迟；给 llama.cpp 查 Flash Attention 与 `offloaded` 日志。
- **用完后停止服务并停止实例，积分按小时扣。** 实例重建后，`/tmp` 里的内容会丢，模型和脚本要确认放在持久目录。

---

## 附录：测速脚本 `bench.py`

用法：`python bench.py <接口地址> <模型名> <并发数>`

```python
import sys, json, time, threading, urllib.request

url, model = sys.argv[1], sys.argv[2]
conc = int(sys.argv[3]) if len(sys.argv) > 3 else 1
PROMPT = "请写一篇不少于 800 字的文章，介绍 AMD ROCm 的发展历史、核心组件和应用场景。"

def one(res, i):
    body = json.dumps({
        "model": model, "temperature": 0, "max_tokens": 512, "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": PROMPT}],
    }).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = None; chunks = 0; usage = None
    with urllib.request.urlopen(req) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]["completion_tokens"]
            ch = d.get("choices") or []
            if ch and (ch[0]["delta"].get("content") or ch[0]["delta"].get("reasoning_content")):
                chunks += 1
                if first is None:
                    first = time.perf_counter()
    t1 = time.perf_counter()
    res[i] = (first - t0, t1 - first, usage if usage else chunks, usage is not None)

def run():
    res = [None] * conc
    ts = [threading.Thread(target=one, args=(res, i)) for i in range(conc)]
    t0 = time.perf_counter()
    [t.start() for t in ts]; [t.join() for t in ts]
    wall = time.perf_counter() - t0
    toks = sum(r[2] for r in res)
    ttft = sum(r[0] for r in res) / conc
    dec = sum(r[2] / r[1] for r in res) / conc
    print(f"并发={conc} 首token延迟={ttft:.2f}s 单请求解码={dec:.1f} tok/s "
          f"总吞吐={toks/wall:.1f} tok/s 总token={toks} usage精确={res[0][3]}")

run()  # 预热，丢弃
for _ in range(3):
    run()
```
