# 命令清单（Radeon Cloud，W7900D / gfx1100）

说明：
- 标 [已验证] 的命令，在本次实例里跑通过。
- 标 [未验证] 的命令，只是写法参考，没有在本次实例里跑过。
- 不要把 `HF_TOKEN` 等密钥写进这个文件。

## 1. 环境检查 [已验证]

```bash
rocminfo | grep -E "Marketing Name|gfx"
amd-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import vllm, torch; print(vllm.__version__, torch.__version__, torch.cuda.is_available())"
```

在 Notebook 单元格里，每行前加 `!`，或第一行写 `%%bash`。长任务建议在终端里运行。

## 2. 出网诊断 [已验证]

```bash
echo $HTTPS_PROXY $HTTP_PROXY
curl -sI -m 10 https://wheels.vllm.ai/rocm/vllm/ | head -3   # 本次返回 403，被白名单拦截
curl -sI -m 10 https://pypi.org/simple/ | head -3            # 本次返回 200
pip config list                                              # 默认源是清华镜像
```

## 3. 下载模型

下载 BF16 的 GGUF（用于 llama.cpp）[已验证]：
```bash
cd /workspace
modelscope download --model bartowski/google_gemma-4-E4B-it-GGUF \
  google_gemma-4-E4B-it-bf16.gguf \
  --local_dir ./models/gemma-4-E4B-it-GGUF/
```

Q4_K_M 的 GGUF：把文件名换成 `google_gemma-4-E4B-it-Q4_K_M.gguf` [未验证：我没有看到你下载它用的命令，请对照你自己的历史]。

vLLM 用的原始权重 `google/gemma-4-E4B-it` [未验证：我没有看到你下载它用的命令，请对照你自己的历史]。需要先在 Hugging Face 页面同意协议，并使用你自己的 token。

确认文件大小：
```bash
ls -lh /workspace/models/gemma-4-E4B-it-GGUF/
```
本次的大小：Q4_K_M 约 5.1 GB，BF16 约 15 GB。

## 4. 解压 llama.cpp 预构建包 [已验证]

容器里没有 `unzip`，用 Python 解压：
```bash
python -m zipfile -e llama-gfx110X.zip llama-gfx110X
chmod +x llama-gfx110X/*
```
包来自 llama.cpp 的 ROCm 预构建发布页，选 gfx110X 的 Linux 版本。GitHub 在实例里下载很慢，可以在 Mac 下载后上传。

## 5. 启动服务

llama.cpp [已验证]：
```bash
./llama-server -m /workspace/models/gemma-4-E4B-it-GGUF/google_gemma-4-E4B-it-Q4_K_M.gguf \
  -ngl 99 -c 16384 -np 8 2>&1 | tee /workspace/llama-server.log
```
- BF16 的 GGUF 只需要换 `-m` 后面的文件名。
- 端口看启动日志里的 `listening on`，本次是 9931。

vLLM [已验证，模型路径按你的实际目录修改]：
```bash
vllm serve ~/models/gemma-4-E4B-it \
  --served-model-name gemma-4-E4B-it \
  --max-model-len 4096 2>&1 | tee /workspace/vllm.log
```
本次 vLLM 的端口是 8000。

## 6. 调用接口 [已验证]

要用聊天接口，`/v1/completions` 不会套聊天模板：
```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma-4-E4B-it","messages":[{"role":"user","content":"用一句话介绍 ROCm。"}],"max_tokens":1024}'
```
`max_tokens` 不要设太小，回答被截断时先看 `finish_reason`。

## 7. 测速 [已验证]

脚本见 `2026-11-17-rocm-cloud-deploy-notes.md` 的附录，保存为 `/workspace/bench.py`。

```bash
python /workspace/bench.py http://127.0.0.1:8000/v1/chat/completions gemma-4-E4B-it 1
python /workspace/bench.py http://127.0.0.1:8000/v1/chat/completions gemma-4-E4B-it 4
python /workspace/bench.py http://127.0.0.1:8000/v1/chat/completions gemma-4-E4B-it 8
```
- 每组输出的第一行是预热，读数时跳过。
- 一次只开一个服务。

## 8. 清理

```bash
amd-smi                 # 确认显存已释放
ps -p <PID> -o pid,cmd  # 查进程名，amd-smi 里进程名显示为 N/A 是正常的
```
测完停掉服务，并在平台上停止实例，积分按小时扣。
