# 在本地运行 LFM2-350M（Liquid AI）

本目录的脚本让你在**普通 CPU 机器**上运行 Liquid AI 的 LFM2-350M 模型，
输出连贯语言，无需 GPU。

## 为什么是 LFM2-350M

LFM2 是 Liquid AI 的混合架构（门控短卷积 + 分组查询注意力），以高效著称：

- **350M 参数**，Q4 量化后仅 **229 MB**，可在 8 GB 内存的普通笔记本运行；
- 在 CPU 上生成速度约 **40–55 token/秒**（i7-7500U 实测）；
- 英文问答、解释、对话质量良好；
- 中文能力较弱（350M 小模型的常见局限）。

> 说明：LFM2 的官方训练用了 10 万亿 token，是**大规模预训练**的结果。
> 本项目不重新训练它，而是**直接运行**官方发布的量化模型。

## 依赖

无需 Python 包，只需：

1. `data/LFM2-350M-Q4_K_M.gguf`（模型，约 229 MB）
2. `build/llama-cpu/bin/llama-cli.exe`（推理引擎，从源码编译）

### 下载模型

```bash
python tools/lfm/download_model.py
```

### 编译推理引擎（llama.cpp，CPU 版）

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git tools/llama.cpp
"D:/Software/VisualStudio/2022/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe" ^
  -S tools/llama.cpp -B build/llama-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF
"D:/Software/VisualStudio/2022/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe" ^
  --build build/llama-cpu --config Release
```

## 使用

### 单次提问

```bat
chat_lfm2.bat "Explain what a neural network is in two sentences."
```

### 交互对话

```bat
chat_lfm2.bat
```

进入后输入问题，`/exit` 退出。

### 直接调用 llama-cli

```bat
build\llama-cpu\bin\llama-cli.exe -m data\LFM2-350M-Q4_K_M.gguf -p "your question" -n 256 -t 4 -st
```

参数：

| 参数 | 含义 |
|---|---|
| `-m` | 模型文件 |
| `-p` | 提问 |
| `-n` | 最多生成 token 数 |
| `-t` | 线程数（4 = 你的 4 线程） |
| `-st` | 单轮后退出 |
| `-cnv` | 对话模式（交互） |

## 硬件要求

| 项 | 最低 | 你的机器 |
|---|---|---|
| 内存 | 2 GB | 8 GB ✅ |
| CPU | 支持 AVX2 | i7-7500U（AVX2）✅ |
| GPU | 不需要 | 940MX 用不上，纯 CPU 即可 |

## 与 Astrax 的关系

- **Astrax**：本项目自研的“有状态、会演化的非纯预测系统”骨架（见 `docs/research/`）。
- **LFM2 运行**：一个**可直接使用的语言模型**，让你在本机获得连贯输出。

两者独立：Astrax 是架构研究，LFM2 是可用的语言能力。
