# Astrax 训练指南

Astrax 提供两条训练路径，可同时使用：

- **CPU 路径**：仓库内原生 C++ 训练器（`astrax_train.exe`），数据并行，多核。
- **GPU 路径**：`tools/torch/train_astrax.py`，用 PyTorch 在 GPU 上训练相同架构。

两条路径训练的是同一个模型，权重可以互相转换。

---

## 语料准备

真实 Wikipedia 语料通过脚本下载（约 100 MB，中英各 ~50 MB）：

```powershell
python tools\wiki\extract_wiki_corpus.py --language zh --target-mib 50
python tools\wiki\extract_wiki_corpus.py --language en --target-mib 50
```

输出 `data/wiki_zh.txt` 与 `data/wiki_en.txt`，每行一篇完整文档。

## 子词表

输入编码器使用 BPE 子词表，离线学习：

```powershell
python tools\bpe\learn_bpe.py --merges 8000 --max-megabytes 40 --max-words 200000
```

输出 `data/astrax_subwords.bin`（C++ 读取）与 `.txt`（可读）。

---

## CPU 路径

```powershell
.\build\Release\astrax_train.exe [选项]
```

选项：

| 选项 | 说明 | 默认 |
|---|---|---|
| `--document-epochs N` | 文档重建 epoch 数 | 4 |
| `--pair-epochs N` | 输入-目标对微调 epoch 数 | 48 |
| `--resume` | 从已有 checkpoint 继续训练 | 关 |
| `--checkpoint PATH` | checkpoint 路径 | artifacts/astrax_training.astrax-model |

长时间训练示例：

```powershell
# 首次
.\build\Release\astrax_train.exe --document-epochs 200 --pair-epochs 200

# 中断后继续（会加载已有 checkpoint）
.\build\Release\astrax_train.exe --document-epochs 200 --resume
```

CPU 训练是数据并行的，会自动使用所有 CPU 核心。训练进度可从进程 CPU 时间观察；
最终结果写入 `artifacts/offline_training_report.txt` 与 checkpoint。

---

## GPU 路径

见 `tools/torch/README.md`。

---

## 评估

`astrax_chat.exe` 从 `artifacts/astrax_training.astrax-model` 加载 checkpoint 并对话：

```powershell
.\build\Release\astrax_chat.exe "What is anarchism?"
```

训练报告中的 `validation_codepoint_accuracy` 等指标只描述独立验证集，
不代表通用语言能力。
