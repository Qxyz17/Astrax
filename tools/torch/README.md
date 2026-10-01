# Astrax GPU 训练（PyTorch）

这套脚本用 PyTorch 在 GPU（或 CPU）上训练与 C++ 版本**完全相同的架构**：

- 子词嵌入 → 池化输入特征
- Osten condition 向量
- 共享隐藏投影 + 每 slot 的 slot/round 嵌入
- 密集字节分类头（256 字节 + 1 stop）
- 多轮**并行整段去噪**（非 Transformer、非 next-token）

权重可在 PyTorch 与 C++ 之间互换。

## 依赖

```bash
pip install torch
```

## 训练

```bash
python tools/torch/train_astrax.py --document-epochs 200 --device cuda
```

常用参数：

| 参数 | 说明 | 默认 |
|---|---|---|
| `--document-epochs` | 训练轮数 | 200 |
| `--batch-size` | 批大小 | 16 |
| `--learning-rate` | 学习率 | 2e-3 |
| `--max-documents` | 最多使用文档数 | 100000 |
| `--max-slots` | 每文档最大字节数 | 64 |
| `--device` | `cuda` 或 `cpu` | 自动 |
| `--checkpoint` | 输出权重路径 | artifacts/torch_dialogue.pt |
| `--resume` | 从 checkpoint 续训 | 关 |

训练前请先准备语料与子词表（见 `docs/TRAINING.md`）。

## 导出为 C++ checkpoint

训练完成后，导出成 C++ 运行时能加载的格式：

```bash
python tools/torch/export_checkpoint.py \
  --input artifacts/torch_dialogue.pt \
  --output artifacts/astrax_training.astrax-model
```

之后 `astrax_chat.exe` 即可使用 GPU 训练出的对话模型。

> 注意：导出的 checkpoint 中，离线 RL 模型（状态预测器 / 动作价值 / 内在奖励）
> 使用确定性初始值；若需要它们也被训练，请再运行一次 C++ `astrax_train.exe`。
