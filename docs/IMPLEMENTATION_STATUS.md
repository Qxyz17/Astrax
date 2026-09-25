# Astrax 当前实现状态（诚实记录）

日期：2026-09-21

## 已完成并验证

- `include/astrax/charset.hpp`：完整 Unicode 码点空间 + 因子化映射（high 1088 / low 1024）。
- 因子化输出层替换数据驱动词表（`dialogue.hpp` / `dialogue.cpp`）。
- 真实 Osten condition 注入：`AstraxModel::conditions_for()` 对每个文档调用 `engine_.tick`。
- 多线程数据并行训练（replica averaging，4 核约 3.4× 加速）。
- 真实 Wikipedia 语料：`data/wiki_zh.txt` + `data/wiki_en.txt`，共 104.9 MB，6048 文档。
- 完整训练跑通：5928 文档 × 4 epoch，checkpoint 2.1 MB，`checkpoint_verified=1`。
- 全部单元测试通过。

## 实测结果（失败点）

`astrax_chat.exe` 对任意输入输出相同的字符级乱码，例如：

```text
输入: Anarchism / What is anarchism? / hello world
输出: Aeteaddetrt stre, ˺){rtemaina+t{ r tstdr:"ͮut <<rwHrlrai w.rIt!
```

特征：
- 不同输入 → 输出**相同**。
- 同输入连续两次 → 输出**相同**。
- 输出包含训练语料的局部字符 n-gram 片段，但无语义、无句法。

## 根因

1. 训练目标是"完整文档重建自身"（source == target），接近恒等映射，
   模型只需拟合高频字符模式即可降低 loss。
2. 哈希编码器没有语义表示能力（同义词、词形、中英混合在编码层无共享结构）。
3. Osten 固定随机权重未训练，`action` 恒为 4、`action_value` 恒为 0，
   条件信号对输出无实际影响，因此输出不随输入或思考状态变化。
4. `text_codepoint_accuracy` 只衡量逐位置 argmax 命中率，不能代表生成质量。

## 待决策

见 `docs/MAJOR_REFACTOR_PROPOSAL.md`。核心瓶颈是编码器（方向一）与训练目标。
