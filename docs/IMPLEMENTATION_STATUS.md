# Astrax 当前实现状态（诚实记录）

日期：2026-09-25

## 本轮完成

- 完整 Unicode 码点空间 + 因子化输出（high 1088 / low 1024），
  模型可从完整 UTF-8 中自选字符（`charset.hpp`）。
- 真实 Osten condition 注入（`conditions_for` 每文档 `engine_.tick`）。
- 多线程数据并行训练（replica averaging，4 核约 3.4×）。
- 真实 Wikipedia 语料 104.9 MB（`data/wiki_zh.txt` + `wiki_en.txt`，6048 文档）。
- BPE 子词表 8000 merges（`tools/bpe/learn_bpe.py`，真实子词如 中国/人民/共和）。
- 子词输入编码器 + 可训练嵌入表（`embedding.hpp` / `embedding.cpp`），
  嵌入梯度经 `train_pass` 反向传播更新。
- 掩码文档去噪训练目标（`mask_document`，遮盖约 20% 输入码点）。
- 训练跑通：5928 文档 × 4 epoch，checkpoint 9.8 MB，checkpoint_verified=1。
- 全部单元测试通过。

## 仍未达成（关键）

`astrax_chat.exe` 输出仍是字符级乱码，不是连贯语言。例如：

```text
输入: What is anarchism?
输出: #it:lddti<i,stieȡ b)ire maind) b> ȡ:dnud:uiut n< "dild \ nt  d
```

- `validation_codepoint_accuracy = 0`
- `validation_exact_match = 0`
- `validation_answer_accuracy = 0`

## 根因判断

1. 输出空间为完整 21 位 Unicode（~111 万码点），经因子化后仍很稀疏，
   字符级学习效率低。
2. 模型容量小（hidden 192，4 epoch，6048 文档），训练信号不足以学出语法。
3. 并行整段去噪相对 next-token 更难优化，在有限算力下收敛慢。
4. Osten 为固定随机权重，其状态对输出的影响有限。

## 待决策方向

- A: 接受当前为“架构骨架 + 训练链路”，如实记录，停止。
- B: 输出单位改为 byte（256 类），大幅降低学习难度（仍为并行 slot 去噪）。
- C: 大幅增大模型与训练量。
