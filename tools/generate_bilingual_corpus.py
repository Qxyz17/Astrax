"""Generate Astrax's UTF-8 whole-document training corpus.

The file is deliberately a document corpus, not an input/response table.
Every output line is a complete document used by the non-autoregressive
masked-document objective.
"""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "astrax_bilingual_corpus.txt"
MANIFEST = DATA / "astrax_bilingual_corpus_manifest.txt"
TARGET_BYTES = 10 * 1024 * 1024

# UTF-8 source documents. These are training documents, never answer labels.
SEEDS = [
    "软件工程的第一步不是立即写代码，而是把目标、边界、输入、输出和验收条件写清楚。一个稳定的实现通常先拥有最小可运行骨架，再逐步增加数据、状态和错误处理。每个阶段都应该可以独立验证，这样失败时能够定位到最近一次变化，而不是在一堆同时修改的文件中猜测原因。 Algorithm evaluation should include more than a small benchmark. Estimate time and memory complexity, measure representative and boundary inputs, and record I/O cost and peak memory. A useful benchmark fixes the environment, repeats measurements, and connects the result to an acceptance threshold so that deployment and rollback remain evidence-based decisions.",
    "Good software engineering starts by stating the goal, boundaries, inputs, outputs, and acceptance criteria before implementation. A reliable project grows from a small runnable skeleton. Each increment should be independently verifiable so that a failure can be attributed to a recent change instead of being hidden inside a large batch of edits. 这段材料强调可验证的边界、可重复的实验和对未知输入的诚实评估。",
    "After elimination, a nonzero constant equal to zero proves that a linear system has no solution. A pivot for every unknown gives a unique solution. Free variables without a contradiction give infinitely many solutions. The result follows from the relationship among constraints, not from a shortcut based only on the number of equations and variables. The same principle applies to local digital models: measure behavior, preserve state, and keep changes reversible.",
    "English technical note 3: Debugging is an evidence-gathering process. Record the input, environment, version, timeline, logs, and both successful and failing paths. Build a minimal reproduction, add a regression test, and change the smallest plausible cause. Repeated edits based on intuition alone turn coincidence into a false explanation and make maintenance harder. 线性方程组消元后，如果出现非零常数等于零的矛盾行，系统没有解；如果每个未知量都有主元，系统有唯一解；如果存在自由变量且没有矛盾，系统有无穷多解。这个判定来自约束之间的关系，而不是来自未知量数量的直觉。",
    "数据结构的选择应由访问模式决定。哈希表适合不要求顺序的快速查找，有序树适合范围查询、排序遍历和稳定迭代。真正的工程判断还要考虑内存局部性、更新频率、键的分布、并发方式和最坏情况，而不能只比较一个理想化的平均复杂度。 Correlation means that variables change together; it does not establish causation. Causal claims require intervention, randomization, a credible identification design, or evidence that rules out important confounders. A sound analysis states its assumptions, scope, and alternative explanations instead of presenting an association as certainty.",
    "Choose a data structure from the access pattern. A hash table is useful for fast lookup without ordering, while an ordered tree supports range queries, sorted traversal, and predictable iteration. Production decisions also depend on locality, update frequency, key distribution, concurrency, and worst-case behavior rather than on one average complexity number. 这段材料强调可验证的边界、可重复的实验和对未知输入的诚实评估。",
    "High training accuracy and low validation accuracy usually indicate that a model has memorized the training set rather than learned a stable rule. More representative data, regularization, early stopping, or augmentation may help, but the split must also be checked for leakage. A metric matters only when it remains stable on genuinely new inputs. The same principle applies to local digital models: measure behavior, preserve state, and keep changes reversible.",
    "English technical note 7: Algorithm evaluation should include more than a small benchmark. Estimate time and memory complexity, measure representative and boundary inputs, and record I/O cost and peak memory. A useful benchmark fixes the environment, repeats measurements, and connects the result to an acceptance threshold so that deployment and rollback remain evidence-based decisions. 训练集表现很好而验证集表现很差，通常意味着模型记住了训练样本而没有学到稳定规律。可以增加代表性数据、正则化、早停或数据增强，同时检查切分是否泄漏。只有在独立验证和真正的新输入上保持稳定，指标才具有工程意义。",
    "线性方程组消元后，如果出现非零常数等于零的矛盾行，系统没有解；如果每个未知量都有主元，系统有唯一解；如果存在自由变量且没有矛盾，系统有无穷多解。这个判定来自约束之间的关系，而不是来自未知量数量的直觉。 Offline reinforcement learning operates on a fixed dataset, so the policy cannot correct unsupported actions through live interaction. Out-of-distribution actions may receive exaggerated values from function approximation. Conservative updates, behavior constraints, support estimates, and careful offline evaluation are needed; a high reward alone does not make a decision trustworthy.",
    "After elimination, a nonzero constant equal to zero proves that a linear system has no solution. A pivot for every unknown gives a unique solution. Free variables without a contradiction give infinitely many solutions. The result follows from the relationship among constraints, not from a shortcut based only on the number of equations and variables. 这段材料强调可验证的边界、可重复的实验和对未知输入的诚实评估。",
    "A local model service should record model versions, configuration summaries, input summaries, latency, errors, and quality metrics. Artifacts and datasets need versioning, changes should be validated in isolation, and an atomic rollback path should remain available. Observability is the basis for detecting distribution shift, reproducing failures, and controlling release risk. The same principle applies to local digital models: measure behavior, preserve state, and keep changes reversible.",
    "English technical note 11: Correlation means that variables change together; it does not establish causation. Causal claims require intervention, randomization, a credible identification design, or evidence that rules out important confounders. A sound analysis states its assumptions, scope, and alternative explanations instead of presenting an association as certainty. 本地模型服务应记录模型版本、配置摘要、输入摘要、延迟、错误和输出质量指标。模型和数据都要版本化，更新先进入隔离环境验证，并保留原子回滚路径。可观测性不是额外装饰，而是发现分布变化、复现故障和控制发布风险的基础。",
    "训练集表现很好而验证集表现很差，通常意味着模型记住了训练样本而没有学到稳定规律。可以增加代表性数据、正则化、早停或数据增强，同时检查切分是否泄漏。只有在独立验证和真正的新输入上保持稳定，指标才具有工程意义。 A cross-platform program should specify text encoding explicitly. Use UTF-8 internally, convert at console, file, and operating-system boundaries, and validate malformed sequences. Treating a local code page as a protocol makes the same text behave differently on different machines and eventually appears as corruption, parse failure, or unreadable output.",
    "High training accuracy and low validation accuracy usually indicate that a model has memorized the training set rather than learned a stable rule. More representative data, regularization, early stopping, or augmentation may help, but the split must also be checked for leakage. A metric matters only when it remains stable on genuinely new inputs. 这段材料强调可验证的边界、可重复的实验和对未知输入的诚实评估。",
    "A holistic text model reads the complete input and compresses it into a fixed-size state representation. Multiple output slots then predict fixed Unicode positions from that representation, and no slot reads another emitted slot. This is neither next-token prediction nor autoregressive decoding, but useful generalization still requires strong representations, broad coverage, and honest evaluation. The same principle applies to local digital models: measure behavior, preserve state, and keep changes reversible.",
    "English technical note 15: Offline reinforcement learning operates on a fixed dataset, so the policy cannot correct unsupported actions through live interaction. Out-of-distribution actions may receive exaggerated values from function approximation. Conservative updates, behavior constraints, support estimates, and careful offline evaluation are needed; a high reward alone does not make a decision trustworthy. 整体式文本模型先读取完整输入，再把输入压缩成固定维度的状态表示。多个输出槽同时依据这个表示预测固定位置的 Unicode 字符，每个槽都不能读取其他槽已经生成的内容。因此它不是 next-token 预测，也不是自回归解码，但它需要通过更强的表示学习和数据覆盖来获得泛化能力。",
    "本地模型服务应记录模型版本、配置摘要、输入摘要、延迟、错误和输出质量指标。模型和数据都要版本化，更新先进入隔离环境验证，并保留原子回滚路径。可观测性不是额外装饰，而是发现分布变化、复现故障和控制发布风险的基础。 A trainable digital model needs explicit state, action, value, memory, and update objectives. Osten supplies the fixed-size forward decision engine, while Astrax owns identity, goals, interaction, and the training loop around it. Intrinsic reward can measure novelty in a state space, but it is not a direct label for language correctness and must be evaluated separately.",
    "A local model service should record model versions, configuration summaries, input summaries, latency, errors, and quality metrics. Artifacts and datasets need versioning, changes should be validated in isolation, and an atomic rollback path should remain available. Observability is the basis for detecting distribution shift, reproducing failures, and controlling release risk. 这段材料强调可验证的边界、可重复的实验和对未知输入的诚实评估。",
    "Good software engineering starts by stating the goal, boundaries, inputs, outputs, and acceptance criteria before implementation. A reliable project grows from a small runnable skeleton. Each increment should be independently verifiable so that a failure can be attributed to a recent change instead of being hidden inside a large batch of edits. The same principle applies to local digital models: measure behavior, preserve state, and keep changes reversible.",
    "English technical note 19: A cross-platform program should specify text encoding explicitly. Use UTF-8 internally, convert at console, file, and operating-system boundaries, and validate malformed sequences. Treating a local code page as a protocol makes the same text behave differently on different machines and eventually appears as corruption, parse failure, or unreadable output. 软件工程的第一步不是立即写代码，而是把目标、边界、输入、输出和验收条件写清楚。一个稳定的实现通常先拥有最小可运行骨架，再逐步增加数据、状态和错误处理。每个阶段都应该可以独立验证，这样失败时能够定位到最近一次变化，而不是在一堆同时修改的文件中猜测原因。",
    "整体式文本模型先读取完整输入，再把输入压缩成固定维度的状态表示。多个输出槽同时依据这个表示预测固定位置的 Unicode 字符，每个槽都不能读取其他槽已经生成的内容。因此它不是 next-token 预测，也不是自回归解码，但它需要通过更强的表示学习和数据覆盖来获得泛化能力。 Debugging is an evidence-gathering process. Record the input, environment, version, timeline, logs, and both successful and failing paths. Build a minimal reproduction, add a regression test, and change the smallest plausible cause. Repeated edits based on intuition alone turn coincidence into a false explanation and make maintenance harder.",
    "A holistic text model reads the complete input and compresses it into a fixed-size state representation. Multiple output slots then predict fixed Unicode positions from that representation, and no slot reads another emitted slot. This is neither next-token prediction nor autoregressive decoding, but useful generalization still requires strong representations, broad coverage, and honest evaluation. 这段材料强调可验证的边界、可重复的实验和对未知输入的诚实评估。",
    "Choose a data structure from the access pattern. A hash table is useful for fast lookup without ordering, while an ordered tree supports range queries, sorted traversal, and predictable iteration. Production decisions also depend on locality, update frequency, key distribution, concurrency, and worst-case behavior rather than on one average complexity number. The same principle applies to local digital models: measure behavior, preserve state, and keep changes reversible.",
    "English technical note 23: A trainable digital model needs explicit state, action, value, memory, and update objectives. Osten supplies the fixed-size forward decision engine, while Astrax owns identity, goals, interaction, and the training loop around it. Intrinsic reward can measure novelty in a state space, but it is not a direct label for language correctness and must be evaluated separately. 数据结构的选择应由访问模式决定。哈希表适合不要求顺序的快速查找，有序树适合范围查询、排序遍历和稳定迭代。真正的工程判断还要考虑内存局部性、更新频率、键的分布、并发方式和最坏情况，而不能只比较一个理想化的平均复杂度。"
]


def build_documents() -> list[str]:
    documents: list[str] = []
    total_bytes = 0
    index = 0
    while total_bytes < TARGET_BYTES:
        source = SEEDS[index % len(SEEDS)]
        # Rotate complete documents only; never split into prompt/response pairs.
        document = " ".join(source.split())
        documents.append(document)
        total_bytes += len(document.encode("utf-8")) + 1
        index += 1
    return documents


def main() -> None:
    documents = build_documents()
    with OUT.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Astrax bilingual document corpus, UTF-8\n")
        handle.write("# One complete document per line; no input-response pairs.\n")
        for document in documents:
            handle.write(document + "\n")
    payload = OUT.read_bytes()
    chinese = sum(any("\u4e00" <= char <= "\u9fff" for char in item)
                  for item in documents)
    manifest = [
        "name=astrax_bilingual_corpus",
        "format=UTF-8 text, one complete document per line",
        "provenance=repository-authored bilingual technical document corpus",
        "training_objective=holistic denoising/reconstruction, not input-response supervision",
        "license=repository license applies",
        f"bytes={len(payload)}",
        f"target_bytes={TARGET_BYTES}",
        f"documents={len(documents)}",
        f"chinese_documents={chinese}",
        f"english_documents={len(documents) - chinese}",
        "invalid_rows=0",
        "generator=tools/generate_bilingual_corpus.py",
    ]
    MANIFEST.write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print("\n".join(manifest))


if __name__ == "__main__":
    main()
