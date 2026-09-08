# Astrax

Astrax 是运行在 Osten 之上的本地数字模型上层。
Osten 是固定规模的决策引擎；Astrax 负责把它组织成具有身份、状态、目标、记忆、自省和训练流程的模型。

实现方式：`C++17 + MSVC + 原生 Visual Studio 工程`  
构建系统：不使用 CMake

## 当前已经实现

- 调用 Osten 进行每一步核心前向决策
- 状态管理：版本、迭代次数、运行时间、记忆数量、目标和上次输出
- 有界向量记忆：写入、相似度检索、使用次数更新和遗忘
- 目标设置、清除和优先级限制
- 结构化自省：读取迭代、目标、运行时间和记忆
- 状态预测模型
- 动作价值模型
- 基于状态新颖度的内在奖励
- 基础 Offline RL / Q-learning 训练器
- Text、Code、State 三种输出模式；Text/Code 使用并行整段迭代去噪生成，不使用 next-token 或自回归生成
- `Text / Code / Image / Audio / Video / Binary` 输入模态接口

## 当前能力边界

`astrax_demo.exe` 会先训练一组确定性的 starter offline dataset。这个训练证明：

1. 状态预测器参数会根据数据更新；
2. 动作价值会根据奖励和下一状态更新；
3. 内在奖励会统计状态新颖度；
4. Osten 会根据编码后的输入、状态、目标、记忆上下文和心跳继续前向运行。

当前实现提供非 Transformer、非 next-token 文本/代码模型的实验路径。文本训练使用输入-目标对做整体文档去噪重建；它仍是小规模研究模型，实际通用问答和代码能力必须以独立评测结果为准。

因此，当前可以说 Astrax 已经具备**可训练的数字模型骨架、动作价值决策、外部反馈接口和非自回归文本生成路径**。当前训练规模不足以证明达到成熟通用自然语言模型或可靠代码生成模型水平，也不代表具备类似人的意识。

## 构建和运行

在 Visual Studio Developer PowerShell 中执行：

```powershell
.\build-msvc.ps1
```

或者直接打开 `Astrax.sln`，选择 `Release | x64` 生成。输出：

```text
build\Release\astrax.lib
build\Release\astrax_tests.exe
build\Release\astrax_train.exe
build\Release\astrax_demo.exe
```

运行：

```powershell
.\build\Release\astrax_tests.exe
.\build\Release\astrax_train.exe
.\build\Release\astrax_demo.exe
.\build\Release\astrax_demo.exe "write a small hello world program"
```

## 训练

### 在线反馈

`step()` 产生一个待反馈转移。环境执行动作后，调用：

```cpp
const auto output = model.step(input);
model.observe_feedback(next_state, reward, terminal);
```

其中 `next_state` 必须包含 `ModelConfig::state_dim` 个位于 `[-1, 1]` 的有限值。该接口会使用真实的 `reward` 和 `next_state` 更新动作价值模型。没有外部反馈时，不应把内在新颖度当作任务成功奖励。

`astrax_train.exe` 会生成并训练一组符合约束的离线轨迹：

```text
episodes = 128
horizon = 32
state_dim = 32
action_count = 8
samples = 4096
```

数据会写入：

```text
data\astrax_offline_transitions.csv
artifacts\offline_training_report.txt
artifacts\astrax_training.astrax-model
```

每一行都是合法的 `state/action/reward/next_state/terminal` 转移。状态有限且位于 `[-1, 1]`，动作位于 `[0, action_count)`，终止标记只取 `0/1`。训练前会重新从 CSV 加载并执行完整校验。
数据字段和生成规则见 `data\README.md`。

### 输入-目标文本训练

自然语言和代码训练样本位于 `data\astrax_dialogue_pairs.tsv`，格式为：

```text
input<TAB>target
```

训练程序会将输入和目标分开编码与监督，目标不会泄漏到输入。当前样本用于验证训练链路，不足以证明通用语言能力；中文、英文和代码质量必须通过独立评测确认。

执行：

```powershell
.\build\Release\astrax_train.exe
```

底层训练调用：

```cpp
const auto dataset = astrax::make_starter_dataset(
    model.config().state_dim,
    model.config().action_count);

const astrax::TrainingReport report =
    model.train_offline(dataset, 24);
```

也可以从 CSV 加载离线转移数据：

```text
action,reward,terminal,state,next_state
0,1.0,0,0;0;0;0,0.1;0;0;0
```

```cpp
model.train_offline_csv("dataset.csv", 10);
```

CSV 每行包含 `action,reward,terminal,state,next_state`；`state` 和 `next_state` 必须包含 `ModelConfig::state_dim` 个以分号分隔的浮点数。

训练程序会保存可审计的训练报告和 checkpoint。chat 单文件程序不依赖这些外部训练产物，而是在可执行文件内置的配对语料上启动时训练。

starter dataset 只是验证训练闭环的数据，不代表真实语言或代码能力。当前语料文件包含重复行，训练入口会先去重并受配置上限限制；报告中的 `text_training_documents_unique` 是实际进入训练的独立文档数量。要训练出可靠的通用输出，需要继续准备真实的输入-目标文本数据、状态-动作-奖励-下一状态轨迹，并实现完整评测。

训练报告中的 `validation_codepoint_accuracy`、`validation_answer_accuracy`、`validation_exact_match`、`validation_non_repetition` 和 `validation_code_compile_rate` 只描述独立验证集结果。当前代码编译率评测尚未执行编译器验证，因此报告中的该项为 `0`，不能解释为代码质量证明。

## 架构

```text
输入文本 / 特征 / 未来模态
            │
            ▼
       Astrax embed
            │
            ├── MemoryStore：写入和提供上下文
            ├── StatePredictor：预测状态转移
            ├── ActionValueModel：估计动作价值
            ├── IntrinsicRewardModel：计算状态新颖度
            └── OfflineRLTrainer：离线训练
            │
            ▼
       Osten::OstenEngine::tick
            │
            ▼
        ControlledRenderer
        Decision / State
```

Osten 的固定规模网络产生状态转移和动作候选，Astrax 的动作价值模型对候选动作进行最终评估。记忆检索只作为上下文，不能直接选择答案或动作。Text/Code 由独立的并行整段迭代去噪模块生成，不是核心动作选择器，也不使用 next-token 机制。

## 单文件 Chat

`astrax_chat.exe` 是自包含构建：配对训练样本和训练配置编译进可执行文件，chat 启动时在内存中训练，不读取外部 corpus、checkpoint 或 artifacts 文件。Release 构建使用静态 MSVC runtime；产品版本号不写入源代码或运行时状态，只通过 Git commit 和 release 管理。

## 后续路线

1. 更大且经过许可审查的真实文本/代码文档语料；
2. 真实离线轨迹数据集与数据校验；
3. 文本/代码质量评测和结构化解码指标；
4. 状态预测、动作连续性、目标达成率、记忆利用率和输出一致性评测；
5. 图像、音频、视频编码器，统一接入 `MultimodalInput::features`；
6. 经验证、可回滚的权重热更新。

这些属于 Astrax 层，不改变 Osten 的核心拓扑，也不引入 CMake。

## 目录

```text
include\astrax\   公共 C++ 接口
src\              Astrax 实现
tests\            原生 C++ 测试
Astrax.sln        Visual Studio 解决方案
*.vcxproj         MSVC 工程文件
todolist.md        需求和硬约束
```
