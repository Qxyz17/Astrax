# Astrax

Astrax 是运行在 Osten 之上的本地数字模型上层。
Osten 是固定规模的决策引擎；Astrax 负责把它组织成具有身份、状态、目标、记忆、自省和训练流程的模型。

当前版本：`0.1.1-alpha`  
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
- Text、Code、State 三种受控输出模式
- `Text / Code / Image / Audio / Video / Binary` 输入模态接口

## 当前能力边界

`astrax_demo.exe` 会先训练一组确定性的 starter offline dataset。这个训练证明：

1. 状态预测器参数会根据数据更新；
2. 动作价值会根据奖励和下一状态更新；
3. 内在奖励会统计状态新颖度；
4. Osten 会根据编码后的输入、状态、目标、记忆上下文和心跳继续前向运行。

这还不是大语言模型，也不是已经学会自然语言和通用编程的模型。当前文本和代码由 `ControlledRenderer` 根据 Osten 的动作、置信度和 Astrax 状态生成，是受控呈现结果，不是通用 next-token 生成器。

因此，当前可以说 Astrax 已经具备**可训练的数字模型骨架和决策闭环**，不能说它已经像人一样思考，或已经具备 ChatGPT 级文本/代码生成能力。

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

训练结束后会保存 `artifacts\astrax_training.astrax-model`，然后创建一个全新的 `AstraxModel` 重新加载，并逐值验证状态预测和动作价值输出。checkpoint 当前保存 Astrax 的状态预测器、动作价值模型和内在奖励统计，不修改 Osten 核心拓扑。

starter dataset 只是验证训练闭环的数据，不代表真实语言或代码能力。要训练出有用的输出，需要继续准备真实的状态-动作-奖励-下一状态轨迹，并实现文本/代码结构化解码和完整评测。

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
       Text / Code / State
```

核心决策仍由 Osten 的固定规模网络前向产生。记忆检索和规则只提供上下文或边界保护，不直接挑选答案。

## 后续路线

1. 持久化 checkpoint 保存和加载；
2. 真实离线轨迹数据集与数据校验；
3. 文本/代码动作参数的结构化解码器；
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
