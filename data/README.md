# Astrax Offline RL 数据

`astrax_offline_transitions.csv` 是由 `astrax_train.exe` 生成的可复现 starter offline dataset。

## 数据格式

```text
action,reward,terminal,state,next_state
```

- `state`：长度为 `state_dim` 的浮点向量；
- `action`：范围为 `[0, action_count)` 的离散动作；
- `reward`：有限浮点奖励；
- `next_state`：执行动作后的状态，长度与 `state` 相同；
- `terminal`：回合结束标记，只能是 `0` 或 `1`。

当前生成配置：

```text
state_dim   = 32
action_count = 8
episodes    = 128
horizon     = 32
samples     = 4096
```

每个回合包含 32 条连续转移，最后一条的 `terminal` 为 `1`，其他记录为 `0`。状态值均为有限数，并限制在 `[-1, 1]`。奖励由动作执行前后到目标状态的距离变化计算，不是随机标签。

训练程序会先生成 CSV，再重新加载 CSV，并在训练前校验维度、动作范围、奖励有限性、状态范围和终止标记。
