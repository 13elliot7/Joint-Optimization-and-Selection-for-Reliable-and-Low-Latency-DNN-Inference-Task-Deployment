# 方案 A：决策时动态带宽，接纳后固化运行时长

本文档描述在当前动态云边端 DNN 部署模型中加入动态链路带宽的最小侵入方案。

## 1. 目标

当前系统已经考虑链路负载、链路热度和动态可靠性，但链路带宽仍主要作为静态容量使用。方案 A 的目标是：

- 在新 DNN 的候选部署评价阶段，使用当前 slot 的动态有效带宽估计传输时延、链路负载和过载风险。
- DNN 被接纳后，将接纳时估计出的 `estimated_runtime` 固化为运行块时长。
- 后续 slot 不重新计算已接纳 DNN 的 DAG 任务进度或剩余时延。
- 保留现有 `customized_proposed` 主流程，不重写种群初始化、交叉、变异、排序、局部搜索和最终接纳逻辑。

整体语义为：

```text
候选部署评价
  -> 使用当前系统状态下的 effective_bandwidth 估计 delay
  -> 接纳后保存 estimated_runtime
  -> 后续 slot 只更新负载、热度、有效带宽和可靠性
```

## 2. 动态带宽模型

每条链路保留静态采样带宽 `base_band_width`，并在运行时维护 `effective_band_width`。

有效带宽采用平滑衰减模型：

```text
effective_bandwidth =
  base_bandwidth / (1 + bandwidth_heat_gamma * heat + bandwidth_load_gamma * load_ratio)
```

并设置下界：

```text
effective_bandwidth >= max(1.0, base_bandwidth * min_bandwidth_ratio)
```

默认参数：

```text
bandwidth_heat_gamma = 0.5
bandwidth_load_gamma = 0.3
min_bandwidth_ratio = 0.2
```

含义：

- `heat` 表示链路历史负载形成的热度记忆。
- `load_ratio` 表示当前运行中 DNN 对该链路造成的平均负载。
- 有效带宽不会无限衰减，最低保留静态带宽的一定比例。

## 3. 影响范围

方案 A 只修改系统模型层，算法主流程保持不变。动态带宽影响以下计算：

1. 链路负载：

```text
load_ratio = used_bandwidth / effective_bandwidth
```

2. 路径传输时延：

```text
path_delay = data_amount * sum(1 / effective_bandwidth(link))
```

3. 候选路径状态预测：

```text
predicted_load =
  current_load + new_traffic / runtime / effective_bandwidth
```

4. 约束违反度：

候选部署导致的预测链路过载使用动态带宽口径。

5. 最终部署选择：

动态带宽会改变 `value3` 时延收益、链路可靠性预测、路径可行性和 CV，从而影响 Pareto 排序和最终接纳。

## 4. 不做的事情

方案 A 明确不引入以下复杂机制：

- 不在每个 slot 重算已运行 DNN 的 DAG 任务剩余时延。
- 不追踪每个 DAG 任务的完成状态、传输进度或排队状态。
- 不做运行中迁移。
- 不基于动态带宽重新计算最短路径结构。

路径结构仍基于初始化时静态带宽倒数预计算；动态的是路径上每条链路的有效带宽。

## 5. 实现位置

主要改动集中在：

- `models.py`
  - `LinkNode` 增加 `base_band_width` 和 `effective_band_width`。
- `core/environment.py`
  - `Environment` 增加动态带宽参数。
  - 增加 `get_effective_bandwidth()`、`refresh_effective_bandwidth()` 和链路传输时延辅助函数。
  - 在 slot 推进时刷新有效带宽。
  - 将链路负载、路径传输时延、候选路径预测、部署时延估计中的链路带宽读取改为有效带宽。

`proposed.py` 不需要重构。`customized_proposed` 通过现有环境接口获取时延、链路状态、目标值和约束违反度，因此会自然受到动态带宽影响。

## 6. 运行时间影响

该方案不会显著增加算法运行时间，原因是：

- 最短路径结构仍然缓存，不在候选评价中重复跑最短路。
- 每个 slot 刷新有效带宽只是遍历链路一次。
- 路径时延从静态缓存因子改为沿缓存路径累加有效带宽倒数，当前拓扑规模下路径较短，开销可控。

如后续大规模实验发现路径时延计算成为瓶颈，可在每次刷新有效带宽后预计算所有节点对的动态路径 delay factor，使候选评价恢复为 O(1) 查询。
