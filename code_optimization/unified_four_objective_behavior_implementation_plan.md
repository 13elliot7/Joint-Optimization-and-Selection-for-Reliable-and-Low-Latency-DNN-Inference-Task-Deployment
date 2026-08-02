# 四目标算法行为统一修改与实施方案

## 1. 目标

统一精度、运行可靠性、时延和能耗在以下算法环节中的行为：

1. 目标表示；
2. 优化方向；
3. 归一化；
4. Pareto 支配与拥挤距离；
5. 最终偏好评分。

同时保留以下合理差异：

- `adaptive` 及显式偏好模式使用不同目标权重；
- 时延、资源、层级和链路过载继续作为独立硬约束；
- reliability、latency、energy、feasibility 子膜及其专用搜索算子继续使用各自启发式。

## 2. 统一目标模型

每个部署方案同时保存原始指标和算法目标。

原始指标用于日志、实验结果和部署审计：

```text
accuracy
operation
estimated_delay
total_energy
deadline_feasible
```

算法统一使用四个 `[0, 1]` 最大化满意度目标：

```text
accuracy_satisfaction
operation_satisfaction
delay_satisfaction
energy_satisfaction
```

其中：

```text
delay_satisfaction =
    clamp((deadline - estimated_delay) / deadline, 0, 1)

energy_satisfaction =
    clamp((energy_upper - total_energy)
          / (energy_upper - energy_lower), 0, 1)
```

时延目标值为零不等价于不可行。是否超过截止时间由
`deadline_feasible` 和约束违反度独立判断。

## 3. 算法环节修改

### 3.1 候选评价

在 `proposed.py` 中新增 `ObjectiveValues`，集中保存原始指标、统一目标向量和截止时间可行性。
`_evaluate_assignment()` 与评价缓存统一返回该结构。

### 3.2 Pareto 支配

新增统一支配函数：

```python
_dominates_objectives(left, right)
```

四个维度全部采用：

```text
left[k] >= right[k]
```

并要求至少一个维度严格更大。种群排序、约束支配和 Pareto 点过滤复用同一口径。

### 3.3 拥挤距离

`function1_values` 至 `function4_values` 全部保存统一满意度，因此拥挤距离直接在统一的四维
`[0, 1]` 最大化空间中计算。

### 3.4 最终偏好评分

最终评分直接复用搜索阶段的统一目标，不再对原始时延和能耗进行第二套归一化：

```text
score =
    w_a * accuracy_satisfaction
  + w_r * operation_satisfaction
  + w_t * delay_satisfaction
  + w_e * energy_satisfaction
```

偏好权重定义保持不变。`adaptive` 原始权重输入增加 `[0, 1]` 截断，避免超出参考范围的请求产生
负权重。

### 3.5 硬约束与专用机制

- 基础算法中，满足截止时间的候选优先于超时候选；两个超时候选按原始时延选择。
- 定制算法继续使用 ε 约束支配和最终严格可行性检查。
- 精英局部搜索继续要求约束不恶化、Pareto 不劣且偏好评分提高。
- 子膜角色和候选节点专用评分不改为统一权重。

### 3.6 Pareto 导出与 HV

Pareto 点同时导出原始指标和四个满意度目标。HV 后处理直接复用满意度字段，不再根据某个算法
产生的当前前沿二次归一化。

结果版本升级为：

```text
4d_unified_v2
```

旧版本结果不能与新结果追加到同一输出文件。

## 4. 修改文件

- `proposed.py`
  - 新增统一目标结构；
  - 修改评价、支配、拥挤距离输入、局部搜索和最终评分；
  - 分离时延目标与时延可行性；
  - 保留原始时延和能耗缓存。
- `experiment_runner.py`
  - 初始化新增缓存；
  - HV 直接复用统一满意度；
  - 更新 Pareto CSV 字段与结果版本。
- `tests/test_four_objective_optimization.py`
  - 验证四目标统一最大化；
  - 验证评价范围、最终评分和 HV 使用同一表示；
  - 验证时延硬约束与时延满意度相互独立。

## 5. 验证标准

1. 四个搜索目标均位于 `[0, 1]`；
2. 四个维度全部按最大化执行支配判断；
3. 最终评分等于统一满意度与偏好权重的直接点积；
4. 超时方案不能因目标归一化而被当作可行方案；
5. 部署指标继续输出原始时延和原始总能耗；
6. 四种 Pareto 算法均可完成小规模运行；
7. 全部单元测试通过。

## 6. 实施状态

状态：已实现并完成单元测试及小规模联调。

