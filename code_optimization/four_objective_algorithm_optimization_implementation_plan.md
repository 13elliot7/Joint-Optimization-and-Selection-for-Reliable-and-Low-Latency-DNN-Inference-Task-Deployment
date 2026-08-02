# DAREED 四目标算法优化实现方案

## 1. 文档状态

- 状态：已实施并完成小规模回归验证（2026-07-19）
- 适用代码：`proposed.py`、`core/environment.py`、`experiment_runner.py` 及实验绘图脚本
- 优化对象：`run_customized_proposed()` 对应的 DAREED 定制化演化算法
- 核心约束：Hypervolume（HV）只在 `pareto` 实验中收集和计算

本方案替代旧方案中“能耗只参与最终加权选择、不参与 Pareto 搜索”的设计。旧方案保留在
`code_optimization/archive/implemented plans/` 中，仅作为历史记录，不再作为后续实现依据。

### 1.1 实际落地摘要

- `proposed.py`：总能耗已作为第四个最小化目标进入支配排序、拥挤距离、局部搜索和偏好选解。
- `core/environment.py`：新增稳定能耗参考边界、能耗满意度，以及不修改环境状态的一步负载/热度/可靠性预测。
- 初始化：使用启发式模板、扰动与随机可行解混合，并以 75% 唯一种群为目标。
- 交叉与变异：交叉概率已生效；交叉块改为 DAG 后继闭包；偏斜变异使用归一化评分和稳定 softmax。
- repair：显式返回成功状态、修改规模和失败原因；失败时不再把无效修复结果写入种群。
- 子膜：采用持久角色标签、角色配额和周期性全局父代选择实现专化与迁移，不额外复制四套完整种群。
- 约束：历史最优档案只接纳严格可行解，最终部署前重新计算完整约束违反度。
- `experiment_runner.py`：只有 `pareto` suite 收集前沿点和计算 4D HV；普通 suite 不输出任何 Pareto 字段。
- 结果兼容：`pareto_hv_version=4d_v1`，旧 3D-HV 或 Pareto/非 Pareto 混合结果会被明确拒绝。
- 绘图：Pareto 图已标明四目标；原三维散点图明确标为四目标前沿的 3D 投影。

### 1.2 已完成验证

- 11 项单元测试全部通过，覆盖四目标支配、4D HV、HV 门控、结果格式、初始化多样性、DAG 闭包、交叉概率和 repair 回退。
- 四种 Pareto 算法（`proposed`、`customized`、`random`、`sa`）均完成小规模联调，前沿点统一包含能耗维度。
- 普通 `overall` 实验在 5、10、20、30、40 个 DNN 场景完成回归，未生成 HV 字段或 `pareto_points.csv`。
- 一步动态预测的三次配对小测：关闭时平均 176.3 ms，开启时平均 194.0 ms，增幅约 10.0%，低于 20% 性能预算。
- 上述运行仅用于实现回归，不替代正式的多种子消融与统计显著性实验。

## 2. 优化目标

本轮优化需要解决以下问题：

1. 将总能耗真正接入演化搜索，使算法形成四目标优化，而不是只在最终选解时弱化地使用能耗。
2. 修正能耗归一化尺度，避免 `1 / energy` 在 `[0, 1]` 区间内接近零、导致能耗权重失效。
3. 保证历史最优解、最终选解和 Pareto 档案均遵守完整约束。
4. 提高 DAG 感知初始化的种群多样性。
5. 将当前按任务编号后缀交换的交叉改为真正的 DAG 闭合子图交叉。
6. 使交叉概率和变异概率在定制化算法中真实生效。
7. 修正偏斜变异的评分尺度和采样机制。
8. 使四个子膜具有真实、可保持的角色专化，而不是按数组位置临时命名。
9. 逐步将部署后的预测负载、热度和可靠性接入候选评价。
10. 将 HV 完全移出普通实验，只允许 `pareto` suite 收集 Pareto 点并计算四维 HV。

## 3. 总体设计决策

### 3.1 四目标定义

单个部署方案使用以下四个目标：

| 目标 | 字段 | 方向 |
|---|---|---|
| 动态精度可靠性 | `accuracy` | 最大化 |
| 动态服务可靠性 | `operation` | 最大化 |
| 时延收益 | `delay_utility` | 最大化 |
| 总能耗 | `total_energy` | 最小化 |

总能耗保持当前口径：

```text
total_energy = compute_energy + link_energy
```

当前能耗口径已经升级为输入上传、DAG 中间传输和最终结果回传三部分，与时延、链路负载及 OSS 的通信边界保持一致。对应结果必须使用 `stability_fidelity_v2_return_energy` 语义版本，不得与未计回传能耗的旧实验混合。

### 3.2 内部直接保存能耗，不再用 `1 / energy` 作为主值

建议将候选评价中的第四个值由：

```text
energy_utility = 1 / total_energy
```

改为：

```text
energy_cost = total_energy
```

原因：

- 原始能耗可直接解释和审计。
- 避免数值压缩到 `1e-4` 附近。
- Pareto 支配可以明确使用 `energy_i <= energy_j`。
- 最终打分可以单独使用稳定的能耗归一化，不再混淆“目标原值”和“偏好得分”。

为减少一次候选评价中的重复计算，建议评价结果统一为：

```python
ObjectiveValues(
    accuracy,
    operation,
    delay_utility,
    total_energy,
    estimated_delay,
)
```

可以使用 `@dataclass(frozen=True)`，也可以保留元组，但必须统一字段顺序并补充类型注释。

### 3.3 HV 仅属于 Pareto 实验后处理

必须明确区分：

- 算法搜索目标：四目标；
- 最终在线部署选择：根据请求偏好从四目标候选中选择一个方案；
- HV：仅用于 `pareto` 实验评价，不参与每代搜索，也不参与普通实验。

以下 suite 不计算 HV：

- `overall`
- `dynamic`
- `sensitivity`
- `ablation`
- `scale`
- `preference`

只有：

```text
suite == "pareto"
```

时才允许：

- 收集 Pareto 点；
- 写入 `pareto_points.csv`；
- 计算 `pareto_point_count`；
- 计算四维 `pareto_hypervolume`；
- 将 Pareto 指标写入 `raw_results.csv` 和 `summary.csv`。

## 4. 阶段一：四目标评价与选择修正

### 4.1 修改候选评价结构

文件：`proposed.py`

修改 `_evaluate_assignment()` 和 `_evaluate_customized_assignment()`：

```python
accuracy = env.count_dynamic_accuracy_by_assignment(...)
operation = env.count_dynamic_operation_by_assignment(...)
delay = env.estimate_delay_from_assignment(...)
delay_utility = ...
total_energy = env.count_total_energy_by_assignment(...)
```

返回：

```python
(accuracy, operation, delay_utility, total_energy, delay)
```

缓存键继续使用：

```python
(dnn_index, tuple(assignment))
```

能耗当前已经在候选评价阶段计算，因此本改动不会增加一次评价中的能耗遍历次数。

### 4.2 四目标约束支配

修改 `_dominates_with_cv()`。

当两个候选都满足当前 ε 约束时，`i` 支配 `j` 的条件为：

```text
accuracy_i      >= accuracy_j
operation_i     >= operation_j
delay_utility_i >= delay_utility_j
total_energy_i  <= total_energy_j
```

并且至少一个目标严格更优。

建议新增统一函数，避免搜索、局部搜索和 Pareto 导出分别实现不同口径：

```python
def _dominates_objectives(left: ObjectiveValues, right: ObjectiveValues) -> bool:
    ...
```

以下位置统一调用该函数：

- `_dominates_with_cv()`
- `_dominates_point()`
- `_filter_pareto_points()`
- 局部搜索的 `pareto_not_worse`
- 历史最优候选比较

### 4.3 四目标拥挤距离

修改 `get_res()`：

```python
objectives = (
    function1_values,
    function2_values,
    function3_values,
    function4_values,
)
```

拥挤距离只依赖排序和目标范围，因此能耗虽然是最小化目标，也可以按数值升序计算距离，不需要先取倒数。

边界点仍设置为无穷大距离。

### 4.4 修正 ε 可行候选的环境选择

当前 `update_res_with_cv()` 按：

```text
rank -> raw CV -> crowding distance
```

排序。该逻辑会在两个候选都满足 ε 时仍优先选择 CV 更小者，削弱 ε 放宽约束的探索作用。

建议改为：

```text
1. Pareto rank
2. 如果 CV > epsilon，则按 CV 升序
3. 如果 CV <= epsilon，则按 crowding distance 降序
```

实现时将当前代 `epsilon` 显式传入 `update_res_with_cv()`。

### 4.5 历史最优档案必须遵守完整约束

当前每代更新 `best_value` 时主要检查时延目标是否为正，可能保存其他约束仍被违反的候选。

修改原则：

- 代内最优候选必须通过约束支配选择。
- 历史最优只允许保存 `CV <= epsilon` 的候选。
- 最终部署只能接受 `CV == 0`（允许数值容差，例如 `1e-12`）的候选。
- 最终接纳前重新计算一次完整 CV，不依赖旧缓存状态。

建议新增：

```python
def _is_strictly_feasible(..., tolerance: float = 1e-12) -> bool:
    return self._constraint_violation(...) <= tolerance
```

最终拒绝原因应拆分记录：

- `no_deadline_feasible_solution`
- `resource_violation`
- `hierarchy_violation`
- `link_overload`
- `initialization_failed`

### 4.6 最终偏好选择中的能耗归一化

不要再把 `1 / energy` 直接放入 `[0,1]` 固定区间。

推荐由 `Environment` 为每个 DNN 提供与算法无关的确定性能耗参考边界：

```python
energy_lower_bound, energy_upper_bound = env.energy_reference_bounds(dnn_index)
```

边界根据合法节点、任务计算量、节点功率/算力比和路径能耗系数计算，不能根据某一个算法产生的种群单独确定，否则不同算法的分数和 HV 不可比较。

能耗满意度定义为：

```text
energy_score =
    clamp((energy_upper_bound - total_energy)
          / (energy_upper_bound - energy_lower_bound), 0, 1)
```

最终四目标得分：

```text
score =
    w_a * accuracy_score
  + w_r * operation_score
  + w_t * delay_score
  + w_e * energy_score
```

`preference` 对应权重可保留现有设置，但必须增加测试，确认：

- `energy_sensitive` 下能耗权重确实主导选解；
- `delay_sensitive` 下不会因能耗量纲改变而意外反转；
- 所有权重非负且总和为 1。

## 5. 阶段二：DAG 初始化、交叉和变异修正

### 5.1 初始化多样性控制

当前六种初始化模式循环生成60个个体，确定性模式会产生大量重复解。

改为“模板种子 + 扰动种子 + 随机可行个体”：

```text
10% 确定性高质量模板
40% 模板扰动个体
50% 随机可行个体
```

具体要求：

1. 使用 `set[tuple[int, ...]]` 去重。
2. 每个模板最多直接保留一个副本。
3. 模板扰动至少改变一个任务部署节点。
4. 新个体与已有个体重复时最多重试固定次数。
5. 重试失败后回退到随机可行初始化，而不是立即使整个DNN搜索失败。
6. 初始化完成后输出：
   - `initial_unique_count`
   - `initial_duplicate_count`
   - `initialization_retry_count`

验收标准：

```text
population_size = 60 时，唯一初始个体数至少达到 45；
建议目标达到 50。
```

### 5.2 真正的 DAG 闭合子图交叉

废弃当前：

```text
选择关键路径任务编号 boundary
交换所有 task_idx >= boundary 的任务
```

改为以下两类块之一：

#### 后继闭包块

随机选择锚点任务 `u`：

```text
block = descendants(u) ∪ {u}
```

#### 分支块

从分支起点选择一个后继分支，直到最近公共汇合点之前：

```text
block = one_branch_between_split_and_join
```

第一版优先实现后继闭包块，逻辑简单且可测试。

交叉过程：

```text
child_a[block] <- parent_b[block]
child_b[block] <- parent_a[block]
repair(child_a)
repair(child_b)
```

必须记录：

- 交叉前后实际变化的基因数；
- repair修改的基因数；
- repair后是否退化为父代；
- 子代是否与父代重复。

如果 repair 修改比例超过设定阈值，例如块大小的50%，应回退到父代或重新选择交叉块，避免“先破坏、再完全重构”。

### 5.3 让交叉概率真实生效

在 `_write_customized_child()` 中首先判断：

```python
if random.random() < self.cross_over_pm:
    crossover(...)
else:
    copy parents
```

交叉关闭时不能额外消耗用于选交叉点的随机数，保证不同概率实验具有清晰语义。

验收：

- `cross_over_pm = 0` 时交叉次数为0；
- `cross_over_pm = 1` 时每一对子代都执行交叉；
- 固定种子下交叉次数可复现。

### 5.4 修正偏斜变异

所有候选评分分量先归一化到 `[0,1]`：

- CPU余量；
- 执行时延；
- 前驱/后继链路代价；
- 预测节点可靠性；
- 预测路径可靠性；
- 热度；
- 能耗；
- 可行性。

排除当前节点，确保触发变异时至少尝试产生实际变化：

```python
candidates = [node for node in candidates if node != current_node]
```

使用带温度的 softmax：

```text
p(node_j) = exp(score_j / tau) / Σ exp(score_k / tau)
```

推荐：

- 前期使用较高 `tau`，保留探索；
- 后期降低 `tau`，增强方向性；
- 对所有分数先减去最大值，避免指数溢出。

若变异后 repair 失败：

1. 尝试次优候选；
2. 回退原个体；
3. 不允许将非法个体直接写入联合种群。

### 5.5 修复机制返回显式状态

将 `_repair_assignment()` 的返回值由单个部署向量改为：

```python
RepairResult(
    assignment,
    success,
    changed_gene_count,
    reason,
)
```

避免当前修复失败时静默返回原始非法个体。

`_is_assignment_valid()` 与 `_constraint_violation()` 的约束定义必须对齐：

- CPU资源；
- 用户节点限制；
- DAG层级流向；
- 节点索引合法性；
- 链路容量；
- 时延约束是否作为硬约束由调用方明确决定。

## 6. 阶段三：真实子膜专化

### 6.1 子膜身份不能依赖数组位置

建议为每个角色维护独立种群：

```python
role_populations = {
    "reliability": [...],
    "latency": [...],
    "energy": [...],
    "feasibility": [...],
}
```

每个子膜内部执行：

- 父代选择；
- 角色偏斜变异；
- 环境选择；
- 小型角色档案维护。

不能在全局选择后简单把前四分之一重新命名为“可靠性膜”。

### 6.2 子膜内部选择规则

所有子膜仍使用完整四目标约束支配，角色只用于平局和父代选择，不允许删除其他目标。

建议：

- `reliability`：同层内优先服务可靠性和精度可靠性；
- `latency`：同层内优先时延；
- `energy`：同层内优先低能耗；
- `feasibility`：早期优先低CV，严格可行后恢复拥挤距离。

### 6.3 周期性交换

每隔固定代数进行一次膜间迁移，例如每5代：

```text
每个子膜导出1个角色最优解和1个高拥挤距离解；
发送至相邻或随机子膜；
接收方去重后替换最差个体。
```

四个子膜总种群规模保持60，不因拆分而增加候选评价次数。

若该阶段改动风险过高，可以先取消“膜算法”表述，采用统一种群四目标演化；不能保留只有名称而无独立机制的子膜设计。

## 7. 阶段四：部署后动态状态预测

### 7.1 当前问题

正式可靠性目标使用部署前的当前可靠性，而候选接纳后新增负载和热度会改变运行期间的可靠性。

### 7.2 第一版低开销预测

新增：

```python
env.predict_candidate_state(
    dnn_index,
    assignment,
    estimated_runtime,
)
```

返回：

- 预测节点负载；
- 预测链路负载；
- 下一时隙预测热度；
- 预测节点服务可靠性；
- 预测节点精度可靠性；
- 预测链路可靠性。

使用一次状态转移近似，不为每个候选复制完整环境，也不逐时隙模拟整个生命周期。

候选的正式可靠性目标改为预测状态下的可靠性。

### 7.3 性能约束

预测函数必须：

- 只遍历候选使用的节点和链路；
- 复用 assignment 评价缓存；
- 不修改真实环境；
- 不分配完整 `Environment` 副本；
- 不进行候选级全生命周期仿真。

如果加入预测后候选评价时间增加超过15%，应先保留当前状态目标，并把预测可靠性作为约束或局部搜索辅助量，等待进一步优化。

## 8. HV 仅在 Pareto 实验中计算

### 8.1 配置开关

在 `AllDNNRefactor` 中新增：

```python
self.collect_pareto_points = False
```

在 `_configure_search()` 中设置：

```python
refactor.collect_pareto_points = scenario.suite == "pareto"
```

只有开关为 `True` 时，算法才调用：

- `_append_current_pareto_points()`
- `_assignment_pareto_point()`
- `_filter_pareto_points()`

普通实验中：

```text
pareto_points 始终为空；
不做非支配点导出；
不产生 Pareto 点内存和后处理开销。
```

注意：四目标非支配排序是主算法机制，仍在所有 suite 中运行；这里只关闭实验用的前沿导出和HV计算。

### 8.2 suite 感知的结果字段

将 `experiment_runner.py` 中统一的 `METRIC_FIELDS` 拆分为：

```python
BASE_METRIC_FIELDS = (
    "avg_estimated_delay",
    "avg_dynamic_operation_reliability",
    "avg_dynamic_accuracy_reliability",
    "avg_total_energy",
    "rejected_or_failed_count",
    "runtime_ms",
)

PARETO_METRIC_FIELDS = (
    "pareto_point_count",
    "pareto_point_count_mean",
    "pareto_hypervolume",
    "pareto_hypervolume_sum",
    "pareto_hypervolume_std",
)
```

规则：

```python
metric_fields = BASE_METRIC_FIELDS
if suite == "pareto":
    metric_fields += PARETO_METRIC_FIELDS
```

非 Pareto suite 的 `raw_results.csv` 和 `summary.csv` 不写 Pareto 字段，而不是写入容易被误解的0。

### 8.3 `_result_row()` 的强制门控

修改为：

```python
if scenario.suite == "pareto" and algorithm in PARETO_ALGORITHMS:
    row.update(_pareto_hypervolume_by_dnn(output.pareto_points))
```

其他情况不得调用 `_pareto_hypervolume_by_dnn()`。

建议 `_pareto_hypervolume_by_dnn()` 增加防御式参数：

```python
def _pareto_hypervolume_by_dnn(points, *, enabled: bool):
    if not enabled:
        raise RuntimeError("HV is only available in pareto suite")
```

防止后续维护时意外在普通实验中恢复HV计算。

### 8.4 `pareto_points.csv` 门控

只有 `suite == "pareto"` 时创建：

```text
experiment_results/pareto/pareto_points.csv
```

其他结果目录不得创建该文件。

### 8.5 四维 HV 目标

Pareto实验使用：

```text
accuracy_norm
operation_norm
delay_satisfaction
energy_satisfaction
```

四个目标均为 `[0,1]` 且越大越好，参考点固定为：

```text
(0, 0, 0, 0)
```

能耗满意度必须使用第4.6节中与算法无关的确定性参考边界。

按 `dnn_index` 分别计算HV，再聚合：

- `pareto_hypervolume`：成功DNN的平均四维HV；
- `pareto_hypervolume_sum`：成功DNN的四维HV总和；
- `pareto_hypervolume_std`：DNN级四维HV标准差；
- `pareto_point_count_mean`：每个成功DNN的平均非支配点数。

不得把不同DNN的点混合成一个全局前沿。

### 8.6 HV算法与运行开销

禁止在每代进化中计算HV。

四维精确网格容易出现组合爆炸，建议：

1. 小前沿使用确定性精确4D算法；
2. 超过阈值后使用固定样本集的准蒙特卡洛估计；
3. 所有算法、场景和重复使用同一参考点和同一采样点；
4. 输出 `pareto_hypervolume_is_approx`；
5. HV后处理时间不计入算法 `runtime_ms`，可另记 `pareto_postprocess_ms`。

## 9. 实验与绘图适配

### 9.1 `plot_results.py`

- 普通 suite 只读取基础指标。
- 只有 `pareto` suite 尝试读取 Pareto 指标。
- 字段缺失时不补0，不绘制空Pareto图。

### 9.2 `plot/generate_paper_figures.py`

- `make_pareto_figures()` 读取四维HV。
- `make_ablation_figure()` 不再读取或暗示HV。
- overall、dynamic、scale、preference 和 sensitivity 图不显示HV。
- 图注明确：

```text
HV is computed only in the Pareto experiment using four normalized objectives.
```

### 9.3 旧结果兼容

旧CSV中的三维HV不得与新四维HV拼接汇总。

建议在 Pareto 结果中新增：

```text
pareto_hv_version = "4d_v1"
```

绘图脚本发现版本缺失或不一致时应拒绝合并，并提示重新运行 Pareto 实验。

## 10. 测试方案

### 10.1 四目标支配测试

至少覆盖：

1. 三个质量目标相同、能耗更低的候选应支配能耗更高候选。
2. 能耗更低但可靠性更差时应互不支配。
3. 四目标完全相同时互不严格支配。
4. 非可行解不能支配严格可行解。
5. 两个非可行解按CV比较。

### 10.2 能耗归一化测试

- 下界映射为1；
- 上界映射为0；
- 超出边界时正确裁剪；
- 上下界相等时不产生除零；
- `energy_sensitive` 权重能够改变最终选解。

### 10.3 初始化测试

- 60个初始个体至少45个唯一；
- 所有个体满足硬资源和节点约束；
- 固定种子可复现；
- 单个模板修复失败不会导致整个DNN失败。

### 10.4 交叉与变异测试

- DAG交叉块满足后继闭包；
- `cross_over_pm=0/1` 行为正确；
- 触发偏斜变异后优先产生实际基因变化；
- repair失败时不写入非法子代；
- 交叉和变异统计计数与实际执行一致。

### 10.5 HV门控测试

对每个非 Pareto suite 运行最小实验，断言：

- `_pareto_hypervolume_by_dnn()` 调用次数为0；
- `pareto_points.csv` 不存在；
- raw和summary中不存在Pareto字段；
- 算法 `pareto_points` 不增长。

对 `pareto` suite 断言：

- 输出四目标Pareto点；
- 输出 `pareto_hv_version=4d_v1`；
- 单点 `(0.5, 0.5, 0.5, 0.5)` 的HV为 `0.0625`；
- 两个DNN分别计算后再取平均；
- 不混合不同DNN的前沿。

### 10.6 回归测试

修复后先运行：

```bash
python experiment_runner.py \
  --suite ablation \
  --dnn-count 5 \
  --population-size 20 \
  --iteration-limit 20 \
  --repeats 2
```

确认：

- 不计算HV；
- 四目标排序正常；
- 没有非法最终部署；
- 初始化唯一率达到要求。

再运行：

```bash
python experiment_runner.py \
  --suite pareto \
  --dnn-count 5 \
  --population-size 20 \
  --iteration-limit 20 \
  --repeats 2
```

确认四维HV和Pareto点文件正常生成。

## 11. 性能预算

四目标接入利用已有能耗计算和候选缓存，预计：

- 四目标支配比较：主搜索开销增加约1%–5%；
- 四目标拥挤距离：总体开销增加约1%–3%；
- 稳定能耗归一化：可忽略；
- 初始化去重和统计：小于2%；
- DAG闭合子图构建：通过上下文缓存后小于3%；
- 单步预测可靠性：目标控制在15%以内；
- 普通实验关闭Pareto点收集后，可抵消一部分新增开销。

整体目标：

```text
不启用部署后动态预测时，平均 runtime_ms 增幅不超过10%；
启用低开销动态预测时，平均 runtime_ms 增幅不超过20%。
```

HV只在Pareto实验后处理阶段执行，不计入算法 `runtime_ms`。

## 12. 分阶段实施顺序

### Phase 1：正确性优先

1. 候选评价改为保存原始总能耗。
2. 能耗加入支配关系和拥挤距离。
3. 修正能耗归一化。
4. 历史最优和最终选解执行完整约束。
5. HV增加suite门控，仅Pareto实验计算。

Phase 1完成后即可重新运行小规模实验验证四目标口径。

### Phase 2：算子质量

1. 初始化去重和混合随机个体。
2. 交叉概率接入。
3. DAG后继闭包交叉。
4. 偏斜变异归一化和softmax采样。
5. repair显式返回成功状态。

### Phase 3：算法结构

1. 建立真实子膜种群。
2. 增加膜内选择和周期迁移。
3. 增加部署后动态状态预测。

Phase 3应分别做消融，不与Phase 1、Phase 2一次性混跑，否则无法判断性能变化来源。

## 13. 验收标准

实现完成必须同时满足：

1. 能耗进入非支配排序、拥挤距离、局部搜索和最终选择。
2. 能耗目标使用可解释的原始总能耗，不再因 `1/E` 尺度失效。
3. 所有最终接纳部署的完整CV为0。
4. 60个初始个体中至少45个唯一解。
5. `cross_over_pm` 在定制算法中真实生效。
6. DAG交叉块是拓扑闭合块，不是任务编号后缀。
7. 普通suite完全不收集Pareto点、不计算HV、不输出HV字段。
8. Pareto suite使用统一参考边界计算四维HV。
9. HV不进入每代搜索，不计入算法 `runtime_ms`。
10. 不启用预测动态状态时，主算法平均运行时间增幅不超过10%。
11. 新消融实验至少20个配对种子，并同时报告成功率和成功任务条件均值。

## 14. 预期结果

完成上述优化后，算法描述与代码实现应形成一致关系：

```text
四目标演化搜索
    -> 精度可靠性最大化
    -> 服务可靠性最大化
    -> 时延收益最大化
    -> 总能耗最小化

严格约束筛选
    -> 资源
    -> 层级
    -> 链路负载
    -> 时延

偏好选择
    -> 根据请求类型从四目标候选中选择最终部署

实验评价
    -> 普通实验只报告部署性能
    -> Pareto实验额外报告四维HV与前沿规模
```

这样可以避免普通实验承担无必要的HV开销，同时保证能耗确实参与搜索，并使后续消融实验能够真实评价各定制组件的贡献。
