# P0 模型闭环与投稿语义改造方案（设计来源，已被统一方案取代）

> 状态：不再作为独立实施入口。P0 模型决策已经收敛至 `../pending/unified_periodic_planning_model_closure_implementation_plan.md`。若字段、阶段、实验范围或完成条件与统一方案冲突，以统一方案为准。本文件仅用于追溯 P0 设计细节。

## 1. 文档目的

本文档基于 `code_optimization/model_submission_readiness_issues.md`，将其中 P0 问题转化为可执行的代码改造、测试和实验计划。

本轮改造的目标不是增加新的遗传算子，而是先完成以下模型闭环：

1. 四个优化目标基于同一个候选接纳后预测状态；
2. 候选预测与实际状态更新使用同一套链路 offered-rate 公式；
3. OSS 与 IFS 具有可解释、可区分且不过度声称的语义；
4. 所有正式实验可通过版本字段、配置和随机种子追踪。

完成本文档中的 P0 验收项后，项目才能从“成熟研究原型”进入“可投稿模型”的实现阶段。正式实验和统计材料完成后，方可判定为投稿就绪。

---

## 2. 本轮冻结的模型决策

### 2.1 四目标统一定义

对候选部署决策 `x`，统一优化：

```text
F(x) = [IFS(x), OSS(x), DS(x), ES(x)]
```

其中四项均最大化：

- `IFS`：推理保真性分数；
- `OSS`：运行稳定性分数；
- `DS`：时延满意度；
- `ES`：能耗满意度。

同时保留原始 `estimated_delay` 与 `total_energy`，用于论文表格、约束判断和结果解释。

### 2.2 OSS 的投稿主语义

OSS 改为以下两部分的分层联合评价：

1. 候选执行窗口内的节点可用性预测；
2. 候选接纳后预测状态下的链路传输稳定性。

节点当前是否在线属于硬约束。只要候选使用了当前离线节点，该候选直接不可行，不用一个较低 OSS 替代硬约束。

对候选使用节点集合 `N(x)`，定义：

```text
A_N(x) = exp(sum_n omega_n * log(p_up,n(t, H(x))))
```

其中：

- `p_up,n(t, H)` 表示在时隙 `t` 已知节点在线的条件下，节点在未来执行窗口 `H` 内持续可用的预测值；
- `H(x) = ceil(predicted_delay(x) / slot_length)`；
- `omega_n` 为该节点承载 FLOPs 占候选总 FLOPs 的比例，权重和为 1。

对候选使用的唯一物理链路集合 `L(x)`，定义：

```text
S_L(x) = exp(sum_l nu_l * log(s_l(t+1 | x)))
```

其中 `nu_l` 为该物理链路承载候选传输数据量的归一化权重。上下行共享同一物理链路 ID，只计一次；方向流量仍分别汇总。

最终：

```text
OSS(x) = exp(w_N * log(A_N(x)) + w_L * log(S_L(x)))
```

`w_N`、`w_L` 归一化后使用。若候选没有跨节点通信，则 `OSS(x) = A_N(x)`。

该 OSS 是归一化复合效用，不是严格的端到端成功概率。原因是节点可用性预测值与链路压力稳定性代理的统计含义并不完全相同，且模型未声明组件独立失效。

### 2.3 节点可用性预测

投稿默认采用不依赖未来真实轨迹的经验窗口估计器：

```text
p_up,n(t, H)
  = (continuous_up_windows_n(H) + beta_success)
    / (eligible_online_starts_n(H) + beta_success + beta_failure)
```

约束如下：

- 仅使用时隙 `t` 及以前的可见历史；
- 分母只统计窗口起点在线且拥有足够历史观测的样本；
- 分子统计从该起点连续在线至少 `H` 个时隙的样本；
- 使用 Beta 平滑避免短历史直接产生 0 或 1；
- `H <= 0` 返回 1；
- 当前离线节点不进入预测，而是由可行性检查直接拒绝；
- 所有允许部署的节点必须有明确的可用性模式。

对于未纳入随机上下线过程的基础设施节点，必须通过配置显式选择：

```text
availability_mode = stochastic | always_on
```

不能继续依靠 `availability_node_count` 隐式区分。论文若将云节点设为 `always_on`，需把它列为系统假设，并在敏感性实验中提供非完美云可用性场景。

推荐默认版本：

```text
availability_estimator_version = empirical_window_beta_v1
beta_success = 1.0
beta_failure = 1.0
```

### 2.4 IFS 的投稿主语义

由于项目只抽象建模 DNN DAG，不对具体模型、数据集、量化、早退或软错误机制建模，正式版本采用较严格的静态 profiling 方案：

```text
IFS(x)
  = exp(sum_n omega_n * log(base_inference_fidelity_n))
```

其中 `omega_n` 同样按节点承载 FLOPs 归一化。

实现要求：

- 节点负载和热度默认不再直接降低 IFS；
- `base_inference_fidelity` 表示离线校验或受控合成配置中的参考输出保持能力；
- IFS 不称为准确率，不解释为单次推理成功概率；
- 可保留旧的负载/热度衰减公式作为 `dynamic_proxy_legacy` 消融模式，但不得混入默认正式结果；
- 若未来获得软错误、近似计算或输出数值误差数据，再引入可校准的动态 IFS 模式。

推荐默认配置：

```text
ifs_mode = profiled_static
```

### 2.5 时延与能耗语义

时延必须使用候选接纳后预测有效带宽，包含：

```text
输入上传 + DAG 关键路径计算与中间通信 + 结果回传
```

路由在一次候选预测中保持固定。候选搜索不执行动态重路由。

能耗保持部署级确定性一阶模型：

```text
计算能耗 + 输入上传能耗 + 中间传输能耗 + 结果回传能耗
```

能耗不必人为加入负载预测，只需与同一 assignment、同一路由和相同回传口径保持一致。

---

## 3. 统一候选状态闭环

### 3.1 新增观测快照

在一次规划开始时生成只读 `ObservationSnapshot`。算法只能读取该快照，不得在候选评价期间读取未来状态或修改环境。

建议字段：

```python
@dataclass(frozen=True)
class ObservationSnapshot:
    slot: int
    node_online: tuple[bool, ...]
    node_available_cpu: tuple[float, ...]
    node_loads: tuple[float, ...]
    node_heats: tuple[float, ...]
    node_availability_history: tuple[tuple[bool, ...], ...]
    directional_offered_rates: tuple[float, ...]
    directional_link_loads: tuple[float, ...]
    physical_link_heats: tuple[float, ...]
    effective_bandwidths: tuple[float, ...]
```

快照只包含算法在当前时隙可见的信息。后续真实上下线结果、未来到达请求和未来热度均不在快照中。

### 3.2 新增候选预测快照

新增只读 `CandidateStateSnapshot`，使一次候选评价的所有指标共享同一状态：

```python
@dataclass(frozen=True)
class CandidateStateSnapshot:
    dnn_index: int
    assignment: tuple[int, ...]
    converged: bool
    iteration_count: int
    predicted_runtime: float
    horizon_slots: int
    predicted_node_loads: tuple[float, ...]
    predicted_node_availability: tuple[float, ...]
    directional_offered_rates: tuple[float, ...]
    predicted_directional_loads: tuple[float, ...]
    predicted_physical_loads: tuple[float, ...]
    predicted_physical_heats: tuple[float, ...]
    predicted_effective_bandwidths: tuple[float, ...]
    node_availability_score: float
    link_stability_score: float | None
    operational_stability: float
    inference_fidelity: float
    estimated_delay: float
    total_energy: float
    deadline_feasible: bool
    link_overload: float
```

实现时可用按稳定索引排列的 tuple，也可用不可变映射。禁止把可变 `Node`、`LinkNode` 对象本身当作候选结果的唯一状态载体。

### 3.3 offered-rate 统一内核

新增唯一的方向链路 offered-rate 汇总内核：

```text
existing_rate_l
  = sum(data_running_on_l / max(stored_runtime, slot_length, 1))

candidate_rate_l
  = data_candidate_on_l / max(candidate_runtime, slot_length, 1)

total_rate_l
  = existing_rate_l + candidate_rate_l

directional_load_l
  = total_rate_l / predicted_effective_bandwidth_l
```

建议拆分为以下纯函数或无副作用方法：

```python
collect_running_directional_offered_rates(observation)
collect_candidate_directional_data(dnn_index, assignment)
add_candidate_directional_offered_rates(existing_rates, data, runtime)
directional_loads_from_offered_rates(rates, bandwidths)
physical_loads_from_directional_loads(directional_loads, duplex_mode)
```

`predict_candidate_state()` 与 `update_link_loads()` 必须调用同一底层内核。禁止再次使用：

```text
current_load_ratio + candidate_rate / current_bandwidth
```

### 3.4 带宽—负载—时延有限不动点迭代

候选预测流程固定为：

```text
读取 ObservationSnapshot
  → 校验 assignment、当前节点在线和 CPU 可行性
  → 使用观测有效带宽计算初始 delay_0
  → 根据 delay_k 计算候选 offered rate
  → 汇总现有任务与候选任务的总 offered rate
  → 计算方向负载和唯一物理链路负载
  → 预测物理链路热度与有效带宽
  → 使用预测带宽重算 delay_raw
  → 阻尼得到 delay_(k+1)
  → 收敛后计算窗口节点可用性、OSS、IFS、能耗和约束
```

默认参数：

```text
candidate_fixed_point_max_iterations = 5
candidate_fixed_point_relative_tolerance = 1e-4
candidate_fixed_point_damping = 0.5
candidate_fixed_point_fallback = max_delay_seen
```

收敛条件：

```text
abs(delay_(k+1) - delay_k) / max(abs(delay_k), 1.0) <= 1e-4
```

若达到最大迭代次数仍未收敛：

1. 选择迭代过程中最大时延作为保守运行时长；
2. 用该运行时长重新计算一次 offered rate、负载、热度、带宽与时延；
3. 将 `converged=False` 和迭代次数写入快照及实验诊断；
4. 不静默回退到接纳前带宽。

节点可用性窗口 `H` 在最终时延确定后计算。节点可用性不反向改变带宽，因此不参与不动点循环。

### 3.5 链路热度和双工口径

维持当前全双工假设，但将其机器可读化：

```text
duplex_mode = full_duplex_max_directional_load
```

对同一物理链路的两个方向分别计算 offered rate 和方向负载，物理负载取两方向最大值：

```text
physical_load_l = max(load_forward_l, load_reverse_l)
```

预测热度为：

```text
g_l' = lambda_g * g_l + (1 - lambda_g) * physical_load_l
```

随后由同一个物理负载和预测热度计算链路稳定性及有效带宽。论文必须明确该容量口径；若未来改成半双工求和，必须升级版本字段。

---

## 4. 代码改造清单

### 4.1 `models.py`

新增：

- `ObservationSnapshot`；
- `CandidateStateSnapshot`；
- 必要时新增 `AvailabilityPrediction` 诊断结构。

调整：

- 将 `CandidateQualityScores.node_stability` 改为 `node_availability`；
- `raw_joint_product` 改名为中性诊断字段，例如 `raw_component_product`，不得解释为成功概率；
- `PostAdmissionMetrics` 逐步由 `CandidateStateSnapshot` 替代；若暂时保留，必须从候选快照构造，不允许独立重算；
- `Node.operational_stability` 与 `base_operational_stability` 标记为旧语义兼容字段，正式路径不再读取。

兼容原则：旧字段只允许在加载旧配置或消融模式时使用，不允许在新版结果 Schema 中继续伪装为节点可用性。

### 4.2 `core/environment.py`

新增配置：

```python
objective_semantics_version = "availability_oss_profiled_ifs_v3"
candidate_prediction_version = "fixed_point_offered_rate_v1"
availability_estimator_version = "empirical_window_beta_v1"
ifs_mode = "profiled_static"
routing_mode = "fixed_shortest_path"
duplex_mode = "full_duplex_max_directional_load"
runtime_progress_mode = "fixed_admission_runtime"
energy_model_version = "compute_uplink_intermediate_return_v2"
```

新增或重构方法：

```python
capture_observation_snapshot()
predict_candidate_state(dnn_index, assignment, observation=None)
predict_execution_window_availability(node_idx, horizon_slots, observation)
collect_running_directional_offered_rates(observation=None)
collect_candidate_directional_data(dnn_index, assignment)
directional_loads_from_offered_rates(...)
predict_effective_bandwidths(...)
aggregate_node_availability(...)
aggregate_link_stability(...)
aggregate_operational_stability(...)
aggregate_profiled_inference_fidelity(...)
```

修改：

- `predict_link_load_ratios()` 改为 offered-rate 内核的薄封装，或在所有调用迁移后删除；
- `predict_paths_state()` 使用统一 offered-rate 和物理链路聚合口径；
- `predict_candidate_scores_by_assignment()` 迁移为候选快照的兼容读取函数，最终删除；
- `update_link_loads()` 调用同一 offered-rate 汇总内核；
- 上下线状态覆盖所有允许部署节点，或由每个节点的 `availability_mode` 明确处理；
- 可行性检查加入“候选使用节点当前在线”硬约束；
- `alpha_r`、`beta_r`、`o_min` 从正式 OSS 路径移除，保留在 legacy 配置组；
- `alpha_a`、`beta_a`、`a_min` 从 `profiled_static` 路径移除，保留在 IFS 消融配置组。

重要约束：`predict_candidate_state()` 必须无副作用。调用前后，节点 CPU、节点负载、热度、链路带宽、运行队列和随机数状态均不变化。

### 4.3 `proposed.py`

将 `_evaluate_assignment()` 改为只生成并消费一次候选快照：

```python
snapshot = self.env.predict_candidate_state(dnn_index, assignment_slice)
return ObjectiveValues(
    inference_fidelity=snapshot.inference_fidelity,
    operational_stability=snapshot.operational_stability,
    delay_satisfaction=...snapshot.estimated_delay...,
    energy_satisfaction=...snapshot.total_energy...,
    estimated_delay=snapshot.estimated_delay,
    total_energy=snapshot.total_energy,
    deadline_feasible=snapshot.deadline_feasible,
)
```

修改要求：

- 删除“先单独算 delay，再把 delay 传给质量预测”的主路径；
- `known_delay` 不再允许绕过候选闭环；若仅用于测试，改成明确的 `initial_delay_hint`，且最终结果仍由快照生成；
- 缓存键至少包含 `dnn_index`、assignment、观测快照版本/时隙和所有语义版本；
- 每次环境推进、请求变化或版本变化后清空候选缓存；
- 约束判断使用快照中的预测节点资源、链路超载和时延，不能混用当前环境负载。

### 4.4 基线算法

检查并修改 `rtbl/` 及其他算法入口：

- 所有算法从相同 `ObservationSnapshot` 读取可见状态；
- 所有算法的最终方案统一调用 `predict_candidate_state()` 记录四目标；
- RTBL 可保留原有决策规则，但不能使用不同的指标重算公式；
- 至少提供一个拥有相同可见状态的增强贪心基线，降低“主算法信息更多”的比较偏差；
- 禁止基线读取未来可用性轨迹。

### 4.5 实验与结果 Schema

修改 `experiment_runner.py`、指标汇总和绘图入口，使每行正式结果至少包含：

```text
objective_semantics_version
candidate_prediction_version
availability_estimator_version
ifs_mode
routing_mode
duplex_mode
runtime_progress_mode
energy_model_version
quality_parameter_profile
seed
scenario_id
algorithm
```

候选/运行诊断增加：

```text
candidate_prediction_converged
candidate_prediction_iterations
candidate_predicted_delay
candidate_predicted_link_load
post_admission_actual_link_load
prediction_absolute_error
prediction_relative_error
```

推荐升级：

```text
result_schema_version = semantic_names_v5_p0_closure
pareto_schema_version = 4d_unified_v5_p0_closure
```

旧结果不得通过字段重命名伪装成新版语义。凡 OSS/IFS 数值含义发生变化的结果必须重新运行；旧结果只能归档用于追溯。

### 4.6 配置与说明文档

新增可追踪质量配置，例如：

```text
configs/quality_profiles/synthetic_default_v1.json
```

至少记录：

- 每类节点的 `base_inference_fidelity` 来源或合成规则；
- 可用性模式与历史窗口长度；
- Beta 平滑参数；
- 链路稳定性先验及动态系数；
- OSS 域权重；
- 参数允许范围和默认值。

同步更新：

- `code_optimization/current_system_model_and_algorithm_design.md`；
- `code_optimization/experiments.md`；
- 项目 README、代码注释、图例和表头。

统一使用“运行稳定性 OSS”“推理保真性 IFS”，删除正式展示中的 `reliability probability`、`accuracy` 和“任务成功概率”等过度表述。

---

## 5. 分阶段实施计划

### 阶段 0：冻结基线与版本边界

任务：

1. 保存当前测试结果和最小运行样例；
2. 标记当前语义版本为 legacy；
3. 新增新版版本常量，但暂不切换默认；
4. 为旧结果建立只读归档目录；
5. 确认所有随机数均由显式 seed 驱动。

完成条件：当前测试可复现，旧结果与新版输出目录完全隔离。

### 阶段 1：观测快照与节点可用性

任务：

1. 实现 `ObservationSnapshot`；
2. 将节点上下线状态扩展到全部可部署节点或显式配置 `always_on`；
3. 实现经验连续窗口预测器；
4. 实现当前离线节点硬约束；
5. 实现 FLOPs 加权节点可用性聚合。

完成条件：可用性预测只依赖当前及历史信息，并通过窗口单调性与不可见未来测试。

### 阶段 2：共享 offered-rate 内核

任务：

1. 抽取运行任务 offered-rate 汇总；
2. 抽取候选方向数据量汇总；
3. 统一方向负载、物理负载和双工聚合；
4. 迁移 `update_link_loads()`；
5. 删除归一化负载直接相加的实现。

完成条件：同一带宽/热度快照下，候选预测后接纳与实际重算的负载误差小于 `1e-9`。

### 阶段 3：统一候选不动点快照

任务：

1. 实现 `CandidateStateSnapshot`；
2. 实现固定路由下的有限不动点预测；
3. 增加阻尼、收敛诊断和保守回退；
4. 将 OSS、IFS、时延、能耗和约束装入同一快照；
5. 修改 `_evaluate_assignment()` 和缓存。

完成条件：候选四目标仅从一个快照读取，预测过程无副作用，零流量退化和通信量单调性测试通过。

### 阶段 4：切换 OSS/IFS 正式语义

任务：

1. OSS 节点域切换为执行窗口可用性；
2. OSS 链路域继续使用候选接纳后传输稳定性；
3. IFS 默认切换为 `profiled_static`；
4. 旧动态 OSS/IFS 迁入显式 legacy/ablation 模式；
5. 清理结果字段与文字表述。

完成条件：默认路径不再使用 CPU 负载直接降低节点稳定性或 IFS；OSS/IFS 相关性不再由相同节点负载公式机械产生。

### 阶段 5：基线、结果 Schema 与诊断

任务：

1. 基线统一使用相同观测和最终评价内核；
2. 增加版本字段；
3. 增加预测—实际一致性日志；
4. 增加不动点收敛统计；
5. 增加拒绝原因分解。

完成条件：任意结果行都能确定其模型语义、预测方式、路由、双工、运行时长、能耗版本、参数档案和 seed。

### 阶段 6：正式实验

在代码冻结后执行：

1. 至少 20 个配对随机种子的主对比；
2. 均值、标准差或 95% 置信区间；
3. 配对显著性检验和效应量；
4. 遗传算子消融；
5. 可用性、链路稳定性、OSS 权重和 IFS 先验敏感性；
6. OSS/IFS Pearson、Spearman、散点图和目标冲突率；
7. HV、IGD+ 或 epsilon indicator；
8. 规模、收敛、运行时间；
9. 候选预测—接纳后实际状态误差分布；
10. 不动点收敛率、迭代次数和保守回退比例。

完成条件：所有正式结果均来自同一新版语义，原始数据、汇总、统计和绘图可以一键重建。

---

## 6. 测试计划

### 6.1 新增测试文件

建议新增：

```text
tests/test_node_availability_prediction.py
tests/test_offered_rate_consistency.py
tests/test_candidate_state_snapshot.py
tests/test_p0_model_semantics.py
```

### 6.2 节点可用性测试

- 当前离线节点不能生成可行候选；
- 预测值始终位于 `(0, 1]`；
- 相同历史下，执行窗口变长时可用性预测不应上升；
- 全在线历史的预测接近 1，全离线窗口不会被当成可用起点；
- 固定历史与参数时结果确定；
- 修改未来轨迹不影响当前预测；
- `always_on` 节点预测恒为 1；
- 多节点聚合按 FLOPs 权重计算且不受任务切分数量影响。

### 6.3 offered-rate 一致性测试

- 运行任务方向速率等于数据量除以存储运行时长；
- 候选方向速率与实际接纳后的新增速率一致；
- 同一物理链路双向流量只在物理域计一次；
- 全双工物理负载取两方向最大值；
- 同一热度和带宽快照下，预测负载与实际重算误差 `< 1e-9`；
- 输入上传、中间通信和结果回传均进入链路数据汇总；
- 无跨节点通信时链路增量为零。

### 6.4 候选快照测试

- 调用前后环境对象状态完全一致；
- 四目标、原始时延和能耗来自同一 assignment；
- 零候选流量时退化到当前状态；
- 增大候选通信量时预测 offered rate、链路负载和时延不下降；
- 固定输入下收敛结果可复现；
- 未收敛时使用保守最大时延并设置诊断标志；
- 同一规划时隙缓存命中，不同时隙或不同语义版本缓存失效；
- 时延和链路约束使用快照中的同一预测有效带宽。

### 6.5 OSS/IFS 语义测试

- 节点窗口可用性下降时 OSS 不上升；
- 链路稳定性下降时 OSS 不上升；
- 无物理链路时 OSS 等于节点可用性聚合；
- `profiled_static` 模式下仅改变负载/热度不会改变 IFS；
- 降低某承载节点的基础 IFS 时，DNN 级 IFS 不上升；
- 相同 FLOPs 分配在任务拆分后得到相同聚合结果；
- OSS 和 IFS 均在 `[0, 1]`；
- 结果字段不存在把 OSS/IFS 标为概率或具体 DNN accuracy 的默认名称。

### 6.6 回归测试

- 所有算法仍能完成最小场景；
- 固定 seed 的请求流和环境轨迹一致；
- 主算法与基线的观测快照一致；
- 能耗仍包含结果回传；
- 固定路由缓存不因候选预测被修改；
- 旧结果读取只能进入 legacy 路径，不能写成新版 Schema。

---

## 7. P0 问题—代码—证据验收矩阵

| 问题 | 主要改造 | 必须提供的代码证据 | 必须提供的实验/测试证据 |
| --- | --- | --- | --- |
| P0-01 四目标状态不一致 | `ObservationSnapshot`、`CandidateStateSnapshot`、带宽—负载—时延迭代 | `_evaluate_assignment()` 只读取一个候选快照 | 无副作用、单调性、收敛率、四目标同状态测试 |
| P0-02 链路公式不一致 | 共享 offered-rate 汇总内核 | 预测与 `update_link_loads()` 调用同一底层函数 | 同快照误差 `< 1e-9`，预测—实际误差分布 |
| P0-03 IFS 物理含义不足 | 默认 `profiled_static`，删除负载直接衰减；OSS 节点域改为窗口可用性 | 正式路径不读取 `alpha_a/beta_a`；节点负载不直接生成节点稳定性 | IFS 先验敏感性、代理事件定义、OSS/IFS 区分效度 |
| P0-04 缺正式结果 | 版本化 Schema、统一基线、诊断和一键实验 | 每行结果带完整版本与 seed | 多种子、统计检验、消融、Pareto、规模、收敛、一致性 |

---

## 8. 本轮明确不扩展的范围

为避免 P0 整改失控，本轮不同时引入以下内容：

- 不对具体 DNN、数据集或 Top-1/mAP 建模；
- 不实现动态路由；
- 不实现任务迁移、重试、冗余副本或检查点恢复；
- 不建模节点和链路之间的相关故障；
- 不把 `remaining_slots` 改造成逐时隙剩余计算量/通信量仿真；
- 不引入负载相关动态功耗模型；
- 不把 OSS 强行校准成端到端成功概率。

`remaining_slots` 暂时保留为“接纳时固定服务时长近似”，并通过：

```text
runtime_progress_mode = fixed_admission_runtime
```

明确记录。该项属于后续 P1 扩展，不阻塞本轮 P0 闭环，但论文不得声称已经实现真实逐时隙执行进度模拟。

---

## 9. 迁移与清理策略

### 9.1 双路径过渡

短期保留：

```text
legacy_v2                  仅用于回归和旧结果复核
availability_oss_v3       新版默认与正式实验
```

所有入口必须显式选择语义版本。新版稳定后删除生产代码中的隐式布尔开关，例如仅靠 `use_predicted_quality_scores` 决定整套目标含义。

### 9.2 不允许的迁移

以下操作禁止：

- 将旧 `operational_stability` 数值直接改列名为节点可用性；
- 将旧动态 IFS 数值改列名为 `profiled_static` IFS；
- 把旧结果补写新版版本号；
- 在同一统计汇总中混合不同 `objective_semantics_version`；
- 用测试数据替代正式多种子实验结果。

### 9.3 最终可删除代码

待新版正式路径、测试和实验入口稳定后，再删除：

- 基于 `current_load_ratio + candidate_increment` 的候选链路负载计算；
- 正式路径中的节点 `alpha_r/beta_r/o_min` 动态稳定性公式；
- 正式路径中的 `alpha_a/beta_a/a_min` 动态 IFS 公式；
- 独立重算 OSS/IFS 的旧兼容方法；
- 无版本保护的旧结果字段别名。

删除前至少保留一个独立 commit 或归档标签，便于复核历史实验。

---

## 10. 最终完成定义

只有同时满足以下条件，才能将本文档从 `plans/pending/` 迁入已实施归档：

- [ ] `_evaluate_assignment()` 只消费一个统一候选快照；
- [ ] 候选预测不修改真实环境或随机数状态；
- [ ] 预测与实际链路负载使用同一 offered-rate 内核；
- [ ] 同快照预测—实际链路负载误差小于 `1e-9`；
- [ ] 当前离线节点作为硬约束处理；
- [ ] 节点可用性预测只使用当前及历史可见信息；
- [ ] OSS 使用执行窗口节点可用性与链路稳定性分层聚合；
- [ ] 默认 IFS 不由 CPU 负载或热度直接衰减；
- [ ] OSS/IFS 不被表述为任务成功概率或具体 DNN 准确率；
- [ ] 固定路由、全双工和固定接纳时长均有机器可读版本；
- [ ] 能耗包含输入、中间传输和结果回传；
- [ ] 所有算法最终指标使用同一评价内核；
- [ ] 新增测试及现有回归测试全部通过；
- [ ] 新旧结果目录和版本字段严格隔离；
- [ ] 正式多种子、显著性、效应量、消融、敏感性、Pareto、规模、收敛和一致性实验完成；
- [ ] 原始数据、汇总、统计和绘图可由文档化命令重新生成。

完成代码闭环但尚未完成正式实验时，只能标记为“P0 模型实现完成”；完成全部实验和复现材料后，才能标记为“投稿就绪”。
