# 周期性规划架构下的模型一致性修复方案（设计来源，已被统一方案取代）

> 状态：不再作为独立实施入口。模型一致性修复内容已经收敛至 `../pending/unified_periodic_planning_model_closure_implementation_plan.md`。若语义版本、实施阶段或验收条件与统一方案冲突，以统一方案为准。本文件仅用于追溯详细论证。

## 1. 文档状态

- 状态：待实施
- 适用阶段：当前代码改造阶段
- 适用架构：泊松请求到达、DNN 服务画像目录、异步周期规划、在线快速匹配
- 当前目标语义：`stability_fidelity_v2_return_energy`
- 修复完成后的建议目标语义：`stability_fidelity_v3_profiled`
- 关联问题清单：`code_optimization/model_submission_readiness_issues.md`
- 原始关联周期方案：`../reference/poisson_arrival_periodic_planning_code_modification_plan.md`
- 当前实施入口：`../pending/unified_periodic_planning_model_closure_implementation_plan.md`

本方案解决周期性规划改造前必须闭合的模型一致性问题。核心原则是：

> 周期方案池只能缓存部署结构，不能长期复用基于旧状态计算的性能结论。周期规划和在线复验必须调用同一个统一候选预测器，并使用明确、可验证的状态时点和信息边界。

### 1.1 本方案保留的原 P0 问题

- P0-01：候选四目标没有基于同一个预测状态计算；
- P0-02：候选链路预测与接纳后的实际链路状态更新公式不一致；
- P0-03：IFS 的物理含义不足以支撑“推理准确性”主张。

### 1.2 本方案明确排除的原 P0 问题

原问题清单中的：

```text
P0-04 新语义版本缺少完整的正式实验结果
```

不纳入本轮代码修复阻塞项。原因是当前项目仍处于模型和架构改造阶段，正式统计实验应在代码语义冻结之后执行。

本方案只要求：

- 单元测试；
- 确定性一致性测试；
- 小规模集成测试；
- 固定种子回归测试；
- 性能烟雾测试。

不要求在本轮完成：

- 正式多随机种子对比；
- 显著性检验；
- 完整消融实验；
- 正式论文表格和绘图；
- 投稿级结果归档。

这些工作在模型语义冻结后另行规划。

---

## 2. 修复目标

本轮改造需要达到以下结果：

1. 建立算法只读观测快照，阻止周期规划器和在线调度器直接读取环境内部真值。
2. 建立统一候选状态快照，使 delay、链路负载、OSS、IFS 和约束来自同一个预测状态。
3. 候选预测与实际环境更新共用同一个链路 offered-rate 汇总公式。
4. 在候选内部闭合“带宽—时延—流量速率—负载”的有限次迭代。
5. 将 IFS 改为 DNN 画像—节点相关的可监控代理质量，而不是通用节点准确率。
6. 将节点当前可用性、未来生存概率和在线条件下的 OSS 分开建模。
7. 明确周期规划快照、候选预测状态、在线复验状态和实际执行状态的时间关系。
8. 方案池发布前和请求接纳前均执行统一复验。
9. 明确固定路由、双工容量、固定服务时长和确定性能耗的模型边界。
10. 为所有语义变化建立机器可读版本字段。

---

## 3. 统一状态与时间语义

### 3.1 四类状态

改造后必须区分：

```text
EnvironmentInternalState
  仿真器完整内部状态

ObservationSnapshot
  调度算法可见的当前系统观测

CandidateStateSnapshot
  在 ObservationSnapshot 上假设接纳候选后的预测状态

ExecutionState
  已接纳请求在环境中的实际运行状态
```

### 3.2 时隙语义

对时隙 `t`，建议统一以下事件顺序：

```text
1. 进入时隙 t
2. 结算上一时隙结束事件和节点可用性转换
3. 生成 ObservationSnapshot(t)
4. 暴露本时隙泊松到达请求
5. 按 EDF 或配置策略逐个处理请求
6. 每接纳一个请求后立即更新资源占用
7. 本时隙所有请求处理完成
8. 更新节点/链路负载、热度、有效带宽和质量状态
9. 推进运行任务剩余时长
10. 进入时隙 t+1
```

同一时隙的多个请求共享同一个节点可用性状态，但后处理请求必须看到先接纳请求形成的累计资源占用。

### 3.3 候选预测时点

统一规定 `CandidateStateSnapshot` 表示：

```text
候选在当前时隙被接纳，加入现有运行集合后，
按当前模型执行一次负载/热度更新所对应的预测状态。
```

因此：

- 节点和链路负载包含现有运行请求与候选请求；
- 热度按一次 EMA 更新计算；
- 有效带宽由预测负载和预测热度得到；
- delay 使用该预测有效带宽；
- OSS 使用同一预测负载、热度和有效带宽；
- IFS 使用同一候选部署和当前画像质量表；
- 链路过载约束使用同一预测负载。

### 3.4 周期规划与在线复验

周期规划时：

```text
CandidateStateSnapshot(plan)
  = predict(PlanningObservationSnapshot, candidate)
```

在线请求到达时：

```text
CandidateStateSnapshot(online)
  = predict(CurrentObservationSnapshot, cached_assignment)
```

在线接纳必须使用 `CandidateStateSnapshot(online)`，不得直接使用方案池中保存的规划指标。

---

## 4. P0-01：统一候选四目标状态

### 4.1 当前问题

当前候选评价顺序为：

```text
当前有效带宽计算 delay
  -> 根据 delay 预测候选负载
  -> 预测 OSS/IFS
  -> 计算确定性能耗
```

delay 和 OSS/IFS 因此不属于同一个候选后状态。周期性规划若直接缓存该结果，会把状态语义偏差长期复用。

### 4.2 新增 `CandidateStateSnapshot`

文件：`models.py` 或建议新增 `core/snapshots.py`

```python
@dataclass(frozen=True)
class CandidateStateSnapshot:
    source_snapshot_version: int
    profile_id: str
    request_id: int | None
    assignment: tuple[int, ...]

    existing_node_loads: tuple[float, ...]
    predicted_node_loads: tuple[float, ...]
    existing_link_offered_rates: tuple[float, ...]
    candidate_link_offered_rates: tuple[float, ...]
    predicted_link_offered_rates: tuple[float, ...]
    predicted_link_loads: tuple[float, ...]

    predicted_node_heat: tuple[float, ...]
    predicted_link_heat: tuple[float, ...]
    predicted_effective_bandwidths: tuple[float, ...]

    predicted_delay_ms: float
    predicted_operational_stability: float
    predicted_inference_fidelity: float
    deterministic_total_energy: float

    deadline_feasible: bool
    resource_feasible: bool
    hierarchy_feasible: bool
    link_feasible: bool
    availability_feasible: bool
    constraint_violation: float

    fixed_point_converged: bool
    fixed_point_iterations: int
    prediction_version: str
```

### 4.3 新增统一预测入口

文件：`core/environment.py`

```python
def predict_candidate_state(
    self,
    observation: ObservationSnapshot,
    profile: DNNProfile,
    request: InferenceRequest,
    assignment: Sequence[int],
    *,
    max_iterations: int = 5,
    relative_tolerance: float = 1e-4,
) -> CandidateStateSnapshot:
    ...
```

禁止周期规划器分别调用：

```text
estimate_delay_from_assignment()
predict_candidate_scores_by_assignment()
predict_link_load_ratios()
```

拼装候选结果。上述旧接口可以作为兼容适配层，但内部必须委托给统一预测器。

### 4.4 修改 `_evaluate_assignment()`

文件：`proposed.py`

改为：

```python
candidate_state = self.env.predict_candidate_state(
    observation,
    profile,
    request,
    assignment,
)

return ObjectiveValues(
    inference_fidelity=candidate_state.predicted_inference_fidelity,
    operational_stability=candidate_state.predicted_operational_stability,
    delay_satisfaction=delay_satisfaction(
        candidate_state.predicted_delay_ms,
        request.deadline_ms,
    ),
    energy_satisfaction=energy_satisfaction(
        profile,
        candidate_state.deterministic_total_energy,
    ),
    estimated_delay=candidate_state.predicted_delay_ms,
    total_energy=candidate_state.deterministic_total_energy,
    deadline_feasible=candidate_state.deadline_feasible,
)
```

约束违反度直接读取 `candidate_state.constraint_violation`，不得再次独立估算时延或链路负载。

### 4.5 缓存键

统一候选缓存键必须包含状态版本：

```python
cache_key = (
    observation.snapshot_version,
    profile.profile_id,
    request.initiate_node,
    request.deadline_ms,
    tuple(assignment),
    candidate_prediction_version,
)
```

禁止只使用：

```python
(dnn_index, tuple(assignment))
```

否则跨快照复用会返回过期结果。

### 4.6 验收标准

- 四目标评价只消费一个 `CandidateStateSnapshot`；
- constraint violation 不再重复运行另一套预测；
- 候选预测不修改环境；
- 相同快照、相同候选结果完全可复现；
- 快照版本变化后缓存不会误命中；
- 增加候选数据量时，预测链路负载和预测 delay 不下降；
- 候选无通信时，链路预测退化为当前状态。

---

## 5. P0-02：统一链路 offered-rate 模型

### 5.1 当前问题

当前候选预测在旧 `load_ratio` 上增加候选归一化负载，而实际环境更新会重新汇总所有运行请求。有效带宽变化后两种公式不等价。

### 5.2 统一底层数据

链路动态模型的原始量应是 offered rate，而不是归一化负载：

```text
offered_rate_l = sum(data_on_l / service_runtime)
load_ratio_l = offered_rate_l / effective_bandwidth_l
```

文件：`core/environment.py`

新增：

```python
def aggregate_running_link_offered_rates(
    self,
    running_dnns: Sequence[RunningDNN] | None = None,
) -> dict[int, float]:
    ...


def candidate_link_offered_rates(
    self,
    profile: DNNProfile,
    request: InferenceRequest,
    assignment: Sequence[int],
    predicted_runtime_ms: float,
) -> dict[int, float]:
    ...


def link_loads_from_offered_rates(
    self,
    offered_rates: Mapping[int, float],
    effective_bandwidths: Mapping[int, float],
) -> dict[int, float]:
    ...
```

### 5.3 统一候选预测和实际更新

`predict_candidate_state()`：

```text
predicted_rate
  = aggregate_running_link_offered_rates()
  + candidate_link_offered_rates()
```

`update_link_loads()`：

```text
actual_rate
  = aggregate_running_link_offered_rates()
```

两者必须调用相同函数生成运行请求流量。

### 5.4 有向边与物理链路

建议分别维护：

```text
directed_offered_rate[link_id]
physical_load[physical_link_id]
```

全双工模式：

```text
physical_load = max(forward_load, reverse_load)
```

共享容量模式：

```text
physical_load
  = (forward_rate + reverse_rate) / shared_bandwidth
```

禁止先分别计算归一化负载后再无说明地求和。

### 5.5 单位

统一明确：

```text
data amount       = Kbit
runtime           = ms
offered rate      = Mbit/s
bandwidth         = Mbit/s
load ratio        = dimensionless
```

数值上：

```text
Kbit / ms = Mbit/s
```

建议变量名显式包含单位：

```python
data_kbit
runtime_ms
offered_rate_mbps
effective_bandwidth_mbps
```

### 5.6 验收标准

- 候选预测和实际环境使用同一 offered-rate 汇总函数；
- 在不推进热度的情况下，预测接纳负载与实际接纳负载误差小于 `1e-9`；
- 单方向流量增加时物理链路负载不下降；
- `full` 和 `shared` 两种双工模式均有测试；
- 输入上传、中间传输和结果回传均计入 offered rate；
- 同一物理链路的两个方向不会重复计算故障域。

---

## 6. 带宽—时延—负载固定点预测

### 6.1 闭环关系

候选运行时长影响候选平均流量速率：

```text
runtime -> offered rate -> load -> heat
        -> effective bandwidth -> delay -> runtime
```

因此需要有限次固定点迭代。

### 6.2 推荐算法

```python
delay_ms = estimate_delay(
    assignment,
    observation.link_effective_bandwidths,
)

delay_history = [delay_ms]
converged = False

for iteration in range(max_iterations):
    candidate_rates = candidate_link_offered_rates(
        profile,
        request,
        assignment,
        predicted_runtime_ms=max(delay_ms, slot_length_ms),
    )
    total_rates = add_rates(existing_rates, candidate_rates)
    predicted_loads = solve_link_loads(total_rates, predicted_bandwidths)
    predicted_heat = predict_link_heat(observation, predicted_loads)
    predicted_bandwidths = predict_effective_bandwidths(
        predicted_loads,
        predicted_heat,
    )
    next_delay_ms = estimate_delay(assignment, predicted_bandwidths)
    delay_history.append(next_delay_ms)

    if relative_change(next_delay_ms, delay_ms) <= tolerance:
        delay_ms = next_delay_ms
        converged = True
        break

    delay_ms = next_delay_ms

if not converged:
    delay_ms = max(delay_history)
```

### 6.3 默认配置

```python
candidate_prediction_max_iterations = 5
candidate_prediction_relative_tolerance = 1e-4
candidate_prediction_nonconvergence_policy = "max_delay"
```

### 6.4 路由边界

固定点迭代期间保持路由不变。否则同时改变：

- 部署；
- 路由；
- 带宽；
- 流量；

会显著扩大模型和搜索空间。

### 6.5 数值保护

- `runtime_ms >= 1e-9`；
- `effective_bandwidth_mbps >= min_bandwidth`；
- 负载允许大于 1，用于表示过载；
- 不在预测函数内静默裁剪过载到 1；
- 时延非有限时标记候选不可行；
- 非收敛必须记录诊断字段。

### 6.6 验收标准

- 零候选流量最多一次迭代即稳定；
- 增大数据量不会降低收敛后的 delay；
- 降低基础带宽不会降低 delay；
- 非收敛候选采用保守最大 delay；
- 固定种子结果可复现；
- 固定点预测不修改环境状态。

---

## 7. P0-03：IFS 改为画像—节点保真性

### 7.1 语义调整

IFS 统一定义为：

> DNN 服务画像在指定节点和执行配置上保持参考输出质量的归一化代理分数。

IFS 不是：

- 具体数据集上的 Top-1 accuracy；
- 单次推理成功概率；
- 节点在线概率；
- CPU 利用率的简单反函数。

### 7.2 新增质量注册表

建议新增 `quality/registry.py`：

```python
@dataclass(frozen=True)
class ProfileNodeFidelity:
    profile_id: str
    node_id: int
    execution_config: str
    reference_agreement: float
    sample_count: int
    source: str
    last_updated_slot: int
    confidence: float | None = None


class QualityRegistry:
    def get_profile_node_fidelity(
        self,
        profile_id: str,
        node_id: int,
        execution_config: str,
    ) -> ProfileNodeFidelity:
        ...
```

### 7.3 质量来源

支持以下 `source`：

```text
offline_validation
online_probe
cloud_reference_comparison
output_integrity_check
controlled_synthetic
```

当前没有真实 profiling 数据时，允许使用 `controlled_synthetic`，但必须：

- 固定生成种子；
- 保存完整矩阵；
- 把范围写入场景配置；
- 不声称代表真实模型准确率。

### 7.4 第一阶段推荐 IFS 公式

对画像 `q` 的部署 `x`：

```text
IFS_q(x)
  = exp(
      sum_n workload_fraction(q, n, x)
      * log(profile_node_fidelity(q, n))
    )
```

其中：

```text
workload_fraction
  = assigned_MFLOP_on_node / total_profile_MFLOP
```

第一阶段不再使用未经校准的：

```text
base_ifs * exp(-alpha_a * load - beta_a * heat)
```

### 7.5 动态 IFS 的可选扩展

只有存在 profiling 或明确代理事件时，才允许增加动态项，例如：

```text
fidelity(q,n,t)
  = profiled_fidelity(q,n,config_t)
  * integrity_factor(n,t)
```

其中 `config_t` 可以表示：

- 精度模式；
- 量化位宽；
- 模型版本；
- 早退配置；
- 近似计算模式。

普通CPU负载只影响 delay 和 OSS，不默认直接降低 IFS。

### 7.6 缺失画像处理

若画像—节点组合没有质量记录：

```text
policy = reject | conservative_default | fallback_profile
```

默认建议：

```text
conservative_default
```

并记录：

```text
fidelity_source = conservative_default
```

### 7.7 验收标准

- IFS 查询必须同时包含 `profile_id` 和 `node_id`；
- 同一节点可对不同 DNN 画像具有不同 IFS；
- 负载变化不会在无配置变化时直接改变 IFS；
- IFS 聚合仍按 FLOPs 权重；
- 缺失画像行为可配置且可追踪；
- 代码、CSV 和文档主字段不再使用 `accuracy` 表示 IFS；
- 旧 `a_reliability` 仅保留为 deprecated 兼容字段。

---

## 8. 节点可用性、未来生存概率与 OSS 分离

### 8.1 三层定义

```text
A_n(t) in {0,1}
  节点当前是否在线

P_surv_n(t,H) in [0,1]
  当前在线条件下，未来 H 内保持在线的估计概率

Q_oss_n(t) in [0,1]
  节点在线条件下的运行稳定性质量
```

### 8.2 当前可用性硬约束

候选必须满足：

```text
assignment_task_to_node <= node_available
```

实现上：

```python
availability_feasible = all(
    observation.node_available[node_idx]
    for node_idx in used_nodes
)
```

离线节点不得：

- 进入周期方案；
- 通过在线复验；
- 继续承载已运行请求而不产生结果。

### 8.3 可用性更新时间

节点可用性每时隙只更新一次。禁止在每个请求的调度函数中调用会推进故障过程的接口。

建议拆分：

```python
def advance_availability_one_slot(self) -> None:
    ...

def get_current_availability(self) -> tuple[bool, ...]:
    ...
```

### 8.4 环境真值与算法观测

环境内部可以持有：

```text
remaining_up_or_down_slots
next_failure_slot
random_generator_state
```

算法只能获得：

```text
current_available
online_age_slots
historical_availability_estimate
estimated_survival_probability
```

不得把真实 `remaining_time` 写入 `ObservationSnapshot`。

### 8.5 未来生存概率

若在线时长服从 Weibull：

```text
S(age) = exp(-(age / scale)^shape)
```

则：

```text
P_surv(t,H)
  = S(age + H) / S(age)
```

周期方案可使用机会约束：

```text
product(P_surv_n for n in used_nodes) >= min_plan_survival
```

第一阶段可以只实现当前可用性硬约束，把未来生存概率作为第二阶段功能。

### 8.6 运行期节点下线

时隙边界更新可用性后检查：

```python
for running_dnn in running_dnns:
    if any(not node_available[n] for n in running_dnn.used_nodes):
        fail_running_dnn(running_dnn, reason="node_outage")
```

第一阶段策略：

- 请求执行失败；
- 释放资源；
- 记录失败原因；
- 不迁移、不重试。

周期方案池通过多个节点集合不同的候选降低单节点下线影响。

### 8.7 OSS 定义

OSS 继续表示在线条件下的运行质量：

- 节点负载；
- 节点热度；
- 链路负载；
- 链路热度；
- 节点/链路经验质量先验。

当前二值可用性作为可行性门控，不直接平均进 OSS。

### 8.8 验收标准

- 所有可部署节点均有可用性状态；
- 同一时隙多个请求不会推进多次可用性过程；
- 周期规划和在线接纳均排除离线节点；
- 仿真真实剩余在线时间不进入算法观测；
- 运行节点下线能够产生独立执行失败记录；
- OSS、IFS 和 availability 使用不同字段和不同语义。

---

## 9. OSS 与 IFS 区分效度的代码基础

### 9.1 驱动机制分离

修复后：

```text
OSS drivers
  = operational prior
  + node/link load
  + node/link heat
  + link state

IFS drivers
  = DNN profile
  + node execution profile
  + model/execution configuration
  + integrity monitoring
```

### 9.2 诊断接口

虽然本轮不要求正式统计实验，但代码应预留：

```python
def export_candidate_objective_diagnostics(...) -> dict[str, float]:
    return {
        "operational_stability": ...,
        "inference_fidelity": ...,
        "node_load_component": ...,
        "link_load_component": ...,
        "profile_fidelity_component": ...,
    }
```

便于后续计算：

- Pearson相关系数；
- Spearman相关系数；
- 目标冲突率；
- 去掉OSS或IFS的消融。

### 9.3 当前阶段验收

- 更改负载时 OSS 可变化而 IFS 不必同步变化；
- 更改画像质量表时 IFS 可变化而 OSS 不变化；
- 同一候选的两项目标可以产生不同排序；
- 诊断字段能够区分两者来源。

---

## 10. 质量参数和退化参数配置化

### 10.1 场景配置

把以下参数全部加入 `Scenario` 和结果元数据：

```text
alpha_r
beta_r
alpha_l
beta_l
lambda_h
lambda_g
o_min
l_min
node_stability_weight
link_stability_weight
quality_profile_version
profile_fidelity_catalog_version
availability_profile_version
```

IFS 不再默认使用 `alpha_a/beta_a` 动态负载退化。旧参数保留时标记 deprecated。

### 10.2 参数来源

每个配置必须声明：

```text
measured
literature_based
controlled_synthetic
```

当前没有真实测量时使用：

```text
quality_parameter_source = controlled_synthetic
```

### 10.3 参数校验

- 所有质量值在 `(0,1]`；
- 权重非负且总和大于零；
- EMA系数在 `[0,1]`；
- Weibull和对数正态参数合法；
- 画像质量目录覆盖率可报告；
- 不允许静默使用缺失默认值而不记录来源。

---

## 11. 固定路由与双工容量语义

### 11.1 路由模式

当前主模型保留固定路由：

```text
routing_mode = fixed_base_route_v1
```

含义：

- 路径按基础拓扑和基础带宽预先确定；
- 周期规划器优化任务部署，不优化路由；
- 动态带宽只改变固定路径上的时延和负载；
- 不声称联合部署—路由优化。

### 11.2 双工模式

新增：

```python
duplex_mode: Literal["full", "shared"] = "full"
```

`full`：

```text
两个方向具有独立容量；
共享故障和热度域；
物理压力取较繁忙方向。
```

`shared`：

```text
两个方向共享总容量；
物理 offered rate 为两个方向之和。
```

### 11.3 验收标准

- 路由模式写入实验元数据；
- 周期方案保存使用的固定物理链路 ID；
- 拓扑版本变化时旧方案失效；
- `full/shared` 两种模式公式和测试分离；
- 上传和回传不会因双向边表示重复计数。

---

## 12. 能耗模型和归一化边界

### 12.1 能耗语义

当前继续采用确定性一阶模型：

```text
total_energy
  = compute_energy
  + input_upload_energy
  + intermediate_transfer_energy
  + result_return_energy
```

不因候选动态负载额外预测功耗。

### 12.2 画像级参考边界

周期方案按 `profile_id` 重复使用，能耗参考边界也应按画像固定：

```python
energy_reference_bounds(profile_id, origin_group)
```

边界不能依赖：

- 当前算法搜索出的候选；
- 其他算法的测试结果；
- 未来请求轨迹。

### 12.3 报告字段

方案池保存：

```text
raw_total_energy
energy_satisfaction
energy_reference_version
```

在线复验重新计算请求实际发起位置对应的传输能耗。

### 12.4 规划器能耗

后台规划器的控制器能耗若后续测量，单独记录：

```text
planner_energy
amortized_planner_energy
```

不得混入 DNN 部署执行能耗目标。

---

## 13. 运行时长语义

### 13.1 当前阶段选择

本轮先保留固定服务时长近似：

```text
runtime_progress_mode = fixed_service_time_v1
```

接纳时：

```text
remaining_slots
  = ceil(predicted_delay_ms / slot_length_ms)
```

后续动态带宽影响新请求看到的系统状态，但不追溯修改已接纳请求的完成时间。

### 13.2 为什么暂不同时实施剩余工作量模型

剩余工作量模型需要同时增加：

- DAG任务执行状态；
- 计算进度；
- 中间传输进度；
- 链路容量共享；
- 任务就绪条件；
- 结果回传进度。

与本轮候选预测和周期架构同时修改风险过高。

### 13.3 后续扩展接口

`RunningDNN` 预留：

```python
runtime_progress_mode: str
remaining_slots: int | None
task_execution_states: tuple[TaskExecutionState, ...] | None
```

未来可增加：

```text
runtime_progress_mode = remaining_work_v1
```

### 13.4 当前结论边界

使用固定服务时长时，后续吞吐量只能解释为：

> 接纳时固定服务时长的流体近似下，系统的请求接纳和资源占用吞吐。

不能解释为逐任务、逐链路精细执行仿真的真实吞吐。

---

## 14. 显式算法观测快照

### 14.1 新增 `ObservationSnapshot`

建议新增 `core/snapshots.py`：

```python
@dataclass(frozen=True)
class ObservationSnapshot:
    snapshot_version: int
    slot: int
    topology_version: int

    node_available: tuple[bool, ...]
    node_available_cpu: tuple[float, ...]
    node_max_cpu: tuple[float, ...]
    node_load_ratio: tuple[float, ...]
    node_heat: tuple[float, ...]
    node_operational_stability: tuple[float, ...]
    node_online_age_slots: tuple[int, ...]
    node_survival_estimates: tuple[float, ...]

    directed_link_offered_rates_mbps: tuple[float, ...]
    link_load_ratio: tuple[float, ...]
    link_heat: tuple[float, ...]
    link_effective_bandwidth_mbps: tuple[float, ...]
    link_transmission_stability: tuple[float, ...]

    routing_mode: str
    duplex_mode: str
    observation_version: str
```

### 14.2 不进入快照的信息

- 未来请求；
- 未来节点上下线结果；
- 节点真实剩余在线时间；
- 下一次故障时刻；
- 随机数生成器内部状态；
- 尚未完成的周期规划结果；
- 其他算法的未来决策。

### 14.3 接口限制

周期规划器：

```python
search_plan_pool(snapshot, profile, origin_group)
```

在线调度器：

```python
dispatch(request, snapshot, repository)
```

长期目标是二者均不直接持有完整 `Environment`。第一阶段允许执行接纳动作时通过受限环境接口提交：

```python
env.commit_assignment(request, assignment, expected_snapshot_version)
```

提交时执行版本检查和最终复验。

### 14.4 乐观并发控制

若在线复验后环境版本已经变化：

```text
expected_snapshot_version != current_snapshot_version
```

则：

1. 重新捕获快照；
2. 重新评价候选；
3. 超过在线预算则回退或拒绝。

同一线程顺序仿真中很少发生，但该接口可以防止未来异步实现产生陈旧提交。

---

## 15. 周期方案池与修复后模型的集成

### 15.1 方案池只保存结构和规划参考值

`DeploymentPlan` 保存：

```text
profile_id
origin_group
assignment
used_nodes
used_physical_links
required_cpu
planning_reference_metrics
source_snapshot_version
topology_version
created_slot
expires_slot
prediction_version
```

`planning_reference_metrics` 只能用于：

- 方案池诊断；
- 规划时Pareto筛选；
- 状态漂移比较；
- 热启动排序。

不能直接用于在线接纳。

### 15.2 方案生成

```text
PlanningObservationSnapshot
  -> predict_candidate_state()
  -> ObjectiveValues
  -> strict feasibility filter
  -> Pareto filter
  -> failure-domain diversity filter
  -> PlanRepository candidate pool
```

### 15.3 发布前复验

周期规划完成时：

```text
CurrentObservationSnapshot
  -> 对所有新候选重新 predict_candidate_state()
  -> 删除不可用、超载、超期和非收敛候选
  -> 计算状态漂移
  -> 决定发布或保留旧方案
```

建议发布门槛：

```text
valid_candidate_ratio >= min_publish_valid_ratio
```

### 15.4 在线复验

请求到达时：

```text
按 profile_id/origin_group 查找候选
  -> 当前可用性筛选
  -> 当前CPU快速筛选
  -> 统一候选状态预测
  -> 严格约束筛选
  -> 请求偏好选解
  -> commit_assignment()
```

### 15.5 方案故障域多样性

Pareto候选可能全部使用同一高质量节点。增加结构多样性筛选：

```text
node_set_jaccard(plan_i, plan_j) <= max_plan_overlap
```

或至少保留：

- 最优综合方案；
- 最低时延方案；
- 最低能耗方案；
- 最高OSS方案；
- 高IFS方案；
- 不同关键节点集合的备用方案。

### 15.6 触发重规划

新增触发条件：

- 节点上下线；
- 候选池可行率下降；
- 在线直接命中率下降；
- offered-rate漂移超过阈值；
- 画像质量表版本变化；
- 拓扑版本变化；
- 方案池TTL到期。

必须设置最小重规划间隔，防止规划风暴。

---

## 16. 基线公平性接口

### 16.1 共同输入

所有算法共享：

- DNN画像目录；
- 泊松请求轨迹；
- 节点可用性轨迹；
- 拓扑；
- 初始状态；
- `ObservationSnapshot`；
- 统一后验评价器。

### 16.2 算法是否利用信息

允许不同算法选择不使用某些字段，但必须记录：

```text
algorithm_observation_profile
```

例如：

```text
periodic_dareed_full_state
state_aware_greedy_full_state
rtbl_original_observation
maxresource_resource_only
```

### 16.3 推荐基线

本轮代码至少预留：

1. 原RTBL，保持原核心决策；
2. 同信息状态感知贪心；
3. 周期HEFT或周期贪心方案池；
4. `maxresource_fast`在线回退；
5. 逐请求DAREED，仅作为高开销质量参考。

正式实验不属于本轮阻塞项，但接口必须避免后续重新改造请求轨迹和观测边界。

---

## 17. 预测—实际一致性验证

### 17.1 无热度推进一致性

测试步骤：

```text
保存环境
  -> 捕获ObservationSnapshot
  -> 预测候选
  -> 实际接纳候选但不推进热度
  -> 比较节点负载、offered rate和链路负载
```

期望：

```text
absolute_error < 1e-9
```

### 17.2 一时隙状态一致性

```text
保存环境
  -> 预测候选的一步后状态
  -> 实际接纳
  -> 推进一个时隙
  -> 比较热度、有效带宽、OSS和链路稳定性
```

每个字段必须明确预测时点。

### 17.3 周期方案漂移测试

```text
基于快照S0生成方案
  -> 环境推进若干时隙
  -> 基于St在线复验
  -> 验证旧规划指标不会直接用于接纳
```

### 17.4 节点可用性测试

- 规划时在线、在线接纳时离线：候选被淘汰；
- 接纳时在线、运行时离线：运行请求失败；
- 同一时隙多请求：可用性只转换一次；
- `remaining_time` 不出现在观测快照。

### 17.5 IFS解耦测试

- 相同画像、不同节点：IFS可不同；
- 相同节点、不同画像：IFS可不同；
- 仅改变负载：OSS变化，IFS不必变化；
- 仅改变画像质量表：IFS变化，OSS不变化。

---

## 18. 文件级改造清单

### 18.1 `models.py`

新增：

- `DNNProfile`；
- `InferenceRequest`；
- `ProfileNodeFidelity`；
- `ObservationSnapshot`；
- `CandidateStateSnapshot`；
- `NodeAvailabilityObservation`；
- `DeploymentPlan`必要公共字段；
- `RunningDNN`请求、画像、方案版本和失败信息。

兼容：

- `DNN`保留为物化适配结构；
- `o_reliability/a_reliability`标记deprecated。

### 18.2 `core/topology.py`

- 服务画像目录生成；
- 显式计算量和数据单位；
- 画像质量目录初始化入口；
- 拓扑版本；
- 不再把每个请求都生成成完全不同的DAG。

### 18.3 `core/environment.py`

- `capture_observation_snapshot()`；
- `aggregate_running_link_offered_rates()`；
- `candidate_link_offered_rates()`；
- `predict_candidate_state()`；
- 固定点带宽—时延预测；
- 可用性每时隙更新；
- 运行期节点下线处理；
- `commit_assignment()`版本校验；
- `update_link_loads()`复用offered-rate函数。

### 18.4 `proposed.py`

- `_evaluate_assignment()`只读取统一候选快照；
- `_constraint_violation()`不再重复预测；
- 候选缓存键包含快照和预测版本；
- 周期搜索入口只接受观测快照；
- 在线选解重新评价缓存assignment；
- 新代码只使用OSS/IFS语义名称。

### 18.5 `rtbl/scheduler.py`

- 不再通过请求调用推进可用性；
- 从当前时隙观测读取二值可用性；
- 保留原UCB和节点选择逻辑；
- 使用统一后验评价器报告结果；
- 记录观测配置版本。

### 18.6 `metrics.py`

新增：

- 模型预测非收敛计数；
- 预测—实际误差诊断；
- 节点不可用拒绝；
- 运行期节点下线失败；
- IFS画像缺失回退计数；
- 方案发布复验淘汰计数；
- 语义版本字段。

### 18.7 `experiment_runner.py`

- 传递全部模型版本；
- 分离拓扑、请求、可用性和算法随机数；
- 保存质量目录版本；
- 保存运行时长语义；
- 结果Schema拒绝不同版本混合。

### 18.8 建议新增目录

```text
core/snapshots.py
quality/registry.py
planning/repository.py
planning/periodic_planner.py
planning/online_dispatcher.py
```

---

## 19. 机器可读语义版本

所有新结果至少包含：

```text
objective_semantics_version
  = stability_fidelity_v3_profiled

candidate_prediction_version
  = unified_fixed_point_v1

observation_semantics_version
  = explicit_snapshot_v1

workload_semantics_version
  = poisson_profile_catalog_v1

planning_semantics_version
  = periodic_repository_v1

availability_model_version
  = slot_up_down_v1

runtime_progress_mode
  = fixed_service_time_v1

routing_mode
  = fixed_base_route_v1

duplex_mode
  = full_directional_capacity_v1
    | shared_capacity_v1

energy_model_version
  = deterministic_compute_transfer_return_v1

fidelity_model_version
  = profile_node_reference_agreement_v1
```

方案池本身也必须保存：

```text
candidate_prediction_version
fidelity_model_version
topology_version
source_snapshot_version
```

版本不匹配的方案不得加载。

---

## 20. 测试方案

### 20.1 单元测试

建议新增：

```text
tests/test_observation_snapshot.py
tests/test_candidate_state_snapshot.py
tests/test_offered_rate_consistency.py
tests/test_fixed_point_delay_prediction.py
tests/test_profile_node_fidelity.py
tests/test_node_availability_boundary.py
tests/test_plan_online_revalidation.py
tests/test_model_semantic_versions.py
```

### 20.2 回归测试

保留现有：

- 物理链路双向共享状态；
- 回传时延和能耗；
- 四目标支配；
- OSS几何聚合；
- IFS FLOPs权重；
- 候选预测不修改环境。

IFS预期值需要按新画像—节点矩阵更新。

### 20.3 小规模集成测试

固定种子：

```text
2 profiles
5 nodes
20 slots
Poisson lambda = 0.5
planning interval = 5 slots
population = 8
iterations = 3
```

验证：

- 周期规划完成；
- 方案发布；
- 在线重新评价；
- 多请求同槽接纳；
- 节点下线淘汰方案；
- 指标守恒；
- 运行结果可复现。

### 20.4 当前阶段不要求的测试

本方案不以以下结果作为完成条件：

- 20个以上随机种子正式对比；
- 统计显著性；
- 完整参数敏感性；
- 正式Pareto/HV图；
- 论文级吞吐曲线；
- 投稿表格。

这些属于代码语义冻结后的实验阶段。

---

## 21. 分阶段实施顺序

### 阶段 A：观测边界

1. 新增 `ObservationSnapshot`；
2. 环境提供只读捕获接口；
3. 隐藏未来请求和真实剩余在线时间；
4. 周期规划和在线调度入口改用快照。

验收：算法不再需要直接读取环境私有字段完成候选评价。

### 阶段 B：链路基础量统一

1. 增加offered-rate汇总；
2. 实际链路更新改用offered rate；
3. 候选预测复用相同汇总；
4. 增加full/shared双工配置；
5. 完成预测—实际负载一致性测试。

验收：不推进热度时预测和实际链路负载误差小于 `1e-9`。

### 阶段 C：统一候选快照

1. 新增 `CandidateStateSnapshot`；
2. 实现固定点预测；
3. `_evaluate_assignment()`只消费统一快照；
4. 约束违反度复用统一快照；
5. 缓存键加入快照和预测版本。

验收：delay、OSS、链路负载和约束来自同一预测带宽。

### 阶段 D：IFS画像化

1. 新增质量注册表；
2. 构造profile—node质量矩阵；
3. IFS查询包含画像和节点；
4. 取消未经校准的负载直接退化IFS；
5. 清理主结果字段命名。

验收：OSS与IFS驱动机制分离，IFS不再被代码注释解释为准确率。

### 阶段 E：可用性接入

1. 可用性每时隙更新一次；
2. 快照暴露当前可用性而非真实剩余时间；
3. 规划和在线复验增加硬约束；
4. 运行期节点下线产生独立失败；
5. 节点状态变化触发重规划。

验收：同槽多请求不会推进多次故障状态，离线节点不能被接纳。

### 阶段 F：周期方案池集成

1. 方案生成调用统一候选预测器；
2. 发布前重新评价；
3. 在线到达重新评价；
4. 方案版本和TTL生效；
5. 增加故障域多样性。

验收：旧规划指标不能绕过当前状态复验直接接纳。

### 阶段 G：兼容和清理

1. 更新RTBL可用性接口；
2. 更新指标Schema；
3. 写入机器可读模型版本；
4. 标记旧命名deprecated；
5. 更新README和当前系统模型文档。

验收：所有测试通过，不同语义版本结果无法混合加载。

---

## 22. 完成定义

本轮模型修复在满足以下条件后完成：

- [ ] 周期规划器和在线调度器只读取显式观测快照；
- [ ] 未来请求、真实剩余在线时间和未来故障不进入算法观测；
- [ ] 四目标评价只使用一个统一候选状态快照；
- [ ] 候选链路预测和实际链路更新共享offered-rate公式；
- [ ] 带宽—时延—负载预测有明确收敛和非收敛策略；
- [ ] delay、OSS和链路约束使用同一预测有效带宽；
- [ ] IFS按DNN画像—节点质量聚合；
- [ ] IFS不再默认随普通CPU负载直接指数下降；
- [ ] 节点可用性、未来生存概率和OSS语义分离；
- [ ] 可用性每时隙只更新一次；
- [ ] 运行期节点下线可以形成独立失败事件；
- [ ] 方案发布前和在线接纳前均重新评价；
- [ ] 固定路由、双工模式和服务时长语义机器可读；
- [ ] 候选缓存不会跨快照错误复用；
- [ ] 预测—实际一致性测试通过；
- [ ] 新结果使用新的语义版本；
- [ ] 旧兼容命名不会出现在新主结果字段中；
- [ ] 单元测试、集成测试和固定种子回归通过。

以下内容不属于本轮完成条件：

- 正式多种子实验；
- 显著性检验；
- 投稿图表；
- 完整消融和敏感性实验。

---

## 23. 修复后的统一模型语义

### 23.1 OSS

OSS 是节点当前在线条件下，候选接纳后统一预测状态中的运行稳定性效用。它由节点和物理链路质量分层聚合，不是严格端到端成功概率。

### 23.2 IFS

IFS 是特定 DNN 服务画像在所选节点和执行配置上保持参考输出质量的代理分数，按节点承载FLOPs加权聚合，不是具体数据集准确率。

### 23.3 可用性

节点当前可用性是部署硬约束。未来生存概率可作为周期方案风险约束。两者不与OSS或IFS混用。

### 23.4 delay

delay 是固定路由下，基于候选接纳后预测 offered rate、负载、热度和有效带宽，经有限次固定点迭代得到的端到端服务时长估计。

### 23.5 energy

energy 是由计算操作量和输入、中间、回传数据量得到的确定性一阶估计。后台规划器能耗不进入DNN部署能耗目标。

### 23.6 周期方案

周期方案池保存可复用的部署结构和规划参考指标。任何请求接纳前都必须基于当前观测重新计算候选状态并通过严格约束检查。

---

## 24. 最终实施原则

实施时必须遵守以下顺序：

```text
先修统一观测和候选预测
  -> 再修IFS和可用性语义
  -> 再接入周期方案池
  -> 最后冻结语义并开展正式实验
```

不得先实现方案缓存，再让缓存调用旧的分裂评价公式。否则周期规划只会把当前每请求评价误差转化为长期复用误差。
