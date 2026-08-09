# 周期性规划与 P0 模型闭环统一实施方案

## 1. 文档状态

- 状态：待实施
- 角色：当前唯一实施入口
- 适用阶段：模型与代码改造阶段
- 架构目标：泊松批量到达、异步周期规划、在线快速复验
- 模型目标：P0-01～P0-03 闭环
- 不纳入当前阻塞项：P0-04 正式论文实验不足

本文档收敛并取代以下三个并行实施入口：

- `p0_model_closure_code_modification_plan.md`；
- `poisson_arrival_periodic_planning_code_modification_plan.md`；
- `periodic_planning_model_consistency_remediation_plan.md`。

上述文档保留为设计来源和细节参考，不再分别决定实现顺序、字段名称、语义版本或完成条件。若其内容与本文档冲突，以本文档为准。

联合控制专项实现细节见：

- `../reference/adaptive_joint_planning_budget_control_implementation_plan.md`。

该专项文档用于细化规划周期、分画像候选池规模和在线复验时间预算的联合决策，但仍通过本文档的实施阶段和完成条件进入代码，不构成第二个独立实施入口。

---

## 2. 当前阶段的唯一目标

将当前“每个请求到达后执行一次完整进化搜索”的请求驱动系统，改造成双时间尺度系统：

```text
慢时间尺度：
当前可见状态快照
  -> 周期性 DAREED 搜索
  -> 每类 DNN 画像和接入组的 Pareto 部署结构池
  -> 规划完成时基于最新状态发布复验
  -> 原子发布

快时间尺度：
每时隙泊松批量请求
  -> 画像和接入组匹配
  -> 当前状态硬过滤
  -> Top-K 完整候选预测
  -> 偏好选择、轻量修复或快速回退
  -> 原子接纳或明确拒绝
```

同时完成三个 P0 模型问题：

1. P0-01：IFS、OSS、DS、ES 使用同一个候选接纳后预测状态；
2. P0-02：候选预测和实际环境更新共享同一套 offered-rate 与链路聚合公式；
3. P0-03：IFS 改为可解释的 DNN 画像—节点参考保真性，不再由普通 CPU 负载直接衰减。

本轮不要求完成：

- 正式多随机种子论文对比；
- 显著性检验和效应量；
- 完整消融和参数敏感性；
- 投稿级表格、绘图和结果归档。

这些工作在代码语义冻结后另立实验计划，不作为本文档从 `pending` 移出的条件。

---

## 3. 冻结的系统语义

### 3.1 时隙与事件顺序

系统由请求驱动改为时隙驱动。每个时隙固定执行：

```text
1. 应用本时隙节点和链路状态变化
2. 处理已经完成的运行请求
3. 轮询已完成的后台规划任务
4. 对新方案执行发布复验并原子发布
5. 判断是否启动新的规划任务
6. 暴露本时隙已经到达的请求批次
7. 按确定性策略逐个在线接纳
8. 每次成功接纳后立即提交资源并递增状态版本
9. 时隙末记录指标并推进一次时间
```

同一时隙无论到达多少请求，环境时间只推进一次。后到达处理顺序中的请求必须看到本时隙先前请求已经提交的资源状态。

### 3.2 请求语义

静态 DNN 结构与运行请求实例分离：

```python
@dataclass(frozen=True)
class DNNProfile:
    profile_id: str
    tasks: tuple
    links: tuple
    input_size_kbit: float
    output_size_kbit: float
    semantic_version: str


@dataclass(frozen=True)
class InferenceRequest:
    request_id: int
    profile_id: str
    arrival_slot: int
    initiate_node: int
    deadline_ms: float
    preference_operation: float
    preference_fidelity: float
    priority: int = 0
```

画像目录在仿真初始化时生成或加载；运行时只从画像目录采样请求类型，不为每个请求重新随机生成不可复用的 DAG。

### 3.3 到达语义

每时隙请求总数：

```text
N_t ~ Poisson(lambda_total)
```

每个请求再按配置概率选择 `profile_id`。若使用类型独立到达率，则利用泊松叠加：

```text
N_t,k ~ Poisson(lambda_k)
lambda_total = sum_k lambda_k
```

请求轨迹可以为公平实验预生成，但调度器和规划器只能读取当前及历史到达，不得读取未来真实请求。

### 3.4 运行时长语义

当前阶段保留：

```text
runtime_progress_mode = fixed_service_time_v1
```

资源实际占用时长、节点故障暴露窗口和任务完成时点必须使用同一个 `service_exposure_slots`。在该版本中，节点可用性预测窗口不得仅由预测 delay 隐式决定。

未来若实施剩余工作量模型，必须整体升级运行时长、可用性窗口和完成事件语义，不能只修改其中一项。

---

## 4. 信息边界与状态模型

### 4.1 环境真值

仿真环境可以持有：

- 完整预生成请求轨迹；
- 未来节点上下线抽样结果；
- 未来环境退化过程；
- 隐藏的真实故障时刻。

这些数据只用于推动仿真和事后计算，不得进入算法输入。

### 4.2 算法观测

规划器和在线调度器统一读取不可变 `ObservationSnapshot`：

```python
@dataclass(frozen=True)
class ObservationSnapshot:
    snapshot_version: int
    environment_state_version: int
    topology_version: int
    slot: int
    purpose: str
    node_online: tuple[bool, ...]
    node_available_cpu: tuple[float, ...]
    node_loads: tuple[float, ...]
    node_heats: tuple[float, ...]
    node_availability_history: tuple
    directional_offered_rates: tuple[float, ...]
    directional_link_loads: tuple[float, ...]
    physical_link_heats: tuple[float, ...]
    effective_bandwidths: tuple[float, ...]
```

`purpose` 取值：

```text
planning
publication_revalidation
online_admission
```

规划器允许额外读取：

- DNN 画像目录；
- 配置或历史估计的到达率；
- 上一版本方案池，用于热启动。

不得读取：

- 未来真实请求；
- 真实剩余在线时间；
- 未来故障事件；
- 未来负载和热度；
- 仿真器已经抽样但尚未发生的状态变化。

### 4.3 状态版本

以下事件必须递增 `environment_state_version`：

- 请求接纳并提交资源；
- 请求完成并释放资源；
- 节点上线或下线；
- 拓扑变化；
- 影响候选评价的链路状态变化。

因此，同一时隙内多个请求不会错误复用接纳前的候选评价缓存。

---

## 5. 统一候选预测内核

### 5.1 唯一入口

规划搜索、发布复验、在线复验和基线最终评价统一调用：

```python
predict_candidate_state(
    profile,
    request_context,
    assignment,
    observation_snapshot,
    semantic_config,
) -> CandidateStateSnapshot
```

该函数必须：

- 无副作用；
- 不推进环境；
- 不注册运行任务；
- 不消耗到达或故障随机数；
- 不修改路由缓存；
- 返回四目标、原始指标、约束和诊断所需的全部状态。

### 5.2 候选状态

```python
@dataclass(frozen=True)
class CandidateStateSnapshot:
    observation_snapshot_version: int
    environment_state_version: int
    assignment: tuple[int, ...]
    predicted_node_cpu: tuple[float, ...]
    directional_offered_rates: tuple[float, ...]
    directional_link_loads: tuple[float, ...]
    physical_link_loads: tuple[float, ...]
    predicted_link_heats: tuple[float, ...]
    predicted_effective_bandwidths: tuple[float, ...]
    estimated_delay_ms: float
    total_energy: float
    node_availability_score: float
    link_stability_score: float
    operational_stability: float
    inference_fidelity: float
    delay_satisfaction: float
    energy_satisfaction: float
    availability_horizon_slots: int
    availability_prediction_confidence: float
    fixed_point_converged: bool
    fixed_point_iterations: int
    constraint_violations: tuple[str, ...]
```

四目标统一为：

```text
F(x) = [IFS(x), OSS(x), DS(x), ES(x)]
```

所有目标和约束只从该快照读取，不允许独立重算形成第二套状态。

### 5.3 统一 offered-rate

方向链路的基础动态量统一为：

```text
offered_rate_kbps = transferred_data_kbit / active_duration_s
```

运行任务和候选任务调用同一数据量汇总与路由展开函数。物理链路默认采用：

```text
duplex_mode = full_duplex_shared_heat_v1
physical_load = max(forward_direction_load, reverse_direction_load)
```

若未来改为半双工求和，必须升级语义版本。

### 5.4 带宽—负载—时延固定点

统一流程：

```text
当前有效带宽估计初始 delay
  -> 根据 delay 计算候选 offered rate
  -> 汇总运行任务与候选任务 offered rate
  -> 计算方向和物理链路负载
  -> 预测热度和有效带宽
  -> 重算 delay
  -> 阻尼迭代至收敛或达到上限
```

默认配置：

```text
candidate_prediction_version = unified_fixed_point_v1
max_iterations = 5
relative_tolerance = 1e-4
fallback = max_delay_seen
```

不收敛时返回保守时延并记录诊断，不能静默采用乐观值。

---

## 6. OSS、节点可用性与 IFS

### 6.1 当前可用性硬约束

候选使用的任一节点当前离线时，候选直接不可行：

```text
node_online[n] == True, for all n in used_nodes(x)
```

当前可用性不能用较低 OSS 替代。

### 6.2 未来持续可用性

对当前在线节点，使用当前及历史可见数据估计：

```text
p_up,n(t, H) = P(node n remains online over H slots | visible history at t)
```

默认实现：

```text
availability_estimator_version = empirical_window_beta_v1
```

要求：

- 仅使用 `t` 及以前的历史；
- 使用 Beta 平滑避免短历史产生 0 或 1；
- 新节点同时输出较低预测置信度；
- `H = service_exposure_slots`；
- `always_on` 节点必须显式配置，不能靠节点编号隐式区分。

节点聚合：

```text
A_N(x) = exp(sum_n omega_n * log(p_up,n(t, H)))
```

其中 `omega_n` 为该节点承载 FLOPs 的归一化权重。

### 6.3 链路稳定性与 OSS

链路域：

```text
S_L(x) = exp(sum_l nu_l * log(s_l(t+1 | x)))
```

复合运行稳定性：

```text
OSS(x) = exp(w_N * log(A_N(x)) + w_L * log(S_L(x)))
```

若无跨节点通信，则 `OSS(x) = A_N(x)`。

必须分别输出 `A_N`、`S_L` 和 `OSS`。OSS 是归一化复合效用，不声明为严格端到端成功概率。

可设置独立可用性风险门槛：

```text
A_N(x) >= min_node_availability_score
```

链路稳定性不能掩盖过低的节点可用性。

### 6.4 IFS

默认使用 DNN 画像—节点质量注册表：

```text
base_inference_fidelity[profile_id, node_id]
```

聚合：

```text
IFS(x) = exp(sum_n omega_n * log(base_inference_fidelity[profile, n]))
```

默认模式：

```text
ifs_mode = profiled_static_v1
```

普通 CPU 负载和热度不再直接降低 IFS。IFS 表示参考输出保持能力，不称为具体数据集准确率或单次推理成功概率。

---

## 7. 周期方案池

### 7.1 缓存边界

方案池主要缓存部署结构，而不是可跨时隙复用的最终性能结论：

```python
@dataclass(frozen=True)
class DeploymentPlan:
    plan_id: str
    profile_id: str
    origin_group: int
    assignment: tuple[int, ...]
    required_cpu: tuple
    used_physical_link_ids: tuple[int, ...]
    planning_reference_delay_ms: float
    planning_reference_oss: float
    planning_reference_availability: float
    planning_reference_ifs: float
    planning_reference_energy: float
    planning_snapshot_version: int
    topology_version: int
    created_slot: int
    expires_slot: int
    semantic_versions: tuple[tuple[str, str], ...]
```

`planning_reference_*` 只用于诊断、粗排序和漂移分析，不得直接作为在线接纳结论。

### 7.2 索引

```text
(profile_id, origin_group)
```

第一阶段 `origin_group` 使用终端接入边缘节点，不为每个终端单独执行完整周期规划。

### 7.3 候选池多样性

单请求 Pareto 最优不等于请求流吞吐最优。发布候选池时必须避免所有方案集中在相同节点：

- 保留不同节点集合的候选；
- 保留不同资源占用梯度的候选；
- 保留低时延、高可用、低能耗和低资源方案；
- 在已知故障域时优先保留故障域分散方案；
- 对同构 assignment 去重。

第一阶段不引入多请求联合染色体，使用候选池多样性和在线累计复验控制热点。

### 7.4 版本与失效

方案在以下情况下不能直接进入在线完整评价：

- `topology_version` 不一致；
- 语义版本不一致；
- 超过 TTL；
- assignment 引用不存在的节点或链路。

节点当前离线或资源不足不必删除历史方案对象，但在线硬过滤必须将其淘汰。

---

## 8. 异步周期规划

规划任务读取启动时的 `purpose=planning` 快照，并满足：

- 不冻结仿真时间；
- 不修改在线环境；
- 规划期间继续使用旧方案池；
- 上一版本候选只作为热启动；
- 规划耗时映射到仿真时隙。

```text
ready_slot = start_slot + ceil(planning_runtime_ms / slot_length_ms)
```

规划完成时：

1. 捕获 `publication_revalidation` 快照；
2. 快速检查拓扑、在线状态、资源和语义版本；
3. 删除明显失效候选；
4. 有效候选比例不足时保留旧方案池；
5. 满足发布条件时原子替换仓库版本。

触发条件：

- 固定规划周期到达；
- 节点或拓扑变化；
- 节点可用性预测显著下降；
- 节点或链路负载漂移超过阈值；
- 方案直接命中率过低；
- 在线回退率过高；
- 到达画像分布显著变化；
- 方案池超过 TTL。

必须设置最小重规划间隔，避免抖动和规划风暴。

---

## 9. 在线调度

### 9.1 两级复验

为控制在线开销，使用：

```text
方案池检索
  -> topology、TTL、语义版本过滤
  -> 当前在线状态和 CPU 快速过滤
  -> 使用规划参考值粗排序
  -> 对 Top-K 调用完整 predict_candidate_state()
  -> 偏好选择
  -> 原子检查并提交
```

建议初始配置：

```text
pareto_pool_size = 20
full_revalidation_top_k = 5
online_budget_ms = 50
```

快速过滤不能代替最终完整复验。

### 9.2 同槽累计接纳

每个请求成功接纳后立即：

- 注册运行请求；
- 提交节点 CPU；
- 更新链路 offered rate；
- 递增 `environment_state_version`；
- 使旧在线候选评价缓存失效。

下一请求重新捕获或增量构造 `online_admission` 快照。

### 9.3 在线失败路径

```text
plan_direct
plan_repaired
fast_fallback
rejected
```

拒绝原因至少区分：

- `no_profile_plan`；
- `plan_expired`；
- `node_offline`；
- `insufficient_cpu`；
- `link_overload`；
- `deadline_violation`；
- `availability_below_threshold`；
- `online_budget_exceeded`；
- `fallback_failed`。

周期规划时间不计入单请求在线决策时延，但必须单独报告总规划开销和摊销规划开销。

---

## 10. 缓存和并发控制

候选预测缓存键至少包含：

```text
profile_id
request_context_hash
assignment
observation_snapshot_version
environment_state_version
topology_version
all_semantic_versions
```

方案池可以跨状态版本保存部署结构，候选评价缓存不能跨状态版本复用。

在线最终提交采用乐观并发检查：

```text
评价开始的 environment_state_version
    ==
提交前的 environment_state_version
```

不一致时必须重新评价或拒绝，不能使用旧快照提交资源。

---

## 11. 语义版本

新结果必须至少记录：

```text
objective_semantics_version = stability_fidelity_v3_profiled
candidate_prediction_version = unified_fixed_point_v1
observation_semantics_version = explicit_snapshot_v1
workload_semantics_version = poisson_profile_catalog_v1
planning_semantics_version = periodic_repository_v1
joint_budget_control_version = discrete_mpc_v1
availability_estimator_version = empirical_window_beta_v1
availability_model_version = slot_up_down_v1
ifs_mode = profiled_static_v1
runtime_progress_mode = fixed_service_time_v1
routing_mode = fixed_path_v1
duplex_mode = full_duplex_shared_heat_v1
energy_model_version = deployment_return_energy_v1
result_schema_version = periodic_model_closure_v1
```

任一影响指标含义的版本不一致时，结果不能混合统计，方案评价缓存不能复用。

---

## 12. 文件级改造清单

### 12.1 `models.py`

- 新增 `DNNProfile`、`InferenceRequest`；
- 新增 `ObservationSnapshot`、`CandidateStateSnapshot`；
- 新增 `DeploymentPlan`、`PlanRepository` 或将后两者放入 `planning/repository.py`；
- 拆分规划参考指标与在线最终指标；
- 增加机器可读语义版本。

### 12.2 `core/arrival.py`

- 新增画像目录采样；
- 新增泊松批次请求生成；
- 隔离 arrival RNG；
- 支持预生成公平请求轨迹；
- 禁止算法读取未来请求。

### 12.3 `core/topology.py`

- 固定方向边与物理链路映射；
- 统一全双工共享热度聚合；
- 明确 kbit、kbps、秒、毫秒的转换边界。

### 12.4 `core/environment.py`

- 实现 `capture_observation_snapshot()`；
- 实现 `environment_state_version`；
- 抽取共享 offered-rate 内核；
- 实现无副作用 `predict_candidate_state()`；
- 实际接纳更新调用同一 offered-rate 内核；
- 加入节点当前在线硬约束；
- 实现可见历史可用性估计；
- 每时隙只更新一次节点上下线状态。

### 12.5 `proposed.py`

- `_evaluate_assignment()` 只消费一个候选快照；
- 新增纯函数式 `search_plan_pool()`；
- 搜索不注册任务、不推进时隙；
- 缓存键加入状态和语义版本；
- 旧 OSS/IFS 路径移入显式 legacy 模式。

### 12.6 `planning/repository.py`

- 方案索引、去重、TTL、版本校验；
- 原子发布不可变仓库；
- 候选池多样性筛选。

### 12.7 `planning/periodic_planner.py`

- 固定周期和状态触发；
- 异步规划时序；
- 热启动；
- 发布前复验；
- 最小重规划间隔。

### 12.8 `planning/online_dispatcher.py`

- 快速过滤；
- Top-K 完整复验；
- 偏好选择；
- 轻量修复与快速回退；
- 原子接纳和明确拒绝原因。

### 12.9 `planning/joint_budget_controller.py`

- 固定参数控制器；
- 规则式联合控制器；
- 离散 MPC 联合控制器；
- 联合选择规划周期、分画像候选池配额和在线复验时间预算；
- 安全约束、迟滞、防抖和保守回退；
- 具体接口和数据结构按专项实现文档执行。

### 12.10 `experiment_runner.py`

- 改为时隙事件循环；
- 支持同槽多请求；
- 时隙末统一推进环境；
- 规划计算期间环境继续运行；
- 分离预热和冷启动。
- 支持固定、规则式和离散 MPC 控制模式；
- 隔离控制器训练场景和测试场景。

### 12.11 `metrics.py`

- 分离到达、接纳、完成、拒绝、超期和运行失败；
- 记录方案命中、修复、回退和发布失败；
- 记录规划开销、在线开销和摊销规划开销；
- 记录预测—实际误差与固定点收敛诊断；
- 分别记录节点可用性、链路稳定性和 OSS。
- 记录控制动作、goodput utility、规划器/调度器利用率及预测—实际控制结果。

---

## 13. 实施阶段

### 阶段 0：版本与兼容边界

- 标记当前语义为 legacy；
- 新增统一版本常量；
- 旧结果与新结果隔离；
- 固定最小回归场景。

验收：旧测试可复现，新旧输出不会混合。

### 阶段 1：信息边界与画像请求

- 实现 `ObservationSnapshot`；
- 拆分画像和请求；
- 实现泊松请求轨迹和 RNG 隔离；
- 改造为时隙事件循环。

验收：无法从算法接口读取未来请求或未来故障；同槽多请求只推进一次时间。

### 阶段 2：P0 链路闭环

- 抽取统一 offered-rate；
- 统一方向和物理链路负载；
- 迁移实际链路更新。

验收：相同快照下，预测新增 offered rate 与接纳后实际新增值误差小于 `1e-9`。

### 阶段 3：统一候选快照

- 实现固定点预测；
- 实现 `CandidateStateSnapshot`；
- 四目标和约束统一从快照读取；
- 实现状态版本缓存边界。

验收：预测无副作用，四目标同状态，固定点失败采用保守回退。

### 阶段 4：OSS、可用性和 IFS

- 当前离线节点硬过滤；
- 实现窗口可用性估计；
- 可用性窗口与服务占用时长对齐；
- OSS 分层聚合并保留分量；
- IFS 切换为画像—节点 profiling。

验收：无未来信息泄漏；IFS 不随普通负载机械变化；可用性下降时 OSS 不上升。

实施状态（2026-08-02）：已完成核心模型闭环。

- `Node` 和 `ObservationSnapshot` 已显式携带 `stochastic` / `always_on` 模式；
- 节点状态只由 `advance_time_slot()` 在时隙边界推进一次，RTBL 的兼容接口只读当前状态；
- 连续在线概率只读取观测快照中的当前状态和历史窗口，使用 Beta 平滑并输出置信度；
- `service_exposure_slots()` 统一候选可用性窗口、资源占用时长和完成事件时域；
- `CandidateStateSnapshot` 分别输出节点可用性、链路稳定性、聚合 OSS 和预测置信度；
- `stability_fidelity_v3_profiled` 使用画像—节点 IFS 注册表，普通负载和热度不直接衰减 IFS；
- `stability_fidelity_v2_return_energy` 保留旧动态 OSS/IFS 路径，避免历史实验语义静默改变；
- 对应自动化回归位于 `tests/test_profiled_availability_model.py`。

### 阶段 5：方案池与在线调度

- 实现仓库、TTL 和版本；
- 实现候选多样性；
- 实现两级在线复验；
- 实现同槽累计接纳和原子提交；
- 实现修复、回退和拒绝路径。

验收：方案只缓存结构；在线不直接使用规划指标；多个请求不会超额承诺资源。

实施状态（2026-08-02）：已完成核心同步调度闭环。

- 新增不可变 `DeploymentPlan`、`PlanRepository` 和带版本检查的原子仓库引用；
- 仓库按 `(profile_id, origin_group)` 索引，并校验 TTL、拓扑版本、语义版本和节点引用；
- 规划候选按 assignment 去重，并以节点集合 Jaccard 重叠度执行多样性筛选；
- `search_plan_pool()` 仅使用显式规划快照和局部 RNG，不注册任务、不推进时隙；
- 在线路径执行快速在线/CPU 过滤、规划参考值粗排序、固定 Top-K 完整复验和偏好选择；
- 规划参考指标不作为接纳结论，提交前再次调用统一候选预测器；
- 提供 `plan_direct`、`plan_repaired`、`fast_fallback` 和 `rejected` 四条明确路径；
- `commit_assignment()` 使用环境版本检查、锁内最终复验和原子资源提交；
- 陈旧提交在预算内重新捕获当前快照并复验一次；
- 同槽请求逐个提交并递增环境版本，后续请求不能复用提交前的 CPU 状态；
- 对应自动化回归位于 `tests/test_plan_repository.py` 和 `tests/test_online_dispatcher.py`。

### 阶段 6：异步周期规划

- 实现规划作业时序；
- 旧方案池持续服务；
- 实现发布复验和原子替换；
- 实现固定周期、漂移触发和防抖。

验收：规划不冻结环境，规划耗时正确映射为仿真时隙。

实施状态（2026-08-02）：已完成仿真时隙级异步规划闭环。

- 新增 `PlanningJob`，记录启动槽、就绪槽、源快照/环境/拓扑版本、规划耗时和候选结构；
- 使用 `ready_slot = start_slot + ceil(planning_runtime_ms / slot_length_ms)` 映射规划开销；
- 作业未就绪时仓库引用保持不变，在线调度可持续使用旧版本；
- 上一仓库 assignment 仅作为热启动结构，并在启动快照上重新评价；
- 作业完成时捕获 `publication_revalidation` 快照并对每个候选完整复验；
- 发布候选剔除离线、资源/链路/截止期不可行及固定点不收敛方案；
- 有效比例不足时保留旧仓库，满足门槛后才执行版本化原子替换；
- 支持 `cold_start`、显式 `prewarm()`、固定周期、TTL、拓扑、可用性、负载和画像质量变化触发；
- 支持在线直接命中率、回退率和画像分布漂移反馈触发，并受最小重规划间隔约束；
- 画像—节点质量表内容和版本均进入显式快照，后台作业不读取启动后的质量更新；
- 分离记录规划作业数、发布/保留次数、候选淘汰数和规划总耗时；
- 对应自动化回归位于 `tests/test_periodic_planner.py`。

### 阶段 7：规划周期、候选池与在线预算联合控制

- 先通过统一接口复现固定参数策略；
- 实现分画像/接入组候选池配额；
- 将固定 Top-K 主语义改为请求级时间预算下的 anytime 复验；
- 实现规则式联合控制和防抖；
- 实现有限动作空间、代理预测、安全过滤和离散 MPC；
- 所有动作不可行时回退到保守固定策略。

验收：`T`、`K` 和 `B` 由一个原子决策联合更新；控制器不读取未来信息；固定、规则和 MPC 三种模式通过小规模集成。具体完成项按专项实现文档执行。

实施状态（2026-08-02）：已完成联合控制核心闭环。

- 新增不可变 `PlanningControlSnapshot`、`JointBudgetAction`、预测、决策和 outcome 数据结构；
- 有限动作空间稳定枚举并携带机器可读版本；
- 分画像/接入组配额采用最小配额、最大余数、确定性取整和跨周期平滑；
- 新增固定、规则式和离散 MPC 三种控制器；
- 规则控制具有迟滞、最小控制间隔和单级动作变化；
- MPC 使用解析/历史混合代理、置信度、不确定性惩罚、确定性 tie-break 和安全过滤；
- 所有动作不可行时回退最近安全动作或保守动作，不阻断在线路径；
- `JointBudgetRuntimeCoordinator` 在一个临界区内同时应用规划周期 `T`、分键池配额 `K` 和在线预算 `B`；
- 周期规划器按当前决策使用动态周期和分键配额，并把 `decision_id` 写入规划作业；
- 在线调度以请求级时间预算为主语义，保留固定 Top-K 消融模式，并记录预算、评价数、可行数、耗尽状态、早停原因和效用轨迹；
- 请求预算受截止期余量约束，提交仍保留环境版本最终复验；
- 控制状态聚合和代理在线更新只读取 `available_after_slot <= current_slot` 的历史记录；
- 离线代理拟合拒绝非训练角色数据；预测与实际结果通过 `decision_id` 对齐；
- 控制周期编排器在窗口结束后生成 goodput-cost outcome、因果更新控制器并应用下一动作；
- 新增联合动作空间、配额、规则控制、MPC、安全回退、anytime、信息边界和小规模集成回归。

### 阶段 8：代码级验收与语义冻结

- 单元测试；
- 确定性一致性测试；
- 固定种子回归；
- 小规模泊松集成测试；
- 在线性能烟雾测试；
- 清理默认路径中的旧语义。

验收：本文档第 15 节完成定义和联合控制专项文档的代码完成定义全部满足。

实施状态（2026-08-02）：已完成代码级验收与语义冻结。

- 新增独立周期实验事件循环，不改变旧论文基线 runner 和 legacy 结果口径；
- 每槽按“规划发布/控制 → 当前泊松批次顺序接纳 → 槽末统一推进”执行；
- 当前槽请求逐个原子提交，系统状态只在槽末推进一次；
- 请求到达时由画像物化轻量实例，规划器只使用预先登记的画像代表；
- 执行期节点下线会终止运行请求并释放资源，完成、运行失败、超期和未完成状态分别统计；
- 调度决策和运行结果使用不同的因果可见时点，未完成请求不会提前贡献 goodput；
- 新增周期实验结果 Schema，记录请求守恒、goodput、四类指标、调度路径、规划/在线开销、控制统计和全部语义版本；
- 支持 measured 与 deterministic 在线计时模式，后者用于固定种子完整结果复现；
- 小规模集成显式覆盖 0、1、同槽多个请求、cold-start、prewarm，以及固定/规则/MPC 同轨迹运行；
- 全量测试、差异格式检查和性能烟雾测试通过。

---

## 14. 测试矩阵

### 14.1 信息边界

- 快照中不存在未来请求和未来故障；
- 规划器不能访问完整到达轨迹；
- 候选预测不改变环境或随机数状态。

### 14.2 到达与事件循环

- 泊松样本均值和方差在容差内；
- 同一 seed 的所有算法获得相同轨迹；
- 支持 0、1、多个同槽请求；
- 接纳、拒绝和到达数量守恒。

### 14.3 候选闭环

- 四目标来自同一快照；
- 零通信候选链路增量为零；
- 增大通信量时 offered rate 和预测负载不下降；
- 预测与实际 offered rate 一致；
- 不收敛路径返回保守值。

### 14.4 可用性和质量

- 当前离线节点候选不可行；
- 相同历史下预测窗口增大时持续可用概率不应上升；
- 真实未来下线时间不进入预测；
- `profiled_static_v1` 下只改变普通负载不会改变 IFS；
- `A_N`、`S_L` 和 OSS 均在 `[0, 1]`。

### 14.5 方案池与在线接纳

- TTL、拓扑和语义版本校验生效；
- 发布前复验不会修改在线环境；
- 在线阶段重新计算最终指标；
- 同槽先接纳请求会使后续缓存失效；
- 原子提交前状态变化会触发重新评价；
- 方案池包含不同节点集合或资源梯度。

### 14.6 异步规划

- 规划期间旧仓库保持可读；
- 规划完成时不会覆盖正在读取的对象；
- 发布有效率不足时保留旧仓库；
- 最小间隔能够阻止重规划风暴。

### 14.7 小规模集成

建议固定：

```text
profiles = 2
nodes = 5
slots = 20
poisson_lambda = 0.5
planning_interval_slots = 5
population_size = 8
iteration_limit = 3
```

验证规划、发布、多请求接纳、节点下线过滤、回退、指标守恒和结果复现。

---

## 15. 完成定义

只有以下条件全部满足，本文档才可以从 `pending` 迁出：

- [x] 系统按时隙而不是按请求推进；
- [x] 每时隙可以生成并正确处理多个泊松请求；
- [x] DNN 画像与请求实例已经分离；
- [x] 规划器和在线调度器只能读取显式观测快照；
- [x] 未来请求、未来故障和真实剩余在线时间不进入算法输入；
- [x] 四目标和约束只读取统一候选快照；
- [x] 候选和实际链路更新共享 offered-rate 内核；
- [x] 带宽—负载—时延固定点具有明确收敛与回退策略；
- [x] 当前离线节点作为硬约束；
- [x] 可用性预测只使用当前及历史信息；
- [x] 可用性窗口与实际服务占用时长一致；
- [x] OSS 分别输出节点可用性和链路稳定性分量；
- [x] 默认 IFS 使用画像—节点 profiling；
- [x] 方案池只把指标作为规划参考值；
- [x] 方案发布和在线接纳均重新验证；
- [x] 同槽多个请求不会复用过期候选状态或超额承诺资源；
- [x] 在线阶段不运行完整进化搜索；
- [x] 周期规划不阻塞或冻结在线环境；
- [x] 规划、发布、在线选择和实际执行指标可以区分；
- [x] 规划周期、候选池规模和在线复验时间预算可以联合决策；
- [x] 联合控制具有安全约束、防抖、信息边界和保守回退；
- [x] 固定、规则式和离散 MPC 控制模式可以在相同请求轨迹上运行；
- [x] 所有语义版本机器可读；
- [x] 单元测试、一致性测试、小规模集成和固定种子回归通过。

以下内容明确不属于当前完成条件：

- 正式多随机种子论文实验；
- 显著性检验和效应量；
- 完整消融和敏感性实验；
- 投稿图表和论文级结果归档。

---

## 16. 后续阶段

代码语义冻结后，另行建立正式实验计划，分析：

- 到达率—完成吞吐量和失败率拐点；
- 固定到达与泊松突发性的差异；
- 规划周期、候选池大小和在线 Top-K 的敏感性；
- 周期规划与逐请求完整 DAREED 的质量—开销差异；
- 可用性预测、OSS 分量和 IFS 的区分效度；
- 预测—实际误差与固定点收敛情况。

该阶段用于形成论文证据，不反向改变已经冻结的模型语义；若必须改变语义，应升级版本并重新执行全部结果。
