# 泊松请求到达与周期性部署规划代码改造方案（设计来源，已被统一方案取代）

> 状态：不再作为独立实施入口。周期架构、状态边界和实施顺序已经收敛至 `../pending/unified_periodic_planning_model_closure_implementation_plan.md`。若模型语义、方案缓存、在线复验或完成条件与统一方案冲突，以统一方案为准。本文件仅用于追溯原始周期规划设计。

## 1. 文档状态

- 状态：待实施
- 适用版本：`objective_semantics_version = stability_fidelity_v2_return_energy`
- 核心目标：将当前“每个 DNN 请求到达后执行完整进化搜索”改为“慢时间尺度周期规划 + 快时间尺度在线匹配”
- 请求模型：有限 DNN 服务画像目录，运行时按泊松过程生成请求实例
- 主要涉及文件：`models.py`、`core/topology.py`、`core/environment.py`、`proposed.py`、`experiment_runner.py`、`metrics.py`
- 建议新增文件：`core/arrival.py`、`planning/repository.py`、`planning/periodic_planner.py`、`planning/online_dispatcher.py`

本方案建立在以下现状判断上：

1. 当前 DAREED/Customized 算法对每个请求重新执行完整种群搜索，单请求决策时间约为数秒，明显高于约 500--800 ms 的平均预测推理时延。
2. 当前系统每处理一个请求就推进一个时隙，等价于每时隙确定到达一个请求，无法表达空闲、突发和同一时隙多请求到达。
3. 当前每个 DNN 请求均随机生成独立 DAG，导致不同请求之间几乎不能复用部署方案。
4. 当前 `failure_count` 混合表示拒绝和失败，无法分析高负载下请求失败的具体原因。

本方案不改变当前四目标定义、节点/链路动态退化模型和基本部署约束，而是重构请求生成、规划时序、在线决策和实验统计边界。

---

## 2. 改造目标与非目标

### 2.1 改造目标

1. 引入有限、可重复调用的 DNN 服务画像目录。
2. 每个时隙按泊松分布生成 0 个、1 个或多个请求实例。
3. 使用独立请求轨迹保证所有算法面对完全相同的到达过程。
4. 让 DAREED 在后台周期性生成每类 DNN 的 Pareto 部署候选池。
5. 请求到达时只执行候选匹配、当前状态复验、偏好选解和轻量回退。
6. 规划计算期间环境继续推进，旧方案继续服务，不冻结仿真时间。
7. 支持同一时隙多个请求的累计资源接纳与明确排序策略。
8. 分离到达、接纳、完成、拒绝、超期和执行失败指标。
9. 通过负载扫描确定系统吞吐平台和请求失败率拐点。

### 2.2 第一阶段非目标

第一阶段暂不实现：

- 使用未来真实请求进行滚动时域联合优化；
- 多请求联合染色体和完整批量组合优化；
- 精细排队网络、逐包传输和抢占式执行；
- 实际节点随机故障与任务重试；
- 基于强化学习的在线策略；
- 动态拓扑重构；
- 为每个终端节点单独运行一套完整周期规划。

上述能力可在周期方案池和泊松到达模型稳定后分阶段加入。

---

## 3. 总体系统架构

系统采用双时间尺度控制：

```text
慢时间尺度：周期规划
系统状态快照
  -> DAREED 进化搜索
  -> 每类 DNN 的 Pareto 候选部署池
  -> 完成时基于最新状态复验
  -> 原子发布新版本方案池

快时间尺度：在线请求处理
泊松请求到达
  -> 服务画像和接入位置匹配
  -> 当前资源状态复验
  -> 请求偏好选解
  -> 轻量修复 / 快速基线回退
  -> 接纳或拒绝
  -> 时隙末统一推进环境
```

核心数据流：

```text
DNNProfileCatalog
        |
ArrivalTrace -----> InferenceRequest
        |                   |
        |                   v
SystemSnapshot ---> PlanRepository ---> OnlineDispatcher
        ^                   |                   |
        |                   |                   v
        +---------- Environment <------ RunningDNN
```

### 3.1 信息边界

环境持有：

- 完整、预生成的请求轨迹；
- 当前和未来请求实例；
- 系统内部动态状态；
- 规划任务的开始和完成事件。

在线调度器只能获得：

- 当前时隙已经到达的请求；
- 当前系统快照；
- 已发布的方案池；
- 历史到达统计和历史执行反馈。

周期规划器只能使用：

- 规划开始时的系统快照；
- DNN 服务画像目录；
- 历史估计到达率或配置的先验到达率；
- 当前已发布方案作为热启动种群。

规划器不得读取未来真实请求轨迹。预生成轨迹仅用于跨算法公平复现实验。

---

## 4. 数据模型重构

### 4.1 拆分服务画像与请求实例

当前 `models.DNN` 同时包含 DAG、请求发起节点、截止期和目标偏好，不适合部署方案复用。

建议在 `models.py` 新增：

```python
@dataclass(frozen=True)
class DNNProfile:
    profile_id: str
    tasks: tuple[Task, ...]
    links: tuple[LinkDNN, ...]
    input_size_kbit: float
    output_size_kbit: float
    semantic_version: str = "dnn_profile_v1"


@dataclass(frozen=True)
class InferenceRequest:
    request_id: int
    profile_id: str
    arrival_slot: int
    initiate_node: int
    deadline_ms: float
    preference_operation: float
    preference_fidelity: float
    preference_mode: str = "adaptive"
    priority: int = 0
```

兼容过渡期可保留 `DNN`，并增加适配函数：

```python
def materialize_dnn(profile: DNNProfile, request: InferenceRequest) -> DNN:
    ...
```

第一阶段优先减少对 `Environment.ds` 的全量重写；环境可在请求暴露给调度器时，将画像与请求实例临时物化为当前 `DNN`。

### 4.2 服务画像目录

建议新增：

```python
@dataclass(frozen=True)
class DNNProfileCatalog:
    profiles: dict[str, DNNProfile]
    sampling_probabilities: dict[str, float]
```

约束：

- 概率之和必须为 1；
- `profile_id` 全局唯一；
- 所有任务和链路单位必须显式记录；
- 同一画像在不同请求间保持 DAG 和计算量不变；
- 截止期、偏好和发起位置属于请求实例，不属于服务画像。

### 4.3 方案及方案池

建议在 `planning/repository.py` 新增：

```python
@dataclass(frozen=True)
class DeploymentPlan:
    plan_id: str
    profile_id: str
    origin_group: int
    assignment: tuple[int, ...]
    predicted_delay_ms: float
    operational_stability: float
    inference_fidelity: float
    total_energy: float
    delay_satisfaction: float
    energy_satisfaction: float
    required_cpu: tuple[tuple[int, float], ...]
    used_physical_link_ids: tuple[int, ...]
    snapshot_version: int
    topology_version: int
    created_slot: int
    expires_slot: int
    objective_semantics_version: str


@dataclass(frozen=True)
class PlanRepository:
    repository_version: int
    published_slot: int
    plans_by_key: dict[tuple[str, int], tuple[DeploymentPlan, ...]]
```

索引键：

```text
(profile_id, origin_group)
```

`origin_group` 默认使用请求终端接入的边缘节点 ID，而不是每个终端节点 ID。这样既保留上传/回传路径差异，又控制周期规划数量。

### 4.4 系统快照

建议新增只读结构：

```python
@dataclass(frozen=True)
class SystemSnapshot:
    snapshot_version: int
    slot: int
    topology_version: int
    node_available_cpu: tuple[float, ...]
    node_load_ratio: tuple[float, ...]
    node_heat: tuple[float, ...]
    node_operational_stability: tuple[float, ...]
    node_inference_fidelity: tuple[float, ...]
    link_effective_bandwidth: tuple[float, ...]
    link_load_ratio: tuple[float, ...]
    link_heat: tuple[float, ...]
    link_transmission_stability: tuple[float, ...]
```

`Environment.capture_snapshot()` 必须返回不可变对象，周期规划不得直接持有或修改在线环境。

---

## 5. DNN 服务画像生成

### 5.1 修改位置

文件：`core/topology.py`

将当前：

```python
create_dnns(num, nodes)
```

逐请求随机生成完全不同 DAG 的逻辑，拆分为：

```python
create_dnn_profile_catalog(profile_count, nodes, rng)
generate_request_from_profile(profile, request_parameters)
```

### 5.2 默认画像设置

第一阶段建议设置 5--10 个画像，并覆盖不同规模：

| 类别 | 任务数 | 计算量特征 | 中间数据特征 | 默认占比 |
|---|---:|---|---|---:|
| small | 8--10 | 低 | 低 | 0.30 |
| medium | 11--14 | 中 | 中 | 0.40 |
| large | 15--17 | 高 | 中高 | 0.20 |
| xlarge | 18--19 | 高 | 高 | 0.10 |

实际画像应使用固定种子一次生成并持久化，避免每次实验重新改变服务目录。

### 5.3 画像持久化

建议支持 JSON 输入输出：

```text
experiment_inputs/profiles/profile_catalog_v1.json
```

文件记录：

- 画像版本；
- 任务和链路；
- 数据单位；
- 画像生成种子；
- 类型采样概率。

所有对比算法必须使用同一服务画像目录。

---

## 6. 泊松请求到达模型

### 6.1 新增模块

建议新增 `core/arrival.py`：

```python
@dataclass(frozen=True)
class ArrivalConfig:
    lambda_per_slot: float
    simulation_slots: int
    warmup_slots: int = 0
    profile_probabilities: dict[str, float] | None = None
    ordering_policy: str = "edf"


class PoissonArrivalGenerator:
    def generate_trace(...) -> ArrivalTrace:
        ...
```

每时隙总到达数量：

```text
N_t ~ Poisson(lambda_per_slot)
```

每个到达请求独立选择画像：

```text
profile_k ~ Categorical(p_1, ..., p_Q)
```

若需要按类型配置到达率，可使用泊松叠加：

```text
N_q(t) ~ Poisson(lambda_q)
N(t) = sum_q N_q(t)
```

### 6.2 请求轨迹

建议新增：

```python
@dataclass(frozen=True)
class ArrivalTrace:
    trace_id: str
    seed: int
    lambda_per_slot: float
    requests_by_slot: tuple[tuple[InferenceRequest, ...], ...]
```

轨迹可预生成并持久化：

```text
experiment_inputs/arrival_traces/<scenario>/<seed>.json
```

算法只能通过：

```python
trace.requests_at(current_slot)
```

读取当前请求。不得把完整 `ArrivalTrace` 传给算法对象。

### 6.3 随机数流隔离

至少使用三个独立随机数流：

```text
topology_rng
arrival_rng
algorithm_rng
```

禁止算法搜索消耗到达生成器的随机数，否则不同算法将获得不同请求序列。

### 6.4 默认负载扫描

建议默认：

```text
lambda_per_slot in
{0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0}
```

时隙为 100 ms 时，对应 1--50 req/s。

必须保留两类对照：

1. `deterministic_1_per_slot`：当前固定到达模型；
2. `poisson_lambda_1`：相同平均到达率、不同突发性。

该对照用于分离“平均负载”与“到达方差”的影响。

---

## 7. 时隙事件循环改造

### 7.1 当前问题

当前主循环在处理完一个请求后立即调用：

```python
self.env.advance_time_slot()
```

因此无法在一个时隙内正确接纳多个请求。

### 7.2 新事件循环

建议把请求驱动改为时隙驱动：

```python
for slot in range(simulation_slots):
    planner.poll_completed_jobs(slot, env)
    planner.maybe_start_job(slot, env.capture_snapshot())

    arrivals = workload.requests_at(slot)
    ordered = order_requests(arrivals, policy="edf")

    for request in ordered:
        decision = dispatcher.dispatch(request, env, repository)
        metrics.record_decision(request, decision)

    env.advance_time_slot()
```

观察窗口结束后：

```python
observation_end_slot = simulation_slots
drain_running_dnns_without_new_arrivals()
```

完成吞吐量必须明确采用：

- 观察窗口内完成数量；或
- 到达于观察窗口内且最终完成的数量。

二者不能混合。建议主吞吐指标使用观察窗口内完成数量，另报告排空后的最终完成数量。

### 7.3 同时到达请求排序

第一阶段默认 EDF：

```python
sorted(arrivals, key=lambda request: (request.deadline_ms, request.request_id))
```

实验增加：

- `fcfs`
- `edf`
- `smallest_workload_first`
- `reliability_priority`
- `random_order`

同一时隙中，每接纳一个请求后必须立即更新节点和链路占用，使后续请求看到累计状态。

### 7.4 即时接纳与排队

第一阶段采用即时接纳：

```text
当前没有可行方案 -> 回退 -> 回退失败 -> 拒绝
```

第二阶段再增加截止期感知等待队列：

```text
queue_wait + online_decision + predicted_inference <= deadline
```

---

## 8. 周期规划器

### 8.1 新增模块

建议新增 `planning/periodic_planner.py`：

```python
@dataclass(frozen=True)
class PeriodicPlannerConfig:
    planning_interval_slots: int
    plan_ttl_slots: int
    pareto_pool_size: int
    population_size: int
    iteration_limit: int
    asynchronous: bool = True
    state_drift_threshold: float = 0.25


class PeriodicPlanner:
    def maybe_start_job(self, slot: int, snapshot: SystemSnapshot) -> None:
        ...

    def poll_completed_jobs(self, slot: int, env: Environment) -> None:
        ...
```

### 8.2 规划对象

每个规划周期针对：

```text
profile_id x origin_group
```

执行 DAREED 搜索，输出最多 `K` 个严格可行、互不支配候选。

第一阶段不按请求偏好分别规划。偏好只在在线选择阶段使用，避免重复运行搜索。

### 8.3 从当前 DAREED 中抽离搜索接口

在 `proposed.py` 中新增纯规划入口：

```python
def search_plan_pool(
    self,
    profile: DNNProfile,
    origin_group: int,
    snapshot: SystemSnapshot,
    pool_size: int,
) -> list[DeploymentPlan]:
    ...
```

该函数：

- 不注册 `RunningDNN`；
- 不推进环境时隙；
- 不修改在线节点和链路状态；
- 不进行请求级最终接纳；
- 返回部署向量和完整四目标评价；
- 使用快照版本作为缓存键的一部分。

可复用：

- `_build_dnn_context()`；
- `_initialize_customized_population()`；
- `_evaluate_customized_assignment()`；
- `_constraint_violation()`；
- 非支配排序和拥挤距离；
- `_filter_pareto_points()`。

### 8.4 热启动

新周期规划使用上一版本候选作为部分初始种群：

```text
30% previous feasible plans
40% repaired/perturbed previous plans
30% fresh heuristic/random plans
```

目标：

- 降低规划时间；
- 提高相邻快照间方案稳定性；
- 减少完全重启搜索造成的波动。

拓扑版本变化时禁止直接复用旧部署向量，必须重新校验节点和链路 ID。

### 8.5 异步规划时序

规划开始于 `start_slot`，实际计算耗时为 `planning_runtime_ms`：

```text
ready_slot = start_slot + ceil(planning_runtime_ms / slot_length_ms)
```

规划期间：

- 环境正常推进；
- 在线请求继续使用旧方案池；
- 规划任务只读开始时快照；
- 不允许覆盖仍在使用的方案池对象。

规划完成时：

1. 对新候选使用当前状态执行快速复验；
2. 删除已失效候选；
3. 若有效候选比例低于阈值，保留旧版本并记录发布失败；
4. 否则原子替换 `PlanRepository`。

### 8.6 初始冷启动

支持两种实验口径：

1. `prewarm`：正式观察窗口前完成第一版规划；
2. `cold_start`：系统从空方案池启动，请求使用快速回退，直到第一版规划完成。

两种口径必须分别报告，不得把预热规划时间隐式删除后称为完整在线开销。

### 8.7 周期与触发条件

初始扫描：

```text
planning_interval_slots in {30, 50, 100, 200}
```

考虑当前默认 100 ms 时隙，分别对应 3、5、10、20 秒。

除固定周期外，满足任一条件时可提前触发：

- 最大节点负载漂移超过阈值；
- 最大链路负载漂移超过阈值；
- 方案直接命中率低于阈值；
- 在线回退率超过阈值；
- 节点或链路可用性变化；
- 到达类型分布显著变化；
- 当前方案池超过 TTL。

必须设置最小重规划间隔，避免抖动和规划风暴。

---

## 9. 在线调度器

### 9.1 新增模块

建议新增 `planning/online_dispatcher.py`：

```python
@dataclass(frozen=True)
class OnlineDispatcherConfig:
    ordering_policy: str = "edf"
    enable_local_repair: bool = True
    enable_fast_fallback: bool = True
    online_budget_ms: float = 50.0


class OnlineDispatcher:
    def dispatch(
        self,
        request: InferenceRequest,
        env: Environment,
        repository: PlanRepository,
    ) -> DispatchDecision:
        ...
```

### 9.2 快速匹配流程

```text
1. 计算 origin_group
2. repository[(profile_id, origin_group)] 查找候选
3. 检查 topology_version 和 TTL
4. 使用当前状态重新计算候选指标和约束
5. 删除严格不可行方案
6. 根据请求偏好计算选择得分
7. 选择最高分候选
8. 最终完整复验
9. 注册 RunningDNN
```

在线阶段必须重算：

- 当前 CPU 可行性；
- 层级约束；
- 当前有效带宽下的时延；
- 候选接纳后的节点和链路负载；
- 链路过载约束；
- 当前 OSS/IFS；
- 截止期可行性。

不得直接采用周期规划时保存的指标作为最终接纳依据。

### 9.3 偏好选择

复用当前 `_resolve_preference_weights()` 和四目标统一满意度评分。

请求实例提供：

- 截止期；
- `preference_operation`；
- `preference_fidelity`；
- 可选的显式偏好模式。

周期候选池不因偏好不同重复生成。

### 9.4 局部修复

当没有直接可行候选时，只对最接近可行的前 `R` 个候选执行局部修复：

```text
R <= 3
```

仅允许修改：

- CPU 不足节点上的任务；
- 超载物理链路相关任务；
- 关键路径上导致超期的任务。

修复过程必须受 `online_budget_ms` 限制，不得重新运行完整进化搜索。

### 9.5 快速回退

推荐回退链：

```text
plan pool direct hit
  -> local repair
  -> maxresource_fast
  -> RTBL or configured lightweight baseline
  -> reject
```

第一阶段默认使用 `maxresource_fast`，因为其已有明确搜索预算和 Top-K beam 限制。

### 9.6 同类请求并发

同一部署方案可被多个请求复用，但每次复用前必须重新检查资源。

建议额外保存：

```text
active_instances_by_plan_id
```

并支持可选并发上限：

```text
active_instances(plan) < max_recommended_concurrency(plan)
```

第一阶段可不预计算并发上限，以逐请求累计复验作为安全边界。

---

## 10. 环境层修改

### 10.1 `core/environment.py`

新增或修改：

```python
capture_snapshot()
materialize_request()
evaluate_plan_for_request()
revalidate_plan()
register_request_assignment()
```

### 10.2 请求与运行实体关联

扩展 `RunningDNN`：

```python
request_id: int
profile_id: str
plan_id: str | None
repository_version: int | None
decision_source: str
deadline_ms: float
online_decision_ms: float
```

`decision_source` 取值：

```text
plan_direct
plan_repaired
maxresource_fast
rtbl_fallback
rejected
```

### 10.3 时延口径

在线截止期检查建议使用：

```text
online_decision_ms + predicted_inference_delay_ms <= deadline_ms
```

周期规划时间不进入单请求时延，因为它在后台运行并由多个请求复用；但必须单独报告规划开销和摊销开销。

若在线选择时间远小于截止期，可在主模型中保留推理时延为主要目标，同时在实验中报告包含在线决策的截止期满足率。

### 10.4 高负载时延闭环

当前候选时延先使用现有有效带宽计算，再预测候选负载，未迭代回代有效带宽。

泊松高负载实验中该误差会更明显。建议新增有限次固定点迭代：

```python
delay = estimate_delay(current_bandwidth)
for _ in range(max_delay_iterations):
    predicted_load = predict_link_load(delay)
    predicted_bandwidth = bandwidth_from_load(predicted_load)
    next_delay = estimate_delay(predicted_bandwidth)
    if abs(next_delay - delay) <= tolerance:
        break
    delay = next_delay
```

第一阶段可以先保持原模型，但必须通过实验开关区分：

```text
delay_prediction_mode = current_state | fixed_point
```

正式高负载结论建议使用 `fixed_point`。

---

## 11. 失败原因和指标重构

### 11.1 修改 `metrics.py`

建议新增：

```python
@dataclass
class ThroughputMetrics:
    arrival_count: int
    admitted_count: int
    completed_in_window_count: int
    completed_after_drain_count: int
    rejected_count: int
    deadline_miss_count: int
    execution_failure_count: int

    rejected_no_profile: int
    rejected_no_plan: int
    rejected_stale_plan: int
    rejected_cpu: int
    rejected_link: int
    rejected_deadline: int
    rejected_hierarchy: int
    rejected_online_timeout: int
    repair_failed_count: int
    fallback_failed_count: int

    direct_plan_hit_count: int
    repaired_plan_count: int
    fallback_success_count: int

    planner_run_count: int
    planner_publish_count: int
    planner_publish_failure_count: int
```

### 11.2 派生指标

```text
offered_rate = arrivals / observation_seconds
admission_throughput = admitted / observation_seconds
completion_throughput = completed_in_window / observation_seconds
admission_ratio = admitted / arrivals
rejection_ratio = rejected / arrivals
deadline_satisfaction_ratio = on_time_completed / completed
direct_plan_hit_ratio = direct_plan_hit / arrivals
fallback_ratio = fallback_success / arrivals
```

规划摊销开销：

```text
amortized_planning_ms = total_planning_ms / admitted_count
```

在线时延报告：

- `online_decision_ms_mean`
- `online_decision_ms_p50`
- `online_decision_ms_p95`
- `online_decision_ms_p99`
- `online_decision_ms_max`

规划时延报告：

- `planning_ms_mean`
- `planning_ms_p95`
- `planning_ms_max`
- `plan_age_slots_mean`
- `plan_age_slots_p95`

### 11.3 兼容字段

旧 `failure_count` 暂时保留：

```text
failure_count = rejected_count + execution_failure_count
```

但新实验结果必须使用新的语义版本，禁止与旧结果混合：

```text
workload_semantics_version = poisson_periodic_planning_v1
```

---

## 12. 实验运行器改造

### 12.1 `Scenario` 新增参数

在 `experiment_runner.py` 中增加：

```python
arrival_process: str = "poisson"
lambda_per_slot: float = 1.0
simulation_slots: int = 1000
warmup_slots: int = 100
profile_count: int = 8
profile_catalog_path: Path | None = None
arrival_trace_path: Path | None = None

planning_mode: str = "periodic"
planning_interval_slots: int = 50
plan_ttl_slots: int = 100
pareto_pool_size: int = 20
planner_population_size: int = 60
planner_iteration_limit: int = 200
planner_trigger_enabled: bool = True

online_ordering_policy: str = "edf"
online_budget_ms: float = 50.0
enable_local_repair: bool = True
enable_fast_fallback: bool = True
```

### 12.2 新算法名称

建议保留：

```text
customized_per_request
```

作为高开销质量上界，并新增：

```text
periodic_dareed
periodic_dareed_no_repair
periodic_dareed_no_fallback
maxresource_fast_online
rtbl_online
```

不要继续把逐请求完整 DAREED 作为实际在线方法，而应明确标记：

```text
oracle_quality_reference / per_request_search_reference
```

### 12.3 公平性

每个场景和种子必须先生成：

```text
topology
profile catalog
arrival trace
```

所有算法加载相同输入。算法不得在运行中重新生成请求。

规划器内部随机种子应由：

```text
(scenario_seed, planner_run_index, algorithm_name)
```

确定，便于复现。

---

## 13. 实验设计

### 13.1 负载—吞吐实验

固定规划周期，扫描：

```text
lambda_per_slot = 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5
```

绘制：

- 到达率—完成吞吐量；
- 到达率—接纳率；
- 到达率—拒绝率；
- 到达率—截止期满足率；
- 到达率—节点/链路最大利用率；
- 到达率—直接方案命中率；
- 到达率—回退率。

目标：确定吞吐平台和失败率快速上升的负载拐点。

### 13.2 周期敏感性

```text
planning_interval_slots = 30, 50, 100, 200
```

分析：

- 规划越频繁是否提升方案命中率；
- 规划开销是否超过控制器预算；
- 方案陈旧是否增加回退和拒绝；
- 规划周期与到达负载的交互。

### 13.3 候选池规模

```text
pareto_pool_size = 5, 10, 20, 40
```

分析在线选择开销与候选多样性的折中。

### 13.4 到达突发性对照

在相同平均到达率下比较：

```text
deterministic 1 request/slot
Poisson(lambda=1)
```

验证泊松方差是否增加阻塞和拒绝。

### 13.5 消融实验

- 无周期重规划，仅使用初始方案池；
- 无在线复验；仅用于证明复验必要性，不作为可部署方法；
- 无局部修复；
- 无快速回退；
- 无热启动；
- 固定周期 vs 状态触发；
- 单一最佳方案 vs Pareto 候选池；
- 当前状态时延预测 vs 固定点时延预测。

### 13.6 与逐请求搜索比较

在小规模场景比较：

- 四目标质量损失；
- 接纳率损失；
- 单请求在线决策时间；
- 总控制器计算开销；
- 摊销规划时间；
- 截止期满足率。

逐请求完整搜索只作为质量参考，不纳入严格实时算法排名。

---

## 14. 单元测试与集成测试

### 14.1 到达模型测试

新增 `tests/test_poisson_arrival.py`：

- 固定种子生成轨迹可复现；
- 不同算法读取同一轨迹；
- 当前时隙接口不能读取未来请求；
- 大样本均值接近 `lambda_per_slot`；
- `lambda=0` 时无请求；
- 同一时隙支持多个请求；
- 画像采样比例在统计容差内正确。

### 14.2 服务画像测试

新增 `tests/test_dnn_profile_catalog.py`：

- 画像概率和为 1；
- 画像 ID 唯一；
- 请求物化不修改原画像；
- 同一画像的多个请求共享 DAG，但请求字段独立；
- 不同发起节点产生不同上传和回传路径。

### 14.3 方案池测试

新增 `tests/test_plan_repository.py`：

- 按画像和接入边缘节点正确索引；
- 版本原子替换；
- TTL 生效；
- 拓扑版本不一致时拒绝；
- Pareto 候选不包含严格不可行方案；
- 方案对象不可变。

### 14.4 在线调度测试

新增 `tests/test_online_dispatcher.py`：

- 直接候选命中；
- 当前资源不足时淘汰旧候选；
- 截止期变化触发重新筛选；
- 同时到达请求累计占用资源；
- EDF 顺序稳定；
- 修复受时间预算限制；
- 回退成功和失败原因正确记录；
- 接纳后 `RunningDNN` 记录方案版本和决策来源。

### 14.5 周期规划测试

新增 `tests/test_periodic_planner.py`：

- 固定周期正确触发；
- 规划期间环境继续推进；
- 规划完成前继续使用旧方案；
- 新方案发布前基于当前状态复验；
- 无有效候选时保留旧方案；
- 最小重规划间隔防止触发风暴；
- 热启动不修改旧方案池。

### 14.6 指标测试

新增 `tests/test_throughput_metrics.py`：

- 到达、接纳、完成、拒绝守恒；
- 拒绝原因之和等于拒绝总数；
- 观察窗口和排空完成数分离；
- 吞吐量使用正确时间分母；
- 百分位计算正确；
- `failure_count` 兼容映射正确。

---

## 15. 分阶段实施计划

### 阶段 0：基线冻结

1. 固化当前语义版本和回归测试。
2. 保存当前逐请求算法的小规模基线结果。
3. 记录单请求决策时间、平均 delay 和拒绝率。

验收：现有测试全部通过，基线结果可复现。

### 阶段 1：画像与轨迹

1. 新增 `DNNProfile` 和 `InferenceRequest`。
2. 建立有限画像目录。
3. 实现泊松到达轨迹和随机数流隔离。
4. 保持现有逐请求算法可通过适配器运行。

验收：不同算法使用相同到达轨迹，泊松统计正确。

### 阶段 2：时隙事件循环

1. 改为每时隙读取请求批次。
2. 同一时隙内累计接纳资源。
3. 时隙末只推进一次环境。
4. 分离观察窗口与排空阶段。

验收：同一时隙多请求不会造成重复资源承诺，资源守恒测试通过。

### 阶段 3：候选方案池

1. 新增 `DeploymentPlan` 和 `PlanRepository`。
2. 从 DAREED 抽离不修改环境的搜索入口。
3. 每个画像和接入位置生成 Pareto 候选。
4. 实现版本、TTL 和拓扑校验。

验收：周期搜索不注册任务、不推进时隙、不修改在线环境。

### 阶段 4：在线匹配与回退

1. 实现候选匹配和当前状态复验。
2. 接入偏好选择。
3. 实现受预算限制的局部修复。
4. 接入 `maxresource_fast` 回退。

验收：P95 在线决策时间低于最小截止期的 10%，即默认目标低于 50 ms。

### 阶段 5：异步周期规划

1. 实现规划开始和完成事件。
2. 规划时间映射到仿真时隙。
3. 规划期间继续使用旧方案。
4. 发布前复验和原子替换。
5. 增加热启动和状态触发。

验收：规划期间系统时间正常推进，方案版本切换无中间空状态。

### 阶段 6：指标与正式实验

1. 拆分失败原因。
2. 增加吞吐、命中、回退和规划开销指标。
3. 完成负载、周期、候选池和消融实验。
4. 找出系统容量拐点和大量拒绝开始出现的区域。

验收：所有算法使用相同轨迹完成多种子实验，结果包含均值、标准差和置信区间。

---

## 16. 主要风险与控制措施

### 16.1 画像复用过于理想化

风险：有限画像不能覆盖现实中模型版本和输入形状变化。

控制：将模型版本、输入规格和精度模式纳入 `profile_id`；未匹配请求走快速回退并记录 `rejected_no_profile` 或动态画像注册。

### 16.2 方案陈旧

风险：规划完成时系统已经变化，候选失效。

控制：发布前复验、在线复验、TTL、状态漂移触发、保留多个 Pareto 候选。

### 16.3 规划频率过高

风险：后台规划长期占用控制器 CPU，形成规划风暴。

控制：最小触发间隔、热启动、时间预算、单规划任务并发限制、记录控制器利用率。

### 16.4 同时到达请求顺序偏差

风险：逐个接纳导致先处理请求占据优质资源。

控制：明确 EDF 等排序策略，并进行排序策略敏感性实验；后续再考虑联合批选择。

### 16.5 泊松到达不足以描述强突发业务

风险：真实业务可能具有周期性、相关性或重尾突发。

控制：泊松过程作为第一基线；后续增加非齐次泊松、MMPP 或真实轨迹，不用单一泊松实验外推所有业务。

### 16.6 高负载时延低估

风险：候选负载没有完整反馈到有效带宽和计算速度。

控制：增加固定点时延预测开关，至少在高负载正式实验中使用闭环预测。

### 16.7 云节点资源近似无限

风险：云 CPU 容量设置为极大值，使吞吐瓶颈主要转移到低带宽云链路。

控制：吞吐实验同时扫描有限云 CPU 和边云带宽，避免结论依赖单一理想云配置。

---

## 17. 完成定义

只有满足以下条件，周期规划改造才视为完成：

1. 每个时隙可以正确生成和处理多个泊松请求。
2. 所有算法读取同一请求轨迹且不能访问未来请求。
3. DNN 服务画像可以被多个请求重复引用。
4. 周期规划不阻塞或冻结在线环境。
5. 在线阶段不运行完整进化搜索。
6. 所有部署方案在接纳前基于当前状态重新验证。
7. 同一时隙多个请求不会超额承诺节点或链路资源。
8. 在线匹配失败后有明确修复、回退和拒绝路径。
9. 请求到达、接纳、完成、拒绝和超期指标彼此分离且满足守恒。
10. 能够绘制负载—吞吐量—拒绝率曲线并定位容量拐点。
11. 能够报告规划开销、在线决策开销和摊销规划开销。
12. 新结果带有 `poisson_periodic_planning_v1` 语义版本，不能与旧结果混合统计。

---

## 18. 推荐论文定位

完成改造后，系统可以准确表述为：

> 系统维护一个有限的可重复 DNN 服务画像目录。每个时隙内，推理请求按泊松过程到达，并根据业务比例选择服务画像。在较慢的控制时间尺度上，DAREED 基于最新系统快照周期性生成面向不同服务画像和接入位置的 Pareto 部署候选池；规划在后台异步执行，在线系统继续使用上一版本方案。请求到达后，轻量调度器根据当前资源状态重新验证候选方案，执行偏好感知选择、局部修复或快速回退，并在时隙末统一更新动态节点和链路状态。

逐请求完整 DAREED 应定位为高开销质量参考；实际在线路径由周期候选池、当前状态复验和轻量回退组成。
