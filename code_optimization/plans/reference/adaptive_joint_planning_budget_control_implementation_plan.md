# 规划周期、候选池与在线复验预算联合控制实现方案

> 核心实现状态（2026-08-02）：固定、规则式和离散 MPC 控制器、分键配额、anytime 在线复验、安全回退、原子运行时应用、因果状态聚合及 decision/outcome 对齐已经实现。正式多 seed 实验与完整动作空间扫描仍按本文第 21 节执行，不属于代码改造完成条件。

## 1. 文档状态

- 状态：统一主方案的专项实施细化
- 当前主入口：`../pending/unified_periodic_planning_model_closure_implementation_plan.md`
- 实施位置：固定周期规划、固定候选池和固定 Top-K 在线复验完成之后
- 推荐正式方法：离散模型预测控制（Discrete MPC）
- 必须保留的对照：固定参数控制、规则式自适应控制
- 不建议作为第一版正式方法：强化学习

本文档只细化以下联合控制问题：

```text
规划周期 T
候选池规模 K
在线完整复验预算 B
```

它不改变统一主方案中已经冻结的：

- P0 模型闭环；
- `ObservationSnapshot` 与信息边界；
- `CandidateStateSnapshot`；
- OSS、IFS、delay 和 energy 语义；
- offered-rate 与固定点预测；
- 泊松请求轨迹；
- 方案发布和在线最终复验的安全边界。

若本文档与统一主方案冲突，以统一主方案为准。

---

## 2. 改进目标

当前周期架构使用人工固定参数：

```text
planning_interval_slots = 50
pareto_pool_size = 20
full_revalidation_top_k = 5
online_budget_ms = 50
```

固定参数只能保证一个负载和状态区间内的折中。当到达率、热点画像、节点可用性、链路负载或规划耗时变化时，固定参数可能同时造成：

- 规划过于频繁，后台开销过高；
- 规划过慢，方案大面积陈旧；
- 候选池过小，在线直接命中率低；
- 候选池过大，重复方案和复验开销增加；
- 在线复验不足，错过可行或高质量方案；
- 在线复验过多，侵占请求截止期；
- 高到达率下在线调度器形成计算排队。

本方案的目标是：

> 在不读取未来真实请求和故障的前提下，根据当前及历史监控数据，联合决定下一控制周期的规划间隔、分画像候选池配额和在线复验时间预算，使实际 goodput 最大化，同时控制拒绝、运行失败、超期、规划开销、在线开销和方案陈旧风险。

---

## 3. 决策变量

### 3.1 规划周期

第 `e` 个控制周期选择：

```text
T_e = planning_interval_slots
```

含义是从本次规划启动或发布起，到下一次周期性规划触发的基础间隔。状态事件仍可提前触发规划，但受最小重规划间隔限制。

第一版离散动作：

```text
T_e in {20, 50, 100, 200} slots
```

### 3.2 候选池规模

不再只使用全局固定池大小，而是为每个方案键分配：

```text
K_e[p, g]
```

其中：

- `p`：`profile_id`；
- `g`：`origin_group`。

约束：

```text
K_min <= K_e[p, g] <= K_max
sum_(p,g) K_e[p,g] <= K_total_max
```

第一版离散总预算：

```text
K_total in {20, 40, 80, 160}
K_min = 2
K_max = 40
```

### 3.3 在线完整复验预算

控制器选择周期级基础预算：

```text
B_e = base_online_revalidation_budget_ms
```

第一版动作：

```text
B_e in {5, 10, 20, 50} ms
```

在线调度器再根据请求剩余时限计算请求级预算：

```text
B_r = min(
    B_e,
    max(0, safety_ratio * (deadline_ms - cheap_delay_estimate_ms))
)
```

建议：

```text
safety_ratio = 0.25
```

`B` 是时间预算，`Top-K` 是预算内实际完成完整评价的数量，不再将二者视为相同概念。

### 3.4 联合动作

```python
@dataclass(frozen=True)
class JointBudgetAction:
    planning_interval_slots: int
    total_pool_budget: int
    base_online_budget_ms: float
    pool_allocation_mode: str
```

第一阶段枚举有限动作集合：

```text
|A| = |T| * |K_total| * |B| * |allocation_mode|
```

动作集合必须足够小，使控制器自身开销远低于一次规划任务。

---

## 4. 联合关系

### 4.1 规划周期与候选池

规划周期越长，方案经历的状态漂移越大，需要更多结构多样性进行补偿：

```text
T increases
  -> staleness risk increases
  -> required K tends to increase
```

但候选池不能无限增大，因为陈旧候选本身也会增加发布和在线过滤开销。

### 4.2 候选池与在线预算

```text
K increases
  -> probability of containing feasible plan increases
  -> cheap filtering cost increases
  -> number of potentially valuable full evaluations increases
```

较大的池只有在在线预算允许、粗排序有效且候选具有结构多样性时才有价值。

### 4.3 规划周期与在线预算

较旧的方案需要更多在线评价才能判断其当前价值：

```text
T increases
  -> reference metrics become less reliable
  -> required B tends to increase
```

高负载下又必须降低单请求在线预算，因而可能需要缩短规划周期或提升方案池直接命中质量。

### 4.4 到达率约束

在线调度器利用率近似为：

```text
dispatcher_utilization
  = arrival_rate_per_second * mean_online_decision_seconds
```

必须满足：

```text
dispatcher_utilization <= dispatcher_utilization_limit
```

建议初始上限：

```text
dispatcher_utilization_limit = 0.7
```

否则即使单请求决策未超截止期，也可能因控制器排队造成系统性超期。

---

## 5. 控制状态与信息边界

### 5.1 控制状态

新增不可变控制快照：

```python
@dataclass(frozen=True)
class PlanningControlSnapshot:
    control_epoch: int
    slot: int
    observation_snapshot_version: int
    environment_state_version: int

    arrival_rate_total: float
    arrival_rate_by_key: tuple[tuple[str, int, float], ...]
    arrival_burstiness: float
    profile_distribution_drift: float

    node_load_drift: float
    link_load_drift: float
    availability_drift: float
    topology_changed: bool

    direct_hit_rate: float
    full_revalidation_pass_rate: float
    repair_rate: float
    fallback_rate: float
    rejection_rate: float
    runtime_failure_rate: float
    deadline_violation_rate: float

    valid_plan_ratio: float
    expired_plan_ratio: float
    pool_diversity_score: float
    mean_plan_age_slots: float

    online_decision_mean_ms: float
    online_decision_p95_ms: float
    mean_full_evaluation_ms: float
    planner_runtime_mean_ms: float
    planner_runtime_p95_ms: float
    planner_utilization: float
    dispatcher_utilization: float

    prediction_error_mean: float
    availability_calibration_error: float
```

### 5.2 统计窗口

控制状态只使用已完成的历史窗口：

```text
control_window_slots = max(min_control_window_slots, previous_T)
```

不同动作的评价窗口必须记录实际长度，避免短周期获得更噪声的统计值却被当作同等置信度输入。

### 5.3 禁止输入

控制器不得读取：

- 未来真实请求轨迹；
- 下一次真实节点故障时刻；
- 未来链路退化；
- 当前控制周期之后的真实 goodput；
- 仿真器预先生成但尚未暴露的事件。

离线训练或参数拟合可以使用训练场景完整结果，但测试运行时只能使用因果可见信息。训练 seed 与测试 seed 必须分离。

---

## 6. 优化目标

### 6.1 主结果：goodput utility

单请求效用：

```text
request_utility(r)
  = completed_before_deadline(r)
    * quality_utility(r)
```

其中：

```text
quality_utility
  = preference_aware_utility(IFS, OSS, DS, ES)
```

未按时成功完成的请求主效用为零，避免控制器通过大量拒绝获得较高的成功请求平均质量。

控制窗口 goodput：

```text
goodput_utility
  = sum(request_utility) / control_window_seconds
```

### 6.2 成本项

控制动作的实际窗口回报：

```text
J_e
  = goodput_utility
    - lambda_reject * rejection_rate
    - lambda_failure * runtime_failure_rate
    - lambda_deadline * deadline_violation_rate
    - lambda_planning * planner_utilization
    - lambda_online * dispatcher_utilization
    - lambda_stale * stale_plan_rate
    - lambda_churn * action_change_cost
```

其中：

```text
stale_plan_rate
  = plans_rejected_due_to_state_drift / plans_checked
```

动作变化成本用于抑制频繁调参：

```text
action_change_cost
  = normalized_distance(action_e, action_(e-1))
```

### 6.3 不允许的目标定义

不能只优化：

- 成功请求平均 OSS；
- 成功请求平均 IFS；
- 接纳率；
- 在线决策时间；
- 规划时间。

这些单项都可能诱导退化策略。例如，只最大化接纳率可能造成大量超期和运行失败，只最大化平均 OSS 可能造成过度拒绝。

---

## 7. 安全约束

### 7.1 规划作业约束

```text
predicted_planner_runtime_ms
  <= planning_interval_slots * slot_length_ms
```

且：

```text
predicted_planner_utilization <= planner_utilization_limit
```

建议初始：

```text
planner_utilization_limit = 0.5
max_concurrent_planning_jobs = 1
```

### 7.2 在线预算约束

```text
predicted_online_p95_ms <= online_p95_limit_ms
predicted_dispatcher_utilization <= dispatcher_utilization_limit
```

在线预算不能突破请求剩余截止期。

### 7.3 方案池约束

```text
sum K[p,g] <= K_total_max
K[p,g] >= K_min for active keys
```

无历史的新画像使用先验最小配额，不能因过去没有到达而永久饿死。

### 7.4 安全回退

若所有动作都被预测为不可行，控制器选择：

1. 最近一次安全动作；
2. 若最近动作也不可用，选择保守默认动作；
3. 记录 `joint_control_fallback`；
4. 不得阻断在线调度。

保守默认：

```text
T = 100 slots
K_total = 40
B = 10 ms
```

---

## 8. 候选池配额

### 8.1 配额需求分数

对每个 `(profile_id, origin_group)` 计算：

```text
demand_score[p,g]
  = smoothed_arrival_rate[p,g]
    * (1 + hit_failure_risk[p,g])
    * (1 + availability_risk[p,g])
    * (1 + deployment_diversity_need[p,g])
```

其中：

```text
hit_failure_risk = 1 - direct_hit_rate
availability_risk = 1 - mean_valid_availability_score
deployment_diversity_need = 1 - current_pool_diversity
```

### 8.2 配额分配

先为所有活动键分配 `K_min`，剩余预算按需求分数进行最大余数分配：

```text
remaining = K_total - active_key_count * K_min
extra[p,g] proportional to demand_score[p,g]
K[p,g] = K_min + extra[p,g]
```

所有取整必须确定性执行，以保证固定 seed 结果可复现。

### 8.3 配额平滑

```text
K_e[p,g]
  = clip(
      round(alpha_K * proposed_K + (1-alpha_K) * previous_K),
      K_min,
      K_max
    )
```

建议：

```text
alpha_K = 0.5
max_pool_change_ratio_per_epoch = 0.5
```

### 8.4 多样性不是数量

增加 `K` 时必须通过结构多样性筛选。若新增候选与现有 assignment 高度相似，不应仅为达到数量而保留。

建议相似度：

```text
assignment_similarity(x,y)
  = equal_task_assignments / task_count
```

发布池应同时控制：

- assignment 相似度；
- 使用节点集合重叠；
- 故障域重叠；
- 资源需求梯度。

---

## 9. Anytime 在线复验

### 9.1 粗过滤

按以下顺序执行廉价过滤：

```text
1. topology 和语义版本
2. TTL
3. 当前节点在线
4. CPU 下界
5. 固定路由存在性
6. deadline 乐观下界
7. 可用性风险下界
```

### 9.2 粗排序

粗排序可以使用：

- 规划参考指标；
- 方案年龄惩罚；
- 当前资源余量；
- assignment 节点当前在线率；
- 历史复验通过率；
- 请求偏好。

粗排序不能作为最终接纳结果。

### 9.3 完整评价循环

```python
deadline = monotonic_now_ms() + request_budget_ms
best = None

for plan in ranked_plans:
    if monotonic_now_ms() >= deadline:
        break

    snapshot = current_admission_snapshot()
    candidate = predict_candidate_state(...)

    if candidate.is_strictly_feasible:
        best = better_by_request_preference(best, candidate)

    if should_stop_early(best, remaining_budget_ms):
        break
```

最终提交前再次检查 `environment_state_version`。版本变化时重新评价或放弃，不能用旧快照提交。

### 9.4 早停

第一版只采用安全早停：

- 剩余预算不足以完成一次平均完整评价；
- 已找到严格可行方案且所有剩余方案的乐观上界不优于当前最优；
- 已达到配置的硬候选上限；
- 请求剩余截止期不足。

### 9.5 必须记录

```text
configured_online_budget_ms
request_online_budget_ms
evaluated_candidate_count
feasible_candidate_count
budget_exhausted
early_stop_reason
best_utility_after_each_evaluation
final_online_decision_ms
```

---

## 10. 第一阶段：固定参数基线

在自适应控制之前，必须先完成固定参数版本：

```python
class FixedJointBudgetController:
    def decide(snapshot) -> JointBudgetDecision:
        return configured_action
```

它用于：

- 验证控制接口不改变固定策略结果；
- 建立不同 `(T,K,B)` 组合的性能表；
- 采集代理模型训练数据；
- 作为正式消融基线。

固定网格不要求在代码改造阶段全部运行，但最小集成测试至少覆盖两个不同动作。

---

## 11. 第二阶段：规则式控制

### 11.1 规则

```text
if valid_plan_ratio low or state_drift high:
    decrease T

if planner_utilization high and state_drift low:
    increase T

if direct_hit_rate low and pool_diversity high enough:
    increase K_total

if unused_plan_ratio high or duplicate_ratio high:
    decrease K_total

if online_p95 high or dispatcher_utilization high:
    decrease B

if fallback_rate high and deadline slack sufficient:
    increase B
```

### 11.2 防抖

必须实现：

```text
min_control_interval_slots
min_replanning_interval_slots
metric_hysteresis
max_action_change_per_epoch
```

同一指标必须跨过高阈值才升级，下降到更低阈值才恢复，不能使用同一个边界双向切换。

### 11.3 定位

规则式控制是：

- 联合控制接口的验证版本；
- Discrete MPC 的对照；
- 代理模型不可用时的回退策略。

不建议将规则式控制作为论文唯一核心方法。

---

## 12. 第三阶段：离散 MPC

### 12.1 代理预测

对动作 `a` 和控制状态 `s` 预测：

```python
@dataclass(frozen=True)
class JointActionPrediction:
    predicted_goodput_utility: float
    predicted_rejection_rate: float
    predicted_runtime_failure_rate: float
    predicted_deadline_violation_rate: float
    predicted_planner_utilization: float
    predicted_dispatcher_utilization: float
    predicted_stale_plan_rate: float
    prediction_confidence: float
```

第一版代理模型优先采用：

- 分桶历史均值；
- 正则化线性模型；
- 小型决策树或梯度提升树；
- 明确的解析开销模型。

不要求使用神经网络。

### 12.2 解析开销模型

规划开销预测至少考虑：

```text
profile/origin keys to plan
population_size
iteration_limit
mean DAG task count
mean physical path length
requested pool budget
warm-start ratio
```

在线开销预测至少考虑：

```text
arrival_rate
mean cheap-filtered pool size
mean full candidate evaluation time
online budget
fixed-point iteration statistics
```

### 12.3 动作选择

```python
def choose_action(snapshot, candidate_actions):
    feasible = []

    for action in candidate_actions:
        prediction = predictor.predict(snapshot, action)
        if safety_constraints_pass(prediction, action):
            score = predicted_objective(prediction, action)
            feasible.append((score, action, prediction))

    if not feasible:
        return safe_fallback_decision(...)

    return max(feasible, key=deterministic_tie_break)
```

### 12.4 不确定性惩罚

新动作或数据不足区域使用保守分数：

```text
safe_score
  = predicted_score
    - uncertainty_penalty * prediction_uncertainty
```

不能把缺少失败记录的新动作默认视为安全。

### 12.5 训练与更新

支持两个模式：

```text
offline_fitted
online_causal_update
```

正式比较时：

- `offline_fitted` 只能使用训练场景；
- 测试 seed 和负载轨迹不得参与拟合；
- `online_causal_update` 只能在动作执行并完成观察窗口后更新；
- 更新数据必须记录 `available_after_slot`，避免时间穿越。

---

## 13. 控制时序

推荐控制器在以下时点运行：

```text
规划任务完成并发布后
或
控制窗口结束且没有规划任务正在运行时
```

完整时序：

```text
控制窗口 e 结束
  -> 聚合历史指标
  -> capture PlanningControlSnapshot
  -> 枚举联合动作
  -> 安全过滤
  -> 选择 (T_e, K_e, B_e)
  -> 分配 K_e[p,g]
  -> 更新下一规划触发时点和在线基础预算
  -> 后续实际结果归属于 action_id
  -> 窗口完成后更新代理模型
```

状态事件可以提前触发规划，但不能在每次事件发生时立即重新选择全部参数。动作至少保持一个最小控制窗口。

---

## 14. 数据模型

```python
@dataclass(frozen=True)
class JointBudgetDecision:
    decision_id: str
    control_epoch: int
    created_slot: int
    controller_version: str
    action: JointBudgetAction
    pool_budget_by_key: tuple[tuple[str, int, int], ...]
    predicted_objective: float
    predicted_metrics: JointActionPrediction
    fallback_used: bool
    fallback_reason: str | None


@dataclass(frozen=True)
class JointBudgetOutcome:
    decision_id: str
    observation_start_slot: int
    observation_end_slot: int
    actual_goodput_utility: float
    actual_rejection_rate: float
    actual_runtime_failure_rate: float
    actual_deadline_violation_rate: float
    actual_planner_utilization: float
    actual_dispatcher_utilization: float
    actual_stale_plan_rate: float
    realized_objective: float
```

预测与实际结果通过 `decision_id` 对齐。

---

## 15. 新增模块

### 15.1 `planning/control_models.py`

- `PlanningControlSnapshot`；
- `JointBudgetAction`；
- `JointBudgetDecision`；
- `JointActionPrediction`；
- `JointBudgetOutcome`。

### 15.2 `planning/joint_budget_controller.py`

```python
class JointBudgetController(Protocol):
    def decide(
        self,
        snapshot: PlanningControlSnapshot,
    ) -> JointBudgetDecision:
        ...

    def observe(
        self,
        outcome: JointBudgetOutcome,
    ) -> None:
        ...
```

实现：

```text
FixedJointBudgetController
RuleBasedJointBudgetController
DiscreteMPCJointBudgetController
```

### 15.3 `planning/action_predictor.py`

- 规划开销模型；
- 在线开销模型；
- goodput 与失败代理模型；
- 不确定性估计；
- 离线拟合和因果在线更新。

### 15.4 `planning/pool_allocator.py`

- 分画像/接入组需求分数；
- 最小配额；
- 最大余数分配；
- 配额平滑；
- 多样性约束。

### 15.5 修改 `planning/periodic_planner.py`

- 读取当前 `JointBudgetDecision`；
- 使用动态规划周期；
- 按键使用不同池配额；
- 记录动作下的规划开销；
- 禁止规划重叠。

### 15.6 修改 `planning/online_dispatcher.py`

- 使用请求级时间预算；
- 实现 anytime 完整复验；
- 记录预算耗尽和效用轨迹；
- 保留状态版本原子提交。

### 15.7 修改 `metrics.py`

- 聚合控制状态；
- 输出 action prediction/outcome；
- 计算 goodput utility；
- 计算 planner/dispatcher utilization；
- 计算 stale plan rate；
- 输出控制动作变化。

### 15.8 修改 `experiment_runner.py`

- 支持控制器模式；
- 支持训练场景与测试场景隔离；
- 输出控制周期级结果；
- 固定策略和自适应策略共享请求轨迹。

---

## 16. 配置

```yaml
joint_budget_control:
  mode: fixed  # fixed | rule_based | discrete_mpc

  action_space:
    planning_intervals: [20, 50, 100, 200]
    total_pool_budgets: [20, 40, 80, 160]
    online_budgets_ms: [5, 10, 20, 50]

  pool_allocation:
    min_per_active_key: 2
    max_per_key: 40
    smoothing_alpha: 0.5
    max_change_ratio: 0.5

  safety:
    planner_utilization_limit: 0.5
    dispatcher_utilization_limit: 0.7
    online_p95_limit_ms: 50.0
    max_concurrent_planning_jobs: 1
    min_control_interval_slots: 20
    min_replanning_interval_slots: 20

  request_budget:
    deadline_safety_ratio: 0.25
    hard_candidate_limit: 20

  objective:
    reject_penalty: 1.0
    runtime_failure_penalty: 2.0
    deadline_violation_penalty: 2.0
    planner_utilization_penalty: 0.2
    dispatcher_utilization_penalty: 0.5
    staleness_penalty: 0.5
    action_change_penalty: 0.1

  fallback:
    planning_interval_slots: 100
    total_pool_budget: 40
    online_budget_ms: 10
```

惩罚权重必须配置化、版本化，并在正式实验阶段执行敏感性分析。本轮代码改造只需验证边界和确定性。

---

## 17. 版本与结果 Schema

新增：

```text
joint_budget_control_version
joint_action_space_version
pool_allocation_version
online_anytime_revalidation_version
action_predictor_version
control_objective_version
```

推荐初始版本：

```text
joint_budget_control_version = discrete_mpc_v1
joint_action_space_version = grid_tkb_v1
pool_allocation_version = arrival_risk_diversity_v1
online_anytime_revalidation_version = budgeted_fixed_point_v1
action_predictor_version = regularized_surrogate_v1
control_objective_version = goodput_cost_v1
```

每个控制周期输出：

```text
decision_id
control_epoch
selected_T
selected_K_total
selected_B_ms
pool_budget_by_key
predicted_objective
realized_objective
prediction_error
fallback_used
action_change_distance
```

---

## 18. 测试计划

### 18.1 单元测试

建议新增：

```text
tests/test_joint_budget_action_space.py
tests/test_pool_budget_allocator.py
tests/test_rule_based_joint_controller.py
tests/test_discrete_mpc_joint_controller.py
tests/test_online_anytime_revalidation.py
tests/test_control_information_boundary.py
tests/test_joint_budget_safety_fallback.py
tests/test_joint_budget_semantic_versions.py
```

### 18.2 动作空间

- 枚举结果确定且无重复；
- 不合法的 `T/K/B` 被拒绝；
- 确定性 tie-break；
- 所有动作都带版本。

### 18.3 配额分配

- 总配额守恒；
- 活动键获得最小配额；
- 高频键在其他条件相同时不少于低频键；
- 冷启动键不会饿死；
- 平滑后仍满足上下界；
- 相同输入产生相同整数分配。

### 18.4 安全约束

- 规划时间超过周期的动作被过滤；
- planner/dispatcher 利用率超限的动作被过滤；
- 全部动作不可行时使用安全回退；
- 控制器异常不会阻断在线调度。

### 18.5 Anytime 复验

- 时间预算耗尽后停止新评价；
- 已开始的评价具有明确超时口径；
- 找到可行解后可以安全早停；
- 状态版本变化时禁止旧结果提交；
- 增大预算时实际评价数不应下降；
- 固定预算和固定 Top-K 的兼容基线可复现。

### 18.6 信息边界

- 控制快照不含未来请求；
- 控制快照不含未来故障；
- 在线代理更新只使用 `available_after_slot <= current_slot` 的结果；
- 测试场景不会进入离线拟合数据。

### 18.7 小规模集成

使用统一主方案的小规模场景，执行：

```text
fixed controller
rule_based controller
discrete_mpc controller
```

验证：

- 至少发生一次动作选择；
- 规则控制在构造的漂移场景中改变动作；
- MPC 能过滤故意设置为不安全的动作；
- 选择结果可复现；
- action prediction 与 outcome 正确关联；
- 请求、资源和结果指标守恒。

---

## 19. 实施顺序

### 阶段 A：指标和固定控制接口

1. 实现控制状态聚合；
2. 实现动作、决策和结果数据结构；
3. 实现固定控制器；
4. 将动态参数接入 planner、repository 和 dispatcher；
5. 验证固定动作与原固定配置行为一致。

### 阶段 B：分画像候选池配额

1. 实现需求分数；
2. 实现最小配额和最大余数分配；
3. 实现平滑和变化上限；
4. 接入方案池多样性筛选。

### 阶段 C：Anytime 在线复验

1. 时间预算替代固定 Top-K 主语义；
2. 实现请求剩余时限预算；
3. 实现安全早停；
4. 输出预算耗尽和效用轨迹；
5. 保留固定 Top-K 消融模式。

### 阶段 D：规则式联合控制

1. 实现阈值规则；
2. 实现迟滞、防抖和最大动作变化；
3. 实现安全回退；
4. 构造确定性状态变化测试。

### 阶段 E：动作代理模型

1. 实现规划与在线解析开销预测；
2. 实现分桶/线性初始代理；
3. 实现预测置信度；
4. 实现训练和测试数据隔离；
5. 输出预测误差。

### 阶段 F：离散 MPC

1. 枚举联合动作；
2. 预测动作结果；
3. 过滤不安全动作；
4. 计算保守目标；
5. 确定性选择；
6. 观察结果并因果更新。

### 阶段 G：代码级验收

1. 单元测试；
2. 固定 seed 回归；
3. 小规模集成；
4. 性能烟雾测试；
5. 版本和结果 Schema 检查。

---

## 20. 当前完成定义

- [x] 固定参数通过统一控制器接口执行；
- [x] `T`、`K` 和 `B` 可以由一个原子决策共同更新；
- [x] 候选池配额可以按画像和接入组分配；
- [x] 配额满足总量、上下界和确定性要求；
- [x] 在线复验以时间预算为主语义；
- [x] 请求级预算不会突破剩余截止期；
- [x] 高到达率下显式约束 dispatcher 利用率；
- [x] 规划运行时间和规划周期具有安全约束；
- [x] 规则式控制具有迟滞和防抖；
- [x] 离散 MPC 只从有限安全动作中选择；
- [x] 代理模型输出预测值和置信度；
- [x] 数据不足时执行不确定性惩罚；
- [x] 所有动作不可行时有安全回退；
- [x] 控制器不读取未来请求或未来故障；
- [x] 预测和实际结果通过 `decision_id` 对齐；
- [x] 固定、规则和 MPC 三种模式共享相同请求轨迹；
- [x] 单元测试、固定 seed 回归和小规模集成通过。

正式多 seed 对比、显著性检验、完整动作空间扫描和论文图表不属于当前代码改造完成条件。

---

## 21. 后续正式实验问题

代码语义冻结后验证：

1. 联合控制是否优于最佳固定 `(T,K,B)`；
2. 联合控制是否优于只调整一个变量；
3. 规则控制与离散 MPC 的差距；
4. 与拥有未来轨迹的离线 oracle 的差距；
5. 控制器在负载阶跃、热点迁移、节点故障和网络漂移下的响应；
6. 动作切换频率和系统抖动；
7. 代理预测误差对控制效果的影响；
8. goodput 改进是否来自更少失败，而不是过度拒绝；
9. 规划开销、在线开销和服务质量之间的 Pareto 关系。

建议正式消融：

```text
fixed T, fixed K, fixed B
adaptive T only
adaptive K only
adaptive B only
independent rule adaptation
joint rule adaptation
joint discrete MPC
offline oracle upper bound
```

该实验阶段用于形成投稿证据，不作为当前实现文档的代码完成阻塞项。
