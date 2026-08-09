# Customized 通用方案池增强实施方案

## 1. 文档状态

- 状态：待实施
- 生成日期：2026-08-09
- 实施对象：周期性规划主路径中的 Customized 候选生成、方案池筛选和在线反馈闭环
- 核心目标：把“规划快照上的优质 assignment”增强为“在同一 profile、origin group 和 deadline 类别内具有跨状态复用能力的部署方案”
- 活动优化目标保持不变：`OSS + delay satisfaction + energy satisfaction`

本方案不增加新的第四优化目标，不改变当前节点在线硬约束，也不允许规划器读取未来故障或未来请求。通用性通过鲁棒可行性门控、资源裕量、结构多样性和因果在线反馈实现。

---

## 2. 当前问题

当前 Customized 已具备 DAG 感知初始化、块交叉、偏斜变异、修复、三目标 Pareto archive 和节点集合多样性，因此能够生成同一 `profile_id + origin_group` 下可复用的结构候选。

但其候选主要针对单个规划快照优化：

\[
F(x\mid S_{t_p})=
[OSS(x),D_{sat}(x),E_{sat}(x)]
\]

存在以下不足：

1. 候选可能过拟合规划时刻的空闲 CPU、链路负载和节点历史；
2. 单一代表 deadline 不能覆盖严格与宽松请求；
3. 节点 Jaccard 多样性不能识别共享物理链路和共同故障域；
4. 规划快照下刚好可行的方案可能没有 CPU、链路或 deadline 裕量；
5. 方案池没有利用历史在线复验通过率和实际使用率进行淘汰；
6. 当前候选 Pareto 质量不能直接证明未来时隙的直接命中能力。

因此，增强后的方案池需要同时回答：

```text
该方案在当前快照下是否优质？
该方案在一组因果可构造的扰动状态下是否仍经常可行？
该方案是否为现有方案池提供了新的节点、链路或故障域覆盖？
该方案在真实在线复验中是否被证明有用？
```

---

## 3. 冻结设计决策

### 3.1 通用性的适用范围

通用方案限定为：

```text
同一 profile_id
+ 同一 origin_group
+ 同一 deadline_class
+ 相同拓扑与目标语义版本
```

不追求跨不同 DAG 的 assignment 复用。

### 3.2 三目标保持不变

Customized 的 Pareto 目标仍为：

```python
(operational_stability, delay_satisfaction, energy_satisfaction)
```

以下量只用于门控、tie-break、池筛选和诊断，不进入 Pareto 目标向量：

- 鲁棒可行率；
- 效用均值和下置信界；
- CPU、链路和 deadline 裕量；
- 节点、物理链路和故障域多样性；
- 历史在线复验通过率和实际利用率。

### 3.3 信息边界

鲁棒场景只能由以下信息构造：

- 当前规划快照；
- 截止规划时刻已经观测到的负载与可用性历史；
- 配置中公开的扰动级别；
- 过去已完成在线复验形成的统计量。

禁止使用：

- 规划时刻之后的真实故障轨迹；
- 未来泊松到达批次；
- 为当前实验 seed 预生成但尚未发生的环境状态；
- 请求执行后的未来完成或失败标签。

---

## 4. 总体增强框架

```text
当前规划快照
  -> 因果鲁棒场景生成
  -> Customized 搜索生成 assignment
  -> 当前快照三目标严格可行性
  -> 多场景鲁棒复验
  -> 鲁棒可行率门控
  -> 三目标 Pareto 筛选
  -> 裕量与在线反馈 tie-break
  -> 节点/链路/故障域联合多样性筛选
  -> 分 key 候选池原子发布
  -> 在线复验与实际使用反馈
  -> 下一规划周期更新保留和淘汰决策
```

增强分为三个层次：

1. 候选层：多快照鲁棒复验与裕量诊断；
2. 方案池层：联合结构多样性和 deadline 分桶；
3. 反馈层：使用真实在线结果更新方案价值。

---

## 5. 因果多快照鲁棒复验

### 5.1 鲁棒场景集合

为每个规划快照构造一个小规模、确定性、可审计的场景集合：

```text
S0 nominal：当前规划快照
S1 load_up：节点负载和方向 offered rate 轻度上升
S2 load_high：节点负载和 offered rate 中度上升
S3 availability_stress：依据历史概率降低部分随机节点连续可用性
S4 link_stress：提高已观测高热物理链路的压力
S5 mixed：轻度负载上升与可用性压力组合
```

默认不直接把节点强制设为未来离线。若需要离线压力场景，只能根据当前历史估计进行确定性 worst-case 选择，并在结果中标记为合成压力测试。

### 5.2 场景一致性

扰动快照必须保持字段一致：

- 节点负载升高时同步减少可用 CPU；
- 节点离线时可用 CPU 设为零；
- offered rate 变化后由统一候选预测器重算链路负载和有效带宽；
- 不直接同时手工修改互相依赖的 offered rate、链路负载和带宽；
- 每个场景具有稳定的 `scenario_id`、扰动参数和生成版本。

建议新增：

```python
@dataclass(frozen=True)
class RobustnessScenario:
    scenario_id: str
    observation_snapshot: ObservationSnapshot
    scenario_weight: float
    perturbation_metadata: tuple[tuple[str, float | str], ...]
```

### 5.3 候选鲁棒统计

对 assignment `x` 在 M 个场景中复验：

\[
R_f(x)=\frac{1}{M}\sum_{m=1}^{M}
\mathbf{1}[x\in\mathcal F(S_m)]
\]

同时计算：

\[
\bar U(x)=\sum_m q_m U(x\mid S_m)
\]

\[
U_{lcb}(x)=\bar U(x)-\kappa\sigma_U(x)
\]

新增具名诊断结构：

```python
@dataclass(frozen=True)
class CandidateRobustnessScores:
    scenario_count: int
    feasible_scenario_count: int
    feasible_ratio: float
    mean_utility: float
    utility_std: float
    utility_lcb: float
    worst_delay_slack_ratio: float
    minimum_cpu_slack_ratio: float
    minimum_link_slack_ratio: float
    nonconverged_scenario_count: int
```

### 5.4 鲁棒门控

候选进入方案池必须满足：

```text
nominal snapshot 严格可行
fixed-point 在 nominal snapshot 收敛
robust_feasible_ratio >= configured threshold
nonconverged_scenario_count <= configured limit
```

建议默认阈值从 `0.6` 开始做 sensitivity，不直接冻结为论文参数。

当所有候选都未达到阈值时，允许降低阈值并发布退化池，但必须：

- 记录 `robust_gate_relaxed=1`；
- 记录实际使用阈值；
- 不允许绕过 nominal 严格可行性。

---

## 6. 资源和 deadline 裕量

### 6.1 裕量定义

节点 CPU 裕量：

\[
M_{cpu}(x)=
\min_{n\in N(x)}
\frac{CPU^{available}_n-CPU^{required}_n}
{\max(CPU^{available}_n,\epsilon)}
\]

链路裕量：

\[
M_{link}(x)=
\min_{l\in L(x)}(1-load_l(x))
\]

deadline 裕量：

\[
M_d(x)=\frac{deadline-D(x)}{deadline}
\]

### 6.2 使用方式

裕量不作为新目标，只用于：

1. 同一 Pareto 层且效用接近候选的 tie-break；
2. 阻止近饱和方案占满方案池；
3. 为在线 repair 选择备用方案；
4. 输出通用性诊断。

建议组合为：

\[
M(x)=\min(M_{cpu},M_{link},M_d)
\]

避免平均值掩盖单一严重瓶颈。

---

## 7. 联合结构多样性

### 7.1 当前问题

当前只比较使用节点集合的 Jaccard 重叠，不能识别两个方案是否共享：

- 同一关键物理链路；
- 同一边缘区域或故障域；
- 相同计算瓶颈节点；
- 相同输入/回传路径。

### 7.2 新多样性距离

定义：

\[
D(x_i,x_j)=
\alpha D_{node}
+\beta D_{link}
+\gamma D_{domain}
\]

其中每项为 `1 - Jaccard similarity`。

第一阶段可冻结：

```text
alpha = 0.5
beta  = 0.5
gamma = 0.0
```

若拓扑后续增加显式 failure-domain 标识，再启用 `gamma`。

### 7.3 池筛选顺序

每个 key 的候选池按以下顺序构造：

1. assignment 去重；
2. nominal 严格可行；
3. 鲁棒可行率门控；
4. 三目标非支配筛选；
5. 按 `utility_lcb + margin tie-break + online feedback` 排序；
6. 贪心选择与已选方案联合距离足够大的候选；
7. 若不足 K，再放宽多样性阈值补齐。

---

## 8. Deadline 分桶

### 8.1 分桶原因

单一 `planning_deadline_ms` 可能使方案只适合代表 deadline。建议增加：

```text
strict：  [min_deadline, q33]
normal：  (q33, q67]
relaxed： (q67, max_deadline]
```

分位点只能根据配置的公开 deadline 分布或过去已经到达的请求统计计算。

### 8.2 仓库 key

将可选增强 key 改为：

```text
(profile_id, origin_group, deadline_class)
```

第一阶段保留功能开关：

```python
deadline_bucket_mode: Literal["disabled", "three_quantile"] = "disabled"
```

只有当实验显示 deadline 分桶显著提高 direct hit 或降低 deadline miss，且没有造成候选池严重碎片化时，才设为默认。

---

## 9. 在线使用反馈闭环

### 9.1 反馈统计

按 `plan_id` 维护：

```python
@dataclass
class PlanOnlineFeedback:
    lookup_count: int
    quick_filter_pass_count: int
    full_revalidation_count: int
    full_revalidation_pass_count: int
    direct_commit_count: int
    repair_count: int
    runtime_failure_count: int
    deadline_miss_count: int
    cumulative_realized_utility: float
    last_used_slot: int | None
```

反馈只能在相关事件已经发生后更新。

### 9.2 反馈分数

采用 Beta 平滑避免新方案因样本少被过早淘汰：

\[
P_{pass}(p)=
\frac{pass_p+a_0}{revalidated_p+a_0+b_0}
\]

建议保留探索奖励：

\[
V_{feedback}(p)=
P_{pass}(p)+
c\sqrt{\frac{\log(1+N)}{1+n_p}}
\]

反馈只用于跨周期 retain/evict 和候选排序 tie-break，不覆盖最新快照的完整在线复验结果。

### 9.3 保留与淘汰

新仓库生成时允许保留旧仓库中满足以下条件的方案：

- 拓扑和目标语义一致；
- TTL 可续期；
- 在新 nominal 快照上重新验证通过；
- 在线复验通过率或利用率较高；
- 与新候选不存在完全重复。

长期未使用、复验通过率低或频繁 repair 的方案优先淘汰。

---

## 10. Customized 搜索增强

### 10.1 搜索阶段控制成本

不能对每个遗传评价都执行全部鲁棒场景，否则规划成本可能扩大 M 倍。采用两阶段评价：

```text
遗传迭代阶段：只使用 nominal 快照三目标评价
archive 候选阶段：对去重后的严格可行候选执行多场景鲁棒复验
```

可选优化：只对每代 top-L 或最终 archive 复验。

### 10.2 鲁棒引导精英

若第一阶段候选池鲁棒率过低，可在后续版本将少量鲁棒表现好的候选作为下一轮规划的 warm-start seed，但必须：

- 在最新 nominal 快照上重新修复；
- 不直接复制历史目标值；
- 记录 `warm_start_source_plan_id`；
- 保留随机和结构启发式个体，避免种群退化。

### 10.3 候选数量预算

联合控制中的 K 仍表示最终发布方案池预算。新增内部参数：

```text
robust_revalidation_top_l
robust_scenario_count
robust_gate_threshold
```

这些参数先固定并做 sensitivity，暂不并入 T/K/B 联合动作空间，避免一次扩展过多控制维度。

---

## 11. 文件级改造清单

| 文件/模块 | 修改内容 |
|---|---|
| `models.py` | 新增鲁棒场景、候选鲁棒统计和 deadline class 类型，保持 CandidateStateSnapshot 三目标结构不变 |
| `planning/robustness.py` | 新增因果扰动场景生成、快照一致性校验和多场景候选评价 |
| `proposed.py` | 最终 archive 输出后执行 top-L 鲁棒复验；候选保留三目标元组，附加具名鲁棒统计 |
| `planning/repository.py` | DeploymentPlan 增加鲁棒统计、裕量、物理链路签名和可选 deadline class |
| `planning/periodic_planner.py` | 鲁棒门控、旧方案重验证保留、反馈驱动淘汰和发布审计 |
| `planning/online_dispatcher.py` | 输出 plan 级快速过滤、完整复验和提交反馈，不改变最终在线复验规则 |
| `planning/request_dispatcher.py` | 保持逐请求基线不读取方案反馈，避免不公平的信息扩张 |
| `periodic_experiment.py` | 记录方案反馈、跨时隙存活率和鲁棒预测—实际结果对应关系 |
| `metrics.py` | 增加 direct validity、plan survival、reuse concentration、robust calibration 指标 |
| `periodic_experiment_runner.py` | 增加 generalization 与 robustness_ablation suite |
| `semantics.py` | 增加 robustness、pool diversity、deadline bucket 和 feedback 版本字段 |
| `plot_results.py`、`plot/generate_paper_figures.py` | 增加存活曲线、复验通过率和方案利用分布图 |

---

## 12. 新增语义版本

建议增加：

```text
robustness_scenario_version = causal_snapshot_stress_v1
robust_candidate_gate_version = feasible_ratio_lcb_v1
pool_diversity_version = node_physical_link_v1
plan_feedback_version = beta_smoothed_usage_v1
deadline_bucket_version = disabled_v1 / three_quantile_v1
```

只要场景构造、门控阈值语义或反馈评分变化，就必须升级对应版本。旧方案仓库不能被新版本调度器静默读取。

---

## 13. 实施阶段

### 阶段 A：通用性指标基线

1. 不改变候选生成，仅记录现有方案的在线复验通过率；
2. 记录方案发布时间到后续时隙的存活率；
3. 记录 direct、repair、fallback 和方案使用集中度；
4. 建立 Customized 与 lightweight 的当前通用性基线。

验收：能够量化当前方案池是否真正被重复使用。

### 阶段 B：鲁棒场景与候选门控

1. 实现因果鲁棒场景生成器；
2. 实现快照一致性校验；
3. 对最终 archive top-L 执行多场景复验；
4. 增加鲁棒可行率门控和退化发布审计。

验收：不得读取未来状态；固定 seed 下鲁棒统计完全可复现。

### 阶段 C：裕量与联合结构多样性

1. 计算 CPU、链路和 deadline 最小裕量；
2. 方案池多样性从节点扩展到节点+物理链路；
3. 增加池不足时的可审计阈值放宽；
4. 比较对 direct hit、fallback 和 Goodput 的影响。

验收：方案池不是多个共享同一关键链路的表面多样候选。

### 阶段 D：在线反馈闭环

1. 按 plan_id 记录因果反馈；
2. 实现 Beta 平滑通过率和探索项；
3. 新规划周期重新验证并保留高价值旧方案；
4. 淘汰低通过、低利用和高 repair 方案。

验收：反馈更新不会使用未来结果，也不会绕过最新快照在线复验。

### 阶段 E：Deadline 分桶实验

1. 以功能开关实现三分位 deadline class；
2. 扩展仓库 key、规划 target 和请求 lookup；
3. 对比关闭/开启分桶的收益与池碎片化；
4. 根据实验决定是否设为默认。

验收：严格 deadline 请求的 miss 降低，且规划开销和 no-plan rejection 不出现不可接受增长。

---

## 14. 测试要求

### 14.1 单元测试

- 鲁棒场景只依赖当前及历史观测；
- 相同快照、配置和 seed 生成完全相同场景；
- 节点负载与可用 CPU 扰动保持一致；
- nominal 不可行候选永远不能通过鲁棒门控；
- feasible ratio、utility mean/std/LCB 和三类裕量计算正确；
- 联合多样性能够识别节点不同但物理链路相同的方案；
- Beta 平滑反馈在零样本时有定义；
- 旧方案保留前必须在新快照重新验证；
- deadline bucket 边界确定且无重叠。

### 14.2 集成测试

- 周期 Customized 能发布带鲁棒统计的方案池；
- `--suite all --quick` 保持请求守恒；
- 在线复验最终判断不读取规划鲁棒分数替代当前评价；
- 方案反馈能跨规划周期保留和淘汰候选；
- old repository 因新语义版本被拒绝；
- 同一到达和故障轨迹下可以关闭全部增强恢复当前基线。

### 14.3 性能测试

- 报告鲁棒复验使规划时间增加的比例；
- 验证 top-L 两阶段策略不会使规划成本简单扩大为场景数 M 倍；
- 在线 P95 不因后台鲁棒评价明显上升；
- 反馈表的内存随 plan 数有界增长。

---

## 15. 实验设计

### 15.1 核心对比

```text
C0 current_customized
C1 + robust gate
C2 + robust gate + margin
C3 + robust gate + margin + node/link diversity
C4 + all above + online feedback
C5 + C4 + deadline bucket
```

所有变体共享：

- profile catalog；
- 泊松到达轨迹；
- 节点故障轨迹；
- topology；
- T/K/B 或同一联合控制策略；
- paired seed。

### 15.2 主要指标

通用性主指标：

- `plan_direct_rate`；
- `full_revalidation_pass_rate`；
- `plan_survival_rate_at_1/5/10_slots`；
- `repair_rate`；
- `fast_fallback_rate`；
- `plan_reuse_concentration`；
- `unused_plan_fraction`。

系统结果：

- Goodput；
- completion throughput；
- rejection、runtime failure 和 deadline miss；
- 平均 delay、OSS 和 energy；
- planning runtime、online P95 和候选复验次数。

### 15.3 通用性结论门槛

只有满足以下条件才能声称增强有效：

1. 多数负载场景下 direct rate 或复验通过率显著提高；
2. fallback 或 deadline miss 至少一项显著降低；
3. Goodput 不下降，或下降可由明确的稳定性收益解释；
4. 规划时间增长处于报告并可接受的范围；
5. 提升在 paired multi-seed 统计中稳定；
6. 不是仅靠扩大 K 或 B 获得的收益。

---

## 16. 推荐实施优先级

```text
P0：阶段 A 指标基线
P0：阶段 B 因果鲁棒门控
P1：阶段 C 裕量与节点/链路多样性
P1：阶段 D 在线反馈闭环
P2：阶段 E deadline 分桶
```

优先实现阶段 A 和 B。没有现状通用性指标时，无法判断后续复杂增强是否真正有效；没有因果鲁棒门控时，单纯增加候选池多样性也不能解决单快照过拟合。

---

## 17. 完成定义

- [ ] 当前方案池通用性能够通过直接命中、复验通过率和存活曲线量化；
- [ ] 鲁棒场景生成满足信息边界且可复现；
- [ ] Customized archive 候选经过 nominal 严格可行和多场景门控；
- [ ] Pareto 目标仍严格为 OSS、delay satisfaction、energy satisfaction；
- [ ] 资源裕量不作为隐藏第四目标；
- [ ] 方案池同时具备节点和物理链路多样性；
- [ ] 在线反馈只使用已经发生的因果结果；
- [ ] 在线最终接纳仍由最新快照完整复验决定；
- [ ] 所有增强均可通过开关关闭以恢复当前基线；
- [ ] 完整单元测试和 `--suite all --quick` 通过；
- [ ] 多 seed 实验能够证明通用性收益不是 K/B 扩大造成的。

---

## 18. 风险与处理

| 风险 | 处理方式 |
|---|---|
| 多场景复验显著增加规划时间 | 只复验最终 archive top-L，缓存重复 assignment 结果 |
| 鲁棒门控导致候选池为空 | 可审计地降低阈值，但绝不放宽 nominal 严格可行性 |
| 压力场景与真实分布不一致 | 同时报告合成鲁棒指标和未来真实在线通过率，不把前者当真实概率 |
| 裕量偏好变成隐藏新目标 | 仅在 Pareto 层和效用近似相同时 tie-break，并做独立消融 |
| 多样性导致候选质量下降 | 池不足时逐级放宽距离阈值，记录放宽过程 |
| 在线反馈造成热门方案垄断 | Beta 平滑和探索奖励，保留新方案最小配额 |
| deadline 分桶造成方案池碎片化 | 默认关闭，通过实验决定是否启用 |
| 历史反馈跨语义污染 | 反馈记录绑定 topology、objective、robustness 和 pool-diversity 版本 |

本方案的核心原则是：Customized 继续负责发现三目标优质结构，鲁棒门控负责检验跨状态可行性，方案池多样性负责提供替代路径，在线反馈负责识别长期有用方案，而最新快照在线复验始终保留最终决定权。
