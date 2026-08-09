# 后台规划复用原有 Customized 算法生成候选的执行方案

## 1. 文档状态

- 状态：已实施（2026-08-03）
- 适用架构：周期性后台规划、在线快速复验与接纳
- 默认启动方式：`prewarm`
- 核心决策：后台规划必须复用原有 `customized` 进化搜索生成候选
- 关联总方案：`unified_periodic_planning_model_closure_implementation_plan.md`

本文档细化周期规划中的“候选如何生成”，不改变已经冻结的泊松批量到达、不可变系统快照、异步发布复验和在线接纳语义。

实施结果：

- 已新增固定快照、无环境提交副作用的单 DNN customized 搜索入口；
- 周期规划器已默认使用 customized 跨代可行 Pareto 档案生成方案池；
- `run_customized_proposed()` 已改为调用同一搜索内核并选择单个执行解；
- lightweight 生成器保留为显式模式和可选 fallback；
- 方案与发布审计已记录生成器模式、版本、搜索种子和来源代数；
- 批量实验入口已增加生成器模式、种群、代数和档案容量参数；
- 全量测试及 `--suite all --quick` 周期实验入口已经通过。

原有 `customized` 算法仍是系统的核心部署优化算法。周期性规划不是用新的轻量启发式替换它，而是改变它的运行位置和输出形式：

```text
原流程：请求到达 -> customized 完整搜索 -> 选出一个方案 -> 立即提交

新流程：规划触发 -> customized 完整搜索 -> 生成多个候选 -> 后台方案库
        请求到达 -> 匹配候选 -> 当前状态复验 -> 快速选择并提交
```

---

## 2. 当前问题

当前代码已经具备周期性规划框架，但后台候选生成仍调用 `proposed.py::search_plan_pool()`。该函数只生成以下简单候选：

- 全部任务部署到某个合法节点；
- 轮询部署；
- 随机可行部署；
- 对上述方案执行可行性过滤、Pareto 过滤和多样性选择。

它没有执行原有 `run_customized_proposed()` 中的完整搜索过程，包括：

- customized 种群初始化；
- 角色与膜结构演化；
- DAG 感知交叉；
- 动态约束阈值和约束支配排序；
- 精英保留与局部搜索；
- 多代搜索形成的高质量 Pareto 解集。

因此，当前周期规划虽然复用了相同的目标计算和 `AllDNNRefactor` 类，却没有真正复用原 customized 优化算法。若直接用于正式实验，会产生两个问题：

1. “周期规划方案建立在原算法之上”的论述与实现不完全一致；
2. 周期方案与逐请求 customized 基线的质量差异会同时混入“规划架构变化”和“候选生成器退化”两种因素。

---

## 3. 目标与非目标

### 3.1 实施目标

1. 抽取原 customized 算法的单 DNN、无副作用搜索内核；
2. 后台规划默认使用该内核生成多个严格可行的 Pareto 候选；
3. 所有搜索计算绑定规划启动时的同一份 `ObservationSnapshot`；
4. 搜索不得接纳请求、占用资源、推进时隙或读取未来环境真值；
5. 保留原 `run_customized_proposed()` 作为逐请求对照入口，但改为调用同一搜索内核；
6. 保留现有轻量 `search_plan_pool()`，仅用于快速测试、故障回退和消融实验；
7. 记录真实搜索开销、搜索预算和候选质量，使后台规划延迟可解释。

### 3.2 本轮非目标

- 不重写 customized 的目标函数和搜索算子；
- 不改变 IFS、OSS、delay、DS、ES 的统一候选预测公式；
- 不让规划器读取未来节点状态或未来请求；
- 不把单次搜索的最佳解直接视为在线接纳结果；
- 不在本轮把搜索代数或种群规模纳入联合控制器的动作空间；
- 不要求本轮完成投稿级完整实验。

---

## 4. 冻结的架构决策

### 4.1 双时间尺度职责

```text
后台慢路径
ObservationSnapshot(planning)
  -> 单 DNN customized 搜索
  -> 可行非支配档案
  -> 去重和多样性筛选
  -> DeploymentPlan 候选池
  -> 规划完成时 publication_revalidation
  -> 原子发布 PlanRepository

在线快路径
InferenceRequest
  -> profile_id + origin_group 定位方案池
  -> TTL/语义/节点在线硬过滤
  -> 预算内 Top-K 完整复验
  -> 偏好选择或轻量修复
  -> 原子接纳或明确拒绝
```

后台 customized 只负责生成具有部署质量和结构多样性的候选。在线阶段仍负责根据最新资源、链路和节点可用性决定某个请求能否实际采用候选。

### 4.2 默认生成器

新增配置：

```python
candidate_generator_mode: str = "customized"
```

允许值：

- `customized`：正式默认路径；
- `lightweight`：现有简单候选生成器，用于消融和冒烟测试；
- `customized_with_fallback`：customized 未在预算内产生可行解时，追加轻量候选。

正式主实验必须使用 `customized`。是否启用 fallback 必须显式记录，且 fallback 生成的候选要带来源标签，不能无痕混入主结果。

### 4.3 Prewarm 语义

默认 `prewarm` 也必须运行 customized 候选生成器。预热阶段允许在正式统计时钟开始前构建第一版方案库，但必须：

- 使用初始可观测快照；
- 记录搜索耗时、评价次数和候选数量；
- 不推进仿真时隙；
- 不预先接纳任何请求；
- 不读取预生成请求轨迹中的未来到达。

---

## 5. 搜索内核改造

### 5.1 新增搜索配置

在 `proposed.py` 或独立的 `planning/customized_generator.py` 中定义不可变配置：

```python
@dataclass(frozen=True)
class CustomizedSearchConfig:
    population_size: int
    generations: int
    elite_count: int
    local_search_count: int
    max_evaluations: int | None = None
    wall_time_budget_ms: float | None = None
    archive_capacity: int = 64
    random_seed: int = 0
```

第一阶段优先读取原 customized 算法现有默认参数，确保算法含义不变。`max_evaluations` 和 `wall_time_budget_ms` 先作为停止条件与统计字段，不立即加入联合控制动作。

### 5.2 新增结果类型

```python
@dataclass(frozen=True)
class CustomizedCandidate:
    assignment: tuple[int, ...]
    objectives: tuple[float, float, float, float]
    constraint_violation: float
    generation: int
    source: str


@dataclass(frozen=True)
class CustomizedSearchResult:
    candidates: tuple[CustomizedCandidate, ...]
    evaluations: int
    generations_completed: int
    feasible_evaluations: int
    archive_size_before_selection: int
    termination_reason: str
    wall_time_ms: float
    seed: int
```

`source` 至少区分：`initialization`、`evolution`、`local_search` 和 `lightweight_fallback`。

### 5.3 单 DNN 搜索入口

新增纯搜索入口：

```python
def generate_customized_candidate_pool(
    self,
    *,
    dnn_index: int,
    snapshot: ObservationSnapshot,
    profile_id: str,
    origin_group: int,
    search_config: CustomizedSearchConfig,
    output_pool_size: int,
    semantic_versions: SemanticVersions,
) -> CustomizedSearchResult:
    ...
```

该入口复用原算法的初始化、交叉、变异/修复、支配排序、精英局部搜索与目标评价，不包含以下副作用：

- `_register_running_dnn(...)`；
- `env.advance_time_slot(...)`；
- 资源扣减或释放；
- 请求完成事件；
- 方案库发布；
- 全局随机数状态修改。

### 5.4 从原入口抽取，而不是复制算法

`run_customized_proposed()` 当前同时承担搜索、选解、提交和推进环境四种职责。改造后拆为：

```text
_run_customized_search_core(...)
  -> 返回可行非支配候选与搜索诊断

run_customized_proposed()
  -> 调用 search core
  -> 按原规则选出单个解
  -> 提交到环境并推进原基线流程

PeriodicPlanner
  -> 调用同一个 search core
  -> 保留多个多样化解
  -> 转换为 DeploymentPlan，不提交环境
```

禁止复制一份“periodic customized”算法。两条路径必须共享同一内核，避免后续算子修复只作用于其中一条路径。

---

## 6. 规划快照绑定

### 6.1 当前风险

`_candidate_state_for_assignment()` 当前可能在评价过程中重新捕获 `online_admission` 快照，并使用实时环境版本作为缓存条件。后台搜索运行期间，仿真环境可能继续变化；若每次评价读取实时状态，同一代种群将基于不同系统状态比较，搜索结果失去明确含义。

### 6.2 改造要求

为一次搜索创建只读评价上下文：

```python
@dataclass(frozen=True)
class SearchEvaluationContext:
    dnn_index: int
    snapshot: ObservationSnapshot
    profile_id: str
    origin_group: int
    semantic_versions: SemanticVersions
```

以下调用链必须显式接收该上下文或其中的 `snapshot`：

- 种群初始化和可行性修复；
- `_candidate_state_for_assignment()`；
- `_evaluate_assignment()`；
- `_count_customized_values()`；
- 子代生成后的评价；
- 精英局部搜索；
- 最终候选重评。

候选预测缓存键至少包含：

```text
(dnn_index, assignment, snapshot_version,
 environment_state_version, topology_version,
 objective_semantics_version)
```

搜索开始后即使环境状态发生变化，本次搜索仍完整使用启动快照。环境变化只在规划完成后的发布复验阶段处理。

### 6.3 信息边界

搜索上下文可以读取：

- 规划快照中的节点、负载、链路和历史可用性监控数据；
- DNN 画像、DAG 和资源需求；
- 已发布方案池，用于热启动；
- 配置或历史估计到达率。

不得读取：

- 未来请求到达；
- 节点真实未来下线时刻；
- 仿真器已经抽样但尚未发生的退化值；
- 规划完成时才会出现的真实环境状态。

---

## 7. 多候选档案与方案池

原算法最终只需选择一个执行解；周期规划需要输出多个候选。因此搜索过程中维护有界档案：

1. 每代收集严格可行解；
2. 按 assignment 去重；
3. 维护跨代非支配解，而非只保留最后一代；
4. 将局部搜索产生的可行精英加入档案；
5. 搜索结束后用同一规划快照统一重评；
6. 删除失效、不可行和被支配解；
7. 使用现有 `select_diverse_plans()` 从档案中选择 `K_g` 个结构和目标空间均有差异的方案。

每个 `DeploymentPlan` 继续记录：

- assignment、资源和链路占用；
- reference delay、OSS、availability、IFS、energy、satisfaction；
- planning snapshot/environment/topology version；
- TTL 和语义版本；
- `profile_id` 与 `origin_group`。

建议新增：

```text
generator_mode
generator_version
search_seed
search_job_id
source_generation
```

这些字段用于复现、消融和发现 fallback，不参与在线目标计算。

---

## 8. 预算语义

必须区分三类预算：

| 预算 | 含义 | 当前控制方式 |
|---|---|---|
| `K_g` | 每个画像/接入组发布的候选数量 | 联合控制器决策 |
| `B` | 在线复验时间或候选数量预算 | 联合控制器决策 |
| `P, G, E_max` | customized 种群、代数和最大评价次数 | 第一阶段配置固定 |

`K_g` 是输出方案池配额，不等于 customized 的种群规模。不得为了发布 4 个候选就把种群规模设为 4。

后台规划耗时由实际 customized 搜索测量或通过校准模型映射到仿真时隙：

```text
planning_ready_slot = launch_slot + calibrated_planning_slots
```

至少记录：

- 墙钟搜索时间；
- 仿真规划完成时隙；
- 评价次数；
- 完成代数；
- 可行解比例；
- 档案规模；
- 最终发布候选数；
- 终止原因。

若超过预算，应返回当前档案并标记 `evaluation_budget` 或 `wall_time_budget`，不能丢弃整个规划任务。若档案为空，则继续使用上一版有效方案库或显式触发配置允许的 fallback。

---

## 9. 随机性与可复现性

后台搜索必须使用局部 RNG，种子由稳定字段派生：

```text
scenario_seed
+ planning_job_id
+ profile_id
+ origin_group
+ generator_version
```

要求：

- 不调用或重置模块级全局 `random` 状态；
- 相同快照、配置和种子产生相同候选及排序；
- 不因规划任务执行顺序改变某个 key 的搜索轨迹；
- `customized` 与 `lightweight` 消融共享相同场景、请求轨迹和环境退化轨迹。

---

## 10. 文件级改造清单

| 文件 | 改造内容 |
|---|---|
| `proposed.py` | 抽取单 DNN customized 搜索内核；评价函数接收显式快照；返回多候选档案和诊断；原在线基线调用共享内核 |
| `planning/periodic_planner.py` | 默认调用 customized 生成器；传入规划快照、搜索配置和输出 `K_g`；支持显式 lightweight/fallback 模式 |
| `planning/repository.py` | 为方案增加生成器来源和搜索追踪字段；保持发布原子性与 TTL 语义 |
| `planning/config.py` 或现有配置模块 | 增加 `CustomizedSearchConfig`、生成器模式和版本 |
| `planning/semantics.py` 或语义版本定义处 | 增加 `candidate_generator_version`，避免只修改代码而不使旧方案失效 |
| 周期实验配置与 CLI | 暴露生成器模式、种群、代数、评价预算和 fallback 开关；默认 customized |
| 结果记录模块 | 输出搜索开销、候选质量、档案规模、fallback 次数和发布复验结果 |
| `tests/` | 增加无副作用、快照一致性、确定性、内核复用、规划集成和 prewarm 测试 |

若新增独立模块，建议使用 `planning/customized_generator.py` 封装配置、结果和适配器；搜索算子本身仍保留在 `AllDNNRefactor` 或逐步迁移，避免一次重构范围过大。

---

## 11. 分阶段执行顺序

### 阶段 0：冻结基线

- 记录当前 `run_customized_proposed()` 的固定种子输出和关键指标；
- 记录当前周期 lightweight 生成器输出；
- 确认现有测试全量通过；
- 将现有结果作为重构回归基线，不修改目标公式。

完成标准：能够识别“算法重构引入的变化”和“原随机搜索本身的波动”。

### 阶段 1：定义配置、结果和语义

- 新增 `CustomizedSearchConfig`；
- 新增候选和搜索结果类型；
- 增加 `candidate_generator_mode/version`；
- 补充结果字段，但暂不切换规划器默认路径。

完成标准：配置可序列化，旧实验配置具有明确默认值。

### 阶段 2：绑定规划快照

- 为评价链路增加显式 `SearchEvaluationContext`；
- 修改候选状态预测缓存键；
- 消除搜索期间隐式捕获 live/online 快照的路径；
- 保证修复、局部搜索和最终重评使用同一快照。

完成标准：搜索启动后修改环境，不改变该次搜索在固定种子下的结果。

### 阶段 3：抽取 customized 搜索内核

- 从 `run_customized_proposed()` 抽取单 DNN 搜索循环；
- 保留原初始化、DAG 交叉、约束支配、精英和局部搜索；
- 增加跨代可行非支配档案；
- 返回多候选及诊断；
- 让原 `run_customized_proposed()` 改调共享内核。

完成标准：原基线仍可完成单解选择和环境提交，搜索内核自身没有环境副作用。

### 阶段 4：接入周期规划器

- 将 `_search_at_snapshot()` 默认生成器切换为 customized；
- 将 `K_g` 仅用于档案末端多样性筛选；
- 保留上一版方案热启动逻辑；
- 实现空档案处理和显式 fallback；
- prewarm 与普通周期规划走相同生成器入口。

完成标准：周期规划生成的 `DeploymentPlan` 可追踪到 customized 搜索任务，在线路径不运行进化搜索。

### 阶段 5：规划耗时和结果闭环

- 用实际 customized 墙钟时间校准规划时长；
- 规划任务在 ready slot 前不可发布；
- 发布时继续对最新状态执行复验；
- 记录搜索预算、候选质量、过期率和 fallback；
- 输出生成器版本以保证旧池失效。

完成标准：算法开销不再被计入毫秒级推理 delay，但会作为后台规划时效性和系统性能成本独立报告。

### 阶段 6：实验入口和文档

- 在统一 `--suite all` 入口中加入生成器参数；
- 增加 `customized` 对 `lightweight` 消融；
- 增加逐请求 customized 对周期 customized 的架构对照；
- 更新系统模型、算法流程和实验说明文档。

完成标准：正式运行命令无需手工修改源码，结果中可以确认实际使用的生成器。

---

## 12. 测试计划

### 12.1 单元测试

1. **无环境副作用**：调用搜索内核前后，时隙、资源、运行请求和环境版本不变；
2. **快照一致性**：搜索中途改变 live 环境，固定快照结果不变；
3. **确定性**：相同输入和种子返回相同 assignment、目标和顺序；
4. **严格可行**：输出候选全部通过统一候选预测内核的硬约束；
5. **Pareto 档案**：输出中不存在重复 assignment 和明显被支配候选；
6. **多样性**：有足够可行解时，候选池不退化为同一部署结构；
7. **预算终止**：达到评价或时间预算后返回当前档案并标记原因；
8. **RNG 隔离**：调用搜索内核不改变外部全局随机序列。

### 12.2 回归测试

1. 原 `run_customized_proposed()` 仍可执行并提交方案；
2. 在相同配置和种子下，原入口与共享内核的候选质量保持一致；
3. lightweight 模式保持可用；
4. 原目标计算、delay 迭代和 OSS 结果不因结构重构发生非预期变化。

### 12.3 集成测试

1. 周期规划默认 `generator_mode=customized`；
2. prewarm 生成初始池但不推进正式时钟；
3. 后台任务运行期间在线请求仍只执行快速复验；
4. 规划完成后基于最新状态发布复验；
5. 拓扑/语义版本变化使旧候选失效；
6. customized 无可行档案时按配置保留旧池或触发带标签 fallback；
7. 多请求同槽接纳仍逐次递增环境版本并重新评价。

### 12.4 验证命令

每阶段至少执行：

```bash
python -m pytest tests -q
python run_periodic_experiments.py --suite smoke
```

阶段 6 完成后执行统一正式入口。实际脚本名与参数以当前批量入口为准，不在实现前硬编码另一套入口。

---

## 13. 实验与投稿口径

完成改造后，主方法应表述为：

> 在慢时间尺度上，周期规划器基于可观测系统快照运行原 customized 多目标约束搜索，为各 DNN 画像和接入组维护多样化 Pareto 部署候选；在快时间尺度上，请求仅对当前候选执行最新状态复验、偏好选择和原子接纳。

后续实验至少区分：

1. `Per-request Customized`：每请求执行原 customized 搜索；
2. `Periodic Customized`：本文主方法；
3. `Periodic Lightweight`：候选生成器消融；
4. `Periodic Customized without joint control`：固定 `T/K/B`；
5. `Periodic Customized with joint control`：完整方法。

公平比较时应固定请求轨迹、节点退化轨迹和随机种子，并同时报告：

- 请求成功率、吞吐、deadline violation；
- delay、OSS、availability、IFS、energy；
- 在线决策时延；
- 后台搜索墙钟开销和规划新鲜度；
- 单位成功请求的总计算开销；
- 候选池命中率、复验淘汰率和 fallback 率。

这样可以证明改造目标不是用低质量候选换取速度，而是将原有高开销优化从请求关键路径迁移到可控的后台慢路径。

---

## 14. 风险与处理

| 风险 | 处理方式 |
|---|---|
| customized 搜索超过规划周期 | 任务不重入；限制评价预算；返回当前档案；联合控制器拉长周期或缩小覆盖范围 |
| 单 DNN 抽取改变原算法行为 | 先冻结种子基线；原入口与周期入口共享内核；逐阶段回归 |
| 搜索读取变化中的 live 状态 | 全链路显式传入不可变 planning snapshot；禁止内部重抓快照 |
| 候选档案全部过期 | 发布复验；保留上一有效池；提前触发重规划；记录空池时间 |
| 方案池候选同质化 | 跨代档案、assignment 去重和目标/结构联合多样性选择 |
| fallback 掩盖 customized 失败 | 候选和结果显式标记来源；正式实验分别统计 |
| prewarm 获得未来信息 | 只读初始可观测快照和静态画像，不读未来请求与故障轨迹 |
| 计算开销被错误计入推理 delay | 推理 delay 与后台 planning runtime 分开建模、分开报告 |

---

## 15. 最终验收条件

以下条件全部满足后，本方案才可从 `pending` 移出：

- [x] 周期规划默认使用原 customized 搜索核心生成候选；
- [x] 原在线 customized 基线和周期规划共享同一搜索内核；
- [x] 搜索内核不提交资源、不推进时隙且不修改全局 RNG；
- [x] 一次搜索的全部评价严格绑定同一 planning snapshot；
- [x] 输出来自跨代严格可行非支配档案，而非简单随机候选；
- [x] `K_g` 与搜索种群/代数预算语义分离；
- [x] prewarm 使用 customized 且无未来信息泄漏；
- [x] 规划耗时、评价次数、档案规模和生成器版本可追踪；
- [x] lightweight 仅作为显式回退或消融模式；
- [x] 发布复验、在线 Top-K 复验和原子接纳行为保持不变；
- [x] 单元、回归和周期集成测试全部通过；
- [x] 实验入口能够明确选择并记录候选生成器。

---

## 16. 推荐的首个实施切片

首个代码切片只完成以下闭环：

1. 定义 `CustomizedSearchConfig/Result`；
2. 让 `_candidate_state_for_assignment()` 可显式使用给定快照；
3. 从 `run_customized_proposed()` 抽取单 DNN、无提交的搜索入口；
4. 用固定种子验证原入口与新内核；
5. 暂不切换周期规划器默认路径。

该切片通过后，再接入 `PeriodicPlanner`。这样可以先证明“原算法已被无损抽取”，再处理异步规划、候选池和发布语义，降低一次性改造风险。
