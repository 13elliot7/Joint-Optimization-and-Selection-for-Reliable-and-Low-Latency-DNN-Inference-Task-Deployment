# Customized Proposed 单机大规模实验优化实施方案

## 1. 文档目标

本文面向当前 `customized proposed` 算法，给出一套可落地的性能优化和单机大规模实验支持方案。

优化目标不是改变算法定义，而是在保持实验口径和搜索行为可对比的前提下：

- 显著降低单个 DNN、单代搜索耗时。
- 支持大量随机种子、参数组合和算法对照实验。
- 保证实验可复现、可恢复、可追踪。
- 控制单机内存占用和日志 I/O。
- 为后续算法优化提供稳定的性能基准。

涉及的主要文件：

- `proposed.py`
- `core/environment.py`
- `core/topology.py`
- `core/dag_generator.py`
- `runner.py`
- 建议新增 `experiment_runner.py`

## 2. 当前性能基线

使用以下命令对当前实现进行最小规模剖析：

```bash
python -m cProfile -o /tmp/customized_profile.prof \
  runner.py --algorithm customized --tmax 1 --iteration-limit 1
```

本次剖析结果：

| 热点函数 | 调用次数 | 累计时间 | 说明 |
|---|---:|---:|---|
| `Environment.get_shortest_path()` | 4,102 | 9.97s | 几乎占据全部运行时间 |
| `Environment.get_arrive_link()` | 18,577 | 9.99s | 反复触发最短路径计算 |
| `AllDNNRefactor._evaluate_assignment()` | 277 | 7.11s | 同一代中大量重复评价 |
| `Environment.count_delay_random()` | 397 | 4.09s | 递归计算 DAG 时延 |
| `Environment._collect_link_weights()` | 566 | 4.56s | 反复查询相同路径 |
| `AllDNNRefactor._local_search_elites()` | 1 | 2.33s | 每代重新评价父代和邻域 |
| `dominated_sort_with_cv()` | 1 | 0.004s | 当前不是主要瓶颈 |

一次 DNN、一次迭代耗时约 10 秒。若简单线性估算默认实验：

```text
40 DNN × 120 代 × 约 10 秒/代 ≈ 13.3 小时/次实验
```

实际耗时受 DNN 规模、局部搜索候选数、动态环境状态影响，但当前实现无法有效支撑大规模重复实验。

## 3. 当前瓶颈优先级

### P0：最短路径重复计算

当前 `get_arrive_link()` 每次都会调用 `get_shortest_path()`。网络拓扑在搜索过程中没有改变，但相同节点对的路径仍被计算数千次。

`get_shortest_path()` 当前使用多重循环松弛，单次调用成本较高；随后还会遍历全部链路，将节点路径转换为链路序列。

### P0：候选方案重复评价

每代联合种群评价后，局部搜索又重新评价全部父代。约束违反度计算还可能再次估计相同方案的时延。

当前同一个 assignment 可能重复执行：

- 动态精度可靠性计算
- 动态运行可靠性计算
- 链路收集
- 能耗计算
- 时延计算
- 约束违反度计算

### P0：旧时延模型递归重复遍历 DAG

`count_delay_random()` 通过递归 `count_tran_random()` 计算关键路径时延。存在以下重复工作：

- 相同子路径被递归计算多次。
- 每次递归扫描 DNN 链路。
- 每次查找任务索引。
- 每次扫描 one-hot 矩阵寻找任务部署节点。
- 每次任务间传输都重新查询网络最短路径。

### P1：局部搜索每代成本过高

`_local_search_elites()` 每代重新评价全部父代，并对精英个体的关键路径任务尝试多个候选节点。局部搜索产生的评价次数可能超过联合种群评价次数。

### P1：种群结构随 `t_max` 扩张

当前种群结构为：

```text
population_size × len(env.ds) × max_dnn_num
```

但搜索时只使用当前 DNN 对应的一层。`t_max` 增大后，每代都会分配和复制大量未使用数据。

### P2：批量实验基础能力不足

当前缺少：

- 独立随机种子管理
- 静默运行模式
- 结构化结果输出
- 多进程批量执行
- 断点续跑
- 失败任务重试
- 场景复用与公平算法比较

## 4. 优化原则

### 4.1 保持算法口径

第一阶段优化必须保证：

- 候选方案的可靠性、能耗和时延计算语义不变。
- 相同场景、相同随机种子下，优化前后评价结果在浮点容差内一致。
- 不直接修改 `proposed` baseline 的搜索行为。
- customized 的搜索算子和最终接纳规则保持不变。

### 4.2 优先减少重复工作

当前主要问题不是 Python 单行代码执行速度，而是相同路径和相同 assignment 被重复计算。因此优先采用：

- 预计算
- 缓存
- 动态规划
- 数据结构收缩

不建议第一步就引入 NumPy、Cython 或复杂并行。

### 4.3 实验级并行优先

算法运行属于 CPU 密集型任务。单机并行应优先并行不同实验任务，而不是在单个进化循环内部并行候选评价。

推荐：

```text
一个进程 = 一个完整实验任务
```

这样可以避免共享动态环境、缓存同步和候选序列化成本。

## 5. 阶段一：路径预计算

### 5.1 目标

将搜索循环内的 `get_shortest_path()` 调用次数降为 0。

### 5.2 修改位置

主要修改 `core/environment.py`：

- `Environment.__post_init__()`
- `Environment.get_arrive_link()`
- `Environment.get_shortest_path()`
- `_collect_link_weights()`
- 可新增 `_build_path_cache()`

### 5.3 新增缓存

环境初始化后，对所有节点对预计算：

```python
self.path_links: dict[tuple[int, int], tuple[LinkNode, ...]]
self.path_delay_factor: dict[tuple[int, int], float]
self.path_energy_factor: dict[tuple[int, int], float]
```

含义：

```text
path_delay_factor[s, e] = Σ(1 / link.band_width)
path_energy_factor[s, e] = Σ(link.energy_per_mb)
```

之后可直接计算：

```text
传输时延 = data_amount × path_delay_factor[s, e]
传输能耗 = data_amount × path_energy_factor[s, e]
```

### 5.4 推荐实现

节点数量当前约为 30，适合在初始化阶段执行全节点对预计算。可以保留现有最短路径算法，以降低行为变化风险：

```python
def _build_path_cache(self) -> None:
    for start in range(len(self.nodes)):
        for end in range(len(self.nodes)):
            links = self._compute_arrive_links_uncached(start, end)
            self.path_links[(start, end)] = tuple(links)
            self.path_delay_factor[(start, end)] = sum(
                1.0 / max(float(link.band_width), 1.0)
                for link in links
            )
            self.path_energy_factor[(start, end)] = sum(
                link.energy_per_mb
                for link in links
            )
```

`get_arrive_link()` 改为只读缓存。

### 5.5 验收标准

- 搜索阶段 `get_shortest_path()` 调用次数为 0。
- 对随机生成的节点对，缓存路径与原始路径完全一致。
- 优化前后相同 assignment 的时延和能耗结果一致。
- 单 DNN、单代运行时间显著降低。

## 6. 阶段二：统一候选评价与缓存

### 6.1 目标

确保同一 DNN、同一动态环境状态下，相同 assignment 最多完整评价一次。

### 6.2 修改位置

主要修改 `proposed.py`：

- `_evaluate_assignment()`
- `count_values()`
- `_constraint_violation()`
- `_local_search_score()`
- `_local_search_elites()`
- `run_customized_proposed()`

### 6.3 新增评价结果结构

建议新增：

```python
@dataclass(frozen=True)
class AssignmentEvaluation:
    accuracy: float
    operation: float
    delay_utility: float
    energy_utility: float
    delay: float
    total_energy: float
```

缓存键使用当前 DNN 的实际 assignment：

```python
cache_key = tuple(assignment[:task_count])
```

缓存生命周期：

```text
开始搜索当前 DNN：清空
当前 DNN 搜索期间：复用
advance_time_slot() 后：失效
```

### 6.4 合并评价调用

当前：

```python
self.count_values(t, m)
self.constraint_violations[m] = self._constraint_violation(t, assignment, context)
```

建议改为：

```python
evaluation = self._evaluate_assignment_cached(t, assignment)
self._write_evaluation(m, evaluation)
self.constraint_violations[m] = self._constraint_violation(
    t,
    assignment,
    context,
    delay=evaluation.delay,
)
```

必须将已经得到的 `delay` 传给 `_constraint_violation()`，避免再次计算时延。

### 6.5 局部搜索复用

环境选择后保留父代对应的：

- 目标值
- delay
- CV
- 综合得分

`_local_search_elites()` 不再重新评价全部父代，只评价真正生成的新邻域候选。

### 6.6 验收标准

- 相同 assignment 在当前 DNN 搜索阶段最多完整评价一次。
- `_constraint_violation()` 不再重复计算已有 delay。
- 局部搜索不再为筛选精英重新评价全部父代。
- 缓存命中率、唯一评价次数可通过 debug 统计输出。

## 7. 阶段三：DAG 动态规划时延评价

### 7.1 目标

使用一维 assignment 和拓扑序，将时延评价复杂度降低为：

```text
O(task_count + edge_count)
```

### 7.2 修改位置

主要修改：

- `proposed.py:_build_dnn_context()`
- `core/environment.py:estimate_delay_from_assignment()`

可为环境层新增轻量时延上下文，或将现有 `DNNContext` 中的拓扑信息传入评价函数。

### 7.3 计算方式

预计算或直接读取：

```text
execution_time[task] = task.float_num / assigned_node.float_rate
transfer_time[pred, task] =
    edge_data[pred, task] × path_delay_factor[pred_node, task_node]
```

按拓扑序计算：

```python
finish_time[task] = execution_time[task]
if predecessors[task]:
    finish_time[task] += max(
        finish_time[pred] + transfer_time[pred, task]
        for pred in predecessors[task]
    )
```

最终时延：

```text
输入上传时延
+ 最终任务完成时间
+ 结果回传时延
```

### 7.4 兼容性验证

在正式替换旧模型前，保留双实现：

```python
legacy_delay = estimate_delay_legacy(...)
fast_delay = estimate_delay_fast(...)
```

对至少 1,000 个随机 assignment 验证：

```text
abs(legacy_delay - fast_delay) <= 1e-9
```

若旧递归模型存在特殊语义或错误，应先明确实验口径，再决定保持兼容还是修正。

### 7.5 验收标准

- 时延评价不再构建 one-hot 矩阵。
- 时延评价不再递归调用 `count_tran_random()`。
- 随机 assignment 回归测试通过。
- 单次候选评价耗时稳定随 DAG 规模线性增长。

## 8. 阶段四：控制局部搜索预算

### 8.1 目标

保留关键路径局部搜索能力，同时限制其最坏运行成本。

### 8.2 建议参数

新增：

```python
self.local_search_interval = 5
self.local_search_evaluation_budget = 30
self.local_search_stagnation_limit = 3
```

执行条件：

```python
if generation % self.local_search_interval == 0:
    run_local_search(...)
```

### 8.3 搜索约束

每代局部搜索最多评价固定数量的新候选：

```text
evaluation_budget = 30
```

某个精英连续若干任务无改进后，提前结束该个体的邻域搜索。

优先对以下个体执行：

- 第一 Pareto 层个体
- 本代新进入父代的个体
- CV 较低且综合得分较高的个体

### 8.4 实验验证

需要比较以下配置：

```text
local_search_interval = 1 / 5 / 10
local_search_budget = 10 / 30 / 60
```

评价：

- 最终目标值
- 接纳率
- 平均运行时间
- 单位时间内得到的最好解

## 9. 阶段五：收缩种群数据结构

### 9.1 目标

使单代内存和复制成本只与当前 DNN 任务数有关，不随 `t_max` 增长。

### 9.2 当前结构

```python
res[population][dnn_index][task_index]
```

当前搜索只处理一个 DNN，因此中间的 `dnn_index` 维度没有必要。

### 9.3 建议结构

```python
parents: list[list[int]]
children: list[list[int]]
combined: list[list[int]]
```

维度：

```text
population_size × current_task_count
```

联合种群可以直接构造：

```python
combined = parents + children
```

同步将以下结构改为当前代一维数组：

```python
pareto_rank: list[int]
crowding_distance: list[float]
constraint_violations: list[float]
evaluations: list[AssignmentEvaluation]
```

### 9.4 迁移策略

该改动范围较大，建议：

1. 先仅迁移 `run_customized_proposed()`。
2. 保留 `run_proposed()` 使用旧三维结构，确保 baseline 不变。
3. 为 customized 新增独立辅助函数，避免同时修改两套算法。

### 9.5 验收标准

- 单代种群分配量不随 `t_max` 增长。
- 删除父代复制的三层循环。
- `tmax=1` 和 `tmax=1000` 时，单代搜索内存基本一致。

## 10. 阶段六：单机批量实验框架

### 10.1 实验任务定义

每个实验任务应由以下字段唯一标识：

```text
algorithm
scenario_seed
algorithm_seed
tmax
population_size
iteration_limit
local_search_interval
local_search_budget
其他算法参数
```

任务 ID 可以由配置字段生成稳定哈希。

### 10.2 随机种子隔离

当前全局使用 `random`，不利于公平比较和多进程实验。

建议拆分：

```python
scenario_rng = random.Random(scenario_seed)
algorithm_rng = random.Random(algorithm_seed)
```

- `scenario_rng`：节点、链路、DNN 请求生成。
- `algorithm_rng`：初始化、父代选择、交叉和变异。

相同 `scenario_seed` 下，不同算法必须使用同一环境场景。

### 10.3 场景持久化

为了保证公平比较，建议支持：

```text
generate scenario -> 保存 JSON/pickle -> 多个算法加载同一场景
```

场景至少包含：

- 节点属性
- 链路属性
- DNN 任务与依赖
- deadline、preA、preR
- 初始动态状态

优先建议使用 JSON 或明确版本的结构化格式。若使用 pickle，应记录代码版本且只加载可信文件。

### 10.4 新增批量运行入口

建议新增 `experiment_runner.py`，支持：

```bash
python experiment_runner.py \
  --algorithms proposed customized random \
  --seeds 1:100 \
  --tmax 40 \
  --iteration-limit 120 \
  --workers 8 \
  --output results/experiment.jsonl \
  --resume \
  --quiet
```

### 10.5 多进程模型

使用：

```python
concurrent.futures.ProcessPoolExecutor
```

原则：

- 一个进程运行一个完整实验任务。
- `workers` 默认建议为物理核心数减 1。
- 不在单个算法内部再创建进程池。
- 每个进程拥有独立 Environment、随机数生成器和缓存。

### 10.6 结果格式

推荐 JSONL，每完成一个任务立即追加一行：

```json
{
  "task_id": "stable-hash",
  "status": "success",
  "algorithm": "customized",
  "scenario_seed": 42,
  "algorithm_seed": 1001,
  "tmax": 40,
  "population_size": 60,
  "iteration_limit": 120,
  "avg_delay": 304.35,
  "avg_operation": 0.91,
  "avg_accuracy": 0.98,
  "avg_energy": 3868.31,
  "failure_count": 2,
  "runtime_ms": 120000
}
```

JSONL 优点：

- 可流式写入。
- 单条失败不破坏整个文件。
- 易于断点续跑。
- 后续可转换为 CSV 或 DataFrame。

### 10.7 断点续跑和失败重试

启动时读取现有结果，建立已完成 `task_id` 集合：

```text
已成功任务：跳过
失败任务：根据参数决定重试
不存在任务：执行
```

失败结果必须记录：

- 异常类型
- 异常消息
- 任务配置
- 随机种子
- 运行时长

## 11. 阶段七：日志与性能观测

### 11.1 日志级别

新增：

```bash
--log-level quiet
--log-level summary
--log-level debug
```

批量实验默认 `quiet`：

- 不打印每个 DNN 信息。
- 不打印每代日志。
- 不打印每个时隙摘要。
- 只记录最终结果和错误。

### 11.2 性能统计

建议每次实验额外记录：

```text
unique_assignment_evaluations
evaluation_cache_hits
local_search_evaluations
path_cache_hits
accepted_dnn_count
search_runtime_ms
slot_runtime_ms
```

这些统计能够判断优化是否真正降低计算量，而不只是偶然缩短运行时间。

### 11.3 基准脚本

建议新增固定基准：

```bash
python benchmark_customized.py \
  --scenario-seed 42 \
  --algorithm-seed 42 \
  --tmax 1 \
  --iteration-limit 1
```

每次性能改动后记录：

- 总运行时间
- 单代时间
- 唯一评价次数
- 每秒候选评价数
- 峰值内存

## 12. 测试方案

### 12.1 单元测试

需要覆盖：

- 缓存路径与原始路径一致。
- 路径时延因子正确。
- 路径能耗因子正确。
- 快速 DAG 时延与旧实现一致。
- 相同 assignment 缓存命中。
- 动态环境推进后评价缓存失效。
- 固定随机种子生成相同场景。

### 12.2 回归测试

固定场景和随机种子，比较优化前后：

```text
每个 assignment 的四目标值
constraint violation
最终选择 assignment
接纳/拒绝结果
ExperimentMetrics
```

浮点指标使用明确容差，例如：

```python
math.isclose(before, after, rel_tol=1e-9, abs_tol=1e-9)
```

### 12.3 性能测试

建议建立三档固定基准：

| 基准 | tmax | iteration_limit | 用途 |
|---|---:|---:|---|
| smoke | 1 | 1 | 快速验证 |
| medium | 5 | 20 | 日常性能回归 |
| large | 40 | 120 | 正式吞吐验证 |

## 13. 推荐实施顺序

### 第一批：低风险、高收益

1. 预计算所有节点对路径。
2. 增加 assignment 评价缓存。
3. 将已有 delay 传给 `_constraint_violation()`。
4. 局部搜索复用父代评价结果。
5. 增加 quiet 日志模式。

预期结果：

- 搜索阶段不再计算最短路径。
- 单 DNN、单代耗时从约 10 秒下降到明显低于 1 秒。

### 第二批：评价核心重构

1. 使用拓扑序动态规划计算时延。
2. 移除评价路径中的 one-hot 矩阵构建。
3. 合并可靠性、能耗、时延和 CV 评价。
4. 增加局部搜索执行间隔和评价预算。

预期结果：

- 单次 assignment 评价复杂度接近 `O(tasks + edges)`。
- 单 DNN、单代目标耗时低于 0.2 秒。

### 第三批：大规模实验支持

1. customized 种群改为当前 DNN 二维结构。
2. 增加随机种子隔离和场景持久化。
3. 新增多进程 `experiment_runner.py`。
4. 增加 JSONL 输出、断点续跑和失败重试。
5. 增加性能基准和运行统计。

预期结果：

- 单机可以稳定执行数百至数千个实验任务。
- 实验可复现、可暂停、可恢复。

## 14. 建议验收指标

| 指标 | 当前状态 | 第一阶段目标 | 最终目标 |
|---|---:|---:|---:|
| 单 DNN、单代时间 | 约 10s | 小于 1s | 小于 0.2s |
| 搜索阶段最短路径计算次数 | 4,102 次/代 | 0 | 0 |
| 同一 assignment 完整重复评价 | 大量存在 | 每环境状态最多 1 次 | 每环境状态最多 1 次 |
| 局部搜索执行频率 | 每代 | 可配置 | 按预算自适应 |
| 单代内存与 `t_max` 的关系 | 线性增长 | 暂时保留 | 基本无关 |
| 随机种子支持 | 无 | 支持 | 场景与算法种子隔离 |
| 批量实验 | 手工串行 | 静默结构化输出 | 多进程、可恢复 |

## 15. 风险与注意事项

### 15.1 缓存失效

节点和链路动态可靠性会随时隙变化，因此完整 assignment 评价缓存不能跨 `advance_time_slot()` 使用。

路径本身和带宽当前不随时隙变化，所以路径、时延因子和能耗因子缓存可以跨时隙复用。如果未来引入动态带宽，需要同步设计缓存失效机制。

### 15.2 算法公平性

不同算法对同一场景进行比较时，必须使用独立 Environment 副本。不能只重置节点 CPU，而忽略：

- 链路动态状态
- 节点热度
- 动态可靠性
- 运行队列
- 随机数状态

### 15.3 并行资源竞争

多进程数量不宜直接设置为逻辑核心总数。正式实验前应测试：

- 物理核心数
- 内存占用
- 磁盘写入压力
- 单任务运行时间随 workers 数变化

推荐从 `workers = 物理核心数 - 1` 开始测试。

### 15.4 性能优化与算法修改分离

路径缓存、评价缓存和数据结构优化属于性能优化；局部搜索间隔、评价预算和提前停止会影响搜索行为。

正式实验中应分别记录：

```text
behavior_preserving_optimization
search_budget_optimization
```

避免将计算加速带来的收益与算法搜索预算变化混为一谈。

## 16. 最终目标架构

```text
ExperimentRunner
  ├── 读取实验任务配置
  ├── 跳过已完成任务
  ├── ProcessPoolExecutor
  │     └── 单进程运行一个 ExperimentTask
  │           ├── 加载固定 Scenario
  │           ├── 创建独立 Environment
  │           ├── 初始化路径缓存
  │           ├── 运行指定算法
  │           │     ├── 当前 DNN 二维种群
  │           │     ├── assignment 评价缓存
  │           │     ├── DAG 动态规划时延评价
  │           │     └── 有预算的局部搜索
  │           └── 返回指标与性能统计
  └── 流式写入 JSONL 结果
```

完成以上优化后，当前 customized proposed 才具备稳定开展单机大规模参数实验、消融实验和多随机种子对照实验的基础。
