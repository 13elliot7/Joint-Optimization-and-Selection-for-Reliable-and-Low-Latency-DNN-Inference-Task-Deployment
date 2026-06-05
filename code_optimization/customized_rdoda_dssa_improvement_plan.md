# 面向 DNN 拓扑与边缘网络特性的定制化 RDODA-DSSA 改进方案

## 1. 目标

在保留现有 `proposed` 算法作为基线的前提下，设计一个新的改进算法，用于回应“当前 RDODA 本质上是经典 NSGA-II 直接应用，缺乏针对 DNN 拓扑依赖和边缘网络特性的定制化设计”的问题。

本方案遵循以下约束：

- 保留当前 `proposed` 作为 baseline，不破坏其现有实验口径。
- 新算法仍采用 `RDODA + DSSA` 两阶段框架。
- `RDODA` 阶段仍负责三目标 Pareto 搜索：
  - 动态精度可靠性
  - 动态运行可靠性
  - 时延收益
- `DSSA` 阶段再加入能耗偏好：
  - 能耗不进入 RDODA 的非支配排序
  - 能耗只作为最终候选方案筛选时的偏好项
- 新增的定制化设计主要体现在：
  - DAG 拓扑感知初始化
  - DAG/层级约束修复
  - 边缘资源感知偏斜变异
  - 精英局部邻域搜索

## 2. 当前 baseline 保留范围

现有 `proposed` 算法保留为对照组，建议命名为：

```text
baseline: RDODA-DSSA
improved: C-RDODA-DSSA
```

其中 `C` 表示 customized。

baseline 保留以下逻辑：

- 当前一维部署染色体：

```text
assignment[i] = node_index
```

- 当前随机初始化逻辑
- 当前 `cross_over1()` 后缀交换逻辑
- 当前 `_mutate_assignment()` 随机变异逻辑
- 当前三目标非支配排序
- 当前 DSSA 加权最终选择

新算法建议新增独立入口，而不是直接覆盖 `run_proposed()`：

```python
run_customized_proposed()
```

这样后续实验可以直接比较：

```text
random
max_resource
local_first
baseline RDODA-DSSA
customized RDODA-DSSA
```

## 3. 新算法总体流程

单个 DNN 的改进搜索流程如下：

```text
输入: 当前 DNN, 当前边缘网络动态状态

1. 构建当前 DNN 的拓扑辅助信息
   - 拓扑序
   - 前驱集合
   - 后继集合
   - 关键路径任务集合
   - 每个任务的合法候选节点集合

2. 定制化初始化种群
   - 全云解
   - 发起端/边缘优先解
   - 资源优先解
   - 可靠性优先解
   - DAG 拓扑连续性启发式解
   - 少量随机可行解

3. 每代进化
   - 选择父代
   - DAG 块感知交叉
   - 边缘资源感知偏斜变异
   - 约束感知修复
   - 三目标评价
   - 三目标 Pareto 非支配排序
   - 拥挤距离更新父代
   - 对少量精英执行局部邻域搜索

4. DSSA 最终选择
   - 在 RDODA 产生的 Pareto 候选中
   - 根据用户期望和实际边界计算三目标权重
   - 加入能耗偏好项
   - 选择最终部署方案
```

## 4. 设计一：DAG 拓扑辅助信息

### 4.1 动机

当前代码中的 DNN 已经是 DAG，而不是线性链式网络。因此不适合直接照搬线性 DNN 的 partition chromosome。

更合理的做法是保留当前一维部署向量，但在初始化、交叉、变异和修复中引入 DAG 拓扑信息。

### 4.2 新增辅助结构

建议为每个 DNN 构建：

```text
topological_order: List[int]
predecessors[task_idx]: List[int]
successors[task_idx]: List[int]
critical_path_tasks: Set[int]
candidate_nodes[task_idx]: List[int]
```

候选节点集合需要过滤掉明显非法节点：

- CPU 不足的节点
- 非发起端用户节点
- 在层级约束下明显不可行的节点

对应代码可新增在 `AllDNNRefactor` 中：

```python
def _build_dnn_context(self, dnn_index: int) -> DNNContext:
    ...
```

## 5. 设计二：DAG 拓扑感知初始化

### 5.1 当前问题

当前首代初始化以随机采样为主，可能产生大量结构上不合理的部署：

- 相邻依赖任务反复跨节点传输
- 关键路径被放到低算力节点
- 分支任务没有利用并行部署机会
- 初始化阶段耗时不稳定

### 5.2 改进方案

初始化种群由启发式种子和随机个体混合构成。

建议比例：

```text
10% 全云/云优先种子
20% 发起端附近边缘优先种子
20% 最大资源优先种子
20% 可靠性优先种子
20% DAG 拓扑连续性种子
10% 随机可行种子
```

### 5.3 DAG 拓扑连续性种子

对拓扑序中的任务按依赖关系部署：

```text
如果 task 是关键路径任务:
    优先选择高算力、高可靠、低链路代价节点
否则:
    优先选择与其前驱部署节点通信代价低的节点
```

这样可以减少无意义的数据往返，同时保留 DAG 分支并行部署的可能。

## 6. 设计三：DAG/层级约束感知修复

### 6.1 动机

当前 `_is_assignment_valid()` 只检查：

- CPU 资源约束
- 非发起端用户节点不可部署

但没有显式处理边云端层级流向。如果希望体现云-边-端架构特性，可以引入约束修复机制。

### 6.2 层级约束

定义节点层级：

```text
user = 1
edge = 2
cloud = 3
```

对依赖边 `u -> v`，可以设置软约束：

```text
level(assignment[v]) >= level(assignment[u]) - rollback_tolerance
```

第一版建议设置：

```text
rollback_tolerance = 0
```

即数据沿 DAG 推理过程不从高层级回流到低层级。

如果必须保留结果回传到发起端，则结果回传只作为输出阶段，不参与 DAG 内部任务层级约束。

### 6.3 修复流程

每次交叉和变异后，对子代执行：

```text
for task in topological_order:
    for pred in predecessors[task]:
        if level(task_node) < level(pred_node):
            将 task 修复到同级或更高级候选节点
    如果 CPU 约束违反:
        将过载节点上的部分非关键任务迁移到可行候选节点
    如果仍不可行:
        回退到父代或执行一次低成本重采样
```

### 6.4 约束处理口径

为了保留 baseline 的三目标 RDODA，不建议把能耗约束放进非支配排序。

但可以为时延和资源引入约束违反度：

```text
CV = max(0, delay - deadline) / deadline
   + resource_violation_ratio
   + hierarchy_violation_ratio
```

如果后续需要增强选择压力，可在同一 Pareto 层内优先保留 CV 小的个体。

第一版可以先只做 repair，不改 `dominated_sort()`。

## 7. 设计四：边缘资源感知偏斜变异

### 7.1 当前问题

当前 `_mutate_assignment()` 是随机选择一个任务，再随机选择一个节点。该过程没有利用边缘网络状态，容易把任务变异到：

- CPU 紧张节点
- 高热度节点
- 低可靠节点
- 与前后继通信代价高的节点
- 能耗明显偏高的节点

### 7.2 改进方案

将随机变异替换为 skew mutation。

对被选中的任务 `i`，候选节点 `j` 的采样权重定义为：

```text
score(i,j) =
    w_cpu  * cpu_score(j)
  + w_time * exec_time_score(i,j)
  + w_link * link_score(i,j)
  + w_rel  * reliability_score(j)
  + w_heat * heat_score(j)
  + w_e    * energy_score(i,j)
```

其中各项均归一化到 `[0,1]`，并保持“越大越好”。

建议第一版权重：

```text
w_cpu  = 0.20
w_time = 0.20
w_link = 0.20
w_rel  = 0.20
w_heat = 0.10
w_e    = 0.10
```

注意：这里的能耗只影响变异方向，不进入 RDODA 的 Pareto 支配关系。因此仍满足“RDODA 三目标 Pareto，DSSA 加入能耗偏好”的主设定。

### 7.3 链路评分

对任务 `i`，链路评分由其前驱和后继决定：

```text
link_cost(i,j) =
    Σ pred_data / bandwidth(node(pred), j)
  + Σ succ_data / bandwidth(j, node(succ))
```

然后转成：

```text
link_score = 1 / (1 + link_cost)
```

## 8. 设计五：DAG 块感知交叉

### 8.1 当前问题

当前 `cross_over1()` 使用后缀交换。对于 DAG 来说，拓扑序后缀不一定对应结构内聚的子图，交叉后可能打散紧密依赖任务。

### 8.2 第一版轻量改造

不引入复杂的双层染色体，只把交叉点从“随机下标”改为“DAG 边界下标”。

可选边界包括：

- 入度/出度发生明显变化的位置
- 关键路径节点附近
- 分支汇合节点之前或之后

仍保持一维 assignment 编码：

```text
child_a[block] = parent_b[block]
child_b[block] = parent_a[block]
```

交叉后统一执行 repair。

### 8.3 第二版增强

如果第一版实验有效，再加入 macro-module：

```text
module = 一组拓扑相邻且依赖紧密的 task
```

进化算子优先以 module 为单位交叉和变异，减少搜索空间。

## 9. 设计六：精英局部邻域搜索

### 9.1 动机

GA 的全局搜索能力较强，但收敛后常常需要局部微调。当前算法没有针对精英个体的局部搜索。

### 9.2 改进方案

每一代完成 Pareto 更新后，选择少量精英个体：

```text
elite_count = max(1, pop_size * 5%)
```

对每个精英个体执行邻域搜索：

```text
for task in critical_path_tasks first, then other tasks:
    尝试将 task 移动到 top-k 候选节点
    执行 repair
    如果三目标 Pareto 不劣，且 DSSA 临时分数更优:
        接受该邻域解
```

第一版只对关键路径任务做局部搜索，避免成本过高。

## 10. DSSA 阶段的能耗偏好保留方案

### 10.1 RDODA 阶段不变

RDODA 的非支配排序继续只比较：

```text
f1 = dynamic_accuracy
f2 = dynamic_operation_reliability
f3 = delay_utility
```

能耗不参与：

- `dominated_sort()`
- `get_res()` 的拥挤距离
- Pareto rank

### 10.2 DSSA 最终选择加入能耗

在最终候选集中，DSSA 使用四项打分：

```text
score =
    w_a * norm(accuracy)
  + w_r * norm(operation_reliability)
  + w_t * norm(delay)
  + w_e * norm(energy_utility)
```

其中：

```text
energy_utility = 1 / total_energy
```

第一版保持当前实现中的固定能耗权重：

```text
w_e = 0.2
```

其余权重由用户期望和边界动态归一化得到。

第二版可以将 `w_e` 改成由应用能耗期望动态生成。

## 11. 推荐代码落点

建议新增方法，而不是直接覆盖 baseline。

### 11.1 `proposed.py`

新增：

```python
def run_customized_proposed(self) -> ExperimentMetrics:
    ...

def _build_dnn_context(self, dnn_index: int) -> DNNContext:
    ...

def _initialize_customized_population(self, dnn_index: int, context: DNNContext) -> None:
    ...

def _repair_assignment(self, dnn_index: int, assignment: List[int], context: DNNContext) -> List[int]:
    ...

def _skew_mutate_assignment(self, dnn_index: int, assignment: List[int], context: DNNContext) -> List[int]:
    ...

def _dag_block_crossover(self, dnn_index: int, parent_a: List[int], parent_b: List[int], context: DNNContext) -> tuple[List[int], List[int]]:
    ...

def _local_search_elites(self, dnn_index: int, context: DNNContext) -> None:
    ...
```

### 11.2 `runner.py`

新增命令行算法选项：

```text
customized
```

并映射到：

```python
algorithm.run_customized_proposed()
```

### 11.3 `metrics.py`

如果当前指标已经包含 `avg_energy`，无需修改。

如果实验输出中仍未打印能耗，需要补充：

```text
avg_energy
```

## 12. 实验对比设计

建议至少比较以下算法：

```text
Random
MaxResource
LocalFirst
RDODA-DSSA baseline
C-RDODA-DSSA improved
```

对比指标：

- 平均估计时延
- 平均动态运行可靠性
- 平均动态精度可靠性
- 平均能耗
- 失败/拒绝数量
- 运行时间

建议额外记录：

- 平均每个 DNN 搜索耗时
- 首代初始化耗时
- repair 次数
- skew mutation 接受次数
- local search 改善次数

这些指标可以证明改进不只是“效果更好”，还体现了定制化机制实际发挥作用。

## 13. 预期效果

相比 baseline，改进算法预期具有以下优势：

1. 搜索空间更小  
   DAG 拓扑初始化和块感知交叉减少明显不合理的部署组合。

2. 可行解比例更高  
   repair 机制减少交叉和变异后直接废弃子代的情况。

3. 收敛速度更稳定  
   skew mutation 利用 CPU、链路、可靠性、热度和能耗信息指导搜索。

4. 更贴合边缘网络状态  
   变异和局部搜索能够避开高负载、高热度、低可靠节点。

5. 论文创新点更清晰  
   改进点不再只是“把 NSGA-II 套上去”，而是体现 DNN DAG 与边缘网络的联合定制。

## 14. 实施优先级

建议分三步实施。

### 第一阶段：低风险增强

- 新增 `run_customized_proposed()`
- 复制 baseline 主流程
- 加入定制化初始化
- 加入 skew mutation
- 保留原有交叉和三目标 Pareto

### 第二阶段：结构约束增强

- 加入 DAG context
- 加入 repair
- 加入 DAG 边界交叉
- 记录 repair 和 mutation 统计

### 第三阶段：性能与创新增强

- 加入关键路径识别
- 加入精英局部搜索
- 尝试 macro-module 分组
- 优化评价缓存

第一阶段即可形成一个可运行的改进算法版本；第二阶段开始能体现较强的“定制化算法设计”；第三阶段适合用于最终论文实验强化。
