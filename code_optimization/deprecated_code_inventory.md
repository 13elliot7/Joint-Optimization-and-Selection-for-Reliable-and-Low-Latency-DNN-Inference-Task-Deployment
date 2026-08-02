# 当前项目已弃用代码清单

> 审查范围：当前工作区内的 Python 源码、测试、实验导出与绘图调用  
> 当前目标语义：`stability_fidelity_v2_return_energy`  
> 审查日期：2026-08-02  
> 本文只整理弃用情况，不执行代码删除。

## 1. 判定口径

本文将代码分为四类：

| 状态 | 含义 | 处理原则 |
| --- | --- | --- |
| D0：可删除 | 定义之外没有仓内调用，且已被新接口完整替代 | 补充最小回归测试后删除 |
| D1：迁移后删除 | 语义已弃用，但测试、基线、导出或绘图仍在使用 | 先迁移调用方，再删除兼容层 |
| D2：整组淘汰 | 单个函数仍有调用，但调用链整体属于旧执行路径 | 按功能簇整体删除，不能零散删除 |
| K：有意保留 | 旧语义只用于诊断、结果隔离或兼容检查 | 不参与主目标，暂不删除 |

“仓内无调用”通过全仓静态引用检索判断。Python 可能存在反射调用或仓外使用者，因此删除公开方法前仍应完成一次运行入口和外部脚本确认。

## 2. 摘要

当前弃用代码主要来自五次演进：

1. Java 风格数据访问器迁移为 Python 属性；
2. `accuracy/reliability` 术语迁移为 IFS/OSS；
3. 三维部署矩阵迁移为一维 assignment；
4. RTBL 静态执行入口迁移为共享时隙环境下的 `run_dynamic()`；
5. 三目标/倒数能耗口径迁移为统一四目标满意度。

优先级建议：

- 第一批删除：无调用的 Java 风格 getter、孤立包装器和旧遗传算子；**已于 2026-08-02 完成**；
- 第二批删除：RTBL 静态运行链、三维矩阵评价函数和旧图矩阵查询支线；**已于 2026-08-02 完成**；
- 第三批迁移：`accuracy/operation/reliability` 字段、CSV 兼容列和绘图旧列；**已于 2026-08-02 完成**；
- 长期保留：`raw_joint_product` 诊断链和旧结果拒绝逻辑。

第三批迁移后，全量 41 项单元测试已通过。以下 D0/D1 清单作为已删除或已迁移记录保留。

## 3. D0：本轮已删除的仓内无调用代码

### 3.1 `models.py` 中无调用的 Java 风格访问器

以下方法原来只存在定义、没有仓内调用，现已删除；代码直接访问 dataclass 字段或使用新的语义属性。

| 类型 | 已删除方法 | 替代方式 |
| --- | --- | --- |
| `Node` | `getCpu()` | `node.cpu` |
| `Node` | `setCpu()` | `node.cpu = value` |
| `Node` | `getMaxCpu()` | `node.max_cpu` |
| `Node` | `getLevel()` | `node.level` |
| `Node` | `getoReliability()` | `node.operational_stability` |
| `Node` | `getaReliability()` | `node.inference_fidelity` |
| `Node` | `getFloatRate()` | `node.float_rate` |
| `Task` | `getCpuNeed()` | `task.cpu_need` |
| `Task` | `getFloatNum()` | `task.float_num` |
| `LinkNode` | `getsNode()` | `link.s_node` |
| `LinkNode` | `geteNode()` | `link.e_node` |
| `LinkNode` | `getBandWidth()` | `link.band_width` 或有效带宽接口 |
| `LinkDNN` | `getsTask()` | `link.s_task` |
| `LinkDNN` | `geteTask()` | `link.e_task` |
| `DNN` | `getTasks()` | `dnn.tasks` |
| `DNN` | `getLinks()` | `dnn.links` |

删除前位置：`models.py:75-99`、`models.py:126-130`、`models.py:258-266`、`models.py:290-298`、`models.py:332-340`。

注意：`DNN.getDelay()` 和 `LinkDNN.getFloatTran()` 目前仍被 RTBL 使用，不属于可直接删除项。

### 3.2 `core/environment.py` 中孤立的兼容包装器

| 方法 | 当前情况 | 新接口/建议 |
| --- | --- | --- |
| `_build_assignment_matrix()` | Environment 版本无调用；`proposed.py` 有独立同名实现 | 删除 Environment 版本 |
| `collect_used_links()` | 无调用，仅代理 `collect_used_directed_links()` | 直接使用新接口 |
| `count_energy_utility_by_assignment()` | 无调用，仍返回旧式 `1 / total_energy` | 使用 `total_energy` 与 `energy_satisfaction()` |
| `count_dynamic_operation_by_assignment()` | 无调用 | `count_operational_stability_by_assignment()` |
| `count_dynamic_accuracy_by_assignment()` | 无调用 | `count_inference_fidelity_by_assignment()` |
| `check_delay_by_assignment()` | 无调用的布尔包装器 | 调用 `estimate_delay_from_assignment()` 后显式比较 deadline |
| `predict_path_state()` | 无调用，仅把单条 transfer 包装成列表 | 直接调用 `predict_paths_state()` |
| `get_shortest_path()` | 无调用；当前运行使用初始化时缓存的 `_path_links` | 若无调试用途则删除 |
| `generate_task()` | 无调用的旧 RTBL 独立任务生成器 | 当前 DNN 请求统一由 `core/topology.py` 生成 |
| `generate_accuracy()` | 无调用的旧 RTBL 精度参数生成器 | 当前节点 IFS 来自拓扑节点先验 |

删除前位置：`core/environment.py:372-380`、`550-552`、`655-658`、`1008-1036`、`1085-1087`、`1318-1320`、`1455-1467`。

### 3.3 `proposed.py` 中已脱离运行路径的旧实现

| 项目 | 当前情况 | 替代实现 |
| --- | --- | --- |
| `self.gen` | 只在构造函数赋值，未读取 | `iteration_limit` |
| `self.res_next` | 多次分配但从未读取 | 当前子代直接写入联合种群结构 |
| `_membrane_range()` | 无调用 | 当前使用角色列表和显式配额/角色索引 |
| `mutate()` | 无调用的旧三维种群变异入口 | `_mutate_assignment()`、`_skew_mutate_assignment()` |
| `cross_over()` | 无调用的旧单点交叉入口 | `cross_over1()` 或 DAG block crossover |
| `count_values1()` | 无调用的单目标分步写入 | `count_values()` / `_evaluate_assignment()` |
| `count_values2()` | 同上 | `count_values()` / `_evaluate_assignment()` |
| `count_values3()` | 同上 | `count_values()` / `_evaluate_assignment()` |

删除前位置：`proposed.py:82`、`114`、`1061-1066`、`1416-1431`、`1464-1492`、`1494-1508`。

`res_next` 字段及 `run_proposed()`、`run_customized_proposed()` 中仅分配、不消费该容器的语句已同步移除。

### 3.4 `rtbl/scheduler.py` 中孤立成员

| 项目 | 当前情况 | 建议 |
| --- | --- | --- |
| `RTBLScheduler.reset_q()` | 无调用 | 删除，或仅在确有跨 episode 重置需求时纳入 `_reset_scheduler()` |
| `self.d_max` | 只由 `config.D_Max` 赋值，之后未读取 | 删除；当前 `step()` 使用 DNN 自身 deadline |

删除前位置：`rtbl/scheduler.py:26`、`51-53`。

## 4. D2：本轮已整组删除的旧功能链

### 4.1 RTBL 旧静态 `run()` 链

`RTBLRunner.run()` 没有被当前运行器、实验脚本或测试调用。当前可比基线入口是 `RTBLRunner.run_dynamic()`。

旧静态入口存在以下旧语义：

- 使用三维 `suiji[population][dnn][task]` 容器；
- 通过 `release_resource()` 永久扣减剩余 CPU；
- 分别调用 `count_values1_random()`、`count_values2_random()`、`count_values3_random()`；
- 未纳入当前运行队列、时隙推进、回传能耗和统一接纳后评价；
- 输出能耗固定为 `0.0`；
- 不应与当前动态实验结果比较。

已整体删除：

| 文件 | 删除对象 |
| --- | --- |
| `rtbl/scheduler.py` | `RTBLRunner.run()` |
| `core/environment.py` | `release_resource()` |
| `core/environment.py` | `count_values1_random()` |
| `core/environment.py` | `count_values2_random()` |
| `core/environment.py` | `count_values3_random()` |
| `core/environment.py` | `count_accuracy()` |
| `core/environment.py` | `count_operation()` |
| `core/environment.py` | `count_delay()` |
| `core/environment.py` | `count_tran()` |
| `core/environment.py` | `check_delay_random_3d()` |

其中 `count_tran_delay()` 原来同时被 `check_delay_random_3d()` 和旧三维 `count_tran()` 使用，现已随上述调用链删除。

### 4.2 旧图矩阵最短路查询链

当前 `_build_path_cache()` 直接基于链路列表构建邻接表并缓存全部路径。另一套 `graph_node` 矩阵只服务于 `_get_shortest_path_parents()` 的无 adjacency 回退，以及无调用的 `get_shortest_path()`。

当前运行时不再提供独立公开最短路查询，以下支线已经整体删除：

- `Environment.graph_node`；
- `get_graph_matrix()`；
- `_get_shortest_path_parents()` 的 `adjacency=None` 分支；
- `get_shortest_path()`。

保留：`_build_path_cache()`、显式 adjacency 调用和 `_path_links`。固定路由主模型不受影响。

## 5. D1：第三批已完成的语义迁移

### 5.1 节点的旧可靠性/准确性存储字段

| 旧字段 | 新语义字段 | 当前使用情况 |
| --- | --- | --- |
| `Node.o_reliability` | `Node.operational_stability` | 已迁移并删除旧字段 |
| `Node.a_reliability` | `Node.inference_fidelity` | 已迁移并删除旧字段 |
| `Node.base_o_reliability` | `Node.base_operational_stability` | 已迁移并删除旧字段 |
| `Node.base_a_reliability` | `Node.base_inference_fidelity` | 已迁移并删除旧字段 |

dataclass 主字段、构造、clone、环境更新、算法和测试均已切换到新名称；旧 property 已删除。迁移覆盖：

- `core/environment.py` 的 reset、initialize 和日志；
- `proposed.py` 的启发式可靠性评分；
- `rtbl/scheduler.py` 的 `a_j` 构造；
- `stability_diagnostics.py`；
- `core/topology.py` 的构造参数；
- 测试中的旧字段断言。

本项目没有稳定的外部 Python API 发布约束，因此未继续保留会污染论文语义的 deprecated property。

### 5.2 物理链路的旧 `reliability` 字段

| 旧字段/属性 | 新语义 | 当前情况 |
| --- | --- | --- |
| `PhysicalLinkState.base_reliability` | `base_transmission_stability` | 新属性仍代理旧存储 |
| `PhysicalLinkState.reliability` | `transmission_stability` | 环境日志、诊断和测试仍直接访问 |
| `LinkNode.base_reliability` | 通过 `physical_state.base_transmission_stability` 访问 | 兼容 property |
| `LinkNode.reliability` | 通过 `physical_state.transmission_stability` 访问 | 兼容 property |
| `LinkNode.heat` | 通过 `physical_state.heat` 访问 | 兼容 property，语义本身未弃用 |

主字段现为 `base_transmission_stability` / `transmission_stability`，有向链路旧代理已删除。这里迁移的是命名，不是物理链路状态本身。

### 5.3 `ObjectiveValues` 的旧属性别名

| 旧属性 | 新属性 |
| --- | --- |
| `ObjectiveValues.accuracy` | `inference_fidelity` |
| `ObjectiveValues.operation` | `operational_stability` |

生产代码和测试已迁移到新属性，旧别名已删除。

### 5.4 `ExperimentMetrics` 的旧存储字段与报告列

| 旧项 | 新项 | 状态 |
| --- | --- | --- |
| `avg_operation` | `avg_operational_stability_score` | 已迁移并删除旧字段 |
| `avg_accuracy` | `avg_inference_fidelity_score` | 已迁移并删除旧字段 |
| `avg_dynamic_operation_reliability` | `avg_operational_stability_score` | 已停止兼容输出 |
| `avg_dynamic_accuracy_reliability` | `avg_inference_fidelity_score` | 已停止兼容输出 |

dataclass 主字段、所有构造调用和报告列均已迁移；两个 `avg_dynamic_*` 旧列不再输出。

### 5.5 Pareto 点与绘图使用的旧字段

当前 Pareto 数据同时写出新旧两套字段：

| 旧字段 | 新字段 |
| --- | --- |
| `accuracy` | `inference_fidelity` |
| `operation` | `operational_stability` |
| `accuracy_satisfaction` | `inference_fidelity_satisfaction` |
| `operation_satisfaction` | `operational_stability_satisfaction` |
| `accuracy_norm` | `inference_fidelity_norm` |
| `operation_norm` | `operational_stability_norm` |

这些旧字段已经从以下生产链移除：

- `experiment_runner.py` 的默认 HV 目标、归一化、Schema 和 CSV 导出；
- `plot_results.py` 的 Pareto 3D 与散点图；
- `plot/generate_paper_figures.py` 的 Pareto 绘图；
- `tests/test_four_objective_optimization.py` 的兼容 Schema 测试。

当前 Schema 为 `semantic_names_v4_global`，Pareto/HV Schema 为 `4d_unified_v4_global_semantics`。历史版本化四维 CSV 可由 `migrate_legacy_result_schema.py` 转换；未版本化或三维前沿会被拒绝，避免把不同语义仅靠改列名伪装成新结果。

### 5.6 仍被调用的 Java 风格方法

| 方法 | 当前调用者 | 建议 |
| --- | --- | --- |
| `DNN.getDelay()` | 原调用者为 `rtbl/scheduler.py` | 已改用 `dnn.delay` 并删除 |
| `LinkDNN.getFloatTran()` | 原调用者为 `Environment.calculate_task_delay_costs()` | 已改用 `link.float_tran` 并删除 |

### 5.7 接纳后指标的旧元组协议

以下兼容链已删除：

- 已删除的 `PostAdmissionMetrics.__iter__()` 四元组解包；
- 已删除的 RTBL `hasattr()` 判断和旧四元组回退。

RTBL 和测试桩现统一使用 `PostAdmissionMetrics` 具名属性，不再接受旧四元组。

### 5.8 候选可靠性旧包装接口

`Environment.predict_candidate_reliability_by_assignment()` 的最后一个测试调用已迁移到 `CandidateQualityScores` 具名字段，该包装器已删除。

### 5.9 `_score()` 的旧边界参数

`AllDNNRefactor._score()` 的 `*_reference_bounds` 参数及所有调用点的八个无效边界实参已经删除。

当前签名为：

```text
_score(value1, value2, value3, value4, w_a, w_r, w_t, w_e)
```

边界仅在确实需要归一化或偏好计算的位置保留。

### 5.10 第三批完成后的重新分析

全仓重新检索生产 Python 代码后，旧 `o_reliability`、`a_reliability`、`accuracy_norm`、`operation_norm`、`avg_operation` 和 `avg_accuracy` 只存在于历史 CSV 转换器的显式映射中，不再被运行、评价、导出或绘图主链消费。旧列测试只用于验证迁移与拒绝边界。

重新识别出的后续候选如下：

| 优先级 | 候选 | 依据 | 建议批次 |
| --- | --- | --- | --- |
| 已完成 | `_score(*_reference_bounds)` 及调用方的八个边界实参 | 形参完全未读取 | 第四批已删除 |
| 已完成 | `DNN.getDelay()`、`LinkDNN.getFloatTran()` | 均为字段的无逻辑包装 | 已改用字段并删除 |
| 已完成 | `PostAdmissionMetrics.__iter__()` 与 RTBL `hasattr`/四元组 fallback | 内部环境固定返回具名对象 | 已统一具名属性访问 |
| 已完成 | `DNN.preR/preA` 及 `preference_operation` 代理 | 旧名称会误导为概率或 operation 指标 | 已改为 `preference_stability` / `preference_fidelity` 主字段 |
| 已完成 | `use_predicted_reliability`、`reliability_sensitive`、`"reliability"` 启发式角色 | 属于活代码语义迁移 | 已改为 quality/stability 命名并提升 Schema 版本 |
| 已完成 | `_infer_link_base_reliability()`、`refresh_dynamic_reliability()` | 行为是质量先验/稳定性状态更新，不是故障成功概率 | 第三批已改为 `_infer_link_base_stability()` / `refresh_dynamic_stability()` |
| K | `count_raw_link_product_by_assignment()` | 只服务 `raw_joint_product` 旧模型诊断 | 已明确 raw-product 命名并保留诊断行为 |
| 已完成 | `RunningDNN.arrival_time/start_time/used_*` | 全仓只有构造写入，没有推进、评价、导出或诊断读取 | 第四批已删除；未来若建模排队等待需以明确语义重新引入 |

此外，`check_delay_random()`、`count_delay_random()`、`count_tran_random()` 等名称虽旧，但仍被 RTBL 和位置基线调用，属于“需要重命名的活代码”，不是死代码。`raw_joint_product` 和旧 Schema 拒绝逻辑继续列为 K，不应清除。

## 6. K：旧语义但应有意保留的代码

### 6.1 `raw_joint_product` 诊断链

以下代码反映旧式“节点和物理链路质量连乘”，但当前只用于展示规模偏置和模型迁移效果：

- `CandidateQualityScores.raw_joint_product`；
- `PostAdmissionMetrics.raw_joint_product`；
- `Environment.count_raw_link_product_by_assignment()`；
- `Environment.count_raw_joint_product_by_assignment()`；
- `stability_diagnostics.py`；
- 对应单元测试。

它们应继续满足：

- 不进入四目标向量；
- 不用于支配排序、约束或最终选解；
- 输出名称必须包含 `raw` 或 `diagnostic`；
- 论文只把它作为旧模型对照，不解释为任务成功概率。

因此，该功能簇属于“旧主指标已弃用、诊断实现继续保留”，不应与真正死代码一起删除。

### 6.2 旧结果 Schema 拒绝逻辑

`experiment_runner.py` 中对旧三维 HV、缺失语义版本和旧 product-based 结果的拒绝逻辑必须保留。它不是旧算法残留，而是防止新旧实验混算的数据安全边界。

### 6.3 `run_proposed()`

`run_proposed()` 是当前 NSGA-ED 演化基线，仍被 `runner.py`、`experiment_runner.py` 和诊断脚本调用。它不是已弃用的 DAREED 旧版本，不能因名称为 `proposed` 就删除。

### 6.4 二维矩阵时延链

以下名称虽然含 `random`，但当前仍服务于 RTBL 动态基线和 local-first 搜索：

- `check_delay_random()`；
- `count_delay_random()`；
- `count_delay_random1()`；
- `count_tran_random()`；
- `count_tran_delay_random()`。

它们暂不属于弃用代码。后续可以统一命名为 `*_matrix`，但必须先迁移 RTBL 和 location placement 调用。

## 7. 本轮已删除的未使用字段和配置项

### 7.1 已删除字段

| 文件 | 字段 | 说明 |
| --- | --- | --- |
| `proposed.py` | `AllDNNRefactor.gen` | 只赋值，从未读取 |
| `proposed.py` | `AllDNNRefactor.res_next` | 只分配，从未读取 |
| `models.py` | `Task.weight_file` | 拓扑生成时赋值，后续没有读取 |
| `core/environment.py` | `Environment.k` | 初始化采样后没有读取 |

### 7.2 已随旧 RTBL 生成器删除的配置

| 文件 | 配置项 | 原用途 |
| --- | --- | --- |
| `core/environment.py` | `k_range` | 生成未使用的 `self.k` |
| `core/environment.py` | `i_range`, `c_range` | 旧 `generate_task()` |
| `core/environment.py` | `alpha_1_range`, `alpha2_range` | 旧 `generate_accuracy()` |
| `rtbl/config.py` | `D_Max` | 只赋给未读取的 `self.d_max` |
| `rtbl/config.py` | `T` | 无调用 |
| `rtbl/config.py` | `I_range`, `C_range` | 仅转发给旧任务生成配置 |
| `rtbl/config.py` | `alpha_1_range`, `alpha2_range` | 仅转发给旧精度生成配置 |
| `rtbl/config.py` | `k_range` | 仅转发并生成未读取的 `self.k` |
| `rtbl/config.py` | `G_range` | 无调用 |

上述配置项已从 `SimulationConfig`、`Environment` 和 `RTBLRunner.__init__()` 的内部构造调用中同步移除。

### 7.3 已删除的未消费运行记录

`RunningDNN.arrival_time`、`start_time`、`used_nodes`、`used_links` 和 `used_physical_links` 原来只有构造写入，没有任何推进、评价、导出或诊断读取，现已删除。当前模型不计排队等待；未来若增加等待时延或预测—实际审计，应以明确的数据流和测试重新引入，而不是保留无消费者字段。

## 8. 非源码清理项

仓库中存在以下生成物，不属于业务代码，但建议加入 `.gitignore` 并从版本管理中清理：

- `__pycache__/`；
- `*.pyc`；
- `.DS_Store`；
- 编辑器交换文件 `*.swp`；
- 临时 PDF 处理目录 `tmp/`，若其内容只是一次性分析产物；
- 本地生成的 `output/`，若没有作为正式实验制品归档。

由于当前工作树已有大量用户修改和归档操作，清理这些文件时不能使用会覆盖工作树的命令，也不应顺带删除 `experiment_results/archive/`。

## 9. 推荐删除顺序

### 阶段 1：低风险死代码

已完成：

1. 删除无调用的 Java getter/setter；
2. 删除 `collect_used_links()`、`count_energy_utility_by_assignment()` 等孤立包装器；
3. 删除 `gen`、`res_next`、`_membrane_range()`、旧 `mutate()`、旧 `cross_over()` 和三个分目标写入方法；
4. 删除无调用的 RTBL 生成器和配置；
5. 通过全量单元测试和三个关键算法入口烟雾测试。

### 阶段 2：旧 RTBL 静态链

已完成：

1. 删除 `RTBLRunner.run()`；
2. 删除三维矩阵评价函数簇；
3. 删除 `release_resource()`；
4. 删除旧 graph matrix 查询链；
5. 通过 37 项单元测试以及 `proposed`、`customized`、`run_dynamic()` 烟雾测试。

### 阶段 3：语义字段迁移

已完成：

1. 迁移 Node、PhysicalLinkState 和 ExperimentMetrics 主字段；
2. 迁移 `ObjectiveValues.accuracy/operation`；
3. 迁移 Pareto、CSV、HV 和绘图字段；
4. 增加有版本校验的一次性旧 CSV 转换脚本；
5. 删除旧字段兼容输出和 fallback。

### 阶段 4：发布边界清理

已完成：

1. 删除 `PostAdmissionMetrics` 元组协议；
2. 删除 RTBL 四元组 fallback；
3. 候选可靠性旧包装接口已提前在第三批删除；
4. 保留并重命名 raw product 诊断与旧 Schema 拒绝逻辑；
5. Schema 提升为 `semantic_names_v4_global`。

## 10. 每批删除的验收标准

- [ ] `python -m unittest discover -s tests -v` 全部通过；
- [ ] `runner.py` 中当前八个算法入口至少完成缩小规模烟雾测试；
- [ ] `experiment_runner.py` 能生成新 Schema 的 raw、summary 和 Pareto 文件；
- [ ] 新 Pareto/HV 只读取 IFS/OSS 新字段；
- [ ] RTBL 仅保留共享动态环境入口；
- [ ] 代码中不再出现未使用的倒数能耗主评价；
- [ ] `raw_joint_product` 仍不进入优化目标；
- [ ] 旧结果与新结果混合时仍明确报错；
- [ ] 删除前后的固定种子核心指标在允许误差内一致；
- [ ] README、系统模型和实验文档同步更新。

## 11. 建议最终保留的核心公开接口

完成清理后，环境评价层建议只保留以下主接口：

```text
estimate_delay_from_assignment()
predict_candidate_scores_by_assignment()
evaluate_post_admission_metrics()
count_operational_stability_by_assignment()
count_inference_fidelity_by_assignment()
count_total_energy_by_assignment()
energy_satisfaction()
add_running_dnn()
advance_time_slot()
```

矩阵形式接口仅限确有需要的基线适配器，并统一使用 `*_matrix` 命名。主算法、实验导出和论文绘图只使用 `operational_stability`、`inference_fidelity`、`estimated_delay` 和 `total_energy` 新语义。
