# `src_python` 实验代码说明

## 指标语义

本项目把 DNN 推理任务抽象为 DAG，不对具体模型或数据集进行精度建模。四目标部署评价统一为：推理保真度分数（IFS）、运行稳定性分数（OSS）、时延满意度和能耗满意度，且均按“越大越好”处理。

- OSS 是节点与唯一物理链路状态相关质量因子的分层几何聚合，不表示任务成功概率。
- IFS 是按节点承载 FLOPs 加权的几何聚合，不表示具体 DNN 的测试集准确率。
- 节点与链路初值是经验质量先验；`heat` 是历史压力/负载记忆代理量，不表示器件物理寿命。
- 候选方案按接纳后一步预测状态评价，预测过程不修改真实环境。
- `raw_joint_product` 仅作为旧式连乘诊断量，不参与新版主目标。
- 新实验结果使用 `objective_semantics_version=stability_fidelity_v2_return_energy`，不得与旧连乘或未计回传能耗的结果混合统计。

生产代码与新实验 CSV 仅使用 `operational_stability` 和 `inference_fidelity` 语义字段。旧列不再双写；历史结果须通过 `migrate_legacy_result_schema.py` 显式转换，并接受语义版本校验。

## 1. 目录组成

```text
src_python/
├── README.md
├── __init__.py
├── models.py
├── metrics.py
├── runner.py
├── proposed.py
├── core/
│   ├── __init__.py
│   ├── dag_generator.py
│   ├── topology.py
│   └── environment.py
└── rtbl/
    ├── __init__.py
    ├── config.py
    ├── sota_selector.py
    └── scheduler.py
```

## 2. 各部分职责

### 2.1 数据模型

文件: models.py

定义实验中的基础对象：

- `Node`: 云/边/端节点，包含 CPU、层级、运行稳定性、推理保真度和计算速率
- `Task`: DNN 子任务，包含 CPU 需求和浮点运算量
- `LinkNode`: 节点间链路，包含带宽
- `LinkDNN`: DNN 内任务依赖边，包含中间数据传输量
- `DNN`: 一个 DNN 推理任务图，包含任务集合、依赖边、时延约束、发起节点等

### 2.2 DAG 生成器

文件: core/dag_generator.py

对应原始代码中的 `T.java`，负责：

- 生成无环任务图
- 补齐入度为 0 和出度为 0 的中间节点连接
- 输出邻接矩阵形式的 DNN 任务结构

核心入口：

- `DAGGenerator.get_graph()`

### 2.3 拓扑与 DNN 任务生成

文件: core/topology.py

对应原始代码中的 `CreateNode.java`，负责：

- `create_nodes()`
  - 创建 1 个云节点
  - 创建 10 个边缘节点
  - 创建边缘节点之间、边缘到云之间的链路
  - 为部分边缘节点挂接若干用户节点
- `create_dnn()`
  - 根据 DAG 生成任务集合和任务依赖边
  - 随机设置任务时延约束、质量目标偏好、输入/输出数据量和发起节点
- `create_dnns()`
  - 批量生成 DNN 实验任务

### 2.4 共享实验环境

文件: core/environment.py

这是整个实验的共享核心，对应原始代码中的：

- `ALLDNN.java` 中的公共计算函数
- `RTBL_DNNs/Environment.java`

主要职责：

- 初始化实验环境
- 维护 `nodes`、`link_nodes`、`ds`
- 预计算固定拓扑的节点对最短路径缓存
- 生成节点可用性历史和在线可用性状态
- 提供所有算法共享的约束检查与性能计算函数

关键函数：

- `check_resource()`: 资源约束检查
- `check_delay_random()`: 检查当前部署是否满足时延约束
- `count_delay_random()`: 计算单个部署方案时延
- `count_inference_fidelity_by_assignment()`: 计算部署 IFS
- `count_operational_stability_by_assignment()`: 计算部署 OSS
- `predict_candidate_scores_by_assignment()`: 预测接纳后 OSS、IFS 及诊断量
- `evaluate_post_admission_metrics()`: 统一计算候选接纳后指标
- `count_total_energy_by_assignment()`: 计算部署总能耗
- `get_arrive_link()`: 获取两节点间最短路径对应的链路序列
- `calculate_task_delay_costs()`: RTBL 调度每一步使用的任务-节点时延代价

### 2.5 指标结构

文件: metrics.py

定义 `ExperimentMetrics`，统一封装实验输出：

- `avg_delay`
- `avg_operational_stability_score`
- `avg_inference_fidelity_score`
- `avg_energy`
- `failure_count`
- `runtime_ms`

## 3. 算法模块

### 3.1 主算法与三类基线

文件: proposed.py

对应原始代码中的 `ALLDNN.java`。

类：

- `AllDNNRefactor`

包含四类实验流程：

- `run_proposed()`: 主算法，NSGA 迭代搜索 + 多目标选择
- `run_random()`: 随机部署基线
- `run_max_resource()`: 最大剩余资源优先基线
- `run_local_first()`: 本地优先基线

主算法内部核心函数：

- `_mutate_assignment()`: assignment 级随机变异
- `_skew_mutate_assignment()`: DAG/资源偏斜变异
- `cross_over1()`: 交叉
- `count_values()`: 一次性计算统一四目标值
- `dominated_sort()`: 非支配排序
- `get_res()`: 拥挤度计算
- `update_res()`: 从 `P+Q` 中选出下一代种群

基线内部核心函数：

- `max_resource_placement()/max_resource_placement1()`
- `location_placement()/location_placement1()`

### 3.2 RTBL 基线

目录: [rtbl]

对应原始代码中的 `src_backup_origin/RTBL_DNNs/`。

#### 配置

文件: [rtbl/config.py]

定义 `SimulationConfig`，包括：

- 节点数 `M`
- 可靠性上下界
- 时延预算和最大时延
- 历史窗口 `H_off`
- 输入规模、计算规模、精度范围等参数

#### 节点选择器

文件: [rtbl/sota_selector.py]

定义 `SOTASelector`，负责：

- `compute_f1()`: 计算收益项
- `compute_f2()`: 计算代价项
- `select()`: 选择当前最优单节点策略
- `select_multiple()`: 返回 10 个候选策略

#### 在线调度器

文件: [rtbl/scheduler.py]

定义：

- `RTBLScheduler`
- `RTBLRunner`

职责：

- `RTBLScheduler`
  - 维护 `Q`、`hj`、`rj_bar`、`rj_tilde`
  - 在线更新可靠性估计
  - 在每个子任务调度时调用 `Environment.calculate_task_delay_costs()`
  - 用 `SOTASelector` 选择可行节点
- `RTBLRunner`
  - 初始化环境和调度器
  - 按 DNN、按子任务推进整个 RTBL 基线实验

RTBL 的动态实验指标使用候选部署接纳后的预测负载和热度计算，与
DAREED 的动态可靠性口径保持一致。若 RTBL 实现发生变化，可使用独立入口
只重跑包含 RTBL 的部署实验：

```bash
python3 rtbl_experiment_runner.py
```

默认自动检测 `overall,dynamic,scale` 中已经存在 `raw_results.csv` 的 suite，
避免意外创建只有 RTBL 的不完整对比集。新结果先写入临时目录，全部完成后
才原子替换各 `raw_results.csv` 中 `algorithm=rtbl` 的记录，其他算法结果不会
重跑或覆盖。仅重跑指定 suite：

```bash
python3 rtbl_experiment_runner.py --suites overall
```

## 4. 调用关系

### 4.1 全局入口

文件: [runner.py]

统一命令行入口：

- `main()`

支持算法：

- `proposed`
- `random`
- `maxresource`
- `localfirst`
- `rtbl`
- `all`

### 4.2 主算法调用链

```text
runner.main()
  -> Environment(...)
    -> create_nodes()
    -> create_dnns()
      -> create_dnn()
        -> DAGGenerator.get_graph()
  -> AllDNNRefactor(env)
  -> run_proposed()
    -> _mutate_assignment()
    -> cross_over1()
    -> count_values()
      -> _evaluate_assignment()
        -> Environment.estimate_delay_from_assignment()
        -> Environment.predict_candidate_scores_by_assignment()
        -> Environment.count_total_energy_by_assignment()
    -> dominated_sort()
    -> get_res()
    -> update_res()
    -> Environment.add_running_dnn()
    -> Environment.advance_time_slot()
```

### 4.3 三个基线调用链

```text
runner.main()
  -> Environment(...)
  -> AllDNNRefactor(env)
  -> run_random()
    -> Environment.check_resource()
    -> Environment.check_delay_random()
    -> _evaluate_assignment()
    -> Environment.add_running_dnn()

  -> run_max_resource()
    -> max_resource_placement()/max_resource_placement1()
      -> Environment.check_delay_random()
      -> Environment.check_resource()
    -> _evaluate_assignment()
    -> Environment.add_running_dnn()

  -> run_local_first()
    -> location_placement()/location_placement1()
      -> count_delay_random1()
        -> Environment.count_delay_random1()
      -> Environment.check_delay_random()
    -> _evaluate_assignment()
    -> Environment.add_running_dnn()
```

### 4.4 RTBL 调用链

```text
runner.main()
  -> RTBLRunner()
    -> Environment(...)
    -> RTBLScheduler(config, env.rj_history, env)
      -> SOTASelector(...)
  -> RTBLRunner.run_dynamic()
    -> Environment.generate_availability()
    -> RTBLScheduler.step()
      -> update_rj_tilde()
      -> Environment.calculate_task_delay_costs()
        -> Environment.get_arrive_link()
      -> SOTASelector.select_multiple()
        -> select()
          -> compute_f1()
          -> compute_f2()
      -> Environment.check_resource()
    -> Environment.check_delay_random()
    -> Environment.evaluate_post_admission_metrics()
    -> Environment.add_running_dnn()
    -> Environment.advance_time_slot()
```

## 5. 运行方式

在 `src_python/` 目录执行：

```bash
python runner.py --algorithm proposed
python runner.py --algorithm random
python runner.py --algorithm maxresource
python runner.py --algorithm localfirst
python runner.py --algorithm rtbl
python runner.py --algorithm all
```

可选参数：

- `--tmax`: 控制 `proposed/random/maxresource/localfirst` 的 DNN 数量
- `--iteration-limit`: 控制主算法 `run_proposed()` 中的迭代上限，默认值保持与原始代码一致

示例：

python runner.py --algorithm proposed
python runner.py --algorithm proposed --tmax 1 --iteration-limit 120
```

## 6. 代码组织原则

本目录按“共享环境 + 算法实现 + 统一入口”组织：

- `models.py` 只负责数据结构
- `core/` 只负责任务/拓扑生成和共享评价逻辑
- `proposed.py` 负责主算法与三个来自 `ALLDNN.java` 的基线
- `rtbl/` 负责 `RTBL_DNNs` 体系下的配置、选择器和在线调度器
- `runner.py` 负责命令行入口和实验调度

这样拆分后，能够在不改动原始实验逻辑的前提下，把模型、环境和算法边界分开，便于单独检查和对照。
