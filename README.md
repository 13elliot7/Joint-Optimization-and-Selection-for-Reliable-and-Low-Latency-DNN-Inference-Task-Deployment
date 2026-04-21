# `src_python` 实验代码说明

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

- `Node`: 云/边/端节点，包含 CPU、层级、运行可靠性、精度可靠性、计算速率
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
  - 随机设置任务时延约束、预期精度/可靠性、输入/输出数据量、发起节点
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
- 构建节点图矩阵 `graph_node`
- 生成节点可用性历史和在线可用性状态
- 提供所有算法共享的约束检查与性能计算函数

关键函数：

- `check_resource()`: 资源约束检查
- `check_delay_random()`: 检查当前部署是否满足时延约束
- `count_delay_random()`: 计算单个部署方案时延
- `count_accuracy()`: 计算部署精度可靠性
- `count_operation()`: 计算运行可靠性
- `get_arrive_link()`: 获取两节点间最短路径对应的链路序列
- `get_shortest_path()`: 最短路径计算
- `count_values1_random()/2_random()/3_random()`: 基线算法的三目标评估
- `calculate_task_delay_costs()`: RTBL 调度每一步使用的任务-节点时延代价

### 2.5 指标结构

文件: metrics.py

定义 `ExperimentMetrics`，统一封装实验输出：

- `avg_delay`
- `avg_operation`
- `avg_accuracy`
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

- `mutate()`: 变异
- `cross_over1()`: 交叉
- `count_values1()/2()/3()`: 计算种群三目标值
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
    -> mutate()
    -> cross_over1()
    -> count_values1()/2()/3()
      -> Environment.count_accuracy()
      -> Environment.count_operation()
      -> Environment.count_delay()
        -> Environment.count_tran()
        -> Environment.get_arrive_link()
        -> Environment.get_shortest_path()
    -> dominated_sort()
    -> get_res()
    -> update_res()
    -> Environment.release_resource()
```

### 4.3 三个基线调用链

```text
runner.main()
  -> Environment(...)
  -> AllDNNRefactor(env)
  -> run_random()
    -> Environment.check_resource()
    -> Environment.check_delay_random()
    -> Environment.count_values1_random()/2_random()/3_random()
    -> Environment.release_resource()

  -> run_max_resource()
    -> max_resource_placement()/max_resource_placement1()
      -> Environment.check_delay_random()
      -> Environment.check_resource()
    -> Environment.count_values1_random()/2_random()/3_random()
    -> Environment.release_resource()

  -> run_local_first()
    -> location_placement()/location_placement1()
      -> count_delay_random1()
        -> Environment.count_delay_random1()
      -> Environment.check_delay_random()
    -> Environment.count_values1_random()/2_random()/3_random()
    -> Environment.release_resource()
```

### 4.4 RTBL 调用链

```text
runner.main()
  -> RTBLRunner()
    -> Environment(...)
    -> RTBLScheduler(config, env.rj_history, env)
      -> SOTASelector(...)
  -> RTBLRunner.run()
    -> Environment.generate_availability()
    -> RTBLScheduler.step()
      -> update_rj_tilde()
      -> Environment.calculate_task_delay_costs()
        -> Environment.get_arrive_link()
        -> Environment.get_shortest_path()
      -> SOTASelector.select_multiple()
        -> select()
          -> compute_f1()
          -> compute_f2()
      -> Environment.check_resource()
    -> Environment.check_delay_random()
    -> Environment.count_values1_random()/2_random()/3_random()
    -> Environment.release_resource()
```

## 5. 运行方式

在项目根目录执行：

```bash
python -m src_python.runner --algorithm proposed
python -m src_python.runner --algorithm random
python -m src_python.runner --algorithm maxresource
python -m src_python.runner --algorithm localfirst
python -m src_python.runner --algorithm rtbl
python -m src_python.runner --algorithm all
```

可选参数：

- `--tmax`: 控制 `proposed/random/maxresource/localfirst` 的 DNN 数量
- `--iteration-limit`: 控制主算法 `run_proposed()` 中的迭代上限，默认值保持与原始代码一致

示例：

```bash
python -m src_python.runner --algorithm proposed --tmax 1 --iteration-limit 120
```

## 6. 代码组织原则

本目录按“共享环境 + 算法实现 + 统一入口”组织：

- `models.py` 只负责数据结构
- `core/` 只负责任务/拓扑生成和共享评价逻辑
- `proposed.py` 负责主算法与三个来自 `ALLDNN.java` 的基线
- `rtbl/` 负责 `RTBL_DNNs` 体系下的配置、选择器和在线调度器
- `runner.py` 负责命令行入口和实验调度

这样拆分后，能够在不改动原始实验逻辑的前提下，把模型、环境和算法边界分开，便于单独检查和对照。
