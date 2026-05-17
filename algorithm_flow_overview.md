# 当前环境初始化与算法调用流程梳理

## 1. 总入口流程

程序入口在 `runner.py` 的 `main()`，整体调用链如下：

```text
main()
  -> 解析命令行参数
  -> 如果 algorithm == "rtbl"
       -> RTBLRunner().run()
       -> print_metrics()
  -> 否则
       -> Environment(t_max=args.tmax)
       -> AllDNNRefactor(env)
       -> env.clone_nodes()
       -> 根据 algorithm 调用:
            run_proposed()
            run_random()
            run_max_resource()
            run_local_first()
       -> print_metrics()
```

如果 `algorithm == "all"`，则按顺序运行：

```text
run_proposed()
-> env.reset_nodes(initial_nodes)
-> run_random(initial_nodes)
-> env.reset_nodes(initial_nodes)
-> run_max_resource(initial_nodes)
-> env.reset_nodes(initial_nodes)
-> run_local_first(initial_nodes)
-> RTBLRunner().run()
```

其中 `print_metrics()` 只负责把 `ExperimentMetrics` 按动态模型口径打印为：

- `avg_estimated_delay`
- `avg_dynamic_operation_reliability`
- `avg_dynamic_accuracy_reliability`
- `rejected_or_failed_count`
- `runtime_ms`

## 2. 环境初始化流程

### 2.1 Environment 初始化主链

非 RTBL 算法的环境由 `Environment.__post_init__()` 完成初始化，调用顺序如下：

```text
Environment.__post_init__()
  -> create_nodes()
  -> create_dnns(self.t_max, self.nodes)
  -> get_graph_matrix()
  -> _random_uniform(*self.k_range)
  -> _generate_rj_history()
       -> _sample_uptime()
       -> _sample_downtime()
  -> 初始化 up / remaining_time
  -> 初始化 running_dnns / current_slot
  -> _initialize_dynamic_state()
       -> _reset_link_state()
            -> _infer_link_base_reliability()
```

### 2.2 拓扑初始化流程

`create_nodes()` 的职责是生成系统中的节点与链路：

```text
create_nodes()
  -> 创建 1 个云节点
  -> 创建 10 个边缘节点
  -> add_bidirectional_link() 连接边缘-边缘
  -> 手工创建边缘-云链路
  -> 为部分边缘节点随机挂接用户节点
  -> 返回 nodes, link_nodes
```

这里的关键结果是：

- `nodes`：云节点、边缘节点、用户节点的完整列表
- `link_nodes`：双向链路列表

### 2.3 DNN 请求初始化流程

`create_dnns()` 负责批量生成请求：

```text
create_dnns(num, nodes)
  -> DAGGenerator()
  -> 循环 num 次:
       -> create_dnn(task_num, nodes, dag_generator)
```

单个 DNN 的生成过程是：

```text
create_dnn(task_count, nodes, dag_generator)
  -> dag_generator.get_graph(task_count)
       -> _generate_graph()
       -> _fill_graph()
  -> 为每个任务生成 Task
  -> 根据 DAG 边生成 LinkDNN
  -> 设置 delay / preA / preR / startFloat / backFloat
  -> 从用户节点中随机指定 initiateNode
  -> 返回 DNN
```

### 2.4 动态状态初始化流程

`_initialize_dynamic_state()` 会把环境切到动态模型的初始状态：

```text
_initialize_dynamic_state()
  -> 为每个节点补齐 base_o_reliability / base_a_reliability
  -> 节点 load_ratio = 0, heat = 0
  -> 节点当前可靠性恢复到基线值
  -> _reset_link_state()
       -> 对每条链路设置:
            base_reliability
            reliability
            load_ratio = 0
            heat = 0
```

### 2.5 最短路矩阵初始化流程

`get_graph_matrix()` 使用链路带宽构造图矩阵：

```text
get_graph_matrix()
  -> 遍历 link_nodes
  -> graph_node[s][e] = 1 / band_width
```

该矩阵随后会被：

- `get_shortest_path()`
- `get_arrive_link()`
- `count_delay_random()`
- `count_delay()`

等函数复用，用于路径选择和时延计算。

## 3. 动态环境在运行时的共享调用流程

主算法和三个基线共用一套动态运行底座。

### 3.1 DNN 接纳后的注册流程

当某个算法找到一个可行部署后，会调用：

```text
_register_running_dnn(dnn_index, assignment, delay)
  -> _assignment_slice()
  -> math.ceil(delay / env.slot_length)
  -> env.add_running_dnn(...)
       -> collect_used_nodes()
       -> collect_used_links()
            -> _collect_link_weights()
                 -> get_arrive_link()
                      -> get_shortest_path()
```

结果是把当前 DNN 注册为一个 `RunningDNN`，加入 `env.running_dnns`。

### 3.2 时隙推进流程

动态模型的核心推进函数是 `advance_time_slot()`：

```text
advance_time_slot()
  -> release_finished_dnns()
       -> free_running_resources()
       -> allocate_running_resources()
  -> 对 running_dnns 扣减 remaining_slots
  -> allocate_running_resources()
       -> update_node_loads()
       -> update_link_loads()
            -> _collect_link_weights()
  -> update_node_heat()
  -> update_link_heat()
  -> refresh_dynamic_reliability()
  -> current_slot += 1
```

这意味着每推进一个时隙，系统都会统一重算：

- 节点剩余 CPU
- 节点负载率 `load_ratio`
- 链路负载率 `load_ratio`
- 节点热度 `heat`
- 链路热度 `heat`
- 节点运行可靠性 `o_reliability`
- 节点精度可靠性 `a_reliability`
- 链路可靠性 `reliability`

### 3.3 候选部署的统一评价流程

主算法和三个基线在评价候选部署时都走 `_evaluate_assignment()`：

```text
_evaluate_assignment(dnn_index, assignment)
  -> _assignment_slice()
  -> env.count_dynamic_accuracy_by_assignment()
  -> env.count_dynamic_operation_by_assignment()
       -> collect_used_nodes()
       -> count_dynamic_link_reliability_by_assignment()
            -> collect_used_links()
  -> env.estimate_delay_from_assignment()
       -> _build_assignment_matrix()
       -> count_delay_random()
            -> get_arrive_link()
            -> count_tran_random()
            -> count_tran_delay_random()
```

输出四个值：

- `value1`：动态精度可靠性
- `value2`：动态运行可靠性
- `value3`：时延收益
- `delay`：估计时延

## 4. 主算法 run_proposed() 的函数调用流程

`run_proposed()` 是当前最复杂的算法入口，整体结构如下：

```text
run_proposed()
  -> 初始化种群容器和指标累计量
  -> while next_dnn_index < t_max or env.running_dnns:
       -> 如果没有新 DNN 了:
            env.advance_time_slot()
            continue
       -> 为当前 DNN 初始化本轮搜索状态
       -> while index <= iteration_limit:
            -> 初始代构造
                 -> check_resource()
                 -> check_delay_random()
            -> 生成 res_pq
            -> cross_over1()
                 -> _build_assignment_matrix()
                 -> check_resource()
                 -> mutate()
            -> count_values1/2/3()
                 -> _evaluate_assignment()
            -> _score()
            -> dominated_sort()
            -> get_res()
            -> update_res()
       -> 在当前代结果和历史 best 中选最终部署
       -> 若失败:
            failure_dnn += 1
            next_dnn_index += 1
            env.advance_time_slot()
       -> 若成功:
            _evaluate_assignment()
            _register_running_dnn()
            更新统计量
            next_dnn_index += 1
            env.advance_time_slot()
  -> _advance_until_drained()
       -> env.advance_time_slot()
  -> 返回 ExperimentMetrics
```

### 4.1 主算法的关键辅助函数

`run_proposed()` 主要依赖这些内部函数：

- `_new_population()`：创建种群容器
- `_build_assignment_matrix()`：部署向量转矩阵
- `mutate()`：变异
- `cross_over1()` / `cross_over()`：交叉
- `count_values1()` / `count_values2()` / `count_values3()`：计算三目标
- `dominated_sort()`：非支配排序
- `get_res()`：拥挤距离计算
- `update_res()`：更新下一代
- `_score()`：按 `preA / preR / delay` 做最终加权选择

### 4.2 主算法的外层时序语义

与原始静态版本相比，`run_proposed()` 当前的外层语义已经变成：

```text
接纳一个 DNN
-> 注册为运行块
-> 推进一个时隙
-> 再处理下一个 DNN
```

而不是：

```text
部署一个 DNN
-> 永久扣资源
-> 立即处理下一个 DNN
```

## 5. 随机基线 run_random() 的函数调用流程

`run_random()` 的调用链最直接：

```text
run_random(reset_snapshot)
  -> env.reset_nodes(reset_snapshot)
  -> while next_dnn_index < t_max or env.running_dnns:
       -> 若没有新 DNN:
            env.advance_time_slot()
       -> 随机生成部署矩阵 xs
            -> check_resource()
            -> check_delay_random()
       -> 若失败:
            failure_dnn += 1
            env.advance_time_slot()
       -> 若成功:
            _evaluate_assignment()
            _register_running_dnn()
            env.advance_time_slot()
  -> _advance_until_drained()
  -> 返回 ExperimentMetrics
```

这个算法的特点是：

- 搜索策略最简单，纯随机试探
- 评价阶段与主算法完全共用动态模型接口

## 6. 最大资源基线 run_max_resource() 的函数调用流程

`run_max_resource()` 的部署搜索依赖递归函数：

```text
run_max_resource(reset_snapshot)
  -> env.reset_nodes(reset_snapshot)
  -> while next_dnn_index < t_max or env.running_dnns:
       -> 若没有新 DNN:
            env.advance_time_slot()
       -> clone_nodes()
       -> max_resource_placement()
            -> 递归按剩余 CPU 从高到低尝试
            -> check_delay_random()
            -> check_resource()
       -> 若失败，再尝试 max_resource_placement1()
       -> 若仍失败:
            failure_dnn += 1
            env.advance_time_slot()
       -> 若成功:
            _evaluate_assignment()
            _register_running_dnn()
            env.advance_time_slot()
  -> _advance_until_drained()
  -> 返回 ExperimentMetrics
```

其中：

- `max_resource_placement()`：所有节点都可参与排序
- `max_resource_placement1()`：更偏向非云节点的回退搜索

## 7. 本地优先基线 run_local_first() 的函数调用流程

`run_local_first()` 的搜索依赖链路带宽：

```text
run_local_first(reset_snapshot)
  -> env.reset_nodes(reset_snapshot)
  -> while next_dnn_index < t_max or env.running_dnns:
       -> 若没有新 DNN:
            env.advance_time_slot()
       -> location_placement()
            -> _location_band()
            -> count_delay_random1()
            -> 递归尝试“带宽更近”的节点
       -> 若失败，再尝试 location_placement1()
       -> 若仍失败:
            failure_dnn += 1
            env.advance_time_slot()
       -> 若成功:
            _evaluate_assignment()
            _register_running_dnn()
            env.advance_time_slot()
  -> _advance_until_drained()
  -> 返回 ExperimentMetrics
```

这个算法和 `run_max_resource()` 的差异主要在搜索阶段：

- `run_max_resource()` 按剩余 CPU 排序
- `run_local_first()` 按 `_location_band()` 给出的邻近带宽排序

## 8. RTBL 的函数调用流程

RTBL 目前仍然保留独立流程，没有接入前面那套 `running_dnns + advance_time_slot()` 动态底座。

### 8.1 RTBL 入口流程

```text
RTBLRunner.__init__()
  -> SimulationConfig()
  -> Environment(...)
  -> RTBLScheduler(config, env.rj_history, env)
       -> _initialize_rj_bar()
       -> SOTASelector(...)
```

### 8.2 RTBL 执行流程

```text
RTBLRunner.run()
  -> for t in range(env.t_max):
       -> scheduler.reset_q()
       -> generate_availability()
       -> for each task i:
            -> scheduler.step(t, i, x, a_j, rj_t)
                 -> update_rj_tilde()
                 -> calculate_task_delay_costs()
                 -> selector.select_multiple()
                      -> select()
                           -> compute_f1()
                           -> compute_f2()
                 -> check_resource()
       -> check_delay_random()
       -> count_values1_random()
       -> count_values2_random()
       -> count_values3_random()
       -> release_resource()
  -> 返回 ExperimentMetrics
```

### 8.3 RTBL 的当前特点

RTBL 与前四个算法的主要差异有三点：

1. 它按“任务级在线选择”推进，而不是先得到完整 assignment。
2. 它依赖 `generate_availability()`、`rj_history`、`rj_tilde` 这套在线可用性模型。
3. 它最终仍调用旧的 `count_values*_random()` 和 `release_resource()`，还没有完全切换到新的时隙级动态执行框架。

## 9. 当前代码的两条主线总结

### 9.1 环境主线

```text
Environment 初始化
-> 拓扑生成
-> DNN 生成
-> 图矩阵生成
-> 动态节点/链路状态初始化
-> 运行期不断执行 advance_time_slot()
```

### 9.2 算法主线

当前代码里已经形成两种算法执行范式：

#### 范式 A：主算法 + 三个基线

```text
生成候选部署
-> _evaluate_assignment()
-> _register_running_dnn()
-> env.advance_time_slot()
-> 最后 _advance_until_drained()
```

对应函数：

- `run_proposed()`
- `run_random()`
- `run_max_resource()`
- `run_local_first()`

#### 范式 B：RTBL 独立在线调度

```text
generate_availability()
-> scheduler.step()
-> check_delay_random()
-> count_values*_random()
-> release_resource()
```

对应函数：

- `RTBLRunner.run()`
- `RTBLScheduler.step()`
- `SOTASelector.select_multiple()`

## 10. 建议阅读顺序

如果要继续改代码，建议按下面顺序读：

1. `runner.py:main()`
2. `core/environment.py:Environment.__post_init__()`
3. `core/topology.py:create_nodes()` 与 `create_dnns()`
4. `core/environment.py:advance_time_slot()`
5. `proposed.py:_evaluate_assignment()` 与 `_register_running_dnn()`
6. `proposed.py:run_proposed()`
7. `proposed.py:run_random()` / `run_max_resource()` / `run_local_first()`
8. `rtbl/scheduler.py:RTBLRunner.run()`

这样最容易先建立“共享环境 -> 动态执行 -> 各算法差异点”的整体认识。
