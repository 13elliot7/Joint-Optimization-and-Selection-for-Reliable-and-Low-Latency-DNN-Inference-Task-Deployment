# RTBL 接入动态环境方案

## 1. 背景

当前项目已经形成统一的动态运行模型：算法先为新到达的 DNN 生成部署方案，部署成功后把 DNN 注册为运行块，后续通过时隙推进持续更新节点负载、链路负载、热度、有效带宽和动态可靠性。

已有动态 baseline 的共同流程是：

```text
生成部署方案
-> 使用动态指标评价
-> env.add_running_dnn()
-> env.advance_time_slot()
-> 最后排空 env.running_dnns
```

RTBL 原实现仍是独立静态流程：

```text
RTBLRunner 自建 Environment
-> generate_availability()
-> RTBLScheduler.step()
-> count_values*_random()
-> release_resource()
```

这会导致 RTBL 与其他算法不共享同一批 DNN、同一套动态参数和同一套统计口径。

## 2. 接入原则

本次接入不改变 RTBL 原有算法逻辑，只替换运行壳。

保持不变：

- `RTBLScheduler.step()`
- `SOTASelector.select_multiple()`
- `q` 的更新方式
- `hj / rj_bar / rj_tilde` 的在线可用性估计方式
- `generate_availability()` 驱动的节点可用性观测
- `lambda_a / lambda_r / V_range / D_bug / D_Max` 等 RTBL 参数含义

替换部分：

- RTBL 不再必须自建环境，可以接收外部 `Environment`
- 部署成功后不再调用 `release_resource()`
- 指标不再使用旧的 `count_values*_random()`
- 成功部署的 DNN 进入 `env.running_dnns`
- 运行过程跟随 `env.advance_time_slot()` 推进

## 3. 新运行流程

共享动态环境版 RTBL 流程如下：

```text
RTBLRunner(env).run_dynamic(initial_nodes)
-> env.reset_nodes(initial_nodes)
-> 重置 RTBL 在线估计状态
-> while 仍有新 DNN 或运行中 DNN:
     如果没有新 DNN:
         env.advance_time_slot()
         continue
     对当前 DNN 调用原 RTBL 逐任务选择逻辑
     如果部署失败或超 deadline:
         failure += 1
         env.advance_time_slot()
         continue
     assignment_slice = 当前 DNN 的实际任务部署向量
     delay = env.estimate_delay_from_assignment()
     accuracy = env.count_dynamic_accuracy_by_assignment()
     operation = env.count_dynamic_operation_by_assignment()
     energy = env.count_total_energy_by_assignment()
     env.add_running_dnn()
     env.advance_time_slot()
-> while env.running_dnns:
     env.advance_time_slot()
-> 返回 ExperimentMetrics
```

## 4. 代码改动

### 4.1 `rtbl/scheduler.py`

新增能力：

- `RTBLRunner(env=None, config=None)` 支持注入共享环境
- `_reset_scheduler()` 用于在一次动态运行开始时重置 RTBL 在线估计状态
- `_build_assignment(dnn_index)` 提取原 `run()` 中逐任务 RTBL 部署逻辑
- `run_dynamic(reset_snapshot=None)` 使用共享动态环境运行 RTBL

保留能力：

- `run()` 保留原静态兼容流程
- `RTBLScheduler.step()` 未改变
- `SOTASelector` 未改变

### 4.2 `runner.py`

RTBL 不再走特殊的 `RTBLRunner().run()` 分支，而是复用同一个：

```python
env = Environment(t_max=args.tmax, verbose=not args.quiet)
initial_nodes = env.clone_nodes()
RTBLRunner(env).run_dynamic(initial_nodes)
```

`all` 模式中，RTBL 也在 `env.reset_nodes(initial_nodes)` 后运行。

### 4.3 `experiment_runner.py`

将 `rtbl` 加入 `ALGORITHMS`，并在 `run_once()` 的 runners 表中加入：

```python
"rtbl": lambda: RTBLRunner(env).run_dynamic(initial_nodes)
```

这样批量实验可以用统一的 `Scenario.environment_kwargs()` 生成环境。

## 5. 尺寸与约束

当前 RTBL 配置中的 `M` 表示 RTBL 可选节点数量，也对应 `Environment.availability_node_count` 的可用性建模节点数。

动态接入时需要满足：

```text
config.M <= len(env.nodes)
config.M <= env.availability_node_count
```

本次实现中若不满足会直接抛出 `ValueError`，避免静默跑出错位结果。

默认实验环境中，RTBL 继续使用原有 `M=11`，因此仍只在前 11 个节点上执行原算法选择逻辑；这保持了原 RTBL baseline 的选择范围。若后续希望 RTBL 覆盖扩展拓扑的全部节点，应同步调整 `Environment.availability_node_count` 和 `SimulationConfig.M`。

## 6. 验证方法

建议最小验证：

```bash
python runner.py --algorithm rtbl --tmax 2 --quiet
```

预期：

- 命令正常结束
- 输出 `avg_estimated_delay`
- 输出 `avg_dynamic_operation_reliability`
- 输出 `avg_dynamic_accuracy_reliability`
- 输出 `avg_total_energy`
- 输出 `rejected_or_failed_count`

建议批量入口验证：

```bash
python experiment_runner.py --suite overall --algorithms rtbl --repeats 1 --dnn-counts 2 --output-dir /tmp/rtbl_dynamic_check
```

预期：

- `raw_results.csv` 中包含 `algorithm=rtbl`
- `summary.csv` 中包含 RTBL 聚合结果
- `avg_total_energy` 不再固定为 0，除非所有 DNN 失败

## 7. 有效性判断

接入有效的判断标准是：

- RTBL 与其他 baseline 共用外部 `Environment`
- RTBL 成功部署后会进入 `env.running_dnns`
- 运行中 DNN 会通过 `advance_time_slot()` 影响后续节点和链路动态状态
- 输出指标来自动态环境接口
- 原 RTBL 的逐任务选择和在线估计逻辑保持不变
