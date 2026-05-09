# 基于动态可靠性简化模型的原代码修改方案

## 1. 目标

本文档给出一份面向当前 `src_python` 代码的修改方案，用于将现有静态多 DNN 部署逻辑改造为：

1. 离散时隙推进
2. DNN 块级运行
3. 节点动态可靠性更新
4. 基础链路可靠性更新
5. 基于当前系统状态进行新 DNN 接纳与部署

该方案以 [dynamic_reliability_timing_model.md](./dynamic_reliability_timing_model.md) 中的“简化动态时序模型”为唯一设计依据，不引入完整的子任务级事件仿真，不引入节点可用性建模，但在第一版中实现链路可靠性的基础建模。

## 2. 当前代码的核心问题

当前实现以静态部署评估为主，存在以下问题：

1. DNN 一旦部署成功，就立即通过 `release_resource()` 永久扣减 CPU，而不在后续时隙中继续推进执行。
2. 节点可靠性 `o_reliability` 和 `a_reliability` 在初始化后保持不变，不随负载和历史热度变化。
3. 链路只参与时延计算，不参与运行成功率建模。
4. 所有指标都在“部署瞬间”计算，没有时序推进和系统状态恢复过程。
5. 多个 DNN 之间只有资源竞争，没有动态负载和动态健康度演化。
6. `count_accuracy()` 和 `count_operation()` 直接基于静态节点可靠性计算，不适合动态场景。

因此，需要在保留当前部署搜索框架的前提下，补充一个共享的时序运行层和一层基础链路可靠性建模。

## 3. 总体改造思路

改造原则如下：

1. 尽量不推翻 `proposed.py` 中现有的候选方案生成与多目标搜索框架。
2. 动态状态与时序推进逻辑统一放在 `Environment` 中，不分散到各算法实现。
3. 算法模块只关心：
   - 当前时刻的候选方案是否可行
   - 当前时刻的动态可靠性评价值
   - 若接纳成功，如何将 DNN 注册为一个正在运行的块级实体
4. 第一版同时做节点动态可靠性和链路基础可靠性建模，但不做传输级事件仿真与细粒度排队。

最终系统应表现为：

- 每个时隙先推进已运行 DNN
- 再更新当前节点负载、链路负载与热度
- 再刷新节点与链路动态可靠性
- 再尝试部署一个新的 DNN

## 4. 数据结构修改方案

### 4.1 修改 `models.py` 中的 `Node`

在现有 `Node` 数据结构上增加以下字段：

- `base_o_reliability`
- `base_a_reliability`
- `load_ratio`
- `heat`

建议语义如下：

- `base_o_reliability`：节点运行可靠性的静态先验
- `base_a_reliability`：节点精度可靠性的静态先验
- `o_reliability`：当前时刻动态运行可靠性
- `a_reliability`：当前时刻动态精度可靠性
- `load_ratio`：当前时刻节点负载率
- `heat`：节点热度状态

### 4.2 修改 `LinkNode`

在现有 `LinkNode` 数据结构上增加以下字段：

- `base_reliability`
- `reliability`
- `load_ratio`
- `heat`

建议语义如下：

- `base_reliability`：链路静态先验可靠性
- `reliability`：当前时刻链路动态可靠性
- `load_ratio`：当前时刻链路负载率
- `heat`：当前时刻链路热度

基础链路先验可由链路类别和带宽映射得到，不需要真实链路可靠性标签。

### 4.3 新增 `RunningDNN` 数据类

建议在 `models.py` 中新增一个 `RunningDNN` 数据类，字段至少包括：

- `dnn_index`
- `assignment`
- `arrival_time`
- `start_time`
- `estimated_runtime`
- `remaining_slots`
- `used_nodes`
- `used_links`

字段说明如下：

- `dnn_index`：对应 `env.ds` 中的 DNN 下标
- `assignment`：该 DNN 的一维部署结果
- `arrival_time`：到达系统的时刻
- `start_time`：被接纳并开始运行的时刻
- `estimated_runtime`：由原始静态时延模型估计得到的运行时长
- `remaining_slots`：离散时隙表示的剩余运行时长
- `used_nodes`：承载该 DNN 的唯一节点集合
- `used_links`：该 DNN 使用的唯一链路集合

## 5. `Environment` 的修改方案

### 5.1 初始化阶段增加动态系统状态

在 `Environment.__post_init__()` 中补充以下初始化逻辑：

1. 将当前 `o_reliability` 复制到 `base_o_reliability`
2. 将当前 `a_reliability` 复制到 `base_a_reliability`
3. 为每个链路初始化：
   - `base_reliability`
   - `reliability = base_reliability`
   - `load_ratio = 0.0`
   - `heat = 0.0`
4. 初始化每个节点：
   - `load_ratio = 0.0`
   - `heat = 0.0`
5. 增加 `self.running_dnns = []`
6. 增加时隙长度参数：
   - `self.slot_length`
7. 增加节点动态可靠性参数：
   - `self.alpha_r`
   - `self.beta_r`
   - `self.alpha_a`
   - `self.beta_a`
   - `self.lambda_h`
   - `self.o_min`
   - `self.a_min`
8. 增加链路动态可靠性参数：
   - `self.alpha_l`
   - `self.beta_l`
   - `self.lambda_g`
   - `self.l_min`

### 5.2 新增时延估计辅助接口

新增：

- `estimate_delay_from_assignment(dnn_index: int, assignment: List[int]) -> float`

功能：

1. 将一维 `assignment` 转换成原有 `x` 矩阵
2. 调用现有 `count_delay_random()` 或 `count_delay()` 逻辑
3. 返回该部署方案的静态关键路径时延估计

### 5.3 新增部署结果解析接口

新增：

- `collect_used_nodes(dnn_index: int, assignment: List[int]) -> List[int]`
- `collect_used_links(dnn_index: int, assignment: List[int]) -> List[LinkNode]`

`collect_used_nodes(...)`
1. 提取承载当前 DNN 至少一个子任务的唯一节点集合
2. 用于后续动态运行可靠性计算和当前运行占用统计

`collect_used_links(...)`
1. 根据输入上传、任务间依赖传输和输出回传，提取当前 DNN 实际经过的唯一链路集合
2. 用于后续链路负载统计与路径级可靠性计算

### 5.4 新增运行期管理接口

新增以下方法：

- `add_running_dnn(dnn_index, assignment, arrival_time, start_time, estimated_runtime, remaining_slots)`
- `advance_time_slot()`
- `release_finished_dnns()`
- `allocate_running_resources()`
- `free_running_resources(running_dnn)`
- `update_node_loads()`
- `update_link_loads()`
- `update_node_heat()`
- `update_link_heat()`
- `refresh_dynamic_reliability()`

各方法职责如下：

`add_running_dnn(...)`
- 创建一个 `RunningDNN` 实例并加入 `self.running_dnns`

`advance_time_slot()`
- 推进一个离散时隙
- 依次调用：
  1. `release_finished_dnns()`
  2. 对所有运行中 DNN 扣减 `remaining_slots`
  3. `update_node_loads()`
  4. `update_link_loads()`
  5. `update_node_heat()`
  6. `update_link_heat()`
  7. `refresh_dynamic_reliability()`

`release_finished_dnns()`
- 找出 `remaining_slots <= 0` 的 DNN
- 释放其占用节点资源与链路占用影响
- 从 `self.running_dnns` 中移除

`allocate_running_resources()`
- 根据当前 `self.running_dnns` 重新统计每个节点和链路的实际占用
- 不再依赖“部署成功瞬间永久扣资源”的旧模式

`free_running_resources(running_dnn)`
- 释放一个运行结束 DNN 对节点和链路的占用

`update_node_loads()`
- 基于当前所有 `running_dnns` 的 `assignment`
- 统计每个节点被分配的总 CPU
- 计算 `load_ratio = used_cpu / max_cpu`

`update_link_loads()`
- 基于当前所有 `running_dnns` 的 `used_links`
- 统计每条链路被多少个运行中 DNN 共享，或按对应传输数据量累加占用权重
- 计算 `load_ratio = used_bw / band_width`

`update_node_heat()`
- 按公式更新热度：
  `heat = lambda_h * heat + load_ratio`

`update_link_heat()`
- 按公式更新链路热度：
  `heat = lambda_g * heat + load_ratio`

`refresh_dynamic_reliability()`
- 对每个节点计算新的动态运行可靠性和动态精度可靠性
- 对每条链路计算新的动态链路可靠性

### 5.5 动态可靠性计算接口

新增：

- `count_dynamic_operation_by_assignment(dnn_index: int, assignment: List[int]) -> float`
- `count_dynamic_accuracy_by_assignment(dnn_index: int, assignment: List[int]) -> float`
- `count_dynamic_link_reliability_by_assignment(dnn_index: int, assignment: List[int]) -> float`

定义如下：

对于唯一节点集合 `N_m` 和唯一链路集合 `E_m`：

- 动态运行可靠性：
  `R_m^run(t) = prod_{j in N_m} o_j(t) * prod_{e in E_m} reliability_e(t)`

- 动态精度可靠性：
  `R_m^acc(t) = prod_{j in N_m} a_j(t)^{w_j}`

- 动态链路可靠性：
  `L_m(t) = prod_{e in E_m} reliability_e(t)`

其中 `w_j` 按该节点承载任务的计算量占比确定。

这两个接口将替代当前静态的：

- `count_accuracy()`
- `count_operation()`

同时，链路可靠性不再隐含在时延里，而是被显式并入运行成功率。

### 5.6 资源接口重构

当前 `release_resource()` 实际是在扣减资源，命名与语义不符。

建议重构为：

- `allocate_resource_snapshot(...)`
- `free_resource_snapshot(...)`

但第一版为了尽量少改动，可保留原函数不再用于动态流程，同时新增专用运行期资源管理接口。动态流程统一由 `running_dnns` 驱动当前资源占用，不再让部署成功直接永久扣减 CPU。

## 6. `proposed.py` 的修改方案

### 6.1 保留内部搜索骨架

以下逻辑尽量保留：

- 初始种群生成
- 交叉 `cross_over1()`
- 变异 `mutate()`
- 非支配排序 `dominated_sort()`
- 拥挤度选择 `get_res()` / `update_res()`

这样可以避免重写主算法核心。

### 6.2 修改 `run_proposed()` 的外层循环语义

当前 `run_proposed()` 的外层 `while t < self.env.t_max:` 表示依次处理每个 DNN。  
改造后应改为：

- `t` 表示系统时隙
- `next_dnn_index` 表示下一个尚未到达且尚未处理的 DNN

每个时隙执行：

1. `self.env.advance_time_slot()`
2. 若 `next_dnn_index < self.env.t_max`，取 `ds[next_dnn_index]`
3. 为该 DNN 执行当前的候选部署搜索
4. 对候选方案使用新的动态指标接口打分
5. 若找到最优可行方案：
   - 计算 `estimated_runtime`
   - 计算 `remaining_slots = ceil(estimated_runtime / slot_length)`
   - 调用 `self.env.add_running_dnn(...)`
6. 若无可行方案：
   - 记为失败
7. `next_dnn_index += 1`

### 6.3 修改候选方案评价函数

当前：

- `count_values1()` 调用静态 `count_accuracy()`
- `count_values2()` 调用静态 `count_operation()`
- `count_values3()` 调用静态时延

改造后：

- `count_values1()` 改为调用 `count_dynamic_accuracy_by_assignment()`
- `count_values2()` 改为调用 `count_dynamic_operation_by_assignment()`
- `count_values3()` 继续用静态 `estimate_delay_from_assignment()`，再映射成：
  - 若超时：负值 `deadline - delay`
  - 若不超时：`1 / delay`

这一步保留原始目标形式，只替换可靠性来源。

### 6.4 提交方案逻辑修改

当前部署成功后：

- 直接调用 `self.env.release_resource(...)`

改造后：

- 不再直接永久扣减 CPU
- 改为将该 DNN 注册到 `self.env.running_dnns`
- 由 `Environment` 在后续时隙统一维护其资源占用与释放

## 7. 三个基线算法的修改方案

以下三个方法：

- `run_random()`
- `run_max_resource()`
- `run_local_first()`

都应采用与 `run_proposed()` 一致的外层时隙框架。

### 7.1 保留的部分

- `run_random()` 的随机生成部署逻辑
- `run_max_resource()` 的最大资源优先递归选点逻辑
- `run_local_first()` 的本地优先递归选点逻辑

### 7.2 必须修改的部分

1. 每个时隙先调用 `self.env.advance_time_slot()`
2. 取下一个 DNN 尝试部署
3. 用动态可靠性接口而非静态可靠性接口计算指标
4. 接纳成功后通过 `add_running_dnn()` 注册运行实例
5. 不再直接永久扣减资源

### 7.3 修改优先级

建议先改 `run_random()` 做通路验证，因为它结构最简单，最适合作为时隙推进模型的第一批测试入口。

## 8. `metrics.py` 与 `runner.py` 的修改方案

### 8.1 `metrics.py`

可保留现有 `ExperimentMetrics` 结构，但更新字段语义：

- `avg_delay`：平均估计时延
- `avg_operation`：平均动态运行可靠性
- `avg_accuracy`：平均动态精度可靠性
- `failure_count`：部署失败或拒绝数
- `runtime_ms`：程序运行耗时

如果希望更清晰，可新增：

- `avg_node_load`
- `avg_link_load`
- `avg_node_heat`
- `avg_link_heat`

### 8.2 `runner.py`

`runner.py` 只需要做最小调整：

1. 保持原有命令行入口不变
2. 输出说明中的指标语义更新为“动态模型下的统计值”
3. 若新增平均负载或热度指标，再补充打印逻辑

## 9. 推荐实现顺序

建议按照以下顺序实施：

1. 在 `models.py` 中扩展 `Node`、`LinkNode`，新增 `RunningDNN`
2. 在 `Environment` 中增加：
   - `running_dnns`
   - 时隙推进逻辑
   - 节点负载率更新
   - 链路负载率更新
   - 节点热度更新
   - 链路热度更新
   - 节点动态可靠性刷新
   - 链路动态可靠性刷新
   - 动态指标接口
3. 先改 `run_random()`，验证：
   - 一个 DNN 可以进入 `running`
   - `remaining_slots` 会递减
   - 运行结束后资源会释放
   - 后续 DNN 能感知前序 DNN 带来的动态负载
4. 再改 `run_max_resource()` 和 `run_local_first()`
5. 最后改 `run_proposed()`，只替换评价和提交逻辑，不重写种群搜索结构

## 10. 测试与验证方案

至少需要覆盖以下测试场景：

### 10.1 单 DNN 场景

验证点：

- 只有一个 DNN 时，动态模型输出应与静态模型接近
- `remaining_slots` 能正确从初值递减到 0
- 运行结束后节点负载回到 0

### 10.2 双 DNN 重叠场景

验证点：

- 第一个 DNN 运行期间，第二个 DNN 接纳时看到的节点可靠性已经下降
- 第二个 DNN 的可行部署结果与静态情况下不同

### 10.3 热度恢复场景

验证点：

- 当后续时隙没有运行 DNN 时，节点 `heat` 按恢复系数衰减
- 节点 `o_reliability` 与 `a_reliability` 回升
- 链路 `heat` 按恢复系数衰减，链路 `reliability` 回升

### 10.4 唯一节点聚合场景

验证点：

- 同一节点承载多个子任务时，动态运行可靠性只按唯一节点集合计算
- 不再按任务数重复连乘同一节点运行可靠性

### 10.5 多算法一致性场景

验证点：

- 主算法和三个基线都通过同一套 `Environment` 动态接口计算可靠性
- 避免不同算法口径不一致

### 10.6 链路路径场景

验证点：

- 两个部署方案若节点集合相近但链路路径不同，其动态运行可靠性应体现路径差异
- 云边路径在高占用时的链路可靠性应低于低占用的边边路径

## 11. 第一版明确不做的内容

为了控制复杂度，第一版实现明确不包含：

1. 节点可用性建模
2. 节点运行中断和故障恢复
3. 传输级状态机
4. 子任务级状态机
5. 子任务迁移与重调度
6. 多个新 DNN 同时到达
7. 子任务完成后的局部资源释放

这些能力应作为后续增强版本，而不是第一版实现目标。

## 12. 总结

这份修改方案的核心是：在不推翻原始部署搜索框架的基础上，增加一个共享的离散时隙运行层，把原本“部署即结束”的静态流程改造成“部署后持续运行、占用资源并影响后续 DNN”的动态流程，并把链路基础可靠性显式并入运行成功率。

第一版实现重点应放在：

1. 运行中 DNN 集合
2. 时隙推进
3. 节点与链路负载率、热度
4. 节点与链路动态可靠性
5. 基于当前系统状态的部署评价

只要这五部分打通，当前代码就能完成从静态多 DNN 部署到带基础链路可靠性的简化动态时序运行模型的首次转换。
