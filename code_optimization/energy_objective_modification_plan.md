# 引入能耗目标的代码修改方案

## 1. 目标

在当前 `proposed` 动态部署框架中增加“能耗”这一优化目标，用于在可靠性与时延之外，进一步衡量部署方案的资源开销。

本方案采用以下约束：

- 保留当前动态可靠性建模
- 保留当前时延估计模型
- 第一版先把能耗接入最终打分，不立即改成四目标 Pareto 排序
- **忽略输出回传能耗**

这里“忽略输出回传能耗”表示：

- 输入上传能耗保留
- DAG 内部依赖边传输能耗保留
- 最终结果从最后一个任务节点回传到发起节点的能耗不计入

## 2. 能耗模型

### 2.1 总能耗定义

对一个部署方案 `x`，定义总能耗为：

```text
E_total = E_comp + E_link
```

其中：

- `E_comp`：节点计算能耗
- `E_link`：链路传输能耗

### 2.2 计算能耗

对任务 `i` 部署到节点 `j`：

```text
E_comp(i,j) = P_j^comp * T_i,j
```

其中：

- `P_j^comp`：节点 `j` 的计算功率
- `T_i,j = float_i / float_rate_j`

因此：

```text
E_comp = Σ_i P_{x(i)}^comp * (float_i / float_rate_{x(i)})
```

### 2.3 链路能耗

链路能耗只统计：

1. 输入上传
2. DAG 内部依赖边的数据传输

不统计：

3. 最终结果回传

对一条传输，若数据量为 `D`，路径为 `path`，则：

```text
E_link(path, D) = Σ_e∈path (eps_e * D)
```

其中：

- `eps_e`：链路 `e` 的单位数据传输能耗

因此：

```text
E_link = E_input_upload + E_inter_task_transfer
```

### 2.4 能耗收益

为了与当前 `proposed` 中“目标值越大越好”的口径一致，定义能耗收益：

```text
f_energy = 1 / E_total
```

能耗越低，收益越高。

## 3. 逐文件修改方案

### 3.1 `models.py`

文件：

- [models.py](/Users/hlz/Documents/Project/DNN inference reliability optim/code filing/experiments/algorithms/src_python/models.py:1)

修改内容：

#### `Node` 增加字段

- `comp_power: float`
- 可选：`idle_power: float = 0.0`

说明：

- `comp_power` 表示节点执行计算时的功率
- 第一版只使用 `comp_power`

#### `LinkNode` 增加字段

- `energy_per_mb: float`

说明：

- 表示链路单位数据传输能耗
- 单位可统一为抽象能耗单位或 `J/MB`

#### 同步修改

- `Node.clone()` 需要复制新增字段
- `LinkNode` 默认初始化需要补齐新增字段

### 3.2 `core/topology.py`

文件：

- [core/topology.py](/Users/hlz/Documents/Project/DNN inference reliability optim/code filing/experiments/algorithms/src_python/core/topology.py:1)

修改内容：

#### 节点初始化补充 `comp_power`

建议按节点层级设置固定值：

- 云节点：较高
- 边缘节点：中等
- 用户节点：较低

推荐做法：

- 不使用大范围随机值
- 使用层级映射，保证实验稳定

#### 链路初始化补充 `energy_per_mb`

建议按链路类型或带宽层级设定：

- 云边链路：较高
- 边边链路：中等
- 同节点或超短路径：最低

### 3.3 `core/environment.py`

文件：

- [core/environment.py](/Users/hlz/Documents/Project/DNN inference reliability optim/code filing/experiments/algorithms/src_python/core/environment.py:1)

这是能耗接入的核心文件。

#### 新增接口一：计算能耗

新增：

- `count_compute_energy_by_assignment(dnn_index, assignment)`

实现逻辑：

1. 遍历 DNN 所有子任务
2. 获取该任务的部署节点
3. 根据：
   - `task.float_num`
   - `node.float_rate`
   - `node.comp_power`
4. 计算该任务的执行时间与能耗
5. 累加返回

公式：

```text
exec_time = float_num / float_rate
energy = comp_power * exec_time
```

#### 新增接口二：链路能耗

新增：

- `count_link_energy_by_assignment(dnn_index, assignment)`

实现逻辑建议直接复用：

- `_collect_link_weights()`

但需要修改 `_collect_link_weights()` 的统计口径：

- 保留输入上传
- 保留 DNN DAG 内依赖边传输
- **去掉最终回传**

也就是在 `_collect_link_weights()` 中删除或跳过：

- 最后一个任务节点到 `initiateNode` 的回传链路统计

然后在 `count_link_energy_by_assignment()` 中：

```text
energy += link.energy_per_mb * data_amount
```

#### 新增接口三：总能耗

新增：

- `count_total_energy_by_assignment(dnn_index, assignment)`

实现：

```python
return (
    self.count_compute_energy_by_assignment(dnn_index, assignment)
    + self.count_link_energy_by_assignment(dnn_index, assignment)
)
```

#### 可选接口：能耗收益

新增：

- `count_energy_utility_by_assignment(dnn_index, assignment)`

实现：

```python
energy = self.count_total_energy_by_assignment(dnn_index, assignment)
return 1.0 / energy if energy > 0 else 0.0
```

建议第一版提供这个接口，便于 `proposed.py` 直接调用。

### 3.4 `proposed.py`

文件：

- [proposed.py](/Users/hlz/Documents/Project/DNN inference reliability optim/code filing/experiments/algorithms/src_python/proposed.py:1)

这是主算法接入能耗目标的核心位置。

#### 扩展 `_evaluate_assignment()`

当前返回：

```python
(value1, value2, value3, delay)
```

建议改成：

```python
(value1, value2, value3, value4, delay)
```

含义：

- `value1`：动态精度可靠性
- `value2`：动态运行可靠性
- `value3`：时延收益
- `value4`：能耗收益
- `delay`：估计时延

新增逻辑：

```python
energy = self.env.count_total_energy_by_assignment(...)
value4 = 1 / energy
```

#### 增加第四个目标缓存

在 `__init__()` 中新增：

- `self.function4_values = [0.0 for _ in range(2 * self.pop_size)]`

在每轮搜索开始时同步重置。

#### 扩展 `count_values()`

当前 `count_values()` 一次写回：

- `function1_values`
- `function2_values`
- `function3_values`

应扩展为同时写回：

- `function4_values`

保留 `count_values1/2/3()` 可不动，但主流程仍建议统一走 `count_values()`。

#### 扩展 `_score()`

当前 `_score()` 只融合：

- 精度可靠性
- 运行可靠性
- 时延收益

建议改为：

```text
score =
w_a * norm(value1)
+ w_r * norm(value2)
+ w_t * norm(value3)
+ w_e * norm(value4)
```

因此 `_score()` 参数需要增加：

- `value4`
- `w_e`
- `min_e`
- `max_e`

#### 在 `run_proposed()` 中增加能耗权重

新增：

- `min_e`
- `max_e`
- `w_e`

建议第一版：

- `w_e = 0.2`

其余 `w_a / w_r / w_t` 可保持现逻辑，再通过缩放或直接并列参与打分。

#### 扩展 `best_value`

当前：

```python
best_value = [0.0, 0.0, -1.0]
```

应改成：

```python
best_value = [0.0, 0.0, -1.0, 0.0]
```

并同步修改：

- 本代最优更新逻辑
- 最终选解逻辑
- 日志输出

建议在日志中增加：

- `value4=...`

便于观察能耗收益是否真的影响了最终选择。

### 3.5 是否立即改成四目标 Pareto

第一版建议 **不修改**：

- `dominated_sort()`
- `get_res()`
- `update_res()`

也就是说：

- Pareto 分层仍按当前 3 目标执行
- 能耗先只在 `_score()` 最终选解阶段参与

这样做的优点：

- 风险低
- 兼容当前已经修正好的非支配排序与拥挤距离逻辑
- 便于先验证能耗目标是否有效

### 3.6 `metrics.py`

文件：

- [metrics.py](/Users/hlz/Documents/Project/DNN inference reliability optim/code filing/experiments/algorithms/src_python/metrics.py:1)

建议新增：

- `avg_energy: float = 0.0`

并增加动态口径别名：

- `avg_total_energy`

`to_report_rows()` 中补充：

- `avg_total_energy=...`

### 3.7 `runner.py`

文件：

- [runner.py](/Users/hlz/Documents/Project/DNN inference reliability optim/code filing/experiments/algorithms/src_python/runner.py:1)

如果 `runner.py` 已统一依赖 `metrics.to_report_rows()`，则这里只需保证：

- `avg_total_energy` 能被输出

不需要大改逻辑。

### 3.8 文档同步

建议同步更新：

- [algorithm_flow_overview.md](/Users/hlz/Documents/Project/DNN inference reliability optim/code filing/experiments/algorithms/src_python/algorithm_flow_overview.md:1)

说明当前算法已经变为：

- 三目标 Pareto 搜索
- 加能耗辅助打分的最终选解

避免后续阅读代码时误以为能耗已经进入 Pareto 排序。

## 4. 推荐实施顺序

建议按以下顺序落地：

1. `models.py` 增加 `comp_power` 和 `energy_per_mb`
2. `core/topology.py` 初始化能耗参数
3. `core/environment.py` 增加计算能耗、链路能耗、总能耗接口
4. 修改 `_collect_link_weights()`，去掉输出回传能耗统计
5. `proposed.py` 扩展 `_evaluate_assignment()` 与 `count_values()`
6. `proposed.py` 扩展 `_score()`，将能耗接入最终打分
7. `metrics.py` / `runner.py` 增加能耗输出
8. 跑实验验证能耗目标是否对选解有稳定影响

## 5. 第二阶段可选升级

如果第一版验证结果良好，再考虑把能耗真正升级成第四个 Pareto 目标。

届时需要修改：

- `dominated_sort()`
- `get_res()`

把当前三目标支配关系和拥挤距离扩展为四目标。

建议顺序仍然是：

1. 先做“能耗参与 `_score()`”
2. 再做“能耗进入 Pareto 分层”

## 6. 总结

当前最稳妥的落地方式是：

- **先把能耗接入最终选解**
- **先不改 Pareto 三目标结构**
- **忽略输出回传能耗，只统计输入上传和中间依赖传输**

这样可以用最小修改成本验证：

1. 能耗目标是否会改变部署决策
2. 时延和可靠性是否被显著破坏
3. 是否值得继续升级为四目标优化
