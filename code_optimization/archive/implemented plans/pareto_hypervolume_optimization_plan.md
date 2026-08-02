# Pareto Hypervolume 计算优化方案

## 1. 背景与当前问题

当前实验中，`experiment_runner.py` 会对 `proposed`、`customized`、`random` 和 `sa` 输出 `pareto_point_count` 与 `pareto_hypervolume`。

现有实现的核心流程是：

- 各算法在 `proposed.py` 中按 DNN 请求导出可行非支配点
- 每个点包含 `accuracy`、`operation`、`delay_utility` 和 `energy_utility`
- Pareto 支配关系只使用前三个目标
- `experiment_runner.py` 将一次运行内所有 `pareto_points` 合并后直接计算一个三维 Hypervolume

该口径存在以下问题：

1. 不同 DNN 请求的 Pareto 点被混合计算
2. `delay_utility` 使用原始 `1 / delay`，与可靠性目标量纲不一致
3. 大规模前沿会触发随机采样近似，影响结果稳定性
4. `pareto_point_count` 可能统计的是多个 DNN 局部前沿的总点数，而不是单一全局前沿点数

因此，当前 `pareto_hypervolume` 更适合作为粗略内部参考，不适合作为严格的 Pareto 前沿质量论文指标。

## 2. 优化目标

优化后的 Pareto 前沿质量指标应满足：

- 每个 Hypervolume 对应一个明确的多目标优化问题
- 不混合不同 DNN 请求的前沿点
- 三个目标方向统一为越大越好
- 三个目标量纲归一化，避免时延单位主导 HV 数值
- 计算结果可复现，并尽量避免随机近似
- 保留必要的旧字段兼容已有结果，但明确标注其含义

## 3. 推荐指标口径

### 3.1 三目标定义

仍保持当前三目标 Pareto 设置：

- `accuracy`：动态精度可靠性，越大越好
- `operation`：动态运行可靠性，越大越好
- `delay_utility`：时延收益，越大越好

`energy_utility` 继续写入 `pareto_points.csv`，用于能耗散点分析，不参与三目标 Pareto 支配和 Hypervolume。

### 3.2 归一化目标

建议在计算 HV 前将目标转换为归一化空间：

```text
accuracy_norm = clamp(accuracy, 0, 1)
operation_norm = clamp(operation, 0, 1)
delay_norm = clamp((deadline - estimated_delay) / deadline, 0, 1)
```

如果暂时无法在 Pareto 点中保存 `estimated_delay` 和 `deadline`，第一版可继续使用当前 `delay_utility`，但需要按每个 DNN 内的取值范围归一化：

```text
delay_norm = (delay_utility - min_delay_utility) / (max_delay_utility - min_delay_utility)
```

长期建议在 `pareto_points` 中增加：

- `estimated_delay`
- `deadline`
- `delay_satisfaction`

其中：

```text
delay_satisfaction = clamp((deadline - estimated_delay) / deadline, 0, 1)
```

### 3.3 参考点

归一化后三目标均为 `[0, 1]`，参考点固定为：

```text
reference_point = (0, 0, 0)
```

此时 Hypervolume 的范围也是 `[0, 1]`，更便于跨算法、跨场景解释。

## 4. 分组计算方案

### 4.1 每个 DNN 单独计算 HV

一次运行内，应先按 `dnn_index` 分组：

```text
points_by_dnn[dnn_index] = 当前 DNN 的可行非支配点
```

然后对每个 DNN 单独计算：

```text
hv_dnn = hypervolume(points_by_dnn[dnn_index])
```

### 4.2 场景级聚合指标

推荐输出以下指标：

- `pareto_point_count_total`：所有 DNN 的前沿点总数
- `pareto_point_count_mean`：每个成功 DNN 的平均前沿点数
- `pareto_hypervolume_mean`：每个成功 DNN 的平均 HV
- `pareto_hypervolume_sum`：所有成功 DNN 的 HV 总和
- `pareto_hypervolume_std`：DNN 级 HV 标准差

论文主指标建议使用：

```text
pareto_hypervolume_mean
```

原因是不同算法可能成功部署的 DNN 数量不同，直接使用总和会同时混入成功数量差异。成功数量应通过 `rejected_or_failed_count` 单独报告。

## 5. 代码修改点

### 5.1 `proposed.py`

当前 `_assignment_pareto_point()` 只返回目标值。建议补充时延信息：

```python
value1, value2, value3, value4, delay = self._evaluate_assignment(dnn_index, assignment)
```

返回字段增加：

```python
"estimated_delay": delay,
"deadline": self.env.ds[dnn_index].delay,
"delay_satisfaction": max(0.0, (self.env.ds[dnn_index].delay - delay) / self.env.ds[dnn_index].delay),
```

`_append_current_pareto_points()` 也应使用同样字段，避免 `proposed/customized` 与 `random/sa` 的 Pareto 点结构不一致。

### 5.2 `experiment_runner.py`

新增函数：

```python
def _group_pareto_points_by_dnn(points):
    ...

def _normalize_pareto_points(points):
    ...

def _pareto_hypervolume_by_dnn(points):
    ...
```

`_result_row()` 中不再直接对全量 `output.pareto_points` 计算单个 HV，而是：

```python
hv_stats = _pareto_hypervolume_by_dnn(output.pareto_points)
row.update(hv_stats)
```

旧字段处理建议：

- 保留 `pareto_hypervolume`，但改为等同于 `pareto_hypervolume_mean`
- 如需兼容旧实验，可新增 `pareto_hypervolume_legacy_global`

### 5.3 `pareto_points.csv`

字段建议增加：

- `estimated_delay`
- `deadline`
- `delay_satisfaction`
- `accuracy_norm`
- `operation_norm`
- `delay_norm`

这样后续绘图和结果复算不需要重新运行算法。

## 6. 精确计算与近似计算

当前 `_pareto_hypervolume()` 在网格 cell 数超过 `500_000` 时使用固定随机种子的蒙特卡洛近似。

优化后由于按 DNN 分组，每组点数显著减少，通常可以继续使用精确网格积分，减少触发近似分支的概率。

建议：

- DNN 级 HV 默认使用精确计算
- 只有单个 DNN 前沿点过多时才启用近似
- 在输出中增加 `pareto_hypervolume_is_approx` 或日志提示
- 论文实验中优先保证所有算法使用同一计算方式

## 7. 实施步骤

### Step 1：补充 Pareto 点字段

修改 `proposed.py` 中 Pareto 点生成逻辑，增加 `estimated_delay`、`deadline` 和 `delay_satisfaction`。

验收标准：

- `proposed`、`customized`、`random`、`sa` 输出字段一致
- `pareto_points.csv` 包含新增字段
- 原有三目标支配关系不变

### Step 2：实现 DNN 级 HV

在 `experiment_runner.py` 中实现按 `dnn_index` 分组后的 HV 统计。

验收标准：

- `pareto_hypervolume_mean` 可正常输出
- `pareto_hypervolume_sum` 可正常输出
- 不同 DNN 的点不会被直接混合计算

### Step 3：归一化目标

优先使用 `delay_satisfaction` 作为时延归一化目标。

验收标准：

- HV 计算空间固定为 `[0, 1]^3`
- 参考点固定为 `(0, 0, 0)`
- HV 数值可解释为归一化目标空间覆盖体积

### Step 4：更新实验文档与绘图

同步修改：

- `../../experiments.md`
- `plot_results.py`
- `summary.csv` 字段说明

验收标准：

- 文档明确说明 HV 是 DNN 级均值
- 图表标题和坐标不再暗示混合全局前沿
- 旧结果与新结果不会混淆

## 8. 回归验证建议

### 8.1 单元级验证

构造简单点集：

```text
[(0.5, 0.5, 0.5)]
```

期望 HV：

```text
0.125
```

构造两个互不支配点：

```text
[(1.0, 0.5, 0.5), (0.5, 1.0, 0.5)]
```

期望 HV：

```text
0.375
```

### 8.2 分组验证

构造两个 DNN：

```text
dnn_0: [(0.5, 0.5, 0.5)]
dnn_1: [(1.0, 1.0, 1.0)]
```

期望：

```text
hv_mean = (0.125 + 1.0) / 2 = 0.5625
hv_sum = 1.125
```

禁止将两个 DNN 的点混合后只输出一个全局 HV。

### 8.3 实验级验证

运行小规模 Pareto suite：

```bash
python experiment_runner.py \
  --suite pareto \
  --dnn-count 5 \
  --population-size 20 \
  --iteration-limit 30 \
  --repeats 2
```

检查：

- `raw_results.csv` 中新字段存在
- `pareto_points.csv` 中新增时延字段存在
- 同一场景下算法 HV 排名稳定
- 没有大量触发随机近似分支

## 9. 预期影响

优化后，Pareto 前沿质量指标会更严格：

- HV 数值可能与旧结果不一致
- 算法排名可能发生变化
- `pareto_point_count` 会更清楚地区分总点数与平均点数
- 论文中对 Pareto 前沿质量的解释更稳健

建议在论文或实验报告中说明：

```text
Hypervolume is computed per DNN request in normalized objective space and then averaged across successfully processed DNN requests.
```

中文表述：

```text
Hypervolume 在每个 DNN 请求的归一化三目标空间中独立计算，随后对成功处理的 DNN 请求取平均值。
```

## 10. 推荐结论

当前全量混合计算的 `pareto_hypervolume` 不建议作为最终论文指标。

推荐采用：

```text
主指标：pareto_hypervolume_mean
辅助指标：pareto_hypervolume_sum、pareto_point_count_mean、pareto_point_count_total
```

同时保留部署性能指标：

- 平均估计推理时延
- 平均动态运行可靠性
- 平均动态精度可靠性
- 平均总能耗
- 拒绝或失败部署数
- 运行时间

这样可以避免将“搜索前沿质量”和“最终部署性能”混在一个指标中解释。
