# 实验方案与执行手册

本文档是当前项目实验工作的唯一入口，统一说明实验目标、公平性原则、参数口径、执行命令、结果文件、论文产图和统计分析。实验行为以当前代码为准：

- 批量实验入口：`experiment_runner.py`
- 通用结果绘图：`plot_results.py`
- 论文图片生成：`plot/generate_paper_figures.py`
- 系统环境：`core/environment.py`、`core/topology.py`
- 演化与基线算法：`proposed.py`、`rtbl/`

实验论证顺序为：

```text
参数选择 -> Pareto 搜索质量 -> 在线部署性能 -> 动态鲁棒性
         -> 模块贡献 -> 可扩展性 -> 应用偏好适配
```

## 1. 实验口径

### 1.1 算法名称

| 论文名称 | 代码名称 | 用途 |
|---|---|---|
| DAREED | `customized` | 完整定制化算法 |
| NSGA-ED | `proposed` | 不含 DAG 定制算子的演化基线 |
| RDA | `random` | 随机部署基线 |
| SAA | `sa` | 多目标模拟退火基线 |
| RTBL | `rtbl` | 在线调度基线 |
| MRD | `maxresource_fast` | 最大资源部署基线 |
| LPD | `localfirst` | 本地优先部署基线 |

两类实验使用不同的默认算法集合：

- Pareto 前沿质量：`customized,proposed,random,sa`
- 最终在线部署性能：`customized,proposed,random,localfirst,maxresource_fast,rtbl`

`sa` 生成 Pareto 候选集，但不进入默认在线部署对比；`rtbl` 进入在线部署对比，但不计算 Pareto 前沿。

### 1.2 公平性与可复现性

- 同一个“场景 × 重复编号”对所有算法使用相同随机种子。
- 每次运行重新创建环境，使算法面对相同的节点属性、链路属性、DNN DAG 和请求约束。
- 默认每个场景重复 `10` 次，报告均值和样本标准差。
- 正式论文结果建议将重复次数增加到 `20` 或 `30`。
- 默认基础随机种子为 `20260613`。
- `task_id` 包含场景参数、算法、重复编号和种子；重复执行同一命令会跳过已完成任务，修改参数后不会误用旧结果。

### 1.3 默认搜索参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `population_size` | `60` | 搜索多样性和单代成本的折中 |
| `iteration_limit` | `200` | 搜索充分性和运行时间的折中 |
| `mutation_probability` | `0.25` | 参数敏感性实验的中心点 |
| `cross_over_pm` | `0.50` | 当前演化算法交叉配置 |
| `repeats` | `10` | 每个场景的重复次数 |
| `base_seed` | `20260613` | 可复现实验的基础种子 |

这些默认值必须在参数敏感性实验之后再作为论文最终口径确认，不能只凭单次结果宣称最优。

### 1.4 统一指标

在线部署指标：

- `avg_estimated_delay`：平均估计推理时延
- `avg_operational_stability_score`：平均运行稳定性分数
- `avg_inference_fidelity_score`：平均推理保真性分数
- `avg_total_energy`：平均总能耗
- `rejected_or_failed_count`：拒绝或失败部署数
- `runtime_ms`：算法运行时间

Pareto 质量指标：

- `pareto_point_count`：所有成功 DNN 的局部前沿点总数
- `pareto_point_count_mean`：每个成功 DNN 的平均前沿点数
- `pareto_hypervolume`：DNN 级归一化 Hypervolume 均值，论文主指标
- `pareto_hypervolume_sum`：所有成功 DNN 的 Hypervolume 总和
- `pareto_hypervolume_std`：DNN 级 Hypervolume 样本标准差
- `pareto_hypervolume_legacy_global`：混合全部 DNN 点的旧口径，只用于兼容

### 1.5 输出文件

每个 suite 写入 `experiment_results/<suite>/`：

- `raw_results.csv`：每次重复运行的原始结果
- `summary.csv`：按场景和算法聚合后的均值与样本标准差
- `pareto_points.csv`：仅 `pareto` suite 生成，保存可行非支配点

## 2. 推荐实验与论文顺序

| 顺序 | 实验章节 | 核心问题 | 论文图片 |
|---:|---|---|---|
| 1 | Parameter Sensitivity | 默认搜索参数是否合理 | `fig_param_population.pdf`、`fig_param_iteration.pdf`、`fig_param_mutation.pdf` |
| 2 | Pareto Front Quality | DAREED 是否产生更高质量候选集 | `fig_pareto_3d.pdf`、`fig_pareto_hv.pdf` |
| 3 | Overall Performance | 请求负载增加时在线性能是否更好 | `fig_overall_quality.pdf`、`fig_overall_delay_energy.pdf`、`fig_overall_failure_runtime.pdf` |
| 4 | Dynamic Stability Degradation | 动态退化加剧时是否更稳健 | `fig_dynamic_degradation.pdf` |
| 5 | Ablation Study | 定制模块分别贡献了什么 | `fig_ablation.pdf` |
| 6 | Large-scale Network Evaluation | 更大拓扑下是否可扩展 | `fig_scale.pdf` |
| 7 | Application Preference Scenarios | 是否适配不同应用偏好 | `fig_preference.pdf` |

论文小节建议使用同样顺序，避免先展示最终性能、后解释参数选择。

## 3. 实验执行

所有命令均从项目根目录执行。

### 3.1 Parameter Sensitivity

目的：

- 确认 `population_size = 60`、`iteration_limit = 200`、`mutation_probability = 0.25` 是否合理；
- 观察搜索质量与运行时间的权衡；
- 只分析 DAREED 内部参数，不用于证明其优于其他算法。

主实验采用单因素控制法：

| 参数 | 取值 |
|---|---|
| 算法 | `customized` |
| DNN 请求数 | `30` |
| 种群规模 | `20,40,60,80,100` |
| 迭代次数 | `80,120,160,200,240,280` |
| 变异概率 | `0.10,0.15,0.20,0.25,0.30,0.35` |
| 重复次数 | `10` |

```bash
python3 experiment_runner.py \
  --suite sensitivity \
  --dnn-count 30 \
  --repeats 10
```

重点分析：

- 种群规模在 `60` 或 `80` 后是否出现边际收益递减；
- 迭代次数在 `200` 附近是否趋于稳定，而运行时间继续增长；
- 变异概率在 `0.20 ~ 0.30` 范围内是否更稳定。

可选高负载邻域验证：

```bash
python3 experiment_runner.py \
  --suite sensitivity \
  --dnn-count 40 \
  --population-sizes 40,60,80 \
  --iteration-limits 160,200,240 \
  --mutation-probabilities 0.20,0.25,0.30 \
  --repeats 10 \
  --output-dir experiment_results/sensitivity_high_load
```

该命令仍采用单因素控制法，共 `3 + 3 + 3 = 9` 个场景、`90` 次运行。它用于验证默认点在高负载下仍处于稳定邻域，不替代主 sensitivity 实验。

### 3.2 Pareto Front Quality

比较 DAREED、NSGA-ED、RDA 和 SAA：

```bash
python3 experiment_runner.py \
  --suite pareto \
  --dnn-count 30 \
  --repeats 10
```

Pareto 支配和 Hypervolume 使用统一的四目标满意度：

- 推理保真性 `inference_fidelity_satisfaction`
- 运行稳定性 `operational_stability_satisfaction`
- 时延满意度 `delay_satisfaction`
- 能耗满意度 `energy_satisfaction`

Hypervolume 不混合不同 DNN 的点。先按 `dnn_index` 分组，再计算：

```text
inference_fidelity_norm    = clamp(inference_fidelity_satisfaction, 0, 1)
operational_stability_norm = clamp(operational_stability_satisfaction, 0, 1)
delay_norm                 = clamp(delay_satisfaction, 0, 1)
energy_norm                = clamp(energy_satisfaction, 0, 1)
reference                  = (0, 0, 0, 0)
```

最后对成功 DNN 的四维 Hypervolume 取均值。原始总能耗仍通过 `total_energy` 和 `avg_total_energy` 单独报告。

Random 为每个 DNN 采样候选池并过滤四目标非支配解。SAA 使用多组偏好权重运行退火轨迹，将可行候选合并后过滤非支配解。

论文解释：

- `fig_pareto_3d.pdf` 展示前沿分布；
- `fig_pareto_hv.pdf` 用 Hypervolume 和平均点数定量比较候选集质量；
- 不使用 `pareto_hypervolume_legacy_global` 得出论文结论。

### 3.3 Overall Performance

比较所有在线部署算法，DNN 请求数为 `5,10,20,30,40`：

```bash
python3 experiment_runner.py \
  --suite overall \
  --dnn-counts 5,10,20,30,40 \
  --repeats 10
```

重点分析：

- 请求数增加时动态可靠性的下降速度；
- 失败部署数的增长速度；
- 时延和能耗是否保持可控；
- DAREED 的搜索成本是否处于可接受范围。

对应图片：

- `fig_overall_quality.pdf`
- `fig_overall_delay_energy.pdf`
- `fig_overall_failure_runtime.pdf`

### 3.3.1 RTBL 独立重跑入口

RTBL 的最终动态指标按候选部署接纳后的预测节点/链路负载和热度计算，
与 DAREED 使用相同的动态可靠性口径。修改 RTBL 后无需重跑其他算法：

```bash
python3 rtbl_experiment_runner.py
```

该入口在 `overall,dynamic,scale` 中自动检测已经存在 `raw_results.csv` 的
suite，避免意外创建只有 RTBL 的不完整对比集。每类新结果先在临时目录完整
生成，成功后再原子替换目标 `raw_results.csv` 中 `algorithm=rtbl` 的行，并
基于保留的其他算法结果重新生成 `summary.csv`。

只更新 overall：

```bash
python3 rtbl_experiment_runner.py --suites overall
```

小规模隔离验证：

```bash
python3 rtbl_experiment_runner.py \
  --suites overall \
  --dnn-counts 5 \
  --repeats 1 \
  --output-dir /tmp/rtbl_validation
```

### 3.4 Dynamic Stability Degradation

动态 suite 内置三档场景：

| 场景 | 节点退化 | 链路退化 | 热度记忆 |
|---|---|---|---|
| Weak | 较弱 | 较弱 | `lambda_h=lambda_g=0.50` |
| Medium | 环境默认值 | 环境默认值 | `lambda_h=lambda_g=0.70` |
| Strong | 较强 | 较强 | `lambda_h=lambda_g=0.90` |

```bash
python3 experiment_runner.py \
  --suite dynamic \
  --dnn-count 30 \
  --repeats 10
```

重点分析从 Weak 到 Strong 时：

- DAREED 的运行可靠性和精度可靠性是否下降更慢；
- 强退化场景的失败数是否低于基线；
- 动态资源状态是否实际影响部署选择。

对应图片：`fig_dynamic_degradation.pdf`。

### 3.5 Ablation Study

| 变体 | 含义 |
|---|---|
| `full` | 完整 DAREED |
| `wo_dag_init` | 关闭 DAG-aware initialization |
| `wo_block_crossover` | 关闭 DAG block crossover |
| `wo_skew_mutation` | 关闭 skew mutation |
| `wo_elite_local_search` | 关闭 elite local search |
| `nsga_only` | 同时关闭上述四个定制模块 |

```bash
python3 experiment_runner.py \
  --suite ablation \
  --algorithms customized \
  --dnn-count 30 \
  --repeats 10
```

重点比较可靠性、失败数、能耗、时延和运行时间。`full` 与 `nsga_only` 的差异用于说明全部定制模块的整体贡献，其余变体用于解释单个模块作用。

对应图片：`fig_ablation.pdf`。

### 3.6 Large-scale Network Evaluation

| 场景 | 云节点 | 边缘节点 | 用户节点 | 默认 DNN 数 |
|---|---:|---:|---:|---:|
| Small | 1 | 10 | 30 | 20 |
| Medium | 1 | 30 | 120 | 60 |
| Large | 1 | 60 | 300 | 120 |

```bash
python3 experiment_runner.py \
  --suite scale \
  --repeats 10
```

低成本通路验证：

```bash
python3 experiment_runner.py \
  --suite scale \
  --algorithms maxresource_fast \
  --small-dnn-count 1 \
  --medium-dnn-count 1 \
  --large-dnn-count 1 \
  --repeats 1
```

重点分析运行时间增长、失败数、时延和能耗。当前环境使用稀疏图 Dijkstra 路径缓存，避免大规模场景重复求最短路。

注意：当前 `plot/generate_paper_figures.py` 的 `fig_scale` 固定绘制 NSGA-ED。若论文要用该图证明 DAREED 的可扩展性，正式出图前应同步调整绘图脚本的算法选择。

### 3.7 Application Preference Scenarios

偏好场景：

- `delay_sensitive`
- `stability_sensitive`
- `energy_sensitive`
- `balanced`

```bash
python3 experiment_runner.py \
  --suite preference \
  --algorithms customized \
  --dnn-count 30 \
  --repeats 10
```

偏好只影响候选集中的最终方案选择，不改变统一四目标 Pareto 评价口径。重点观察：

- Delay-sensitive 是否降低时延；
- Stability-sensitive 是否提高 OSS，并观察对 IFS 的权衡；
- Energy-sensitive 是否降低总能耗；
- Balanced 是否形成稳定折中。

对应图片：`fig_preference.pdf`。

## 4. 绘图与统计

### 4.1 通用诊断图

```bash
python3 plot_results.py \
  --results-dir experiment_results \
  --output-dir experiment_figures
```

只处理指定 suite：

```bash
python3 plot_results.py \
  --results-dir experiment_results \
  --output-dir experiment_figures \
  --suites overall,dynamic,pareto
```

只绘制指定指标：

```bash
python3 plot_results.py \
  --results-dir experiment_results \
  --output-dir experiment_figures \
  --metrics avg_estimated_delay,avg_operational_stability_score,avg_total_energy
```

输出包括每个 suite 的 SVG 指标图、Pareto 二维/三维图和 `manifest.json`。

### 4.2 论文图片

全部主实验完成后执行：

```bash
python3 plot/generate_paper_figures.py
```

输出目录：

- `plot/figures/*.svg`
- `plot/figures/*.pdf`
- `plot/figures/manifest.json`

该脚本直接执行，没有 `--help` 或 suite 过滤模式，并要求七类实验结果文件已经存在。论文 LaTeX 优先引用 PDF。

### 4.3 统计要求

- 折线图和柱状图必须带样本标准差误差线。
- 同一场景、同一随机种子的算法结果使用配对检验。
- 两算法比较可使用 Wilcoxon signed-rank test。
- 多算法整体比较建议先做 Friedman test，再做带多重比较校正的事后检验。
- 同时报告效应大小或相对改进率，不能只报告 `p` 值。
- 失败数、运行时间、时延和能耗属于越低越好；可靠性和 Hypervolume 属于越高越好。

## 5. 一次性执行清单

```bash
# 1. 参数敏感性
python3 experiment_runner.py --suite sensitivity --dnn-count 30 --repeats 10

# 2. Pareto 前沿质量
python3 experiment_runner.py --suite pareto --dnn-count 30 --repeats 10

# 3. 不同请求负载下的整体性能
python3 experiment_runner.py --suite overall --dnn-counts 5,10,20,30,40 --repeats 10

# 4. 动态可靠性退化
python3 experiment_runner.py --suite dynamic --dnn-count 30 --repeats 10

# 5. 消融实验
python3 experiment_runner.py --suite ablation --algorithms customized --dnn-count 30 --repeats 10

# 6. 大规模网络
python3 experiment_runner.py --suite scale --repeats 10

# 7. 应用偏好
python3 experiment_runner.py --suite preference --algorithms customized --dnn-count 30 --repeats 10

# 8. 论文图片
python3 plot/generate_paper_figures.py
```

资源有限时优先完成 `sensitivity -> pareto -> overall -> dynamic -> ablation`，再补充 `scale` 和 `preference`。

## 6. 参数参考

### 6.1 当前批量入口已暴露参数

| 参数 | 默认值 | 用途 |
|---|---:|---|
| `--suite` | 必填 | `overall,dynamic,sensitivity,ablation,pareto,scale,preference` |
| `--algorithms` | 按 suite 决定 | 逗号分隔算法列表 |
| `--repeats` | `10` | 每个场景重复次数 |
| `--base-seed` | `20260613` | 基础随机种子 |
| `--output-dir` | `experiment_results` | 结果根目录 |
| `--dnn-count` | `30` | 单场景 DNN 请求数 |
| `--dnn-counts` | `5,10,20,30,40` | overall 请求数列表 |
| `--population-size` | `60` | 默认种群规模 |
| `--iteration-limit` | `200` | 默认迭代次数 |
| `--population-sizes` | `20,40,60,80,100` | sensitivity 种群规模列表 |
| `--iteration-limits` | `80,120,160,200,240,280` | sensitivity 迭代次数列表 |
| `--mutation-probabilities` | `0.10,0.15,0.20,0.25,0.30,0.35` | sensitivity 变异概率列表 |
| `--small-dnn-count` | `20` | Small 场景请求数 |
| `--medium-dnn-count` | `60` | Medium 场景请求数 |
| `--large-dnn-count` | `120` | Large 场景请求数 |

查看实际入口参数：

```bash
python3 experiment_runner.py --help
```

### 6.2 系统与动态模型参数

| 类别 | 关键参数 | 当前默认/范围 | 主要影响 |
|---|---|---|---|
| 实验规模 | `t_max` | `40` | 在线负载与运行时间 |
| 拓扑 | `cloud_count,edge_count,user_count` | `1,10,约30` | 网络规模与资源密度 |
| 拓扑 | `edge_link_factor` | `2.0` | 边缘路径丰富度 |
| DNN | `task_num` | `8 ~ 19` | 搜索空间和 DAG 复杂度 |
| DNN | `delay` | `500 ~ 1999` | 时延约束强度 |
| DNN | `cpu_need` | `1 ~ 2` | 资源竞争 |
| DNN | `float_num` | `50 ~ 1000` | 计算时延与能耗 |
| DNN | `float_tran` | `2400 ~ 3992` | 链路时延、拥塞与能耗 |
| 节点 | `comp_power` | 云18、边10、端4 | 计算能耗 |
| 动态时序 | `slot_length` | `100.0` | 运行块占用时隙数 |
| 节点可靠性 | `alpha_r,beta_r` | `0.8,0.5` | 负载和热度对运行可靠性的影响 |
| 精度可靠性 | `alpha_a,beta_a` | `0.5,0.3` | 负载和热度对精度可靠性的影响 |
| 节点热度 | `lambda_h` | `0.7` | 历史负载记忆 |
| 链路可靠性 | `alpha_l,beta_l` | `0.7,0.4` | 拥塞和热度对链路可靠性的影响 |
| 链路热度 | `lambda_g` | `0.7` | 历史链路负载记忆 |
| 动态带宽 | `bandwidth_heat_gamma` | `0.5` | 热度造成的带宽退化 |
| 动态带宽 | `bandwidth_load_gamma` | `0.3` | 当前负载造成的带宽退化 |
| 动态带宽 | `min_bandwidth_ratio` | `0.2` | 有效带宽下界 |

系统压力实验可依次提高 DNN 数量、task 数、计算量和传输量，并收紧 deadline；不要同时改变搜索参数，否则无法区分算法性能变化来自环境还是搜索预算。

### 6.3 DAREED 内部参数

| 参数 | 默认值 | 用途 |
|---|---:|---|
| `use_dag_aware_initialization` | `True` | DAG 感知初始化 |
| `use_dag_block_crossover` | `True` | DAG 块交叉 |
| `use_skew_mutation` | `True` | 资源感知偏斜变异 |
| `use_elite_local_search` | `True` | 精英局部搜索 |
| `cv_initial_epsilon` | `0.35` | 初始约束违反容忍度 |
| `local_search_elite_ratio` | `0.05` | 每代局部搜索精英比例 |
| `local_search_top_k` | `3` | 每个任务尝试的候选节点数 |

子膜角色为 `stability,latency,energy,feasibility`。如果进一步研究内部参数，应优先分析约束阈值和局部搜索预算，不建议在论文主实验中同时扫描所有候选节点评分权重。

### 6.4 RTBL 参数

| 参数 | 默认值 | 主要影响 |
|---|---:|---|
| `M` | `11` | RTBL 参与选择的节点数 |
| `r_min,r_max` | `0.70,0.99` | 可用性估计范围 |
| `D_Max` | `100.0` | 单步时延限制 |
| `lambda_a` | `1.0` | 精度收益权重 |
| `lambda_r` | `10.0` | 可靠性收益权重 |
| `V_range` | `200.0` | 收益相对时延代价的强度 |
| `H_off` | `100` | 离线历史窗口 |

扩大拓扑时应特别检查 `M` 和环境可用节点数量是否仍保持一致。

### 6.5 尚未暴露到 CLI 的参数

动态可靠性、动态带宽、局部搜索和部分拓扑参数目前不能直接通过批量入口调整。需要扩展实验时，可考虑为 `experiment_runner.py` 增加：

```text
--alpha-r --beta-r --alpha-a --beta-a
--alpha-l --beta-l --lambda-h --lambda-g
--edge-count --user-count --edge-link-factor
--slot-length
--bandwidth-heat-gamma --bandwidth-load-gamma --min-bandwidth-ratio
--cv-initial-epsilon
--local-search-elite-ratio --local-search-top-k
```

在这些参数真正接入 CLI 前，文档不得给出无法执行的命令。

## 7. 论文叙述建议

实验章节可以按以下逻辑展开：

> We first conduct parameter sensitivity analysis to determine the default evolutionary search configuration. Based on the selected parameters, we evaluate Pareto front quality to verify the candidate-generation stage. We then compare online deployment performance under different request loads, examine robustness under dynamic stability degradation, analyze the contribution of each customized component, evaluate scalability under larger network topologies, and finally demonstrate adaptability under different application preferences.

对应论证链条：

```text
合理参数 -> 高质量候选集 -> 更好在线部署 -> 动态环境稳健
         -> 定制模块有效 -> 规模扩展可行 -> 偏好选择灵活
```

后续修改实验代码、默认参数或论文算法名称时，应同步更新本文件，不再新建平行的实验计划文档。
