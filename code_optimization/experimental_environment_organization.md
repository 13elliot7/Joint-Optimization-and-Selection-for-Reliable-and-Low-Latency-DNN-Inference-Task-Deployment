# 收敛后的实验环境与结果规范

## 1. 当前状态

- 更新时间：2026-08-09
- 实施入口：`experiments.py`
- 正式计划：`experiment_plans/formal_experiment_v1.json`
- 正式结果根目录：`results/`
- 实施方案：`plans/pending/two_layer_experiment_convergence_execution_plan.md`

项目保留两种不可互换的实验协议，但只维护一个实验产品、一个总入口和一套结果身份。旧 `experiment_runner.py` 只保留兼容调用；`periodic_experiment_runner.py` 是 B 层协议后端，不再单独定义论文矩阵。

---

## 2. 两种实验协议

| 层次 | 协议版本 | 执行语义 | 允许结论 |
|---|---|---|---|
| A：搜索质量 | `isolated_snapshot_v1` | 固定单 DNN 快照；不提交资源；不推进时隙 | Hypervolume、可行率、搜索预算、关键算子贡献 |
| B：系统性能 | `poisson_lifecycle_v1` | 泊松批量到达；动态环境；周期规划；请求生命周期 | Goodput、接纳、实时性、控制、鲁棒性、扩展性 |

冻结原则：

```text
customized算法为什么有效 -> A层
周期规划能否使customized脱离逐请求关键路径 -> B层
```

A/B 原始行不得进入同一 summary 分组；旧顺序到达协议不再产生正式结果。

---

## 3. 五个正式实验模块

```text
search_quality
load
algorithm_baseline
robustness
scale
```

| 模块 | 层次 | 主要问题 |
|---|---|---|
| `search_quality` | A | Pareto 质量和关键算子是否合理 |
| `load` | B | 逐请求 customized 的实时性瓶颈及周期解耦收益 |
| `algorithm_baseline` | B | periodic customized 的端到端效用是否优于基线 |
| `robustness` | B | 故障和快速连续退化下是否保持收益 |
| `scale` | B | 网络规模增加后的质量和计算开销如何变化 |

2026-08-09 口径调整后移除：

- `generator_ablation`：生成器差异由 `load/algorithm_baseline/robustness/scale` 内嵌的 periodic customized 与 periodic lightweight 对比及 direct/repair/fallback 指标解释；
- `joint_control`：自适应 (T,K,B) 联合控制属于独立贡献叙事，正式模块统一以固定 (T=5, K=20, B=10ms) 运行。

更早不再作为独立正式模块：

- A 层 `overall/dynamic/scale/preference`；
- B 层 `search_budget/fixed_tkb/control/preference/bootstrap/planning_runtime/availability`。

其职责分别被合并到当前模块中：

```text
overall -> load + algorithm_baseline
sensitivity + pareto + ablation -> search_quality
availability + continuous drift -> robustness
planning_runtime -> scale
bootstrap -> algorithm_baseline中的cold-start对照
```

---

## 4. 公平性与随机流

同一 comparison group 和 repeat 的算法共享：

- 拓扑与 profile catalog；
- 到达轨迹、deadline 和请求源；
- 节点可用性和连续退化轨迹；
- 目标与候选预测语义；
- 时隙事件顺序。

结果记录以下资产身份：

```text
topology_hash
profile_catalog_hash
workload_trace_hash
environment_trace_hash
```

随机流按用途隔离：

```text
topology / profile / snapshot / arrival / availability / drift / algorithm
```

算法随机调用次数不得改变外生轨迹。

---

## 5. 结果目录

```text
results/
  manifests/
  search_quality/
  system_poisson/
    load/
    algorithm_baseline/
    robustness/
    scale/
  reports/
  figures/
```

每个 suite 的标准产物：

```text
raw_results.csv
summary.csv
run_manifest.json
validation_report.json
```

B 层按需额外生成 `planning_candidate_audits.csv`；统一统计生成 `reports/descriptive_statistics.csv`；统一绘图生成 SVG 和 `figures/manifest.json`。

旧结果已于 2026-08-09 完成可恢复归档并从工作区清理：

```text
archives/experiment_results_legacy/2026-08-09/
```

归档结果全部标记为不可进入正式统计。

---

## 6. 结果身份与禁止混合项

每条正式结果至少包含：

```text
task_id
scenario_id
experiment_layer
protocol_version
suite
scenario
comparison_group
comparison_method
algorithm
repeat
seed
semantic_versions
timing_mode
```

以下结果不得混合：

- A 层与 B 层；
- 不同协议或目标语义版本；
- measured 与 deterministic 计时；
- cold-start 与 prewarm；
- 不同候选生成器或搜索预算；
- fixed-total K 与 fixed-per-key K。

---

## 7. 统一命令

列出正式模块：

```bash
python3 experiments.py list
```

全链路冒烟：

```bash
python3 experiments.py run --layer all --suite all --quick
python3 experiments.py audit
python3 experiments.py summarize
python3 experiments.py plot
```

运行正式计划：

```bash
python3 experiments.py run --plan formal_experiment_v1
```

单独运行：

```bash
python3 experiments.py run --layer search_quality --suite search_quality
python3 experiments.py run --layer system_poisson --suite algorithm_baseline
```

---

## 8. 当前结论边界

当前代码已完成五模块入口、标准产物、续跑、验证、统计报告和无 Matplotlib 依赖的 SVG 绘图闭环。quick 全链路可以运行。

正式 20-seed 计划已部分执行：`search_quality` 完成，`load/algorithm_baseline/scale` 可续跑，`robustness` 尚未开始；整体为 565/920。当前仍不能宣称 periodic customized 显著优于其他基线。正式结论必须来自冻结计划生成的完整 `results/`，并通过配对统计和完整性审计。
