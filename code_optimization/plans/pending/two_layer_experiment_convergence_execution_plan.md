# Customized 核心算法与周期规划实验方案

## 1. 状态与论文定位

- 状态：五模块方案与代码已冻结；正式 20-seed 运行未完成
- 修订日期：2026-08-09
- 正式计划：`experiment_plans/formal_experiment_v1.json`
- 唯一入口：`experiments.py`
- 正式结果根目录：`results/`

本文将贡献层级固定为：

```text
核心算法贡献
  customized 多目标部署优化算法
        ↓
系统实现机制
  周期性离线规划 + 方案池 + 在线校验/修复/回退
        ↓
解决的问题
  避免遗传搜索进入逐请求关键路径，在保留部署质量的同时降低在线时延
```

周期性规划框架不是与 customized 并列的第二个核心算法贡献，也不承担证明搜索算法优越性的职责。它是使 customized 能用于在线 DNN 推理系统的计算解耦机制。

本次修订只重新组织研究问题、证据角色和论文叙事，不改变当前正式计划的场景参数、seed、任务身份或结果 Schema；已完成的正式结果仍可续跑。

---

## 2. 核心研究问题

论文实验集中回答五个问题：

| 编号 | 研究问题 | 正式模块 | 结论边界 |
|---|---|---|---|
| RQ1 | customized 是否比通用搜索与去除定制算子的版本产生更好的部署解？ | `search_quality` | 算法质量与搜索机制 |
| RQ2 | customized 为什么不能直接逐请求执行，周期化是否缓解关键路径开销？ | `load` | 实时性与负载扩展性 |
| RQ3 | periodic customized 的端到端系统效用是否优于代表性基线？ | `algorithm_baseline` | 主系统结论 |
| RQ4 | 预计算方案在故障和连续漂移下是否仍可靠？ | `robustness` | 动态环境适用性 |
| RQ5 | 系统规模增大时，离线规划开销和在线时延如何变化？ | `scale` | 可扩展性与开销边界 |

论文论证顺序必须是 `RQ1 → RQ2 → RQ3 → RQ4 → RQ5`。不能先用周期系统结果替代 customized 算法有效性证明，也不能用单快照搜索时间替代在线响应时间。

---

## 3. 两层协议及职责

### 3.1 A 层：搜索质量协议

- 协议版本：`isolated_snapshot_v1`
- 固定环境快照，不提交资源，不推进时隙
- 只评价搜索质量、可行性、收敛和墙钟开销
- 不产生 Goodput、接纳率或系统吞吐结论

### 3.2 B 层：泊松生命周期协议

- 协议版本：`poisson_lifecycle_v1`
- 包含泊松到达、动态环境、资源提交、执行完成和失败结算
- 评价 Goodput、接纳/完成/失败、在线时延、规划开销和鲁棒性
- 论文系统结论只来自该层

### 3.3 禁止混合

- A/B 原始行不得进入同一统计分组；
- A 层 runtime 不得表述为请求响应时间；
- B 层 Goodput 不得用于宣称搜索器本身收敛更好；
- 不得把旧 `overall/dynamic/scale` 结果改名后纳入当前正式统计。

---

## 4. 五个正式实验模块

### 4.1 Search Quality：证明核心算法

正式预算统一为 `P=12, G=4, archive=64`，每个 repeat 使用 30 个固定快照实例。

| 方法 | 证据角色 |
|---|---|
| `customized` | 完整核心算法 |
| `proposed` | 不使用领域定制机制的 plain NSGA 对照 |
| `sa` | 非进化式启发搜索对照 |
| `no_dag_operators` | 去除 DAG 定制算子的消融 |

主指标：

- Hypervolume；
- 严格可行率；
- 约束拒绝率与不收敛率；
- 评价次数与档案规模；
- 搜索墙钟时间。

主假设：customized 在相同预算下提高 Hypervolume 或可行率；定制 DAG 算子带来可测的增益。若仅速度更慢而质量无优势，不能宣称核心算法贡献成立。

small/large 搜索预算只属于校准记录，不进入正式矩阵。

### 4.2 Load：证明周期化的必要性

负载点冻结为：

```text
lambda = 0.1, 0.25, 0.5, 1, 2 requests/slot
```

正式比较：

| 方法 | 回答的问题 |
|---|---|
| `per-request customized` | 完整 customized 直接进入请求关键路径的代价 |
| `periodic customized` | 将相同核心搜索移到周期规划阶段后的系统表现 |
| `periodic lightweight` | 不使用 customized 离线搜索时，周期框架本身能达到什么水平 |

主指标：在线平均/P95 时间、Goodput、接纳率、完成率、运行失败率和吞吐。

该模块最关键的比较是：

1. `periodic customized` 对 `per-request customized`：证明移出关键路径的实时性收益；
2. `periodic customized` 对 `periodic lightweight`：证明周期化收益之外，customized 搜索质量仍有系统价值。

不能只报告 periodic 与 per-request 的速度差；必须同时报告质量/效用损失，防止形成“单纯用旧方案换速度”的解释。

### 4.3 Algorithm Baseline：主结果表

参考负载为 `lambda=1`，固定周期参数为 `T=5 slots, K_total=20, B=10 ms`。

正式方法：

```text
periodic customized (prewarm)
periodic customized (cold start)
periodic lightweight (prewarm)
per-request customized
Random
SA
Local First
Max Resource Fast
RTBL
```

证据分组：

- 核心主方法：`periodic customized (prewarm)`；
- 机制对照：`periodic lightweight`、`per-request customized`；
- 启动条件对照：`periodic customized (cold start)`；
- 外部基线：Random、SA、Local First、Max Resource Fast、RTBL。

主表必须同时报告 Goodput、接纳/完成/失败、部署质量、在线 P95 和规划开销。不得仅按某个单目标排序后宣布总体优越。

### 4.4 Robustness：证明预计算方案不会快速失效

正式环境：

| 场景 | 语义 |
|---|---|
| `stable` | 较少随机故障、较长 uptime、较弱连续退化 |
| `failure-prone` | 更高故障参与比例、较短 uptime |
| `fast-drift` | 节点/链路/热状态更快连续变化 |

每个场景比较 periodic customized、periodic lightweight 和 per-request customized。

主指标：运行失败率、发布拒绝率、direct/repair/fallback 路径比例、Goodput、完成率以及候选陈旧相关指标。

本模块用于界定周期复用的适用范围，不用于重复宣称 customized 搜索收敛更好。

### 4.5 Scale：证明离线开销可承担

| 规模 | cloud/edge/user | profiles | origin groups | lambda |
|---|---|---:|---:|---:|
| small | 1/2/4 | 1 | 1 | 0.5 |
| medium | 1/5/10 | 2 | 2 | 1.0 |
| large | 1/10/20 | 4 | 4 | 2.0 |

比较 periodic customized、periodic lightweight 和 per-request customized，跨规模统一采用固定每 active key 的候选预算。

主指标：

- 单次规划真实墙钟及其 95% CI；
- 规划 ready-slot 映射；
- 在线平均/P95 时间；
- Goodput、完成率；
- 方案池规模和有效候选数。

规划耗时随规模增长是预期现象。论文需要证明的是该开销位于周期离线路径，同时在线时延保持受控，而不是声称规划耗时不增长。

---

## 5. 被删除或降级的实验

| 原实验 | 处理 | 原因 |
|---|---|---|
| `preference` | 删除正式 suite | 偏好响应不是本文主贡献，可作为案例图 |
| A 层 `overall/dynamic/scale` | 删除 | 与 B 层系统问题重叠且协议不一致 |
| `generator_ablation` | 合并进五模块的方法对照 | customized/lightweight 已在 load、baseline、robustness、scale 中出现 |
| `joint_control` | 删除正式 suite | MPC/规则控制会形成第二条算法贡献主线 |
| `fixed_tkb` | 降为固定参数与可选附录敏感性 | 避免完整笛卡尔积 |
| `planning_runtime` | 合并进 scale | 开销是扩展性维度，不需独立 suite |
| `bootstrap` | 合并进 baseline | 只保留一个 cold-start 对照 |
| `search_budget` | 降为校准 | 正式比较只使用冻结 medium 预算 |

`generator_ablation` 和 `joint_control` 的历史/中间结果不得进入五模块正式统计。若审稿意见要求，可用新版本计划作为附录扩展，不能回填当前 `formal_experiment_v1`。

---

## 6. 统计与图表方案

### 6.1 配对规则

- 正式实验使用 20 个配对 seed，base seed 为 `20260719`；
- 同一场景和 repeat 的算法共享拓扑、profile、到达、故障与漂移资产；
- 配对缺失时不静默删除，先标记不完整；
- 不混合 measured/deterministic、prewarm/cold-start 或不同协议版本。

### 6.2 报告规则

- 描述统计：均值、样本标准差、95% CI；
- 两方法主比较：配对检验、效应量、相对改进；
- 多方法比较：Friedman 总体检验及校正后的事后比较；
- 图表显示有效样本数，不能只显示均值；
- 明确区分搜索墙钟、规划墙钟和在线请求时间。

### 6.3 论文图表映射

| 论文位置 | 数据来源 | 推荐形式 |
|---|---|---|
| 核心算法表/图 | `search_quality` | HV/可行率对比与算子消融 |
| 周期化动机图 | `load` | 在线 P95 与 Goodput 双视图 |
| 主结果表 | `algorithm_baseline` | 多指标表 + 配对统计 |
| 动态可靠性图 | `robustness` | 失败率、Goodput、路径占比 |
| 扩展性图 | `scale` | 规划墙钟、在线 P95 随规模变化 |

---

## 7. 执行与产物

正式命令：

```bash
python3 experiments.py run --plan formal_experiment_v1
python3 experiments.py audit
python3 experiments.py summarize
python3 experiments.py plot
```

每个 suite 必须生成：

```text
raw_results.csv
summary.csv
run_manifest.json
validation_report.json
```

B 层另外保留请求、规划作业、控制动作和候选发布审计日志。任一论文统计必须能反向定位到 `task_id`、场景哈希、seed、语义版本和代码 revision。

---

## 8. 当前正式执行状态

截至本次修订，当前五模块计划共有 920 条正式任务：

| 模块 | 完成 | 状态 |
|---|---:|---|
| `search_quality` | 80/80 | 完整且验证通过 |
| `load` | 226/300 | 可续跑 |
| `algorithm_baseline` | 106/180 | 可续跑 |
| `robustness` | 0/180 | 未开始 |
| `scale` | 153/180 | 可续跑 |
| 合计 | 565/920 | 61.4% |

当前没有 suite 锁。正式统计、显著性结论和论文图只能在五个模块全部完成并通过 audit 后生成。

---

## 9. 完成条件

- [x] 论文贡献层级固定为 customized 核心算法、周期规划系统机制；
- [x] 正式计划收敛为五模块；
- [x] A/B 协议和统计边界明确；
- [x] preference、joint control 和独立 planning runtime 不进入正式主线；
- [x] 搜索预算与负载范围完成校准；
- [x] 旧结果完成可恢复归档和 SHA-256 校验；
- [x] quick 全链路与单元测试通过；
- [ ] 五模块 20-seed 正式结果全部完成；
- [ ] 完整性审计、配对统计和图表全部通过；
- [ ] 结果与计划、代码 revision 和运行环境一同封存。

只有最后三项完成后，本文件才可从 `pending` 移入已实施归档。
