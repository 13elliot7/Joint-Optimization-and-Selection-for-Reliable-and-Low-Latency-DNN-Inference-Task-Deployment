# 五模块实验执行手册

## 1. 冻结口径

- 更新时间：2026-08-09
- 唯一入口：`experiments.py`
- 正式计划：`experiment_plans/formal_experiment_v1.json`
- 默认正式 repeats：20 个配对 seed
- 默认 base seed：`20260719`
- 正式结果目录：`results/`

正式实验只包含：

```text
search_quality
load
algorithm_baseline
robustness
scale
```

五个模块按论文证据链排序，而不是按代码组件排序：

| 论文问题 | 模块 | 主要证明对象 |
|---|---|---|
| customized 是否有效 | `search_quality` | 核心算法质量与定制算子 |
| 为什么需要周期化 | `load` | 逐请求遗传搜索的实时性瓶颈及周期解耦收益 |
| 端到端是否优于基线 | `algorithm_baseline` | 主系统效果 |
| 预计算方案是否耐动态变化 | `robustness` | 校验、修复和回退机制 |
| 离线规划能否扩展 | `scale` | 规划开销与在线时延边界 |

核心主张必须由 `search_quality + load + algorithm_baseline` 共同支撑；`robustness + scale` 用于说明适用范围。周期性规划是 customized 的部署机制，不作为与 customized 并列的第二个核心算法贡献。

2026-08-09 口径调整：项目以 customized 算法为核心贡献，周期性规划框架定位为规避启发式算法在线运行时间的手段。据此删除两个模块：

- `generator_ablation`：customized 与 lightweight 的系统级对比已内嵌于 `load`、`algorithm_baseline`、`robustness`、`scale`；fallback 路径占比在上述模块中作为指标报告。
- `joint_control`：自适应 (T,K,B) 联合控制属于独立贡献叙事，不再进入正式计划；其余模块均以固定 (T=5, K=20, B=10ms) 运行。审稿需要敏感性分析时，T-only/K-only/B-only 仅作附录扩展。

同时裁剪：`search_quality` 删除仅用于预算校准的 customized small/large 场景；`scale` 删除 periodic customized 额外的固定总 K=40 口径运行。被移除模块与场景的既有结果已归档至 `archives/removed_suites/2026-08-09/`，不得进入正式统计。

A 层与 B 层不得混合统计。旧 `experiment_runner.py` 和旧 suite 仅用于代码兼容或历史核对，不进入正式计划。

---

## 2. 执行顺序

### 2.1 环境审计

```bash
python3 experiments.py cleanup-legacy --verify
python3 experiments.py list
```

### 2.2 全链路冒烟

```bash
python3 experiments.py run \
  --layer all \
  --suite all \
  --quick \
  --output-root /tmp/dareed-converged-smoke

python3 experiments.py audit --output-root /tmp/dareed-converged-smoke
python3 experiments.py summarize --output-root /tmp/dareed-converged-smoke
python3 experiments.py plot --output-root /tmp/dareed-converged-smoke
```

quick 使用 1 个 seed、缩短时隙和搜索预算，只验证入口、Schema、守恒、续跑、汇总与绘图，不形成论文结论。

### 2.3 正式运行

```bash
python3 experiments.py run --plan formal_experiment_v1
python3 experiments.py audit
python3 experiments.py summarize
python3 experiments.py plot
```

正式长任务开始前必须确认 plan 文件、代码 revision、语义版本和机器环境已冻结。

---

## 3. Search Quality

协议：`isolated_snapshot_v1`。

每个实例：

```text
固定seed创建单DNN环境
→ low/medium/high背景负载快照
→ 无副作用搜索
→ 不提交资源、不推进时隙
→ 统计候选前沿和搜索开销
```

正式场景：

| 场景 | P/G/archive | 证据角色 |
|---|---|---|
| customized medium | 12/4/64 | 两层共享的正式 customized、预算、算子完整版本 |
| proposed medium | 12/4/64 | plain NSGA 对照 |
| SA medium | 12/4/64 | SA 对照 |
| no-DAG medium | 12/4/64 | 去除 DAG 定制算子 |

指标：Hypervolume、可行率、候选数、评价次数、约束拒绝率、不收敛率、档案规模和墙钟时间。

---

## 4. Load

默认 lambda：

```text
0.1, 0.25, 0.5, 1, 2 requests/slot
```

比较：

```text
periodic customized
periodic lightweight
per-request customized
```

目的：定位轻载、临界和过载区；正式分析 Goodput、接纳率、完成率、运行失败率和在线时间。

---

## 5. Algorithm Baseline

参考负载为 `lambda=1`。正式方法：

```text
periodic customized (prewarm)
periodic lightweight (prewarm)
periodic customized (cold-start对照)
per-request customized
Random
SA
Local First
Max Resource Fast
RTBL
```

`online_fast_fallback` 不再作为完整算法基线；direct/repair/fallback 路径占比在 `load` 与 `robustness` 中作为指标报告。

必须同时报告 Goodput、接纳/完成/失败、部署质量、规划总开销和在线平均/P95 时间。

---

## 6. Robustness

三种联合场景：

| 场景 | 语义 |
|---|---|
| stable | 较少随机节点、较长 uptime、较弱连续退化 |
| failure-prone | 全节点随机可用、较短 uptime |
| fast-drift | 中等故障比例、快速 load/heat/link 退化 |

正式比较只运行 periodic customized、periodic lightweight 和 per-request customized。

指标：运行失败率、发布拒绝、候选陈旧、direct/repair/fallback 和 Goodput。

---

## 7. Scale and Overhead

三个联合预设：

| 规模 | cloud/edge/user | profiles | origin groups | lambda |
|---|---|---:|---:|---:|
| small | 1/2/4 | 1 | 1 | 0.5 |
| medium | 1/5/10 | 2 | 2 | 1.0 |
| large | 1/10/20 | 4 | 4 | 2.0 |

跨算法主比较使用固定每 key 候选数。禁止对规模因子做完整笛卡尔积。

规划运行时间不再单独成 suite，本模块同时报告真实规划墙钟、仿真规划时间、在线 P95、仓库规模和 Goodput。

---

## 8. 标准产物与统计

每个 suite：

```text
raw_results.csv
summary.csv
run_manifest.json
validation_report.json
```

统一命令额外生成：

```text
reports/descriptive_statistics.csv
figures/<suite>_<metric>.svg
figures/manifest.json
```

正式统计要求：

- 至少 20 个配对 seed；
- 均值、样本标准差和 95% CI；
- 主算法比较使用配对检验；
- 多算法比较使用 Friedman 和校正后的事后检验；
- 报告效应量或配对相对改进；
- 缺失运行和失败运行不得静默删除。

当前代码生成描述性统计和 95% CI；正式显著性检验应在结果完整性审计通过后生成。

---

## 9. 旧结果

旧结果和旧生成图已经归档到：

```text
archives/experiment_results_legacy/2026-08-09/
```

2026-08-09 口径调整移除的模块与场景结果归档到：

```text
archives/removed_suites/2026-08-09/
```

验证：

```bash
python3 cleanup_legacy_results.py --verify
```

所有旧归档均为 `eligible_for_formal_statistics=false`，不得补写后进入新正式结果。
