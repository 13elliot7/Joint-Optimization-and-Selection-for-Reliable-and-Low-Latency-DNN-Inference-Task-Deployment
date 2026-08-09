# 删除 IFS 并迁移为三目标优化的执行方案

## 1. 文档状态

- 状态：已实施（2026-08-09）
- 生成日期：2026-08-09
- 实施对象：当前周期性规划主路径、统一泊松基线、A 层算法质量实验
- 核心目标：彻底取消无数据支撑的推理保真度分数（IFS）对算法决策和论文结论的影响
- 目标优化语义：`OSS + delay satisfaction + energy satisfaction`

本方案不把 IFS 合并进 OSS，也不把旧 IFS 重命名为“质量”后继续使用。历史代码、历史 CSV 和归档文档只允许作为只读追溯材料存在。

---

## 2. 修改理由与最终决策

当前 IFS 是画像—节点静态代理分数，缺少真实模型、数据集、量化配置或设备 profiling 数据支撑。继续将其作为 Pareto 目标会产生以下风险：

1. 节点 IFS 数值来源无法复现；
2. IFS 无法与真实 accuracy、数值误差或任务成功概率对应；
3. IFS 与 OSS 缺少稳定的目标冲突证据；
4. 无依据的 IFS 会影响候选生成、在线排序和 Goodput；
5. 四维 Hypervolume 会把该合成维度计入算法优势。

冻结以下决策：

```text
主优化目标：       [OSS, delay_satisfaction, energy_satisfaction]
当前节点在线状态： 硬约束
资源/层级/链路：   硬约束
IFS：              从活动模型、算法、结果和论文主张中删除
历史四目标结果：   只读归档，不迁移为三目标结果
```

删除后的主问题为：

\[
\max_x \left[OSS(x), D_{sat}(x), E_{sat}(x)\right]
\]

满足：

\[
x \in \mathcal{F}_{resource, hierarchy, topology, availability, deadline}
\]

---

## 3. 实施边界

### 3.1 本次必须删除

- 节点和画像的 IFS 先验；
- `profile_node_fidelity` 注册表及版本；
- 候选状态中的 `inference_fidelity`；
- 请求中的 `preference_fidelity`；
- 方案仓库中的 `planning_reference_ifs`；
- Customized/Proposed/Random/SA 的 IFS 目标；
- 在线调度和 Goodput 中的 IFS 权重；
- 结果中的 `avg_inference_fidelity_score`；
- Pareto 点、Hypervolume 和绘图中的 IFS 维度；
- `ifs_mode` 语义版本字段；
- RTBL 节点收益中的 fidelity 项。

### 3.2 必须保留

- OSS 及其节点连续可用性、链路稳定性诊断；
- 当前节点在线硬约束；
- delay、energy、fixed-point 和 offered-rate 语义；
- Poisson 请求、周期候选池、在线复验和联合 T/K/B 控制；
- 历史结果迁移工具中识别旧 IFS 列的能力，但不得将旧 4D-HV 转换为新 3D-HV。

### 3.3 暂不扩展

- 不在本轮引入新的真实 accuracy 指标；
- 不增加模型压缩、量化或 early-exit 决策；
- 不把 OSS 改造成严格联合成功概率；
- 不同时重写链路随机失效模型。

---

## 4. 新的信息边界与数据模型

### 4.1 请求字段

将请求偏好改为三个显式权重：

```python
InferenceRequest(
    preference_stability: float,
    preference_delay: float,
    preference_energy: float,
)
```

要求：

- 三个权重非负；
- 总和必须大于零；
- 构造时归一化或在评分函数中按总和归一化；
- `preference_mode` 可继续提供 stability-sensitive、delay-sensitive、energy-sensitive、balanced 四种模板。

不得继续保留名称含糊的 `preference_operation + preference_fidelity + 固定0.25` 公式。

### 4.2 观测快照

从 `ObservationSnapshot` 删除：

```text
profile_fidelity_version
profile_node_fidelity
```

快照继续包含：节点在线状态、历史可用性、可用 CPU、负载、热度、链路负载和有效带宽。

### 4.3 候选状态

从 `CandidateStateSnapshot` 删除：

```text
inference_fidelity
```

保留以下可解释字段：

```text
node_availability_score
availability_prediction_confidence
link_stability_score
operational_stability
delay_satisfaction
energy_satisfaction
```

### 4.4 节点模型

从活动 `Node` 模型删除：

```text
inference_fidelity
base_inference_fidelity
```

从 `Environment` 删除：

```text
alpha_a
beta_a
a_min
profile_node_fidelity
profile_fidelity_version
_initialize_profile_node_fidelity()
set_profile_node_fidelity()
_profiled_inference_fidelity()
count_inference_fidelity_by_assignment()
```

若旧结果迁移脚本需要识别这些名称，应在迁移工具内部维护字符串常量，不再依赖活动模型字段。

---

## 5. 三目标算法改造

### 5.1 目标值结构

将：

```python
ObjectiveValues(
    inference_fidelity,
    operational_stability,
    delay_satisfaction,
    energy_satisfaction,
)
```

改为：

```python
ObjectiveValues(
    operational_stability,
    delay_satisfaction,
    energy_satisfaction,
)
```

目标向量冻结为：

```python
(operational_stability, delay_satisfaction, energy_satisfaction)
```

建议同时将 `function1_values` 等位置式数组重构为具名三目标容器，避免删除第一维后产生索引错位。若为控制实施风险而暂时保留数组，必须冻结映射：

```text
function1 = OSS
function2 = delay satisfaction
function3 = energy satisfaction
```

不得保留空的第四目标参与支配排序或拥挤距离。

### 5.2 Customized/Proposed

修改：

- 三目标支配关系；
- 三目标 Pareto level sort；
- 三目标 crowding distance；
- archive 去重与非支配筛选；
- elite local search 的候选评分；
- 初始化中的 `quality_score`，只使用 OSS 或与约束/资源组合，不再读取 IFS；
- `CustomizedCandidate.objectives` 从四元组改为三元组；
- 搜索统计字段保持不变。

Customized 的四个膜角色当前为 stability、latency、energy、feasibility，不依赖 IFS 角色，可继续保留。

### 5.3 最终偏好评分

统一使用：

\[
U(x)=\frac{w_s OSS(x)+w_dD_{sat}(x)+w_eE_{sat}(x)}{w_s+w_d+w_e}
\]

以下模块必须调用同一个具名评分函数：

- Customized 最终候选选择；
- 周期方案参考排序；
- 在线完整复验排序；
- 逐请求 Customized；
- Random 和 SA；
- 请求完成后的 Goodput utility。

禁止各模块继续维护不同的隐式常数权重。

### 5.4 Random、SA、Local First、Max Resource

- Random 和 SA 使用三目标候选评分；
- Local First 和 Max Resource 仍按自身启发式顺序生成候选，最终可行性由统一预测器判断；
- 它们不因删除 IFS 而读取新的系统内部信息。

### 5.5 RTBL

当前 RTBL 收益包含 fidelity benefit。删除后改为 availability-delay 版本：

\[
F_1(S)=V\lambda_r\left(1-\prod_{j\in S}(1-\tilde r_j)\right)
\]

删除 `lambda_a` 和 `a_j`。代码和论文名称应明确为：

```text
RTBL-availability adaptation
```

不能声称该适配器与原含 accuracy/fidelity 收益的 RTBL 完全等价。

---

## 6. 周期规划与方案仓库

### 6.1 DeploymentPlan

删除：

```text
planning_reference_ifs
```

保留：

```text
planning_reference_oss
planning_reference_delay_satisfaction
planning_reference_energy_satisfaction
planning_reference_availability
```

方案 ID、仓库缓存键和语义版本必须变化，防止旧四目标方案被新调度器读取。

### 6.2 周期触发

删除 `profile_fidelity_changed` 触发原因及其基线版本记录。继续支持：

- topology change；
- availability drift；
- node/link load drift；
- fixed interval；
- fallback/rejection/profile-distribution drift。

### 6.3 在线调度

删除所有 `planning_reference_ifs` 和 `candidate.inference_fidelity` 排序项。快速过滤、完整复验、repair、fallback 和原子提交顺序不变。

### 6.4 Goodput

Goodput 的单请求效用改为三目标效用。结果中必须额外记录三项平均原值，避免只报告组合效用。

---

## 7. 语义版本与结果 Schema

### 7.1 新版本建议

```text
objective_semantics_version = stability_delay_energy_v4
candidate_prediction_version = unified_fixed_point_v3
result_schema_version = periodic_three_objective_v3
```

从 `SemanticVersions` 删除：

```text
inference_fidelity
ifs_mode
```

保留旧版本常量只用于识别和拒绝历史结果，不允许新实验选择旧 IFS 模式。

### 7.2 删除的结果字段

```text
avg_inference_fidelity_score
inference_fidelity
inference_fidelity_satisfaction
inference_fidelity_norm
planning_reference_ifs
preference_fidelity
ifs_mode
profile_fidelity_version
```

### 7.3 Pareto 与 Hypervolume

新 Pareto 点为：

```text
(operational_stability, delay_satisfaction, energy_satisfaction)
```

Hypervolume 改为三维，版本建议：

```text
pareto_hv_version = 3d_oss_delay_energy_v1
```

禁止：

- 删除旧 CSV 的 IFS 列后重新计算并标记为新结果；
- 将旧 4D-HV 与新 3D-HV 放在同一图或统计检验中；
- 在同一个 `raw_results.csv` 中追加新旧目标语义。

新结果目录建议：

```text
experiment_results_v3/
periodic_experiment_results_v4/
```

---

## 8. 文件级实施清单

| 文件/模块 | 修改内容 |
|---|---|
| `models.py` | 删除 Node、DNN、InferenceRequest、ObservationSnapshot、CandidateStateSnapshot 中的 IFS 字段 |
| `core/environment.py` | 删除 IFS 初始化、更新、画像注册表、预测和聚合；候选预测只返回 OSS/delay/energy |
| `core/arrival.py` | 生成 stability/delay/energy 三权重或偏好模板 |
| `proposed.py` | 四目标改三目标；修改排序、archive、评分、Pareto 导出和所有基线统计 |
| `planning/repository.py` | 删除方案 IFS 参考值并升级缓存/语义校验 |
| `planning/periodic_planner.py` | 删除 fidelity version 触发和 IFS 审计字段 |
| `planning/online_dispatcher.py` | 三目标请求评分 |
| `planning/request_dispatcher.py` | 三目标逐请求 Customized、Random、SA、RTBL 适配 |
| `periodic_experiment.py` | 三目标 Goodput、执行记录和均值 |
| `metrics.py` | 删除 IFS 指标并升级 Schema |
| `experiment_runner.py` | 三目标 sensitivity/Pareto/HV 和独立实例库输出 |
| `periodic_experiment_runner.py` | 删除 IFS 数值列，按新语义分组和输出 |
| `semantics.py` | 新目标、预测和结果版本；删除活动 `ifs_mode` |
| `plot_results.py` | 删除 IFS 图和四维点读取 |
| `plot/generate_paper_figures.py` | Pareto 图改为 OSS-delay-energy 三维图 |
| `rtbl/config.py`、`rtbl/sota_selector.py`、`rtbl/scheduler.py` | 删除 fidelity 收益，标记 availability-only 适配版本 |
| `migrate_legacy_result_schema.py` | 旧结果只识别/归档，不转换为新三目标结果 |
| `README.md`、`code_optimization/*.md` | 更新系统模型、实验命令、指标和结论边界 |

归档目录 `code_optimization/archive/` 和 `plans/reference/` 不做历史文本重写。

---

## 9. 实施阶段

### 阶段 A：冻结新语义与保护旧结果

1. 新增三目标语义版本；
2. 新结果目录升级；
3. CSV 校验拒绝旧 IFS/4D-HV；
4. 增加“新结果不得包含 IFS 字段”契约测试。

验收：新旧结果不可能被静默合并。

### 阶段 B：切断 IFS 的决策影响

1. 在线评分和 Goodput 改为三目标；
2. 方案参考评分删除 IFS；
3. 请求偏好改三权重；
4. RTBL 删除 fidelity benefit；
5. 增加排序单调性测试。

验收：修改任意遗留 IFS 数值不再改变 assignment、plan rank、accept/reject 或 Goodput。

### 阶段 C：三目标搜索核心

1. `ObjectiveValues` 和候选 archive 改三维；
2. 支配、拥挤距离、选择和 SA 接受准则改三维；
3. Hypervolume 改三维；
4. sensitivity 输出三目标质量指标。

验收：三目标前沿中的任意点均不被同集合其他点支配，HV 可复现且单调。

### 阶段 D：模型与快照结构清理

1. 删除 profile fidelity 注册表和版本；
2. 删除 Node、Snapshot、Candidate、Plan 的 IFS 字段；
3. 删除环境 IFS 参数和函数；
4. 删除 fidelity-change 重规划触发。

验收：活动 Python 代码中除旧迁移识别常量外不存在 `inference_fidelity`、`preference_fidelity`、`profile_node_fidelity` 或 `ifs_mode`。

### 阶段 E：实验、绘图与文档

1. 更新 A 层 sensitivity、pareto、ablation；
2. 更新 B 层全部 suite；
3. 更新三维 Pareto 和系统指标图；
4. 删除旧 IFS 论文结论；
5. 运行 quick、完整回归和正式多 seed 实验。

验收：所有正式命令只产生三目标语义结果。

---

## 10. 测试要求

### 10.1 单元测试

- 三目标 `ObjectiveValues.vector` 顺序固定；
- 三目标 dominance/crowding/HV 正确；
- OSS、delay、energy 任一项提高时，固定权重效用不下降；
- 零权重和非法权重被拒绝；
- Snapshot 和 Candidate 不含 IFS；
- profile fidelity API 不再存在；
- RTBL 不读取 Node fidelity；
- 方案缓存拒绝旧语义。

### 10.2 集成测试

- 周期规划能够生成、发布并复验三目标方案池；
- 同时隙多请求保持资源原子提交和请求守恒；
- 九算法共享相同 Poisson 到达数和故障轨迹；
- prewarm/cold-start 与三种控制器正常运行；
- sensitivity 不提交资源、不推进时隙；
- `--suite all --quick` 全部通过。

### 10.3 负向测试

- 新运行器读取 `stability_fidelity_v3_profiled` 时必须报错；
- 新绘图脚本读取 4D Pareto 点时必须报错；
- 新 CSV 中出现 IFS 列时必须报错；
- 旧方案仓库不得被新调度器接受。

---

## 11. 实验重跑顺序

删除 IFS 后，旧算法排名、Pareto 前沿和 Goodput 均失效。按以下顺序重跑：

1. `sensitivity --quick`：验证三目标搜索和 Schema；
2. 正式 `sensitivity`：重新确定 P/G/变异/交叉/archive；
3. `pareto` 与 `ablation`：验证三目标搜索质量；
4. 周期 `search_budget` 与 `generator_ablation`；
5. 周期 `load`：重新定位容量区间；
6. 周期 `algorithm_baseline`；
7. `control`、`availability`、`planning_runtime`、`scale`；
8. 统计检验与论文产图。

所有正式结果至少使用配对 seed。旧结果只用于解释模型迁移，不进入新论文表格。

---

## 12. 完成定义

只有同时满足以下条件才可将本计划标记为完成：

- [x] 活动系统模型不存在 IFS 状态或画像表；
- [x] 所有算法只优化 OSS、delay satisfaction、energy satisfaction；
- [x] 请求偏好和 Goodput 使用统一三权重公式；
- [x] 周期方案、仓库和在线排序不存在 IFS；
- [x] Pareto 和 Hypervolume 为明确版本化的三维口径；
- [x] RTBL 明确为 availability-only 适配；
- [x] 新旧结果目录、Schema 和语义版本严格隔离；
- [x] 全部单元测试和 `--suite all --quick` 通过；
- [x] README、系统模型、实验手册和投稿问题清单已同步；
- [x] 正式实验命令不再输出任何 IFS 字段。

---

## 13. 风险与回退

| 风险 | 处理方式 |
|---|---|
| 删除一维后算法排名明显变化 | 视为目标定义变化的正常结果，重新做 sensitivity 和 baseline |
| Pareto 多样性下降 | 调整三目标 crowding、archive 和结构多样性，不恢复无依据 IFS |
| RTBL 与原论文目标不再完全一致 | 明确标注 availability-only adaptation，保留旧 A 层结果只作历史参考 |
| 历史绘图脚本失效 | 保留只读旧脚本副本，新脚本只接受三目标版本 |
| 大范围字段删除导致中间状态不可运行 | 严格按 A→B→C→D→E 实施，每阶段保持测试可运行 |
| 后续获得真实 accuracy 数据 | 以新的、独立版本重新引入 measured accuracy，不恢复当前合成 IFS |

回退只允许回到实施前代码版本，不允许把新三目标结果重新标记为旧四目标结果。
