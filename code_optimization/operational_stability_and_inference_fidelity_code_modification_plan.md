# 运行稳定性与推理保真度模型代码改造方案

> 状态：核心代码已实施；正式实验结果待按新语义重新生成  
> 制定日期：2026-07-22  
> 适用范围：`src_python` 下的环境模型、四目标遗传算法、基线算法、实验统计与绘图代码  
> 核心目标：在不引入具体 DNN/数据集建模、不改变“抽象 DAG 部署优化”定位的前提下，使可靠性相关指标具备清晰语义、合理尺度、可复现实验和论文可解释性。

### 2026-07-22 实施记录

已完成数据模型语义别名、默认拓扑初始化修复、OSS/IFS 对数域聚合、候选后接纳具名评价接口、主算法与 RTBL 迁移、回传链路能耗、实验指标版本控制、绘图字段更新和模型专项测试。旧字段与接口暂时保留为兼容层。现有及新增 37 项单元测试全部通过，并完成小规模定制化算法烟雾测试。

尚未执行正式消融、敏感性、规模扩展实验，也未重新生成论文中的全部结果表与图；旧实验数据不得与 `stability_fidelity_v2_return_energy` 结果混合。

## 1. 结论与推荐方案

当前项目不应把“运行可靠性”直接解释为一次 DNN 推理任务无故障完成的严格概率，也不应把节点给定的“推理准确性”直接解释为具体模型在测试集上的真实准确率。原因是当前模型没有显式刻画故障事件、失效率、任务执行时长内的生存过程、模型结构、数据集和节点侧模型精度。

推荐保留当前基于经验先验和动态负载退化的评分框架，但完成以下语义和数学改造：

1. 将运行可靠性重命名为 **运行稳定性分数**（Operational Stability Score，OSS）。
2. 将推理可靠性/推理准确性重命名为 **推理保真度分数**（Inference Fidelity Score，IFS）。
3. 节点和链路的初始值均解释为经验先验质量因子，而不是经严格统计估计得到的成功概率。
4. 组件状态仍随时隙、负载和历史压力变化，但把 `heat` 解释为历史压力代理量，不宣称为真实物理老化。
5. 用分层几何平均替代“所有组件可靠性直接连乘”作为主目标，避免 DAG 或部署域数量增加导致分数机械性坍缩。
6. 保留原始连乘值作为诊断量，用于体现联合脆弱性，但不再用作跨任务规模比较和遗传算法主目标。
7. 四目标统一按最大化处理：IFS、OSS、时延满意度、能耗满意度。

最终算法评价向量建议定义为：

```text
F(x) = [IFS(x), OSS(x), D_sat(x), E_sat(x)]
```

其中 `x` 为 DAG 到计算节点的部署决策。

## 2. 改造边界

### 2.1 本次改造包含

- 可靠性相关变量、接口、输出字段和图表标签的语义统一。
- 节点及链路动态质量因子的计算与校验。
- OSS 和 IFS 的新聚合公式。
- 候选部署的“接纳后一步预测”评价接口。
- 遗传算法与基线算法评价接口适配。
- 指标版本控制、兼容字段和旧结果隔离。
- 单元测试、回归测试、消融实验与敏感性实验设计。
- 论文中模型定义、术语、假设和局限性的推荐写法。

### 2.2 本次改造不包含

- 不对 CNN、Transformer 等具体 DNN 结构进行准确率建模。
- 不引入具体数据集、模型切分层或真实推理精度测量。
- 不模拟离散故障事件、故障恢复、任务重试、迁移或冗余副本。
- 不建立基于危险率的任务期生存概率或累计失效概率。
- 不把历史压力代理量解释为不可逆的器件物理老化或寿命损耗。
- 不改变遗传算法“生成 DAG 部署决策”的核心功能。

如未来需要回答“任务在给定执行时间内成功完成的概率”，应另行增加故障率/危险率、生存函数、相关故障和恢复机制；不应通过本次评分模型的改名来替代概率模型。

## 3. 当前实现的主要问题

### 3.1 指标语义超过了模型能力

当前名称容易让读者将运行可靠性理解为任务成功概率，将推理准确性理解为具体 DNN 的分类或检测准确率。但现有值实质上是由经验初值、当前利用率和历史状态共同生成的无量纲质量分数。

当最终运行可靠性只有 `0.2` 时，它在当前连乘模型下可能只是多个小于 1 的组件因子相乘后的结果，不能直接得出“任务只有 20% 概率成功”的结论。

### 3.2 原始连乘存在规模偏置

若部署使用的节点或物理链路集合为 `C(x)`，当前形式近似为：

```text
R_raw(x) = product(q_c), c in C(x)
```

即使每个组件质量完全相同，使用组件越多，结果也会指数下降。例如每个组件均为 `0.9` 时，2 个组件的连乘为 `0.81`，15 个组件约为 `0.206`。因此 `0.2` 可能主要反映组件数量，而非部署质量显著恶化。

这会造成三个问题：

- 不同 DAG 规模之间不可直接比较。
- 算法可能过度偏好少节点、少链路部署。
- 指标数值容易被误解为概率。

### 3.3 先验值定义不充分

节点初始运行可靠性和推理准确性可以作为经验先验值，但必须说明来源、取值范围、动态更新方式和敏感性分析。若直接称为概率或真实准确率，则缺少统计试验、数据集和置信区间支撑。

### 3.4 动态状态的时间语义需要固定

组件质量确实会随时隙推进发生变化。代码和论文必须统一说明评价发生在何时：推荐采用“当前决策时刻，在接纳候选任务并更新资源占用后，对下一状态的部署质量进行一步预测”。该值是后接纳状态分数，不是任务整个运行期内的累计生存概率。

### 3.5 默认拓扑存在基准值初始化风险

当前默认拓扑创建 `Node` 后再修改当前 `operation_reliability` 和 `inference_accuracy`，而 `Node.__post_init__` 已经提前保存了基准值。环境重置时可能把当前值恢复为修改前的旧基准值。

改造时必须保证“基准值”和“当前值”由同一组最终参数初始化，避免实验在首次运行和重置后出现不一致。

## 4. 推荐数学模型

### 4.1 组件经验先验

对计算节点 `n` 定义：

- `s_n^0`：节点基础运行稳定性先验。
- `a_n^0`：节点基础推理保真度先验。

对物理链路 `l` 定义：

- `q_l^0`：链路基础传输稳定性先验。

统一约束：

```text
0 < lower_bound <= prior <= 1
```

这些值可以来自历史统计、设备等级、仿真设定或归一化基准，但论文必须明确它们是经验质量因子。若仅为合成数据，应报告采样区间、分布、随机种子和敏感性实验。

### 4.2 动态节点运行稳定性

推荐保留当前指数型退化结构：

```text
s_n(t) = clip(
    s_n^0 * exp(-alpha_s * rho_n(t) - beta_s * h_n(t)),
    s_n_min,
    s_n^0
)
```

其中：

- `rho_n(t)`：节点在时隙 `t` 的归一化资源利用率。
- `h_n(t)`：节点历史压力代理量或负载记忆状态。
- `alpha_s`：当前负载敏感系数。
- `beta_s`：历史压力敏感系数。

### 4.3 动态节点推理保真度

```text
a_n(t) = clip(
    a_n^0 * exp(-alpha_a * rho_n(t) - beta_a * h_n(t)),
    a_n_min,
    a_n^0
)
```

该分数表示资源竞争、降频、量化配置或运行环境对推理输出质量的抽象影响。由于本项目不建模具体 DNN，该分数不能称为某个数据集上的实际 accuracy。

### 4.4 动态链路稳定性

```text
q_l(t) = clip(
    q_l^0 * exp(-alpha_l * u_l(t) - beta_l * h_l(t)),
    q_l_min,
    q_l^0
)
```

其中 `u_l(t)` 为链路利用率，`h_l(t)` 为链路历史压力代理量。双向逻辑边若共享同一条物理链路，应共享同一状态，并在一个部署决策中只计一次物理故障/稳定性域。

### 4.5 OSS：分层几何平均

设部署 `x` 使用的唯一计算节点集合为 `N_x`，使用的唯一物理链路集合为 `L_x`。

节点子分数：

```text
S_node(x) = exp((1 / |N_x|) * sum(log(s_n))), n in N_x
```

链路子分数：

```text
S_link(x) = exp((1 / |L_x|) * sum(log(q_l))), l in L_x
```

最终运行稳定性分数：

```text
OSS(x) = S_node(x)^omega_n * S_link(x)^omega_l
omega_n + omega_l = 1
```

默认可设置 `omega_n = 0.5`、`omega_l = 0.5`，并在实验中做权重敏感性分析。若部署未使用物理链路，则定义：

```text
OSS(x) = S_node(x)
```

该形式保留几何平均对低质量组件的惩罚，同时减少组件数量对评分尺度的机械影响。

### 4.6 IFS：工作量加权几何平均

设映射到节点 `n` 的 DAG 工作量占比为：

```text
w_n(x) = flops_assigned_to_n / total_dag_flops
```

则：

```text
IFS(x) = exp(sum(w_n(x) * log(a_n))), n in N_x
```

该指标表示整个抽象 DAG 推理工作受到各节点保真度影响后的聚合质量。工作量较大的节点对结果影响更大，但它仍是抽象保真度分数，而非具体 DNN 的真实精度。

### 4.7 原始联合连乘诊断量

保留：

```text
RawJointProduct(x)
  = product(s_n) * product(q_l), n in N_x, l in L_x
```

用途仅限：

- 展示组件数量增加时联合脆弱性的变化。
- 与旧模型做兼容性对照。
- 消融实验和调试。

不得将其作为新版主优化目标，也不宜用于不同 DAG 规模间直接比较。

### 4.8 数值稳定性

所有几何平均均在对数域计算：

```python
safe_value = max(value, eps)
score = math.exp(weighted_log_sum)
```

建议 `eps = 1e-12`，但正常配置应通过校验保证所有下界严格大于 0，而不是依赖 `eps` 掩盖非法输入。

## 5. 推荐数据结构与接口

### 5.1 采用分阶段字段迁移

不建议在同一次提交中同时完成字段全量重命名、聚合公式替换和实验输出升级。推荐先增加语义别名，再逐步移除旧名。

第一阶段保留现有存储字段，以属性形式增加新语义：

| 旧字段 | 新语义字段 | 含义 |
| --- | --- | --- |
| `operation_reliability` | `operational_stability` | 当前节点运行稳定性分数 |
| `base_operation_reliability` | `base_operational_stability` | 节点稳定性经验先验 |
| `inference_accuracy` | `inference_fidelity` | 当前节点推理保真度分数 |
| `base_inference_accuracy` | `base_inference_fidelity` | 节点保真度经验先验 |
| 链路 `reliability` | `transmission_stability` | 当前物理链路稳定性分数 |
| `preR` | `preference_stability` | 用户对 OSS 的偏好权重输入 |
| `preA` | `preference_fidelity` | 用户对 IFS 的偏好权重输入 |

旧属性在一个过渡版本中保留，并注明 deprecated。所有新代码和新输出只使用新名称。第二阶段再做数据类字段的机械重命名。

### 5.2 候选部署质量结构

新增不可变结果结构，替代位置含义不明确的二元组：

```python
@dataclass(frozen=True)
class CandidateQualityScores:
    operational_stability: float
    inference_fidelity: float
    node_stability: float
    link_stability: float | None
    raw_joint_product: float
    used_node_count: int
    used_physical_link_count: int
```

新增主接口：

```python
def predict_candidate_scores_by_assignment(
    self,
    dag: DAG,
    assignment: Mapping[int, int],
) -> CandidateQualityScores:
    """预测候选任务接纳后的下一状态质量，不修改环境真实状态。"""
```

旧接口：

```python
predict_candidate_reliability_by_assignment(...)
count_dynamic_operation_by_assignment(...)
count_dynamic_accuracy_by_assignment(...)
```

先改为调用新接口的兼容包装器，完成全仓调用迁移后再删除。

### 5.3 接纳后评价结构

将 `evaluate_post_admission_metrics` 的元组返回值改为具名结构，避免时延、运行分数、推理分数和能耗发生顺序错误：

```python
@dataclass(frozen=True)
class PostAdmissionMetrics:
    delay: float
    operational_stability: float
    inference_fidelity: float
    total_energy: float
    raw_joint_product: float
```

### 5.4 配置项

建议在环境或统一配置中增加：

```python
node_stability_weight: float = 0.5
link_stability_weight: float = 0.5
quality_log_epsilon: float = 1e-12
objective_semantics_version: str = "stability_fidelity_v2_return_energy"
```

初始化时校验：

- 两个聚合权重非负且总和大于 0；内部归一化到和为 1。
- 所有基础分数和下界均位于 `(0, 1]`。
- 下界不得高于对应基础分数。
- 动态敏感系数不得为负。

## 6. 按文件的代码改造清单

### 6.1 `models.py`

1. 为 Node 和物理链路增加新语义属性或字段。
2. 将 `heat` 的注释改为“历史压力代理量/负载记忆”。
3. 增加基础值、下界和动态系数的参数校验。
4. 明确 `reset()` 恢复的是基础经验先验状态。
5. 增加 `CandidateQualityScores` 和 `PostAdmissionMetrics` 数据类；若项目已有统一 DTO 模块，则放入该模块。

验收条件：旧调用仍可运行，新调用不再出现概率和真实准确率含义的字段名。

### 6.2 `core/topology.py`

1. 修复默认边缘节点的初始化顺序。
2. 优先在 Node 构造函数中直接传入最终稳定性和保真度基础值。
3. 禁止在 `__post_init__` 保存基准值后只修改当前值。
4. 校验双向链路继续共享同一物理状态对象。

验收条件：构造后、环境 reset 后，节点的当前值和基础值保持预期一致。

### 6.3 `core/environment.py`

这是本次改造的核心文件。

1. 将节点/链路动态更新方法内部变量替换为稳定性和保真度语义。
2. 保留“当前利用率 + 历史压力”的指数更新公式。
3. 新增唯一计算节点集合和唯一物理链路集合提取方法。
4. 新增对数域几何平均工具函数。
5. 实现 `S_node`、`S_link`、OSS、IFS 和原始联合连乘诊断量。
6. 实现 `predict_candidate_scores_by_assignment`，保证预测不修改真实环境。
7. 将接纳后评价改为具名结构。
8. 旧计数接口改成兼容包装器并发出弃用提示，或在内部调用点全部迁移后删除。
9. 无物理链路时只使用节点子分数。
10. 对空 DAG、非法节点、断开路径和零总 FLOPs 给出明确异常或既定拒绝结果。

候选预测流程应固定为：

```text
读取当前环境状态
  -> 复制候选涉及组件的资源状态
  -> 加入候选 DAG 的计算与通信负载
  -> 预测下一状态组件分数
  -> 按唯一物理域聚合 OSS
  -> 按 FLOPs 权重聚合 IFS
  -> 返回结果，不提交状态
```

### 6.4 `proposed.py`

1. 将 `_evaluate_assignment` 改为只调用新的候选质量接口一次，避免运行分数和推理分数分别预测造成重复计算或状态不一致。
2. `ObjectiveValues` 使用具名字段：

```python
inference_fidelity
operational_stability
delay_satisfaction
energy_satisfaction
```

3. 保证四个目标统一为“越大越好”。
4. 可在第一阶段保留 `function1`、`function2` 等内部数组以降低迁移风险，但对外输出和注释必须使用明确语义。
5. 更新 Pareto 解、拥挤距离、加权选择和日志导出的字段名。
6. 将旧 `preR`、`preA` 迁移为 `preference_stability` 和 `preference_fidelity`，明确其为偏好输入而非可靠性约束。

### 6.5 基线算法与 `rtbl/scheduler.py`

1. 所有基线统一调用同一 OSS/IFS 环境接口。
2. 禁止某些算法使用旧连乘、另一些算法使用新几何平均。
3. 若某基线只使用部分指标做决策，最终评估仍必须使用统一指标。
4. 更新内部比较方向，防止把分数误当作待最小化成本。

### 6.6 `metrics.py`

1. 输出字段改为 `operational_stability_score` 和 `inference_fidelity_score`。
2. 增加：

```text
raw_joint_product
node_stability_score
link_stability_score
used_node_count
used_physical_link_count
objective_semantics_version
```

3. 旧字段可以在兼容期重复输出，但必须标注 deprecated，且不得与新旧实验混合统计。

### 6.7 `experiment_runner.py` 与结果模式

1. 每次实验写入 `objective_semantics_version=stability_fidelity_v2_return_energy`。
2. 聚合实验结果前验证版本一致。
3. 新旧版本不允许直接合并计算均值、Hypervolume 或显著性检验。
4. 保存随机种子、先验分布、动态系数、聚合权重和 DAG/拓扑规模。
5. 对所有依赖可靠性指标的旧实验重新运行。

### 6.8 `plot_results.py` 与诊断脚本

1. 图例、坐标轴和表头统一使用 OSS/IFS。
2. 纵轴注明 `[0, 1]` 无量纲评分，不写 probability 或 accuracy。
3. 同时绘制主分数和 `raw_joint_product` 时，使用不同子图，避免读者误以为二者含义相同。
4. 散点图建议增加组件数量着色，用于展示旧连乘的规模偏置。
5. 更新 `stability_diagnostics.py` 的术语和断言。

### 6.9 README、论文材料和已有优化文档

1. 在项目 README 中增加“指标语义”小节。
2. 更新所有将运行分数称为成功概率的说明。
3. 更新算法伪代码、目标函数、实验表头和图注。
4. 明确本项目对 DNN 采用抽象 DAG，不声称对不同 DNN 模型的真实准确率进行预测。

## 7. 推荐实施阶段

### 阶段 0：冻结基线和结果版本

- 记录当前测试结果和一组固定种子的基准实验。
- 为旧结果标记 `legacy_product_v0`。
- 保存当前 Pareto 前沿、时延、能耗及可靠性输出，供回归对照。

### 阶段 1：修复初始化并增加语义别名

- 修复默认拓扑基础值/当前值不一致问题。
- 增加稳定性和保真度属性。
- 增加参数校验。
- 暂不改变聚合公式。

该阶段用于把“初始化错误”和“公式变化”分离，便于定位回归。

### 阶段 2：实现统一质量评价接口

- 新增候选质量与接纳后指标数据类。
- 实现对数域几何平均。
- 实现 OSS、IFS 和原始联合连乘诊断量。
- 增加旧接口兼容包装。

### 阶段 3：迁移算法

- 先迁移 `proposed.py`。
- 再迁移所有基线和 RTBL 调度器。
- 确认所有算法评价同一候选时获得完全相同的 OSS/IFS。

### 阶段 4：升级指标模式与绘图

- 更新 `metrics.py`、实验 runner 和 CSV/JSON 字段。
- 加入语义版本检查。
- 更新 Pareto、Hypervolume 和绘图代码。

### 阶段 5：测试与回归

- 完成第 8 节测试。
- 对固定随机种子做新旧结果对照。
- 检查时延和能耗结果未被无关改变。

### 阶段 6：补充论文实验和说明

- 完成聚合方法、动态机制和参数敏感性实验。
- 更新论文模型、算法、实验和局限性部分。
- 删除对外 API 中的旧名称；必要时保留一个发布周期的兼容层。

## 8. 测试方案

### 8.1 组件与动态更新测试

- 基础值、下界和系数非法时应拒绝初始化。
- 零负载、零历史压力时，动态分数等于基础值。
- 利用率增加时，稳定性与保真度不得上升。
- 历史压力增加时，分数不得上升。
- 动态分数始终位于 `[lower_bound, base]`。
- 环境 reset 后恢复到正确基础值。
- 默认边缘节点构造后的基础值与当前值一致。

### 8.2 聚合公式测试

- 所有节点分数相同时，节点几何平均等于该分数。
- 所有链路分数相同时，链路几何平均等于该分数。
- 任一组件分数降低时 OSS 不得升高。
- 单节点、无物理链路部署时 OSS 等于节点稳定性。
- 双向逻辑链路共享一条物理链路时只计一次。
- 多条 DAG 逻辑传输复用同一物理链路时不重复计入稳定性域。
- 在所有组件质量相同的情况下，增加 DAG 节点数量不应导致新版归一化分数机械下降。
- 原始联合连乘应继续随所用组件数量增加而下降。
- FLOPs 权重和为 1，IFS 位于所用节点保真度的最小值与最大值之间。

### 8.3 状态一致性测试

- 候选预测前后，环境真实资源、组件分数和历史压力完全不变。
- 对候选执行预测后再真实接纳并推进一个约定更新步骤，二者结果在容差内一致。
- 同一候选重复预测结果一致。
- 不同评价目标共享同一份候选后接纳状态快照。

### 8.4 算法与结果测试

- `proposed.py` 与各基线对同一 assignment 输出相同 OSS/IFS。
- 四目标支配关系的方向全部正确。
- 旧参数 `preR`、`preA` 不再进入运行链；历史结果通过离线迁移工具处理。
- 新旧实验结果版本混合时，统计程序明确报错。
- 几何平均在大 DAG 下不发生浮点下溢。
- 无可行部署、空链路集、零总 FLOPs 等边界情况有明确行为。

## 9. 实验补充方案

### 9.1 聚合方法消融

比较：

1. 原始全组件连乘。
2. 全组件统一几何平均。
3. 推荐的节点/链路分层几何平均。
4. 算术平均。

报告 OSS、原始联合连乘、所用节点数、所用物理链路数、时延、能耗和接受率。重点验证推荐方法能保留低质量组件惩罚，同时减弱规模偏置。

### 9.2 动态机制消融

比较：

- 静态先验。
- 仅当前负载。
- 仅历史压力。
- 当前负载与历史压力共同作用。
- 当前状态评价与接纳后一步预测评价。

### 9.3 参数敏感性

至少覆盖：

- `alpha_s`、`beta_s`、`alpha_a`、`beta_a`。
- 链路对应动态系数。
- `omega_n`、`omega_l`。
- 历史压力衰减/记忆系数。
- 基础先验的不同分布和方差。

### 9.4 规模扩展实验

改变：

- DAG 节点数和边数。
- 物理节点数和链路密度。
- 到达负载水平。
- DAG 并行度与通信计算比。

目标是证明新版分数对任务规模更可比，并观察算法复杂度与 Pareto 前沿变化。

### 9.5 偏好输入实验

改变 `preference_stability` 和 `preference_fidelity`，验证：

- 偏好变化能够推动最终方案在 OSS 与 IFS 间发生预期移动。
- 偏好参数不是硬约束，论文中不报告为“最低可靠性要求”。
- 若未来需要硬约束，应另增 `min_operational_stability` 和 `min_inference_fidelity`。

## 10. 论文表述建议

### 10.1 推荐术语

- Operational Stability Score（运行稳定性分数，OSS）
- Inference Fidelity Score（推理保真度分数，IFS）
- state-dependent empirical quality factor（状态相关经验质量因子）
- historical stress proxy / load-memory state（历史压力代理量/负载记忆状态）
- post-admission deployment quality（接纳后部署质量）
- abstract DNN DAG（抽象 DNN DAG）

### 10.2 应避免的表述

- “OSS 是本次推理任务成功完成的概率”。
- “IFS 是 DNN 在测试集上的预测准确率”。
- “heat 表示设备不可逆老化或剩余寿命”。
- “0.2 表示任务仅有 20% 成功率”。
- “经验先验已经等价于实测故障概率”。

### 10.3 模型定义示例

可在论文中写为：

> We model node and link quality using state-dependent empirical factors initialized from prior operational knowledge. These factors decrease with instantaneous resource utilization and a historical stress proxy. They are not interpreted as calibrated mission-success probabilities. For a deployment decision, the Operational Stability Score is calculated using a hierarchical geometric aggregation over unique node and physical-link domains, while the Inference Fidelity Score is calculated using a workload-weighted geometric aggregation over assigned nodes.

### 10.4 关于“可靠性感知”的标题或算法名

如果保留 Reliability-aware 一词，应在首次出现时说明：这里的 reliability-aware 指算法在部署决策中显式考虑状态相关的节点和链路稳定性因素，而不是建立严格的任务期失效概率模型。

## 11. 高水平论文投稿所需的证据链

自定义评价指标本身是可以接受的，但必须具备以下证据链：

1. **构念定义**：说明 OSS/IFS 分别代表什么、不代表什么。
2. **数学性质**：给出范围、有界性、单调性、无量纲性和规模行为。
3. **设计依据**：解释指数退化、几何聚合和工作量加权的选择原因。
4. **参数来源**：报告经验先验和动态系数的设置方式。
5. **消融实验**：证明每个模型组成确实影响且改善所关心的行为。
6. **敏感性分析**：证明结论不依赖单一参数点。
7. **基线一致性**：所有算法使用完全相同的环境评价器。
8. **局限性说明**：明确没有具体 DNN 精度与故障事件校准。
9. **可复现性**：公开随机种子、配置、指标版本和计算代码。

若后续能够获得真实设备日志，可进一步使用日志对先验值和动态参数做校准，并报告排序相关性或预测误差；这会增强外部有效性，但不属于本次代码改造的前置条件。

## 12. 验收标准

本次改造完成应同时满足：

- 代码、输出、图表和论文中不再把 OSS 当作任务成功概率。
- 代码、输出、图表和论文中不再把 IFS 当作具体模型真实准确率。
- 主算法和所有基线共用唯一的质量评价实现。
- OSS 使用唯一节点和唯一物理链路的分层几何聚合。
- IFS 使用 DAG FLOPs 权重的几何聚合。
- 原始连乘只作为诊断输出。
- 候选部署评价不修改环境状态。
- 默认拓扑初始化与 reset 行为一致。
- 新结果带有指标语义版本，旧结果不会被误混入。
- 本文第 8 节测试全部通过。
- 完成聚合、动态、敏感性和规模消融实验。
- README、算法伪代码、公式、表格和图注完成统一更新。

## 13. 推荐的最小提交拆分

为便于审查和回滚，建议至少拆分为以下提交：

1. `fix: align base and current node quality initialization`
2. `refactor: add stability and fidelity semantic aliases`
3. `feat: add unified candidate quality score evaluator`
4. `feat: replace product objective with hierarchical geometric OSS`
5. `refactor: migrate proposed and baseline schedulers to OSS and IFS`
6. `feat: version experiment metric schema and diagnostics`
7. `test: cover quality aggregation prediction and reset semantics`
8. `docs: update metric definitions experiments and limitations`

每个提交均应保持测试可运行。尤其不要把初始化修复、公式替换和所有实验结果更新压入同一个不可审查的大提交。
