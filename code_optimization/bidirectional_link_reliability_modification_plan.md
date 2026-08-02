# 双向链路可靠性建模调研与修改方案

## 1. 文档目的

本文针对当前云—边—端 DNN 部署模型中的双向链路可靠性计算问题，回答以下问题：

1. 主流论文如何定义链路可靠性、路径可靠性和双向通信可靠性；
2. 当前实现是否把同一物理链路的两个方向当成两个独立故障而重复连乘；
3. 在不破坏方向相关的流量、时延和能耗计算的前提下，应如何修改数据结构、动态状态和可靠性聚合；
4. 如何通过单元测试、回归实验和论文公式同步验证修改结果。

本文调研和方案形成于 2026 年 7 月 19 日。建议将本文作为后续代码修改、实验重跑和论文公式修订的共同依据。

---

## 2. 当前模型与问题复现

### 2.1 当前实现

当前拓扑通过 `_add_bidirectional_link(...)` 为一条物理连接创建两个独立的 `LinkNode`：

```text
物理连接 e = {u, v}
    ├── LinkNode(u -> v)
    └── LinkNode(v -> u)
```

两个对象分别保存：

- `base_reliability`
- `reliability`
- `load_ratio`
- `heat`
- `effective_band_width`

路径缓存和链路流量统计均使用有向 `LinkNode`。输入上传、DAG 中间数据传输和结果回传经过的有向链路被放入一个以 `LinkNode` 对象为键的字典。随后：

```python
collect_used_links(...)
```

返回该字典的键，并称其为“唯一链路集合”。由于 `LinkNode(eq=False)` 按对象身份区分，`u -> v` 与 `v -> u` 不会被去重。

以下两个最终或预测可靠性入口都会逐对象连乘：

- `predict_candidate_reliability_by_assignment(...)`
- `count_raw_link_product_by_assignment(...)`

因此，当输入上传使用 `u -> v`、结果回传使用 `v -> u` 时，同一物理连接会贡献两个可靠性因子。

### 2.2 最小复现实验

使用固定随机种子创建环境，将一个 DNN 的全部任务部署到同一个远端边缘节点。此时没有任务间跨节点传输，网络流量只包括：

1. 发起节点到远端节点的输入上传；
2. 远端节点到发起节点的结果回传。

实测结果如下：

| 指标 | 当前有向对象口径 | 物理链路故障域口径 |
|---|---:|---:|
| 有向 `LinkNode` 数量 | 10 | 不适用 |
| 物理连接数量 | 不适用 | 5 |
| 链路可靠性乘积 | 0.845825 | 0.919687 |

5 条物理连接的正反方向各被连乘一次，使链路可靠性相对降低约 8.0%。在路径更长、基础可靠性更低或动态退化更强时，偏差会进一步扩大。

### 2.3 当前语义不一致

当前实现同时包含两种不兼容的语义：

- 对同一个有向 `LinkNode`，即使它被多个 DAG 依赖传输重复经过，也只连乘一次；
- 对同一个物理连接，只要正反方向都被使用，就连乘两次。

如果 `reliability` 表示“链路组件在 DNN 运行期间可用”，第一条是合理的，但第二条属于重复计算。

如果 `reliability` 表示“单次有向数据传输成功概率”，第二条可能合理，但第一条又会漏算重复传输事件。因此，当前实现是“组件可用性”和“逐次传输成功率”的混合模型。

---

## 3. 主流论文处理方式

### 3.1 独立链路失效：路径可靠性按所选链路组件连乘

Lee、Modiano 等研究概率链路失效时，将链路 `(i,j)` 的失效概率记为 `p_ij`。一条路径的生存概率写为：

```text
R_path = product_(i,j) (1 - p_ij * x_ij)
```

其中 `x_ij` 是表示链路是否属于该路径的二元变量。每个被选中的链路组件在一条路径中贡献一次可靠性因子。论文还用 `-log(1-p_ij)` 将最大可靠性路径转换为最短路问题。

这一类模型的关键点是：

- 乘积对象是失效组件，而不是代码中偶然创建的对象数量；
- 独立性假设必须落在真实故障域上；
- 同一故障组件不能因为被多个逻辑流或逻辑边引用而被当成多个独立组件。

代表论文：

- H.-W. Lee, E. Modiano, and K. Lee, “Diverse Routing in Networks with Probabilistic Failures,” *IEEE/ACM Transactions on Networking*, 2010, DOI: [10.1109/TNET.2010.2050490](https://doi.org/10.1109/TNET.2010.2050490)。作者版本：[MIT PDF](https://www.mit.edu/~modiano/papers/CV_C_121.pdf)。
- T. Korkmaz and K. Sarac, “Characterizing Link and Path Reliability in Large-Scale Wireless Sensor Networks,” *IEEE WiMob*, 2010, DOI: [10.1109/WIMOB.2010.5644996](https://doi.org/10.1109/WIMOB.2010.5644996)。作者版本：[UT Dallas PDF](https://personal.utdallas.edu/~ksarac/research/publications/WiMOB2010.pdf)。

Korkmaz 和 Sarac 同样把单路径可靠性建模为路径中链路可靠性的乘积，并进一步讨论链路级重传和多路径机制。这种处理属于“每条实际路径边贡献一个因子”，不支持因为同一物理连接在数据结构中拆成两个方向对象就自动视为两个独立物理故障。

### 3.2 无线有向交付率：正反方向相乘需要明确的通信事务语义

De Couto 等提出 ETX 时，显式区分：

- `d_f`：数据包正向成功到达的概率；
- `d_r`：ACK 反向成功到达的概率。

一次“成功并被确认的数据发送”要求正向数据和反向 ACK 都成功，因此：

```text
P_success_per_attempt = d_f * d_r
ETX = 1 / (d_f * d_r)
```

这里正反向相乘不是因为网络拓扑保存了两条有向边，而是因为协议明确规定一次成功事务由“数据 + ACK”两个传输事件组成。论文还通过实测说明无线链路的两个方向可能具有明显不对称性。

代表论文：

- D. S. J. De Couto, D. Aguayo, J. Bicket, and R. Morris, “A High-Throughput Path Metric for Multi-Hop Wireless Routing,” *ACM MobiCom*, 2003；期刊扩展版 DOI: [10.1007/s11276-005-1766-z](https://doi.org/10.1007/s11276-005-1766-z)。作者版本：[MIT PDF](https://pdos.csail.mit.edu/~decouto/my-papers/mobicom03.pdf)。

ETX 对当前模型的启示是：

1. 如果要采用逐次传输成功率，就必须明确一次 DNN 通信包含哪些正向数据、ACK、重传和返回事件；
2. 应按传输事件次数或数据包数量处理，而不是按“用过的有向对象集合”处理；
3. 在没有逐包、ACK 和重传仿真的简化模型中，直接套用正反向交付率乘积并不严谨。

当前项目明确不做逐包排队和重传仿真，因此不宜把 ETX 式事务可靠性作为第一版服务可靠性的主口径。

### 3.3 共享风险链路组：共享物理资源不能假设为独立故障

SRLG（Shared Risk Link Group）研究指出，多条逻辑链路可能共享光纤、管道、供电或其他物理资源。共享资源失效会使相关逻辑链路同时失效，因此不能把这些逻辑链路作为独立事件重复连乘。

Lee 和 Modiano 的概率 SRLG 模型进一步允许共享风险事件以一定概率影响组内链路，明确区分：

- 逻辑链路；
- 物理风险或故障域；
- 条件独立或相关失效。

代表论文：

- H.-W. Lee and E. Modiano, “Diverse Routing in Networks with Probabilistic Failures,” [MIT PDF](https://www.mit.edu/~modiano/papers/CV_C_121.pdf)。
- Z. Zhou, T. Lin, and K. Thulasiraman, “Survivable Cloud Network Design Against Multiple Failures Through Protecting Spanning Trees,” 2016, [arXiv:1608.04002](https://arxiv.org/abs/1608.04002)。
- Z. Zhou, T. Lin, and K. Thulasiraman, “Survivable Cloud Network Mapping with Multiple Failures,” *IEEE ICC*, 2015，[IBM Research 页面](https://research.ibm.com/publications/survivable-cloud-network-mapping-with-multiple-failures)。

虽然当前项目暂时不需要完整 SRLG 优化，但“双向逻辑边共享同一物理连接”正是最简单的共享故障域情形。若正反向对象代表同一网卡、交换端口或物理线路，则把它们作为两个独立可靠性因子会违反独立性假设。

### 3.4 云和虚拟网络：可靠性通常绑定物理资源或副本，而不是逻辑引用次数

可靠虚拟基础设施和云网络映射研究通常把服务器、物理链路、物理路径和备份副本作为可靠性或生存性对象。当多个虚拟链路映射到同一物理资源时，它们会共享该资源的失效影响。

代表论文：

- W.-L. Yeow, C. Westphal, and U. C. Kozat, “Designing and Embedding Reliable Virtual Infrastructures,” 2010，[arXiv:1005.5367](https://arxiv.org/abs/1005.5367)。
- S. Yang, P. Wieder, R. Yahyapour, S. Trajanovski, and X. Fu, “Reliable Virtual Machine Placement and Routing in Clouds,” 2017，[arXiv:1701.06005](https://arxiv.org/abs/1701.06005)。
- J. Li, W. Liang, M. Huang, and X. Jia, “Reliability-Aware Network Service Provisioning in Mobile Edge-Cloud Networks,” *IEEE TPDS*, 2020, DOI: [10.1109/TPDS.2020.2970048](https://doi.org/10.1109/TPDS.2020.2970048)。

这些工作支持在当前项目中区分“DNN 的逻辑依赖传输”和“承载这些传输的物理可靠性资源”。

### 3.5 调研结论

主流处理方式可以归纳为：

| 模型类型 | 可靠性对象 | 正反方向处理 | 同一对象重复经过 |
|---|---|---|---|
| 物理链路/路径可用性 | 物理链路或独立故障组件 | 同一故障域只计一次 | 在同一服务可用性集合中只计一次 |
| 有向传输成功率 | 单次数据、ACK 或分组传输事件 | 按协议要求分别计算 | 按事件次数计算 |
| SRLG/相关失效 | 共享风险事件或物理资源组 | 共享故障域，不独立连乘 | 按风险事件聚合 |
| 备份和虚拟网络可靠性 | 物理节点、物理路径、备份副本 | 取决于物理映射 | 多个逻辑引用共享底层失效 |

当前项目的设计文档定义了唯一节点集合 `N_m` 和唯一链路集合 `E_m`，且不做逐包与重传仿真。因此，最一致的选择是：

> 将链路可靠性定义为 DNN 运行期间物理链路故障域的动态可用性；有向传输边继续用于流量、时延和能耗，但不再直接充当独立可靠性因子。

---

## 4. 推荐目标模型

### 4.1 分离三个概念

建议将当前单一 `LinkNode` 概念拆成三个层次：

1. **有向传输边 `DirectedLink`**
   - 表示 `u -> v` 的传输能力；
   - 保存方向相关带宽、有效带宽、负载和单位数据能耗；
   - 用于最短路、传输时延、流量和能耗计算。

2. **物理链路 `PhysicalLink`**
   - 表示节点 `u` 与 `v` 之间的一条真实连接；
   - 正反两个 `DirectedLink` 引用相同的 `physical_link_id`；
   - 保存共享的基础可靠性、动态可靠性和可靠性热度。

3. **可靠性故障域 `ReliabilityDomain`**
   - 第一版中与 `PhysicalLink` 一一对应；
   - 将来可让多条物理链路共享一个 SRLG ID；
   - 服务可靠性按唯一故障域集合连乘。

第一版可以让 `PhysicalLink` 同时承担 `ReliabilityDomain`，无需立即实现完整 SRLG。

### 4.2 推荐数据结构

推荐新增：

```python
@dataclass(eq=False)
class PhysicalLinkState:
    physical_link_id: int
    node_a: Node
    node_b: Node
    base_reliability: float = 1.0
    reliability: float = 1.0
    load_ratio: float = 0.0
    heat: float = 0.0
```

修改 `LinkNode`：

```python
@dataclass(eq=False)
class LinkNode:
    s_node: Node
    e_node: Node
    band_width: int
    physical_state: PhysicalLinkState
    base_band_width: int | None = None
    effective_band_width: float | None = None
    load_ratio: float = 0.0
    energy_per_mb: float = 0.0
```

其中：

- `LinkNode.load_ratio` 保留为方向负载；
- `PhysicalLinkState.load_ratio` 是用于动态可靠性的物理链路负载；
- `base_reliability`、`reliability` 和 `heat` 从方向对象迁移到共享物理状态；
- 为兼容旧代码，可暂时在 `LinkNode` 上提供只读属性，把可靠性字段转发到 `physical_state`，但新逻辑不得继续按方向对象聚合可靠性。

不能只使用无序节点对 `(min(u,v), max(u,v))` 作为永久 ID，因为将来可能存在两节点间多条并行物理链路。应在创建物理连接时生成显式、稳定的 `physical_link_id`，然后让两个方向对象共享该 ID。

### 4.3 双向负载聚合

当前每个方向具有独立带宽，等价于简化的全双工假设。建议用于物理可靠性退化的负载定义为：

```text
eta_e(t) = max(eta_(u->v)(t), eta_(v->u)(t))
```

理由：

- 不把全双工两个方向的独立容量错误相加；
- 物理链路的退化程度由压力更大的方向主导；
- 比平均值更保守，但不会像求和那样系统性高估负载。

如果后续实验明确把双向流量建模为共享半双工介质，则改为：

```text
eta_e(t) = eta_(u->v)(t) + eta_(v->u)(t)
```

该选择必须成为环境配置，而不能隐式混用。

### 4.4 物理链路动态可靠性

保持当前指数退化形式，但作用于共享物理状态：

```text
g_e(t) = lambda_g * g_e(t-1) + (1-lambda_g) * eta_e(t)

l_e(t) = clip(
    l_e^0 * exp(-alpha_l * eta_e(t) - beta_l * g_e(t)),
    l_min,
    l_e^0
)
```

候选接纳预测使用相同口径：

```text
hat_eta_(u->v) = current_eta_(u->v) + added_traffic_(u->v) / capacity_(u->v)

hat_eta_e = max(hat_eta_(u->v), hat_eta_(v->u))

hat_g_e = lambda_g * g_e + (1-lambda_g) * hat_eta_e

hat_l_e = clip(
    l_e^0 * exp(-alpha_l * hat_eta_e - beta_l * hat_g_e),
    l_min,
    l_e^0
)
```

### 4.5 服务可靠性

对 DNN `m` 的部署方案，定义：

- `N_m`：承载至少一个子任务的唯一节点集合；
- `A_m`：输入上传、DAG 中间传输和结果回传实际经过的有向传输边集合；
- `P_m = { physical(a) | a in A_m }`：对应的唯一物理链路故障域集合。

动态运行可靠性修改为：

```text
R_m^run(t)
    = product_(j in N_m) o_j(t)
    * product_(e in P_m) l_e(t)
```

候选接纳后的预测可靠性为：

```text
hat_R_m^run(t | X)
    = product_(j in N_m) hat_o_j(t | X)
    * product_(e in P_m) hat_l_e(t | X)
```

输入上传和结果回传仍然都进入 `A_m` 并产生方向流量，但如果它们经过同一物理链路，该物理链路在 `P_m` 中只出现一次。

---

## 5. 代码修改范围

### 5.1 `models.py`

修改内容：

1. 新增 `PhysicalLinkState`；
2. `LinkNode` 增加对共享物理状态的引用；
3. `RunningDNN` 建议同时保存：
   - `used_links`：有向链路集合，用于方向流量；
   - `used_physical_links`：物理链路集合，用于可靠性和共享热度。

不建议继续让 `used_links` 同时承担流量对象和可靠性故障域两种职责。

### 5.2 `core/topology.py`

修改 `_add_bidirectional_link(...)`：

1. 创建一个 `PhysicalLinkState`；
2. 为其分配唯一 `physical_link_id`；
3. 创建 `u -> v` 和 `v -> u` 两个 `LinkNode`；
4. 两个方向引用同一个 `PhysicalLinkState`。

当前默认拓扑中还有直接成对创建 `LinkNode` 的代码，应统一改为调用该辅助函数，避免某些链路遗漏共享物理状态。

同时检查参数化拓扑的特殊规模，尤其是只有两个边缘节点时，环形生成逻辑可能重复添加同一个节点对。若确实需要并行链路，应分配不同物理 ID；若不是有意并行，应在拓扑生成阶段去重。

### 5.3 `core/environment.py`

建议新增以下接口：

```python
collect_used_directed_links(dnn_index, assignment)
collect_used_physical_links(dnn_index, assignment)
group_directed_links_by_physical_id(links)
predict_physical_link_states(...)
```

修改以下逻辑：

| 现有接口 | 修改要求 |
|---|---|
| `_reset_link_state()` | 方向带宽状态逐方向初始化；可靠性状态按物理链路初始化一次 |
| `_infer_link_base_reliability()` | 输入改为物理链路或任一方向的端点和基础带宽 |
| `_collect_link_weights()` | 保持有向流量累计，不做物理去重 |
| `collect_used_links()` | 明确重命名或保留为兼容包装，禁止再称为物理唯一链路 |
| `predict_link_load_ratios()` | 返回方向负载预测 |
| `update_link_loads()` | 更新方向负载，再聚合物理负载 |
| `update_link_heat()` | 只更新共享物理链路热度 |
| `refresh_dynamic_reliability()` | 每个物理链路只更新一次 |
| `predict_candidate_reliability_by_assignment()` | 按唯一物理链路预测和连乘 |
| `count_raw_link_product_by_assignment()` | 按唯一物理链路计算诊断用 raw product |
| `add_running_dnn()` | 同时登记方向链路和物理链路 |
| `_print_slot_summary()` | 链路可靠性均值按物理链路统计，方向负载可单独统计 |

### 5.4 `proposed.py`

最终目标 `_evaluate_assignment(...)` 已经通过环境接口取得预测可靠性。只要环境接口修正，该处不需要自行去重。

但 `_node_suitability_score(...)` 会分别对前驱和后继路径调用 `predict_path_state(...)` 并累乘路径可靠性。它属于搜索启发式，不是最终指标，但仍应检查是否与新的物理故障域语义产生过度惩罚。

建议：

- 第一阶段只保证最终目标和候选预测目标一致；
- 第二阶段让节点适应度评分收集本次局部决策涉及的唯一物理链路，再统一计算路径可靠性；
- 不要让启发式评分继续偏好“看起来链路更少、实际只是方向对象更少”的节点。

### 5.5 `rtbl/scheduler.py` 与诊断脚本

RTBL 通过 `evaluate_post_admission_metrics(...)` 取得预测可靠性，环境修复后会自动使用新口径。

`stability_diagnostics.py` 应增加：

- `used_directed_link_count`
- `used_physical_link_count`
- `reverse_pair_count`
- `physical_link_product`
- `directed_load_avg/max`
- `physical_load_avg/max`

这样可以直接观察修复前后的可靠性组成，而不是只报告含义不清的 `used_link_count`。

---

## 6. 分阶段实施方案

### 阶段 A：建立回归测试，不改变结果

1. 固定随机种子；
2. 保存当前最小复现场景结果；
3. 增加检测正反向对象是否共享物理 ID 的测试框架；
4. 记录当前总体实验中：
   - 接纳数量；
   - 平均服务可靠性；
   - 平均精度可靠性；
   - 平均时延；
   - 平均能耗；
   - 被使用的有向链路和物理链路数量。

### 阶段 B：引入物理链路身份

1. 新增 `PhysicalLinkState` 和 `physical_link_id`；
2. 统一双向链路创建过程；
3. 保持所有方向负载、时延和能耗结果不变；
4. 暂时允许旧可靠性字段通过兼容属性访问。

阶段 B 的验收条件是：

- 任意一对由同一次双向链路创建产生的方向对象共享物理 ID；
- 不同的并行物理链路具有不同 ID；
- 最短路径、时延和能耗测试结果与修改前一致。

### 阶段 C：切换可靠性状态和聚合口径

1. 将可靠性先验、热度和动态可靠性迁移到物理状态；
2. 用方向负载聚合得到物理负载；
3. 修改静态、动态和候选预测可靠性；
4. 修改运行 DNN 登记和诊断输出；
5. 删除最终指标中按方向对象连乘的逻辑。

### 阶段 D：统一搜索启发式与论文公式

1. 检查 `predict_path_state(...)` 和 `_node_suitability_score(...)`；
2. 更新设计文档中的 `E_m` 定义；
3. 更新论文问题建模和算法设计章节；
4. 重跑受影响的所有实验；
5. 在消融实验中报告“有向对象口径”和“物理故障域口径”的差异。

---

## 7. 测试方案

### 7.1 单元测试

#### 测试 1：双向对象共享物理状态

创建 `u <-> v`，验证：

```text
link_uv.physical_link_id == link_vu.physical_link_id
link_uv.physical_state is link_vu.physical_state
```

#### 测试 2：上传与回传不重复计算物理可靠性

场景：

- 发起节点为 `u`；
- 所有任务部署到 `v`；
- 上传经过 `u -> v`；
- 回传经过 `v -> u`；
- 物理链路可靠性为 `0.98`。

期望：

```text
used_directed_links = 2
used_physical_links = 1
link_reliability_product = 0.98
```

不能得到 `0.98^2`。

#### 测试 3：同方向被多个 DAG 边复用

多个 DAG 依赖传输经过同一个 `u -> v`：

- 方向流量应求和；
- 时延和能耗按数据量计算；
- 物理可靠性仍只贡献一次。

#### 测试 4：不同物理链路正常连乘

路径 `u - v - w` 的两个物理链路可靠性分别为 `0.98`、`0.97`：

```text
R_path = 0.98 * 0.97
```

#### 测试 5：双向负载不对称

设置：

```text
eta_(u->v) = 0.8
eta_(v->u) = 0.2
```

全双工 `max` 口径下：

```text
eta_e = 0.8
```

物理可靠性只更新一次。

#### 测试 6：候选预测与真实状态更新一致

对空环境预测接纳一个候选，然后实际接纳并推进一个时隙。允许热度时序造成已定义的差异，但预测和真实计算必须使用相同的物理链路集合、负载聚合函数和可靠性公式。

#### 测试 7：并行物理链路

同一节点对之间创建两条真实连接：

- 每条连接拥有不同 `physical_link_id`；
- 每条连接的两个方向共享各自的物理 ID；
- 不能仅按无序端点对错误合并。

### 7.2 集成与回归测试

至少运行：

```bash
python -m unittest discover -s tests
```

并对 proposed、customized proposed 和 RTBL 使用相同随机种子执行小规模回归。

重点验证：

- 服务可靠性应合理上升，但不应超过节点和物理链路乘积的理论范围；
- 时延和能耗不应仅因可靠性故障域修改而发生变化；
- 动态负载和热度仍随运行 DNN 数量变化；
- 三种算法均使用相同的物理链路可靠性口径；
- Pareto 前沿可能变化，这是目标值修正后的预期结果，不应强行保持旧结果。

---

## 8. 实验与论文影响

### 8.1 预期实验影响

修复后，使用上传和回传重合路径的部署方案不再被系统性双重惩罚，因此预计：

- 平均动态服务可靠性上升；
- 长路径部署的可靠性提升幅度更明显；
- 可靠性敏感偏好可能选择更多远端边缘或云节点；
- Pareto 前沿、超体积和接纳方案分布可能变化；
- 时延与能耗公式本身不变，但由于部署选择变化，其最终平均值可能间接变化。

因此，不能只更新代码而沿用旧实验 CSV 和图表。

### 8.2 论文公式建议

明确把原来的 `E_m` 改为物理故障域集合：

```text
A_m(X): deployment X actually traverses the set of directed transmission arcs

P_m(X) = { phi(a) | a in A_m(X) }
```

其中 `phi(a)` 将有向传输边映射到物理链路故障域。

运行可靠性写为：

```text
R_m^run(t | X)
    = product_(j in N_m(X)) o_j(t)
    * product_(e in P_m(X)) l_e(t)
```

并在文字中说明：

- 正反向流量分别参与带宽、时延、负载和能耗计算；
- 共享同一物理连接的两个方向属于同一个基础故障域；
- 物理链路可靠性在一个 DNN 服务可用性计算中只计一次；
- 当前模型不做逐包 ACK、重传和分组级成功概率仿真。

### 8.3 独立性假设声明

第一版仍可假设不同物理链路故障域相互独立，但必须明确这是简化假设。后续若考虑共享交换机、管道、基站或区域灾害，可将多个 `PhysicalLinkState` 映射到同一 SRLG，并按共享风险事件扩展，而不是继续简单连乘。

---

## 9. 不推荐的修复方式

### 9.1 直接删除结果回传链路

不推荐。结果回传是真实传输，会影响：

- 方向流量；
- 链路负载；
- 有效带宽；
- 时延；
- 热度；
- 可能的传输能耗。

问题不在于回传是否存在，而在于可靠性故障域如何聚合。

### 9.2 仅在最终乘积中按无序节点对去重

只能作为临时止血措施，不适合作为最终方案，原因是：

- 候选预测仍可能重复计算；
- 正反方向仍维护两套互不一致的可靠性和热度；
- 并行物理链路会被错误合并；
- 搜索目标、最终统计和运行状态可能继续不一致。

### 9.3 对正反方向可靠性取平均后再乘一次

平均值缺少明确物理含义，并可能掩盖高负载方向。若采用全双工共享物理故障域，建议先用 `max` 聚合方向负载，再通过统一退化公式计算一次物理可靠性。

### 9.4 保持当前模型但修改文档

若声称当前模型是逐次有向传输成功率，则还必须：

- 按每个 DAG 传输事件重复计算；
- 建模数据包数量、ACK、重传和相关性；
- 解释为何输入和返回数据量不同却各只贡献一个相同量级的因子。

这会显著超出当前简化时序模型范围，因此不建议。

---

## 10. 最终建议

建议修复，并采用“有向传输边与物理可靠性故障域分离”的完整方案：

1. 保留两个方向的 `LinkNode`，继续负责路由、方向流量、时延和能耗；
2. 为同一真实连接创建共享 `PhysicalLinkState`；
3. 方向负载按全双工 `max` 规则聚合为物理可靠性负载；
4. 物理热度和动态可靠性只更新一次；
5. 候选预测和最终运行可靠性均按唯一物理链路集合连乘；
6. 搜索启发式、RTBL、诊断脚本、设计文档和论文公式使用同一口径；
7. 重跑所有依赖服务可靠性的实验结果。

该方案符合当前项目“不做逐包和重传仿真、按唯一节点和唯一链路计算服务可靠性”的设计目标，也与路径可用性、云网络物理映射和共享风险故障域文献的主流处理方式一致。

---

## 参考文献

1. H.-W. Lee, E. Modiano, and K. Lee, “Diverse Routing in Networks with Probabilistic Failures,” *IEEE/ACM Transactions on Networking*, vol. 18, no. 6, pp. 1895–1907, 2010. [DOI](https://doi.org/10.1109/TNET.2010.2050490), [author PDF](https://www.mit.edu/~modiano/papers/CV_C_121.pdf).
2. D. S. J. De Couto, D. Aguayo, J. Bicket, and R. Morris, “A High-Throughput Path Metric for Multi-Hop Wireless Routing,” *ACM MobiCom*, 2003. [Author PDF](https://pdos.csail.mit.edu/~decouto/my-papers/mobicom03.pdf). Journal version: [DOI](https://doi.org/10.1007/s11276-005-1766-z).
3. T. Korkmaz and K. Sarac, “Characterizing Link and Path Reliability in Large-Scale Wireless Sensor Networks,” *IEEE WiMob*, 2010. [DOI](https://doi.org/10.1109/WIMOB.2010.5644996), [author PDF](https://personal.utdallas.edu/~ksarac/research/publications/WiMOB2010.pdf).
4. A. B. McDonald and T. Znati, “A Path Availability Model for Wireless Ad-Hoc Networks,” *IEEE WCNC*, 1999. [DOI](https://doi.org/10.1109/WCNC.1999.797781).
5. Z. Zhou, T. Lin, and K. Thulasiraman, “Survivable Cloud Network Mapping with Multiple Failures,” *IEEE ICC*, 2015. [IBM Research](https://research.ibm.com/publications/survivable-cloud-network-mapping-with-multiple-failures).
6. Z. Zhou, T. Lin, and K. Thulasiraman, “Survivable Cloud Network Design Against Multiple Failures Through Protecting Spanning Trees,” 2016. [arXiv:1608.04002](https://arxiv.org/abs/1608.04002).
7. W.-L. Yeow, C. Westphal, and U. C. Kozat, “Designing and Embedding Reliable Virtual Infrastructures,” 2010. [arXiv:1005.5367](https://arxiv.org/abs/1005.5367).
8. S. Yang, P. Wieder, R. Yahyapour, S. Trajanovski, and X. Fu, “Reliable Virtual Machine Placement and Routing in Clouds,” 2017. [arXiv:1701.06005](https://arxiv.org/abs/1701.06005).
9. J. Li, W. Liang, M. Huang, and X. Jia, “Reliability-Aware Network Service Provisioning in Mobile Edge-Cloud Networks,” *IEEE Transactions on Parallel and Distributed Systems*, vol. 31, no. 7, pp. 1545–1558, 2020. [DOI](https://doi.org/10.1109/TPDS.2020.2970048).
