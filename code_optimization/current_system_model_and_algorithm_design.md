# 当前系统模型与算法设计梳理

> 状态：按 2026-07-23 当前代码实现整理  
> 适用版本：`objective_semantics_version = stability_fidelity_v2_return_energy`  
> 主要实现：`models.py`、`core/topology.py`、`core/environment.py`、`proposed.py`、`rtbl/scheduler.py`  
> 文档目的：统一系统假设、数学符号、时隙语义、四目标定义和定制化遗传算法流程，并区分已实现内容、兼容内容与待完善内容。

## 1. 项目定位

本项目研究云—边—端异构网络中的抽象 DNN DAG 部署问题。每个 DNN 请求由一个有向无环图表示，算法决定每个 DAG 子任务部署到哪个物理计算节点，并在资源、层级、时延和链路负载约束下，同时优化：

1. 推理保真度分数（Inference Fidelity Score，IFS）；
2. 运行稳定性分数（Operational Stability Score，OSS）；
3. 时延满意度；
4. 能耗满意度。

项目不建立具体 CNN、Transformer、数据集或模型切分层的精度模型，也不模拟严格的故障发生过程。因此：

- OSS 不是任务无故障完成概率；
- IFS 不是具体 DNN 在测试集上的真实准确率；
- 节点与链路初值是经验质量先验；
- `heat` 是历史压力或负载记忆代理量，而非器件物理寿命。

## 2. 系统总体结构

系统由四层组成：

```text
抽象 DNN DAG 请求层
        ↓
云—边—端物理资源层
        ↓
时隙化动态环境与统一评价层
        ↓
部署搜索算法与实验统计层
```

一次决策的核心数据流为：

```text
当前环境状态 S(t)
  + 新到达 DAG G_k
        ↓
生成候选部署 x_k
        ↓
检查资源、层级、时延和链路负载
        ↓
计算 IFS、OSS、时延满意度、能耗满意度
        ↓
Pareto 搜索与偏好选解
        ↓
接纳部署并登记为 RunningDNN
        ↓
推进时隙、更新负载/历史压力/有效带宽/质量分数
```

## 3. 物理网络模型

### 3.1 云—边—端节点

物理网络记为：

```text
G_p = (V, E_p)
```

节点集合 `V` 分为三级：

| 层级 | 代码值 | 作用 |
| --- | ---: | --- |
| 用户端 | 1 | DNN 请求发起位置，可承担本用户请求的本地计算 |
| 边缘节点 | 2 | 主要异构计算资源 |
| 云节点 | 3 | 高算力集中计算资源 |

每个节点 `n` 当前包含：

- 最大 CPU 容量 `C_n`；
- 当前剩余 CPU；
- 计算速率 `f_n`；
- 计算功率 `P_n`；
- 当前负载率 `rho_n(t)`；
- 历史压力状态 `h_n(t)`；
- 基础运行稳定性先验 `s_n^0`；
- 当前运行稳定性 `s_n(t)`；
- 基础推理保真度先验 `a_n^0`；
- 当前推理保真度 `a_n(t)`。

默认拓扑包含 1 个云节点、10 个边缘节点，以及挂接在部分边缘节点上的若干用户节点。规模实验可通过 `TopologyConfig` 配置云、边、端节点数量和边缘网络连接密度。

### 3.2 物理链路与有向传输边

两个物理节点之间的一条双向连接在代码中表示为：

- 两个 `LinkNode` 有向传输边；
- 一个共享的 `PhysicalLinkState`。

共享状态包含：

- 唯一物理链路 ID；
- 基础传输稳定性先验 `q_l^0`；
- 当前传输稳定性 `q_l(t)`；
- 物理链路负载率 `u_l(t)`；
- 历史压力状态 `h_l(t)`。

同一物理链路的正向和反向传输共享稳定性与历史压力。在一个部署方案中，同一物理链路只作为一个稳定性域参与 OSS 聚合，避免双向逻辑边重复计数。

### 3.3 路由与有效带宽

系统以带宽倒数为边权预计算节点对之间的最短路径：

```text
weight_l = 1 / B_l
```

当前有效带宽为：

```text
B_l_eff(t)
  = max(
      B_l * eta_min,
      B_l / (1 + gamma_h * h_l(t) + gamma_u * u_l_dir(t)),
      1
    )
```

其中 `u_l_dir(t)` 是有向边负载，`h_l(t)` 来自共享物理状态。

当前实现只动态更新已缓存路径上的有效带宽，不会因为带宽退化重新执行动态路由。这是“固定路由、动态链路性能”模型。

## 4. 抽象 DNN DAG 模型

第 `k` 个请求表示为：

```text
G_k = (T_k, A_k)
```

其中：

- `T_k`：DAG 子任务集合；
- `A_k`：任务依赖边集合。

每个子任务 `i` 包含：

- CPU 需求 `c_ki`；
- 浮点运算量 `F_ki`；
- 权重文件大小（当前主要作为任务属性保存）。

每条 DAG 边 `(i,j)` 包含中间数据量 `D_kij`。

每个 DNN 请求还包含：

- 端到端时延约束 `D_k^max`；
- 发起用户节点 `o_k`；
- 输入数据量 `D_k^in`；
- 输出数据量 `D_k^out`；
- 运行稳定性偏好 `preference_stability`；
- 推理保真度偏好 `preference_fidelity`。

上述两个偏好是最终方案选择权重的输入，不是 OSS/IFS 的硬约束。

### 4.1 DAG 生成方式

当前实验通过随机拓扑顺序生成无环边，并补充：

- 没有前驱的中间任务与首任务连接；
- 没有后继的中间任务与末任务连接。

因此生成的 DAG 保持无环，并使首任务、末任务与中间任务形成可执行依赖关系。

## 5. 部署决策与约束

### 5.1 决策变量

定义：

```text
x_kin = 1，表示请求 k 的任务 i 部署在物理节点 n；否则为 0。
```

代码中一个个体主要使用一维整数编码：

```text
assignment_k[i] = n
```

即一个基因对应一个 DAG 子任务，基因值为物理节点索引。

### 5.2 唯一部署约束

每个子任务必须且只能部署到一个物理节点：

```text
sum_n x_kin = 1
```

### 5.3 CPU 容量约束

请求接纳后，一个节点上新增任务的 CPU 需求不能超过当前剩余 CPU：

```text
sum_i c_ki * x_kin <= C_n_available(t)
```

运行中的多个 DNN 通过 `RunningDNN` 队列共同占用 CPU，完成后释放。

### 5.4 用户节点使用约束

用户级节点只能由在该节点发起的 DNN 使用：

```text
level(n) = 1  =>  n = o_k
```

其他用户节点不能作为候选计算节点。

### 5.5 层级非回退约束

对于 DAG 依赖边 `(i,j)`：

```text
level(assignment_k[j]) >= level(assignment_k[i])
```

因此任务可沿“端 → 边 → 云”方向升级，但不允许后继任务从较高层级回退到较低层级。

### 5.6 时延约束

```text
L_k(x_k) <= D_k^max
```

当前代码同时维护：

- `deadline_feasible`：是否严格满足时延约束；
- `delay_satisfaction`：时延目标分数。

### 5.7 链路过载约束

定制化算法的约束违反度还包含预测有向链路负载超过 1 的程度：

```text
CV_link = mean(max(0, predicted_load_l - 1))
```

## 6. 时延模型

端到端时延由三部分构成：

```text
L_k(x)
  = L_upload
  + L_critical_path
  + L_return
```

### 6.1 输入上传时延

输入数据从发起用户节点传输到首任务部署节点：

```text
L_upload = sum_l D_k^in / B_l_eff(t)
```

### 6.2 DAG 关键路径时延

任务计算时延为：

```text
L_comp(ki,n) = F_ki / f_n
```

依赖边通信时延为：

```text
L_comm(kij) = sum_l D_kij / B_l_eff(t)
```

系统递归计算到末任务的最长依赖路径，而不是简单累加所有并行分支。

### 6.3 结果回传时延

末任务输出从其部署节点返回请求发起节点：

```text
L_return = sum_l D_k^out / B_l_eff(t)
```

### 6.4 当前评价时刻

候选评价先使用当前环境的有效带宽计算 `estimated_delay`，随后根据该运行时间预测候选接纳后的节点和链路负载。当前代码没有在每个候选上迭代执行：

```text
候选负载 → 新有效带宽 → 新时延 → 再预测负载
```

因此，准确表述应为：

> 时延是当前状态下的部署时延估计；OSS/IFS 是候选接纳后的下一状态预测分数。

不应把四项指标全部笼统称为完全一致的后接纳预测值。

## 7. 能耗模型

### 7.1 计算能耗

```text
E_comp(k,x)
  = sum_i P_assignment(i) * F_ki / f_assignment(i)
```

### 7.2 传输能耗

```text
E_link(k,x)
  = sum_l epsilon_l * transmitted_data_l
```

当前传输能耗统一统计输入上传、DAG 中间数据传输和最终结果回传：

```text
E_link = E_upload + E_intermediate + E_return
```

### 7.3 总能耗与满意度

```text
E_total = E_comp + E_link
```

环境根据合法节点和路径计算与算法无关的参考边界 `[E_min,E_max]`：

```text
E_sat = clip((E_max - E_total) / (E_max - E_min), 0, 1)
```

因此算法内部的第四目标是越大越好的能耗满意度，同时保留 `total_energy` 原值用于报告和审计。

## 8. 动态状态模型

### 8.1 节点负载

```text
rho_n(t) = used_cpu_n(t) / C_n
```

### 8.2 链路负载

运行 DNN 的数据量按估计运行时间均摊到所经有向链路：

```text
u_l_dir(t) = used_bandwidth_l(t) / B_l_eff(t)
```

同一物理链路的共享负载取两个方向负载的最大值，而不是求和：

```text
u_l(t) = max(u_l_forward(t), u_l_reverse(t))
```

### 8.3 历史压力更新

节点与物理链路均采用指数移动平均：

```text
h_n(t+1) = lambda_h * h_n(t) + (1-lambda_h) * rho_n(t)
h_l(t+1) = lambda_g * h_l(t) + (1-lambda_g) * u_l(t)
```

### 8.4 动态组件质量分数

节点运行稳定性：

```text
s_n(t) = clip(
    s_n^0 * exp(-alpha_r * rho_n(t) - beta_r * h_n(t)),
    s_min,
    s_n^0
)
```

节点推理保真度：

```text
a_n(t) = clip(
    a_n^0 * exp(-alpha_a * rho_n(t) - beta_a * h_n(t)),
    a_min,
    a_n^0
)
```

物理链路传输稳定性：

```text
q_l(t) = clip(
    q_l^0 * exp(-alpha_l * u_l(t) - beta_l * h_l(t)),
    q_min,
    q_l^0
)
```

所有先验和下界必须位于 `(0,1]`，动态敏感系数非负。

## 9. OSS 与 IFS

### 9.1 候选后接纳预测

对于候选部署 `x`，系统在不修改真实环境状态的情况下：

1. 将候选 CPU 需求加入当前节点负载；
2. 将候选通信量按估计运行时间加入链路负载；
3. 预测下一步历史压力；
4. 预测节点稳定性、节点保真度和链路稳定性；
5. 聚合 OSS、IFS 和诊断量。

### 9.2 节点稳定性子分数

设候选使用的唯一节点集合为 `N_x`：

```text
S_node(x) = exp((1/|N_x|) * sum_n log(s_n_pred))
```

### 9.3 链路稳定性子分数

设候选使用的唯一物理链路集合为 `L_x`：

```text
S_link(x) = exp((1/|L_x|) * sum_l log(q_l_pred))
```

### 9.4 运行稳定性分数 OSS

若使用物理链路：

```text
OSS(x)
  = S_node(x)^omega_n * S_link(x)^omega_l
omega_n + omega_l = 1
```

默认 `omega_n = omega_l = 0.5`。若候选未使用物理链路：

```text
OSS(x) = S_node(x)
```

### 9.5 推理保真度分数 IFS

节点工作量权重为：

```text
w_n(x) = assigned_flops_n / total_dag_flops
```

IFS 为：

```text
IFS(x) = exp(sum_n w_n(x) * log(a_n_pred))
```

### 9.6 原始连乘诊断量

```text
RawJointProduct(x)
  = product_n(s_n_pred) * product_l(q_l_pred)
```

该值仅用于诊断旧模型的规模偏置，不是新版优化目标。

## 10. 四目标优化问题

当前统一目标向量为：

```text
F(x) = [IFS(x), OSS(x), D_sat(x), E_sat(x)]
```

四个目标全部最大化。

时延满意度为：

```text
D_sat(x)
  = clip((D_k^max - L_k(x)) / D_k^max, 0, 1)
```

注意：超过时限和恰好达到时限时，时延满意度均可能为 0，但 `deadline_feasible` 仍单独判断是否满足时限。

Pareto 支配定义为：方案 `x` 在四个目标上均不差于 `y`，且至少一个目标严格更优。

## 11. 时隙化运行语义

### 11.1 请求到达与持续执行

实验循环每个决策时隙处理一个新 DNN 请求。成功接纳的请求被转换为：

```text
RunningDNN(
    assignment,
    estimated_runtime,
    remaining_slots,
    used_nodes,
    used_physical_links
)
```

持续时隙数为：

```text
remaining_slots = max(1, ceil(estimated_runtime / slot_length))
```

因此不同 DNN 可以在多个时隙内重叠运行，共享节点与链路资源。

### 11.2 单个时隙推进顺序

当前 `advance_time_slot()` 顺序为：

```text
根据运行队列重算资源占用
  → 更新节点历史压力
  → 更新链路历史压力
  → 更新有效带宽
  → 更新动态稳定性与保真度
  → 所有 RunningDNN 剩余时隙减 1
  → 释放已完成 DNN
  → 时隙编号加 1
```

任务接纳后立即计入当前资源占用，但历史压力和动态质量分数只在推进时隙时正式提交更新。

## 12. 定制化遗传算法设计

当前定制化算法入口为：

```text
AllDNNRefactor.run_customized_proposed()
```

默认主要参数：

| 参数 | 默认值 |
| --- | ---: |
| 种群规模 | 60 |
| 迭代次数 | 200 |
| 交叉概率 | 0.50 |
| 变异概率 | 0.25 |
| 初始 ε 约束阈值 | 0.35 |
| 精英局部搜索比例 | 0.05 |
| 每个任务局部候选数 | 3 |

### 12.1 DAG 上下文预处理

每个新请求到达后，算法构造：

- DAG 拓扑序；
- 每个任务的前驱和后继；
- DAG 边数据量；
- 按任务 FLOPs 识别的关键路径任务集合；
- 每个任务的合法候选节点集合。

### 12.2 混合初始化

初始种群由五类模板构成：

1. 云优先；
2. 本地/邻近优先；
3. 剩余资源优先；
4. 稳定性与保真度优先；
5. DAG 拓扑综合评分优先。

随后通过模板扰动和随机可行部署补足种群，并以至少约 75% 唯一个体为目标。所有个体经过层级与资源修复。

### 12.3 四类子膜角色

种群个体持久分配到四类角色：

- `stability`：偏向 OSS/IFS 的质量引导角色；
- `latency`：偏向时延和通信代价；
- `energy`：偏向计算与传输能耗；
- `feasibility`：偏向资源、层级和链路负载可行性。

角色影响候选节点评分、父代选择和幸存者配额。每 5 代允许一次全局父代选择，形成子膜间的信息交换。

### 12.4 DAG 闭合块交叉

交叉不再简单交换连续编号后缀，而是选择一个 DAG 任务并构造受控的后继闭包块，然后交换父代在该块上的节点部署。

交叉后执行结构修复；若修复失败或修改基因数超过交换块规模的一半，则回退，避免 repair 覆盖主要遗传信息。

### 12.5 偏斜变异

偏斜变异优先从关键路径任务中选择变异位置，并根据角色相关候选评分进行加权采样。评分综合考虑：

- 剩余 CPU；
- 计算时延；
- 节点 OSS/IFS 相关状态；
- 路径通信代价；
- 路径稳定性；
- 历史压力；
- 能耗；
- 资源和链路可行性。

变异结果必须通过结构修复，否则返回原部署。

### 12.6 Repair 机制

Repair 按 DAG 拓扑序处理：

1. 修复非法节点索引；
2. 根据前驱部署确定最低允许层级；
3. 从合法候选节点中选择综合评分最高者；
4. 检查 CPU、用户节点和层级约束；
5. 必要时逐任务尝试替代节点。

Repair 显式返回：

```text
RepairResult(assignment, success, changed_gene_count, reason)
```

失败结果不会被静默写回种群。

### 12.7 动态约束违反度

总违反度为：

```text
CV(x)
  = CV_delay
  + CV_resource
  + CV_hierarchy
  + CV_link_overload
```

其中各项均经过规模归一化。

ε 阈值从 `0.35` 线性下降到 `0`：

```text
epsilon(g) = 0.35 * (1 - g/(G-1))
```

支配规则为：

1. ε 可行解优于 ε 不可行解；
2. 两个 ε 不可行解比较总违反度；
3. 两个 ε 可行解执行四目标 Pareto 支配。

### 12.8 非支配排序与幸存者选择

联合种群 `P ∪ Q` 执行：

- 带 ε 约束的非支配排序；
- 四目标拥挤距离计算；
- 按角色配额保留个体；
- 配额不足时按总体排序补足。

这使算法同时维持 Pareto 多样性和子膜角色多样性。

### 12.9 精英局部搜索

每代对约 5% 的精英个体执行关键路径邻域搜索。对关键任务尝试综合评分最高的前 3 个候选节点，仅接受：

- 约束违反度不增加；
- 四目标均不劣；
- 偏好加权综合分严格提高。

### 12.10 历史最优与最终部署

每代只允许严格可行解更新历史最优。搜索结束后：

1. 从严格可行候选中按请求偏好选解；
2. 重新计算完整约束违反度；
3. 不满足严格约束则拒绝请求；
4. 满足约束则登记为 `RunningDNN`。

### 12.11 算法流程伪代码

```text
for each arriving DNN request k:
    build DAG context
    derive preference weights
    initialize a diverse feasible population P

    for generation g = 0 ... G-1:
        generate offspring Q by membrane-aware parent selection
        apply DAG-closed crossover
        apply role-aware skew mutation
        repair offspring

        evaluate P ∪ Q with [IFS, OSS, delay_sat, energy_sat]
        compute normalized constraint violations
        epsilon = decreasing_constraint_threshold(g)
        perform epsilon-constrained non-dominated sorting
        compute four-objective crowding distance
        select next P with membrane quotas
        improve a small elite set by critical-path local search
        update strictly feasible best archive

    select a strictly feasible solution using request preference
    if no valid solution:
        reject request
    else:
        register deployment as RunningDNN
    advance one time slot

drain all remaining RunningDNN instances
```

## 13. 偏好选解机制

偏好不改变 Pareto 支配关系，主要用于：

- 历史最优更新；
- 局部搜索接受；
- 从 Pareto/候选集合中选择最终在线部署。

显式偏好权重为：

| 偏好模式 | IFS | OSS | 时延 | 能耗 |
| --- | ---: | ---: | ---: | ---: |
| delay-sensitive | 0.10 | 0.15 | 0.60 | 0.15 |
| stability-sensitive | 0.30 | 0.45 | 0.15 | 0.10 |
| energy-sensitive | 0.10 | 0.15 | 0.15 | 0.60 |
| balanced | 0.25 | 0.25 | 0.25 | 0.25 |

自适应模式固定能耗权重为 `0.20`，其余 `0.80` 根据 DNN 的时延、IFS 和 OSS 偏好归一化分配。

## 14. 算法复杂度

设：

- `P`：种群规模；
- `G`：迭代次数；
- `M`：DAG 任务数；
- `A`：DAG 依赖边数；
- `H`：候选部署涉及的平均物理路径长度。

一次候选评价需要遍历任务、DAG 边和所用物理路径，近似为：

```text
O(M + A*H)
```

每代联合种群约为 `2P`，两两非支配比较为：

```text
O(P^2)
```

因此每个 DNN 请求的主要时间复杂度可写为：

```text
O(G * (P^2 + P*(M + A*H)))
```

精英局部搜索只处理约 `0.05P` 个体，但会对关键路径任务检查最多 3 个候选节点，其代价取决于关键路径长度。路径缓存和 assignment 评价缓存能够减少重复计算。

主要内存开销来自种群、联合种群、支配矩阵和评价缓存，近似为：

```text
O(P*M + P^2)
```

论文中的运行时间实验应同时改变 `P`、`G`、DAG 规模和物理拓扑规模，以验证上述趋势。

## 15. 基线算法

当前实验框架包含：

| 算法 | 核心方式 | 是否可导出 Pareto 前沿 |
| --- | --- | --- |
| `proposed` | 普通四目标遗传/非支配搜索 | 是 |
| `customized` | DAG、子膜、ε 约束、局部搜索定制算法 | 是 |
| `random` | 随机可行部署采样 | 是 |
| `sa` | 多偏好种子与模拟退火邻域搜索 | 是 |
| `localfirst` | 本地/近端优先启发式 | 否 |
| `maxresource_fast` | 最大剩余资源的有界快速搜索 | 否 |
| `rtbl` | 基于可用性估计和虚拟队列的逐任务在线选择 | 否 |

所有正式动态实验应通过统一环境评价最终部署的 OSS、IFS、时延和能耗。

RTBL 仅保留共享动态环境的 `run_dynamic()` 入口。旧静态 `run()` 及其三维部署矩阵评价链已经删除，避免旧静态结果混入新版实验。

## 16. Pareto 与实验统计

只有 `pareto` suite 才：

- 收集前沿点；
- 输出 `pareto_points.csv`；
- 计算四维 Hypervolume；
- 统计 Pareto 点数量。

其他实验只输出部署结果，不计算 HV。

正式主指标为：

- `avg_operational_stability_score`；
- `avg_inference_fidelity_score`；
- `avg_estimated_delay`；
- `avg_total_energy`；
- `rejected_or_failed_count`；
- `runtime_ms`。

旧结果列不再由主运行链写出；历史 CSV 由 `migrate_legacy_result_schema.py` 离线转换。

## 17. 当前系统模型的实现边界

### 17.1 已经明确实现

- 抽象 DAG 到异构节点的一对一任务部署。
- 多 DNN 跨时隙并发占用资源。
- 节点、链路负载和历史压力的动态更新。
- 双向有向边共享物理链路稳定性域。
- 候选接纳后的 OSS/IFS 一步预测。
- 分层几何 OSS 和 FLOPs 加权 IFS。
- 四目标最大化、ε 约束支配和偏好选解。
- DAG 感知初始化、交叉、变异、repair 和局部搜索。

### 17.2 没有建模

- 具体 DNN 模型、层切分和数据集准确率。
- 严格故障概率、失效率、生存函数和恢复过程。
- 任务重试、迁移、检查点或冗余副本。
- 故障相关性、共享风险组和级联故障。
- 动态重新路由。
- 排队等待时间和通信调度时序的精细仿真。
- 空闲功耗、冷却能耗和设备开关机能耗。

## 18. 当前代码中需要继续统一的事项

以下事项不否定当前模型，但在论文定稿或正式实验前应处理或明确说明：

1. **候选时延预测不完全闭环**：候选负载会影响 OSS/IFS，但当前候选时延没有基于预测有效带宽重新计算。
2. **固定路径假设**：路径按基础带宽预缓存，动态带宽变化不会触发改路。
3. **链路双向负载取最大值**：这等价于全双工共享状态的简化，应在论文中说明；若物理链路共享总容量，应考虑求和或方向容量模型。
4. **节点可用性耦合有限**：RTBL 使用显式可用性采样，主遗传算法主要依赖资源和稳定性分数，二者不是同一个故障/可用性机制。
5. **经验先验未实测校准**：当前结论只能解释为合成环境下的相对部署质量优化。

## 19. 推荐的论文系统模型结构

论文可按以下顺序组织：

1. 云—边—端物理网络；
2. 抽象 DNN DAG 请求；
3. 时隙化请求到达与持续执行；
4. 部署决策变量；
5. 资源、层级、时延与链路负载约束；
6. 动态节点和链路状态；
7. OSS 与 IFS；
8. 时延与能耗；
9. 四目标优化问题；
10. 定制化遗传算法；
11. 复杂度、实验设置和局限性。

算法名称中若保留 “reliability-aware”，必须说明其含义是考虑状态相关稳定性质量因子，而不是预测严格任务成功概率。

## 20. 代码模块对应关系

| 模型/算法部分 | 当前主要代码 |
| --- | --- |
| 节点、链路、DNN 数据结构 | `models.py` |
| DAG 随机生成 | `core/dag_generator.py` |
| 云边端拓扑与请求生成 | `core/topology.py` |
| 动态环境、时延、能耗、OSS/IFS | `core/environment.py` |
| 普通和定制化遗传算法、启发式基线 | `proposed.py` |
| RTBL 在线基线 | `rtbl/scheduler.py` |
| 统一聚合指标 | `metrics.py` |
| 场景配置、结果版本和 Pareto/HV | `experiment_runner.py` |
| 模型回归测试 | `tests/test_stability_fidelity_model.py` |

## 21. 一句话概括当前设计

> 当前系统在时隙化云—边—端环境中，将抽象 DNN DAG 的子任务映射到异构节点，通过状态相关的节点/链路质量、关键路径时延和计算/传输能耗评价候选部署，并使用融合 DAG 感知算子、角色化子膜、动态 ε 约束和精英局部搜索的四目标遗传算法选择严格可行的在线部署方案。
