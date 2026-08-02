# 更新后 DAREED 算法对应论文修改 Diff

## 1. 文档说明

- 核对对象：`paperwork` 目录中的优化版论文分节、合并稿和算法流程图说明。
- 代码基线：2026-07-19 更新后的 `proposed.py`、`core/environment.py` 和 `experiment_runner.py`。
- 本文档只给出建议修改，不直接改动任何现有论文文件。
- “位置”中的行号基于当前工作区文件；实际替换后行号会发生变化。
- `optimized_full.md` 由 `merge_optimized.py` 根据各个 `* optimized.md` 分节生成。应优先修改分节源文件，再重新合并，避免分别维护两份正文。

## 2. 总体核对结论

当前论文与代码在动态运行、DAG 建模、能耗口径、角色化搜索和约束违反度等方面大体一致，但以下内容必须更新：

1. DAREED 已经是四目标 Pareto 搜索，能耗不再只用于最终偏好选择。
2. 第四目标保存原始总能耗并按最小化方向参与支配关系，不再使用 `1/E` 作为搜索目标。
3. 候选可靠性使用接纳候选后的单步负载和热度预测，而不是只读取部署前的当前可靠性。
4. DAG 交叉块是后继闭包，不是拓扑序中某个边界之后的编号后缀。
5. 拥挤距离、局部搜索和严格可行解选择均使用四个目标。
6. 在线部署维护跨代严格可行的偏好最优解；普通实验不导出 Pareto 点，也不计算 HV。
7. HV 只在 `pareto` 实验中作为后处理计算，并且是包含能耗满意度的四维 HV。
8. `optimized_full.md` 中仍保留多处“三目标”旧表述，与分节文件和当前代码均不一致。

---

## D01【必须】摘要中的搜索目标

### 位置

- `paperwork/abstract optimized.md`：第 3 行，DAREED 方法概述句。
- `paperwork/optimized_full.md`：第 3 行。合并稿当前仍明确写成 three-objective。

### 原文

分节文件：

> DAREED combines DAG-aware evolutionary search, resource-aware variation, constraint-aware selection, and preference-based deployment selection.

合并稿中的旧句：

> DAREED combines DAG-aware three-objective Pareto candidate generation over inference accuracy, service reliability and inference delay with energy-aware search guidance and preference-based deployment selection.

### 建议修改后

> DAREED performs four-objective evolutionary search over inference accuracy, service reliability, inference delay, and total energy consumption, and combines DAG-aware variation, constraint-aware selection, one-step post-admission reliability prediction, and preference-based deployment selection.

### 修改原因

当前代码的非支配排序同时使用精度可靠性、服务可靠性、时延收益和原始总能耗。能耗已是正式 Pareto 目标，不应再被描述为搜索外部的 guidance。

---

## D02【必须】引言贡献点中的“三目标”旧描述

### 位置

- `paperwork/introduction optimized.md`：第 19 行，第二项贡献。
- `paperwork/optimized_full.md`：第 23 行。该行仍包含 `three-objective Pareto candidate generation`，说明合并稿未与分节文件同步。

### 原文

分节文件：

> We propose DAREED, a DAG-aware reliable and energy-efficient evolutionary deployment algorithm. DAREED integrates DAG-aware evolutionary search, resource-aware variation, constraint-aware selection and preference-based deployment selection.

合并稿旧文本：

> DAREED integrates three-objective Pareto candidate generation, DAG-aware evolutionary operators, resource-aware variation, constraint-aware selection and energy-aware preference-based deployment selection.

### 建议修改后

> We propose DAREED, a DAG-aware reliable and energy-efficient evolutionary deployment algorithm. DAREED integrates four-objective Pareto search, successor-closed DAG crossover, resource-aware skew mutation, persistent role specialization, constraint-aware selection, one-step post-admission reliability prediction, and preference-based deployment selection.

### 修改原因

建议在贡献点中明确最新算法相对于普通 NSGA-II 搜索的关键机制，同时删除三目标表述。

---

## D03【必须】能耗模型不再使用倒数效用

### 位置

- `paperwork/problem formulation optimized.md`：第 213–217 行，Energy Model 末尾。
- `paperwork/optimized_full.md`：第 307–311 行。

### 原文

```latex
For the evolutionary search procedure, total energy consumption is also transformed into a maximization-oriented energy utility:
\begin{equation}
\label{eq:energy_utility}
U^i_E=\frac{1}{E^i_{all}}.
\end{equation}
```

### 建议修改后

```latex
In the evolutionary search, the original total energy consumption
$E^i_{all}$ is retained as a minimization objective rather than being
compressed by a reciprocal transformation. For preference scoring and
Pareto-front normalization only, it is mapped to an energy-satisfaction
value:
\begin{equation}
\label{eq:energy_satisfaction}
\bar E^i=
\operatorname{clip}\left(
\frac{E^i_{\max}-E^i_{all}}
{E^i_{\max}-E^i_{\min}},0,1
\right),
\end{equation}
where $E^i_{\min}$ and $E^i_{\max}$ are deterministic request-specific
reference bounds derived from the minimum and maximum computation energy
over legal nodes and an upper bound on path transmission energy. A larger
$\bar E^i$ indicates lower energy consumption, while Pareto dominance is
still evaluated directly using $E^i_{all}$.
```

### 修改原因

当前实现的第四个目标值是原始 `total_energy`，支配方向为越小越好。`1/E` 会严重压缩数值尺度，已经不再用于主搜索。

---

## D04【必须】目标函数部分删除第二处 `1/E`

### 位置

- `paperwork/problem formulation optimized.md`：第 290–310 行。
- `paperwork/optimized_full.md`：第 385–405 行。

### 原文

```latex
Since the evolutionary search evaluates all fitness components in a
maximization form, inference delay is further transformed into a delay
utility ...

Similarly, total energy consumption is transformed into an energy utility:
\begin{equation}
\label{eq:energy_utility_obj}
U^i_E=\frac{1}{E^i_{all}}.
\end{equation}
```

以及分节文件第 310 行：

> In the algorithmic implementation, $T^i_{all}(t)$ and $E^i_{all}$ are evaluated through $U^i_T$ and $U^i_E$ for unified fitness comparison.

合并稿第 405 行还写成：

> $R'_i(t)$, $A'_i(t)$ and $U^i_T$ are used for Pareto dominance, while $U^i_E$ is used for candidate scoring, resource-aware variation, local refinement and final preference-based selection.

### 建议修改后

```latex
Inference delay is transformed into the maximization-oriented utility
$U^i_T$ in \eqref{eq:delay_utility}. In contrast, total energy consumption
is retained in its original form because it is directly minimized.
Accordingly, the objective vector used by the evolutionary search is
\begin{equation}
\label{eq:algorithm_objective_vector}
F_i(X,t)=
\left(\hat A'_i(X,t),\hat R'_i(X,t),U^i_T(X,t),E^i_{all}(X)\right),
\end{equation}
where the first three components are maximized and the fourth component
is minimized. The hatted reliability quantities denote the predicted
post-admission values defined in the algorithm section.

In summary, the deployment problem remains
\begin{equation}
\label{eq:multi_obj}
\left\{
\max_{x,y}R'_i(t),
\max_x A'_i(t),
\min_{x,y}T^i_{all}(t),
\min_{x,y}E^i_{all}
\right\},
\end{equation}
subject to \eqref{eq:single_deploy}--\eqref{eq:delay_constraint} and the
dynamic state-transition equations. In the implementation, $U^i_T$ is
used to represent feasible delay quality, whereas $E^i_{all}$ directly
participates in four-objective dominance, crowding distance, local
refinement, and preference-based selection.
```

### 修改原因

代码不再要求全部目标均为最大化形式；第四目标在支配判断、拥挤距离和选解中都按最小化处理。

---

## D05【必须】算法两阶段框架仍把能耗放在第二阶段

### 位置

- `paperwork/algorithm design optimized.md`：第 6 行。
- `paperwork/optimized_full.md`：第 411 行。

### 原文

> In the first stage, DAREED searches for a high-quality Pareto candidate set by considering service reliability, inference accuracy and inference delay. In the second stage, DAREED selects the final deployment scheme from the candidate set by incorporating task preferences and total energy consumption.

### 建议修改后

> In the first stage, DAREED performs a four-objective evolutionary search that jointly considers predicted service reliability, predicted inference accuracy, inference delay, and total energy consumption. Four-objective constraint-dominance and crowding-distance selection guide population evolution. In the second stage, DAREED applies request-specific preference weights to the strictly feasible candidates maintained throughout the search and selects one deployment. Pareto-front export and hypervolume evaluation are enabled only in the dedicated Pareto experiment and are not part of ordinary online deployment.

### 修改原因

能耗已进入第一阶段搜索。普通实验也不会为了选解而额外收集完整 Pareto 点；代码维护的是跨代严格可行偏好最优候选。

---

## D06【必须】适应度向量与支配方向

### 位置

- `paperwork/algorithm design optimized.md`：第 19–24 行。
- `paperwork/optimized_full.md`：第 424–429 行。

### 原文

```latex
F(X^m_i)=\left(A^m_i(t),R^m_i(t),U^m_T,U^m_E\right),
```

> In DAREED, the first three fitness components are mainly used for Pareto ranking, while $U^m_E$ is further used in candidate scoring, mutation guidance, local search and final deployment selection.

### 建议修改后

```latex
\begin{equation}
\label{eq:fitness_values}
F(X^m_i)=
\left(
\hat A^m_i(t\mid X^m_i),
\hat R^m_i(t\mid X^m_i),
U^m_T,
E^m_i
\right),
\end{equation}
```

```latex
where $\hat A^m_i(t\mid X^m_i)$ and
$\hat R^m_i(t\mid X^m_i)$ are the one-step post-admission inference
accuracy and service reliability predicted for candidate $X^m_i$,
$U^m_T$ is the maximization-oriented delay utility, and $E^m_i$ is the
original total energy consumption. The first three components are
maximized and $E^m_i$ is minimized. All four components participate in
Pareto ranking, crowding distance, local search, and final
preference-based scoring after normalization.
```

### 修改原因

既修正能耗目标，也体现当前候选评价使用接纳后的预测可靠性，而不是部署前静态值。

---

## D07【必须】新增候选接纳后的单步可靠性预测

### 位置

- 建议插入到 `paperwork/algorithm design optimized.md` 第 24 行之后、`\subsection{DAG Context Construction}` 之前。
- 合并稿对应当前位置约为 `paperwork/optimized_full.md` 第 429 行之后。

### 原文

当前算法设计章节没有描述候选部署对节点/链路负载和热度的单步预测。

### 建议新增

```latex
\subsection{One-Step Post-Admission State Prediction}
\noindent
Evaluating a candidate only with the pre-deployment state may
overestimate the reliability of nodes and links that become heavily
loaded after admission. DAREED therefore performs a one-step,
non-mutating prediction for each candidate deployment. Let
$\Delta C_n(X)$ denote the CPU demand added to node $n$ by candidate
$X$. The predicted node load and heat are
\begin{equation}
\hat\rho_n(t\mid X)=
\rho_n(t)+\frac{\Delta C_n(X)}{\bar C^n_{cpu}},
\end{equation}
\begin{equation}
\hat h_n(t\mid X)=
\lambda_h h_n(t)+(1-\lambda_h)\hat\rho_n(t\mid X).
\end{equation}
For link $l$, let $\Delta D_l(X)$ be the candidate traffic carried by
the link and $\hat T_i(X,t)$ be the estimated candidate runtime. Its
predicted load and heat are
\begin{equation}
\hat\mu_l(t\mid X)=
\mu_l(t)+
\frac{\Delta D_l(X)}
{\hat T_i(X,t)\hat B_l(t)},
\end{equation}
\begin{equation}
\hat g_l(t\mid X)=
\lambda_g g_l(t)+(1-\lambda_g)\hat\mu_l(t\mid X).
\end{equation}
The predicted node and link reliability values are then obtained by
substituting $\hat\rho_n$, $\hat h_n$, $\hat\mu_l$, and $\hat g_l$
into the dynamic degradation model. These values are used to evaluate
$\hat R'_i(t\mid X)$ and $\hat A'_i(t\mid X)$ without modifying the
actual environment state. The real state is updated only after a
candidate has been accepted.
```

### 修改原因

这是当前实现相对于原论文描述的重要机制。它增加的是每个候选的一次线性状态估计，不会在搜索期间污染真实环境状态。

---

## D08【建议】初始化需说明去重和多样性目标

### 位置

- `paperwork/algorithm design optimized.md`：第 41–43 行。
- `paperwork/optimized_full.md`：第 446–448 行。

### 原文

> The initial population is generated by combining several heuristic deployment modes instead of relying only on random sampling. Specifically, the population contains cloud-preferred individuals, edge-local individuals, resource-preferred individuals, reliability-preferred individuals, topology-aware individuals and random feasible individuals.

### 建议修改后

> The initial population is generated from cloud-preferred, edge-local, resource-preferred, reliability-preferred, and topology-aware templates, followed by template perturbation and random feasible sampling. Candidate chromosomes are repaired and deduplicated during construction. DAREED attempts to make at least 75% of the initial population unique; if this target cannot be reached within a bounded retry budget, feasible individuals are duplicated only to complete the required population size. This strategy preserves heuristic coverage without allowing a small number of templates to dominate the initial population.

### 修改原因

当前实现不是简单混合若干模式，而是具有明确的去重目标、重试预算和软回退。

---

## D09【建议】角色子种群的实际实现方式

### 位置

- `paperwork/algorithm design optimized.md`：第 52–54 行。
- `paperwork/optimized_full.md`：第 457–459 行。

### 原文

> The population is divided into several role-based subpopulations ... In each generation, parent individuals are mainly selected from the same role subpopulation, while periodic cross-role parent selection is allowed ...

### 建议修改后

> Each individual is assigned a persistent role label: reliability, latency, energy, or feasibility. Role-specific candidate-score weights guide variation, and environmental selection preserves an approximately equal quota for every role. Parents are normally sampled from the same role, while every five generations one parent may be sampled globally to realize cross-role migration. This label-and-quota implementation maintains specialization without allocating four additional full populations.

### 修改原因

原文方向基本正确，但“划分成子种群”容易被理解为四套独立物理种群。代码采用持久标签、角色配额和周期性全局父代选择。

---

## D10【必须】DAG 交叉块不是拓扑序后缀

### 位置

- `paperwork/algorithm design optimized.md`：第 56–66 行。
- `paperwork/optimized_full.md`：第 461–471 行。

### 原文

> The crossover boundary is selected from the critical path or a topological boundary. ... the tasks after the selected boundary in topological order are exchanged.

### 建议修改后

```latex
For an anchor layer $v_a$, DAREED constructs a successor-closed DAG
block
\begin{equation}
\mathcal{B}(v_a)=
\{v_a\}\cup Desc(v_a),
\end{equation}
such that every successor of a layer in $\mathcal{B}$ also belongs to
$\mathcal{B}$. Candidate blocks are size-controlled, with a preferred
upper bound of $\lceil0.4|V_i|\rceil$; if no block satisfies the
preferred range, the smallest available proper successor closure,
possibly a singleton, is used. With probability $p_c$, the node
assignments of all layers in $\mathcal{B}$ are exchanged between two
parents. The offspring are accepted only when structural repair
succeeds and the number of genes changed by repair does not exceed
$\max(1,\lfloor|\mathcal{B}|/2\rfloor)$; otherwise, the corresponding
parent is retained.
```

交叉公式本身可以保留，但应将 $\mathcal{B}$ 的定义改成上述后继闭包。

### 修改原因

当前交叉显式构造后继闭包，避免把“任务编号后缀”误写成 DAG 子图。

---

## D11【建议】偏斜变异的归一化、softmax 和概率

### 位置

- `paperwork/algorithm design optimized.md`：第 68–70 行。
- `paperwork/optimized_full.md`：第 473–475 行。

### 原文

> Then, it samples a new node from the legal candidate set according to the candidate score in \eqref{eq:candidate_score}.

### 建议修改后

> With probability $p_m$, the mutation operator selects a layer, assigning half of the selection probability to critical-path layers and half to all layers. The current node is excluded from the candidate set. Candidate scores are min-max normalized within the selected layer and converted into sampling probabilities by a numerically stable softmax. The mutated chromosome is retained only if structural repair succeeds; otherwise, the parent assignment is restored. Crossover is independently activated with probability $p_c$.

可补充：

```latex
P(n\mid v^i_j)=
\frac{\exp((\tilde S_{i,j,n}-\tilde S_{\max})/\tau)}
{\sum_{q\in\mathcal{C}_{i,j}\setminus\{x_{i,j}\}}
\exp((\tilde S_{i,j,q}-\tilde S_{\max})/\tau)}.
```

### 修改原因

当前实现已避免原始分数尺度不一致和数值溢出，并确保变异确实改变当前节点。

---

## D12【必须】repair 与约束违反度的职责边界

### 位置

- `paperwork/algorithm design optimized.md`：第 72–91 行。
- `paperwork/optimized_full.md`：第 477–496 行。

### 原文

> For individuals that cannot be fully repaired, DAREED does not discard them immediately. Instead, a normalized constraint violation value is calculated ...

### 建议修改后

> Repair explicitly returns the repaired chromosome, a success flag, the number of changed genes, and a failure reason. If crossover or mutation produces a chromosome that cannot be structurally repaired, that offspring falls back to its parent and is not silently inserted as a repaired solution. Structurally valid individuals may still have nonzero delay, CPU, hierarchy, or predicted-link-overload violation. These residual violations are quantified by $CV(X)$ and handled by the generation-dependent tolerance $\epsilon_g$. Thus, structural repair failure and temporary constraint infeasibility are treated separately.

保留原有 `CV` 和 `\epsilon_g` 公式。

### 修改原因

原文可能被理解为 repair 失败的结构非法个体也会直接进入种群。当前代码会回退父代；$\epsilon$ 机制主要负责保留具有有限 CV 的候选。

---

## D13【必须】四目标约束支配、拥挤距离和局部搜索

### 位置

- `paperwork/algorithm design optimized.md`：第 93–114 行。
- `paperwork/optimized_full.md`：第 498–519 行。

### 原文

支配关系：

```latex
\begin{cases}
A_a\geq A_b,\\
R_a\geq R_b,\\
U^a_T\geq U^b_T,\\
A_a>A_b\ \cup\ R_a>R_b\ \cup\ U^a_T>U^b_T.
\end{cases}
```

拥挤距离说明：

> where $Z=3$ in the Pareto sorting stage ...

局部搜索说明：

> is not worse in the three Pareto objectives

### 建议修改后

```latex
\begin{cases}
A_a\geq A_b,\\
R_a\geq R_b,\\
U^a_T\geq U^b_T,\\
E_a\leq E_b,\\
A_a>A_b\ \cup\ R_a>R_b\ \cup\
U^a_T>U^b_T\ \cup\ E_a<E_b.
\end{cases}
```

将拥挤距离后的解释改为：

> where $Z=4$ and the four dimensions are inference accuracy, service reliability, delay utility, and total energy consumption. The energy dimension is sorted in its original minimization direction; the distance magnitude is calculated from neighboring objective values as in the other dimensions.

将局部搜索句改为：

> A neighboring solution is accepted only if it does not increase constraint violation, is not worse in all four Pareto objectives, including no increase in total energy, and achieves a higher normalized preference score.

### 修改原因

这是当前算法描述中最直接的正确性错误。代码中的支配、拥挤距离和局部搜索已经全部包含能耗。

---

## D14【必须】最终偏好评分与严格可行性

### 位置

- `paperwork/algorithm design optimized.md`：第 116–128 行。
- `paperwork/optimized_full.md`：第 521–533 行。

### 原文

```latex
Score(X^m_i)=w_a\bar A^m_i+w_r\bar R^m_i+w_t\bar U^m_T+w_e\bar U^m_E,
```

> The individual with the largest score and positive delay utility is selected as the final deployment scheme. If no individual satisfies the delay requirement, the DNN request is rejected.

### 建议修改后

```latex
\begin{equation}
\label{eq:final_score}
Score(X^m_i)=
w_a\bar A^m_i+
w_r\bar R^m_i+
w_t\bar U^m_T+
w_e\bar E^m_i,
\end{equation}
```

```latex
where $\bar E^m_i=(E^i_{\max}-E^m_i)/
(E^i_{\max}-E^i_{\min})$ after clipping to $[0,1]$. During evolution,
DAREED maintains the highest-scoring strictly feasible candidate found
across generations. After the final generation, it also examines the
strictly feasible final population. Before admission, the selected
chromosome is re-evaluated and accepted only if $CV(X)\leq10^{-12}$ and
$U_T(X)>0$. Otherwise, the request is rejected.
```

### 修改原因

正时延收益不是最终接纳的唯一条件。代码还要求完整 CV 严格为零，并在部署前清除 CV 缓存后重新验证。

---

## D15【必须】在线算法伪代码

### 位置

- `paperwork/algorithm design optimized.md`：第 132–177 行，Algorithm 1。
- `paperwork/optimized_full.md`：第 537–582 行。

### 原文

关键旧行：

```latex
\STATE Evaluate $F(X)=\left(A_i(X,t),R_i(X,t),U_T(X,t),U_E(X,t)\right)$ ...
\STATE $\mathcal{P}_i\gets ND(P_{G_{\max}})$
\STATE $\mathcal{P}^f_i\gets\{X\in\mathcal{P}_i\mid CV(X)=0,\ T_i(X,t)\leq T^i_{limit}\}$
\STATE $X_i^\ast\gets \arg\max_{X\in\mathcal{P}^f_i} Score(X)$
```

### 建议修改后

将初始化前增加：

```latex
\STATE $X_i^{best}\gets\varnothing$, $s_i^{best}\gets-\infty$
```

将代内评价与选择改为：

```latex
\STATE Predict post-admission node/link states for all
$X\in P_g\cup\tilde Q_g$
\STATE Evaluate
$F(X)=\left(\hat A_i(X,t),\hat R_i(X,t),U_T(X,t),E_i(X)\right)$
and $CV(X)$
\STATE Update $(X_i^{best},s_i^{best})$ using candidates satisfying
$CV(X)=0$ and $U_T(X)>0$
\STATE $\mathcal{F}_g\gets ConstraintSort_4D(P_g\cup\tilde Q_g,\epsilon_g)$
\STATE $P_{g+1}\gets RoleQuotaSelect(\mathcal{F}_g,M)$
\STATE $P_{g+1}\gets LocalRefine_4D(P_{g+1},\mathcal{K}_i,\Theta(t))$
```

将最终非支配集选解部分改为：

```latex
\STATE Re-evaluate the final population and update $X_i^{best}$ with
strictly feasible candidates
\IF{$X_i^{best}\neq\varnothing$ and
$RecheckCV(X_i^{best})=0$}
    \STATE $X_i^\ast\gets X_i^{best}$
    \STATE Admit $X_i^\ast$ and register its occupied time slots
\ELSE
    \STATE Reject $task_i$ and record one failed deployment
\ENDIF
```

在伪代码说明后新增：

> In ordinary online experiments, the four-objective ranks are used internally for environmental selection, but Pareto points are not exported. A feasible non-dominated archive is collected only when the experiment suite is explicitly configured for Pareto-front evaluation.

### 修改原因

当前部署解来自跨代严格可行偏好最优记录与最终种群的共同比较，不是只从最后一代非支配集合中选取。

---

## D16【建议】复杂度与能耗/HV 开销边界

### 位置

- `paperwork/algorithm design optimized.md`：第 179–190 行。
- `paperwork/optimized_full.md`：第 584–595 行。

### 原文

> Fitness evaluation requires calculating inference accuracy, service reliability, inference delay and total energy consumption. Considering shortest-path link lookup and dependency transmission evaluation, the complexity can be written as $O(M(v+|E_i|+l))$.

### 建议修改后

> Fitness evaluation calculates inference delay and total energy together with one-step post-admission node/link load, heat, service-reliability, and accuracy prediction. With cached physical paths, this evaluation remains $O(M(v+|E_i|+l))$ per generation. Adding the fourth objective changes only the constant factor of dominance and crowding-distance operations; their asymptotic complexities remain $O(M^2)$ and $O(M\log M)$, respectively. Pareto-point export and four-dimensional hypervolume are disabled in ordinary experiments. They are executed only as postprocessing in the Pareto experiment and are excluded from the reported algorithm runtime.

### 修改原因

能耗计算本身复用候选评价中的任务/路径遍历，不会引入新的阶数量级。HV 不属于在线算法运行开销。

---

## D17【必须】平均可靠性、能耗和运行时间指标定义

### 位置

- `paperwork/performance analysis optimized.md`：第 51–63 行。
- `paperwork/optimized_full.md`：第 646–658 行。

### 原文

> Average service reliability: ... considering ... dynamic reliability ...

> Average inference accuracy: ... dynamically changing computing nodes.

> Average total energy consumption: ... including computation energy and transmission energy.

> Average running time: The wall-clock time consumed by an algorithm in each simulation run.

### 建议修改后

> **Average predicted service reliability**: The average one-step post-admission service reliability of successfully accepted DNN requests, including the predicted reliability of the unique computing nodes and traversed input, dependency, and output links.

> **Average predicted inference accuracy**: The average one-step post-admission inference accuracy of successfully accepted DNN requests, weighted by the computational workload assigned to each used node.

> **Average total energy consumption**: The average computation energy plus input-upload and inter-layer transmission energy of successfully accepted DNN requests. Final result-return energy is excluded consistently with the energy model.

> **Algorithm running time**: The wall-clock time of deployment search and online execution in one simulation run. Pareto-front serialization and hypervolume postprocessing are excluded.

### 修改原因

指标值来自候选接纳后的预测可靠性；能耗口径明确不包含结果回传；HV 后处理时间不进入 `runtime_ms`。

---

## D18【必须】Hypervolume 改成四维且只在 Pareto 实验计算

### 位置

- `paperwork/performance analysis optimized.md`：第 63 行。
- `paperwork/optimized_full.md`：第 658 行。

### 原文

> For each DNN request, the feasible non-dominated candidate set is normalized in the three-dimensional objective space composed of service reliability, inference accuracy and delay satisfaction. The reference point is set to $(0,0,0)$ ...

### 建议修改后

> **Four-dimensional hypervolume**: Hypervolume is calculated only in the dedicated Pareto-front experiment. For each DNN request, strictly feasible non-dominated candidates are represented by normalized inference accuracy, normalized service reliability, delay satisfaction, and energy satisfaction. Delay satisfaction is $(T^i_{limit}-T_i)/T^i_{limit}$ clipped to $[0,1]$, and energy satisfaction is defined in \eqref{eq:energy_satisfaction}. The common reference point is $(0,0,0,0)$. Per-request HV values are averaged across DNN requests, and their sum and standard deviation are also recorded. The implementation marks these results as `4d_v1`. No Pareto points or HV fields are generated for overall, dynamic, sensitivity, ablation, scale, or preference experiments.

### 修改原因

当前实验入口已对 HV 做 suite 门控，并使用四维归一化空间。继续写三维 HV 会导致论文指标定义与结果文件不一致。

---

## D19【必须】Pareto Front Quality 小节

### 位置

- `paperwork/performance analysis optimized.md`：第 141–169 行。
- `paperwork/optimized_full.md`：第 736–764 行。

### 原文

> The objectives used for Pareto dominance and hypervolume calculation are inference accuracy, service reliability and delay satisfaction. Energy utility ... is not used in the three-objective hypervolume calculation.

> \caption{Pareto front comparison in the normalized three-objective space.}

> \caption{Pareto Front Quality Comparison}

### 建议修改后

正文第一段：

> The Pareto front quality is evaluated using DAREED, NSGA-ED, RDA and SAA. Pareto dominance is performed over predicted inference accuracy, predicted service reliability, delay utility, and raw total energy consumption. For cross-request HV calculation, the corresponding normalized dimensions are inference accuracy, service reliability, delay satisfaction, and energy satisfaction. Thus, energy is part of both candidate search and four-dimensional HV evaluation.

三维图说明：

> \autoref{fig_pareto_3d} compares nondominated solution sets for one matched representative repeat and DNN request. The spatial axes show inference accuracy, service reliability, and delay satisfaction; point color encodes energy satisfaction, and black outlines identify members of the joint four-objective reference front.

图标题：

```latex
\caption{Three-dimensional projection of the normalized four-objective Pareto front.}
```

表标题：

```latex
\caption{Four-Objective Pareto Front Quality Comparison\label{tab:pareto_hv}}
```

表头可以保留 `Avg. HV`，但建议在正文注明其版本为 4D HV。

### 修改原因

现有三维图仍可保留，但必须明确它只是四维前沿的投影，不能再称为三目标 Pareto 空间。

---

## D20【建议】消融实验覆盖新增机制

### 位置

- `paperwork/performance analysis optimized.md`：第 260–288 行。
- `paperwork/optimized_full.md`：第 855–883 行。

### 原文

> The complete DAREED algorithm is compared with five variants: w/o DAG-aware initialization, w/o DAG block crossover, w/o skew mutation, w/o elite local search, and a plain evolutionary variant that disables all four customized components.

### 建议修改后

> The current executable ablation suite compares full DAREED with w/o DAG-aware initialization, w/o DAG block crossover, w/o skew mutation, w/o elite local search, and a plain evolutionary variant. Because the updated full algorithm also contains persistent role specialization and one-step post-admission reliability prediction, two additional paired ablations should be reported before attributing gains to these mechanisms: w/o role specialization/migration and w/o post-admission prediction. Each ablation should use the same random seeds as full DAREED and report accepted-request reliability, accuracy, delay, total energy, failed deployments, and runtime.

建议表中新增两行：

```latex
w/o role specialization & TODO & TODO & TODO & TODO\\
w/o post-admission prediction & TODO & TODO & TODO & TODO\\
```

建议给表增加 `Delay` 和 `Runtime` 两列。

### 修改原因

论文算法章节已经描述角色机制，更新后的实现又增加了候选状态预测；当前五项消融无法分别验证这两个机制。注意：在实际补充这两组结果前，需要先给实验入口增加对应开关，不能直接填写或宣称实验已经完成。

---

## D21【必须】算法流程图图注

### 位置

- `paperwork/dareed_algorithm_flow_figure.tex`：第 4 行。

### 原文

> searches for Pareto candidate deployments through customized evolutionary operators, and selects the final scheme according to task preferences and energy utility.

### 建议修改后

> Overview of the proposed DAREED algorithm. For each online DNN inference request, DAREED observes the dynamic cloud-edge-end state, constructs the DAG context, predicts candidate post-admission states, and performs four-objective evolutionary search over inference accuracy, service reliability, delay, and raw total energy. A strictly feasible deployment is selected according to the request preference, registered as a running block, and revalidated before admission. Pareto-front export and four-dimensional hypervolume are enabled only in the dedicated Pareto experiment.

### 修改原因

图注中的 `energy utility` 已过时，也没有体现单步预测、四目标搜索和 HV 门控。

---

## D22【建议】动态退化实验的机制解释

### 位置

- `paperwork/performance analysis optimized.md`：第 230–258 行，尤其第 234 行。
- `paperwork/optimized_full.md`：第 825–853 行。

### 原文

> DAREED is expected to be more stable under the strong degradation scenario, because its mutation, repair and local search are guided by load, heat and dynamic reliability.

### 建议修改后

> DAREED is expected to be more stable under strong degradation because candidate evaluation predicts the load and heat introduced by the candidate itself before computing service reliability and inference accuracy. Role-aware mutation and local refinement then use these predicted values together with delay, energy, and feasibility information. The final conclusion should be stated only after paired-seed results are available.

### 修改原因

更新后的主要动态可靠性机制不只是算子读取当前状态，而是候选接纳后的可靠性预测。

---

## D23【必须】合并稿同步方式

### 位置

- `paperwork/optimized_full.md`：全文。
- `paperwork/merge_optimized.py`：`SECTION_ORDER` 和 `merge_sections()`。

### 当前问题

当前 `optimized_full.md` 与分节文件不完全一致，例如：

- 合并稿第 3 行仍写 `three-objective Pareto candidate generation`，而 `abstract optimized.md` 已无该表述。
- 合并稿第 23 行仍写 `three-objective Pareto candidate generation`，而 `introduction optimized.md` 的对应贡献点已改变。
- 合并稿第 405 行与 `problem formulation optimized.md` 第 310 行内容不同。

### 建议操作

完成分节修改后，通过现有合并脚本重新生成：

```bash
python3 paperwork/merge_optimized.py
```

不建议直接在 `optimized_full.md` 中逐条手工修改，否则下次合并时会再次被覆盖。

---

## 3. 当前不需要因本次算法更新而修改的部分

以下分节的核心论述已经与四目标方向一致，可在合并稿同步后保留：

- `paperwork/motivation optimized.md`：已明确可靠性、精度、时延和总能耗四方面权衡。
- `paperwork/conclusion optimized.md`：已将问题表述为四目标优化。
- `paperwork/related work optimized.md`：没有依赖三目标 HV 或 `1/E` 的算法实现细节。
- `paperwork/problem formulation optimized.md` 中的计算能耗与传输能耗公式：与代码一致，均不计最终结果回传能耗。
- `paperwork/problem formulation optimized.md` 中的时延收益分段公式：与当前代码一致。

## 4. 推荐修改顺序

1. 先修改 `problem formulation optimized.md` 中的能耗表示和四目标向量。
2. 再修改 `algorithm design optimized.md` 中的预测评价、四目标支配、算子和伪代码。
3. 修改 `performance analysis optimized.md` 中的 4D HV、运行时间口径和消融设计。
4. 修改摘要、引言贡献点和流程图图注。
5. 使用 `merge_optimized.py` 重新生成 `optimized_full.md`。
6. 重新运行 `rg -n -i "three-objective|Z=3|U_E|1/E|energy utility"`，确认正文不存在旧算法口径。
7. 正式实验完成后再填充 TODO 表格，并避免把小规模实现回归数据写成论文性能结论。
