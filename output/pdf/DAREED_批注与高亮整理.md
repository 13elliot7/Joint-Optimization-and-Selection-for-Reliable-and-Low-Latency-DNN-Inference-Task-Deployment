# DAREED 论文批注与高亮整理

## 一、文档概况

- 文档：`0724-批注-DAREED__DAG_Aware_Reliable_and_Energy_Efficient_DNN_Inference_Deployment_in_Cloud_Edge_End_Systems__revision_.pdf`
- 总页数：13 页
- 存在批注/高亮的页面：第 1-11 页
- 注释总数：79 条（高亮 75、插入符 2、文本注解 2）
- 含明确批注文字：54 条；仅高亮、无批注文字：25 条
- 批注作者：王颖
- 批注时间：2026-07-22 至 2026-07-24

## 二、优先修改清单

### A. 核心逻辑与模型一致性（优先级：高）

1. **统一优化目标的数量与表述。** 全文多次出现 `three / first three / Z=3`，但论文同时讨论 reliability、accuracy、delay、energy，需明确究竟是三目标还是四目标；同步更新摘要、贡献、问题定义、Pareto 排序、子种群、选择机制和实验描述（第 1、2、8、9、10 页）。
2. **明确能耗在论文中的地位。** 需说明能耗是独立优化目标、偏好选择指标，还是约束/辅助指标，并解释“能耗优化的确切目标”。摘要结论、关键词、动机、效用函数和算法描述必须一致（第 1、5、8-10 页）。
3. **解释并落实 reliability degradation。** 明确“可靠性退化”的物理含义、数学表达及其在算法中的作用，避免仅在文字中出现而模型/算法没有对应（第 1、2、5、7 页）。
4. **重新审视 heat 模型。** 节点热状态可以理解，但“link heat / overheated links”概念存疑；应说明链路退化的真实机制，或改用拥塞、负载、有效带宽等更恰当的状态变量（第 5-7 页）。
5. **核对时延与能耗模型的完整性。** 统一 service duration、running time、inference delay、inter-layer transmission delay、propagation delay 等概念；确认是否包含推理结果回传产生的时延/能耗（第 6、7 页）。
6. **修正优化问题公式。** 为完整优化问题增加公式编号并尽量用一个完整表达式呈现；检查公式 (31) 与 (20) 重复；确认公式 (32) 中两个“最小化”是否应改为效用最大化（第 8 页）。

### B. 章节组织与论证边界（优先级：中高）

1. Related Work 中 DAG 内容的位置与小节标题不匹配，且出现在 `In summary` 之后；需调整结构（第 3 页）。
2. 澄清 A.3 与 B 两处 reliability 相关综述的分工，避免重复（第 3 页）。
3. 文献综述的总结应严格对应所引文献，尤其 B 小节应围绕 reliability 收束，不要得出超出文献覆盖范围的结论（第 4 页）。
4. 问题建模阶段尽量只定义问题，避免过早写入具体 evolutionary search 方法（第 8 页）。
5. 算法阶段、DAG context、topology-aware mode/score、role-based subpopulations 等概念之间的关系需讲清楚并避免机制重复（第 9、10 页）。

### C. 图表、术语与格式（优先级：中）

1. 重画或简化图 1：考虑合并物理平面与灰色平面；替换易误解的云形图标；补全图例及虚线 `originating` 的含义；解释彩色平面与浅蓝平面的关系；统一 `computing node` 命名（第 5 页）。
2. 最终逐项核对表 I 的参数和符号（第 6 页）。
3. 章节编号改为与模板一致的罗马数字（第 2 页）。
4. 修正 `last → the last`、`Constraint → Constraints` 等文字问题，并处理 `??` 占位符（第 5、7 页）。
5. 伪代码中的 InitializePopulation、Repair、LocalRefine、AdvanceRunningSet、UpdateState、AggregateMetrics 等函数应在正文中得到说明（第 11 页）。

## 三、逐页批注整理

### 第 1 页：摘要、关键词与研究动机

1. 高亮 `degradation`：无文字批注；结合第 2 页批注，应统一解释“可靠性退化”。
2. 高亮摘要结论 `The results demonstrate that DAREED achieves a favorable balance ...`：参照期刊其他论文，确认摘要是否需要给出定量结论。
3. 高亮 `Index Terms—DNN`：批注为 `energy-efficient？`，建议检查关键词是否应补入 energy-efficient / energy efficiency。
4. 高亮应用场景 `intelligent security and intelligent inspection`：考虑更新应用场景。
5. 高亮 `Energy consumption`：该段以能耗开头但后续能耗论述较少，逻辑不一致；需明确本文能耗优化的确切目标。

### 第 2 页：研究问题、贡献与文章结构

1. 高亮 `... studies dynamic ...`：`dynamic` 是否应修饰 `environment`，建议检查语序。
2. 高亮 `degradation`：解释可靠性“退化”的含义，以及其在后续模型和算法中的体现。
3. 高亮 `three-objective Pareto candidate generation`：确认是 three 还是 four。
4. 同一贡献句另有高亮：应强调算法设计相对于问题特征的针对性。
5. 高亮章节号 2、3、4、5、6、7：与模板保持一致，正文应使用罗马数字编号。
6. 高亮 `The discussion focuses on the limitations ...`：若需精简篇幅，此句/段可删除。

### 第 3 页：Related Work 的组织

1. 高亮 `preserving`：无文字批注，需结合原句检查措辞或语法。
2. 高亮 `Recent studies ...`：该段 DAG 内容超出原小节标题和概述段范围，且位于 `In summary` 之后，组织逻辑有问题。
3. 高亮小节标题 `Reliability-aware Edge Computing`：说明 B 小节与前面 A.3 的关系，两处均涉及 reliability，需划清边界。

### 第 4 页：综述总结与研究空白

1. 高亮 `However, existing reliability-aware edge computing studies ...`：总结应更有针对性，并围绕 B 小节的 reliability 内容展开。
2. 文本注解：从该处开始的一段结论与前述文献不能完全对应，且结论超出了可靠性范畴。
3. 高亮 `This paper addresses this gap by modeling DNN inference requests as DAGs ...`：无文字批注；建议结合以上两条检查该 gap 是否由综述充分推出。
4. 另有一条空白文本注解，无可提取内容。

### 第 5 页：动机示例、动态环境与图 1

1. 高亮 `total energy consumption`：无文字批注；应与“能耗是否为独立目标”的全文问题一并处理。
2. 高亮链路由充足带宽变为拥塞的例子：该句本身似乎就是支持“动态性”的论据，需检查上下文论证意图。
3. 高亮 `heat state`：无文字批注；需明确热状态的定义与适用对象。
4. 高亮 `??`：存在未解决的引用/占位符，应修复。
5. 图 1 批注：
   - 最上方物理平面是否必要；若保留，考虑放在下方并与灰色平面合并。
   - 替换图标；云形图标容易让读者误解为网络，其他图标也可更生动。
   - 图例似乎不完整；解释虚线 `originating` 的含义。
   - 解释红/黄/绿平面与浅蓝平面的关系，检查是否存在冗余。
6. 高亮 `caused by accumulated heat`：环境变化是否仅由热因素引起？
7. 高亮 `effective bandwidth variation of links`：是否与前文 `time-varying load of transmission links` 重复？
8. 高亮 `computing node`：图 1 中的命名应与正文一致。

### 第 6 页：资源、可靠性与时延模型

1. 高亮 `The link heat`：链路是否存在 `heat` 这一概念？
2. 高亮 `or`：检查 `or` 前后连接的内容是否属于同一语义层级。
3. 高亮 `The running time is estimated by the delay model`：服务持续时间与推理时延是否为同一概念？
4. 高亮 `propagation delay`：该传播时延与前文层间传输时延是否为同一概念？
5. 高亮 `TABLE I PARAMETERS SYMBOLS`：表 I 尚未细查，定稿前需逐项对照检查。

### 第 7 页：可靠性、能耗与优化特征

1. 插入符位于 `last`：补入 `the`，改为 `the last`。
2. 高亮 `overheated links`：同前，链路“过热”概念存疑。
3. 连续高亮 reliability 同时考虑唯一计算节点与唯一链路的句子：无文字批注；建议重点核对公式是否准确实现该表述。
4. 高亮 `This weighted form`：需明确“weighted form”具体指公式中的哪种形式。
5. 高亮 transmission energy 句：与时延模型对应时，是否遗漏推理结果传输？虽然后文有解释，但当前句应包含完整范围，或调整因果/解释顺序。
6. 高亮 `nonlinear reliability products, exponential degradation, reciprocal delay and energy utilities, and dynamic state transitions`：确认这些特征与前文模型描述逐一、明确呼应。
7. 插入符位于 `Constraint`：补 `s`，改为复数 `Constraints`。

### 第 8 页：约束、优化问题与目标效用

1. 高亮 CPU 约束中 `running DNN tasks and the newly accepted task`：无文字批注，作为下一条的对照。
2. 高亮另一约束中的 `all running DNN tasks`：相较上一约束，为何未体现 newly accepted task？
3. 高亮 `subject to (21)-(25) ...`：完整优化问题没有公式编号；考虑用一个完整公式表达最优化问题。
4. 高亮一段效用与算法用途描述，批注 `更新`：需按当前目标定义和算法流程更新。
5. 高亮 `(31)`：公式 (31) 与公式 (20) 重复。
6. 高亮 `(32)`：两个最小化是否应改为相应效用函数的最大化？否则前面对效用函数的解释缺乏必要性。
7. 高亮 `the evolutionary search evaluates`：问题建模阶段是否有必要体现具体方法？并检查前文是否有类似越界。
8. 高亮 `on the current system state`：含义不清，需要具体解释依赖哪些状态、如何依赖。
9. 高亮 `service reliability, inference accuracy and inference delay`：是否应加入/更新能耗目标？
10. 高亮 `task preferences and total energy consumption`：同上，需统一能耗在目标体系中的位置。

### 第 9 页：算法框架与 DAG-aware 机制

1. 高亮 `DAREED incorporates DAG-aware evolutionary operators ...`：该描述是否体现算法第二阶段？
2. 高亮请求被接受并注册为 running block 的段落：检查该段是否必要。
3. 高亮 `the first three`：同前，目标数量需更新。
4. 高亮 `used for Pareto ranking, while U_E ...`：同前，确认是否需按能耗目标更新。
5. 两处高亮 `n ∈ N`：后一个表达是否可直接替换前一个，避免重复。
6. 高亮 `For each layer ...` 段：与前段关系不清，且似乎与 DAG context 关联不大。
7. 高亮 `topology-aware score`：它与上一段的 `topology-aware mode` 是否相同？若不同，当前命名易混淆；也可考虑与小标题呼应。
8. 高亮 role-based subpopulations 使用不同权重向量的句子：可能与下一个机制重复。
9. 高亮 `six`：确认应为 six 还是 seven。

### 第 10 页：层/任务、子种群与偏好权重

1. 高亮 `tasks`：确认此处应为 layers 还是 tasks。
2. 高亮 `hierarchy violation ratio`：需要解释具体含义及计算方式。
3. 高亮 `Z = 3`：是否需更新并纳入能耗？
4. 高亮 `inference accuracy, service reliability or delay utility`：同前，检查目标数量和能耗。
5. 高亮 `three`：同前，检查数字。
6. 三段无文字批注的高亮分别涉及 `task preferences and total energy consumption`、三类权重以及固定能耗偏好 `w_e`：应整体核对能耗是 Pareto 目标还是最终偏好项，并统一符号、权重和文字说明。

### 第 11 页：伪代码函数与正文说明

1. 高亮 `InitializePopulation` 并批注：黄色高亮函数没有体现在正文说明中，检查是否影响对伪代码的理解。
2. 其余无文字批注的黄色高亮函数：`LocalRefine`、`Repair`、`AdvanceRunningSet`（两处）、`UpdateState`、`AggregateMetrics`。建议在伪代码前后按调用顺序逐一说明输入、输出和作用。

### 第 12-13 页

未检测到批注或高亮。

## 四、建议的修改顺序

1. 先确定目标体系：三目标还是四目标，以及能耗的角色。
2. 再统一动态状态与退化机制：节点热、链路拥塞/带宽、可靠性退化。
3. 重审公式和约束：时延、能耗、可靠性、目标方向、编号及重复公式。
4. 按最终模型更新摘要、贡献、问题定义、算法和实验描述。
5. 重组 Related Work 和算法机制说明。
6. 最后处理图 1、表 I、伪代码解释、章节编号、术语和拼写。

> 注：无文字批注的高亮只能依据上下文推断检查方向；修改时建议回到 PDF 原页确认批注者意图。
