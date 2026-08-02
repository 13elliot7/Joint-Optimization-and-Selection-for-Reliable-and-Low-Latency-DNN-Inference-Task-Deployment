# Response Letter 批注整理

## 一、文档概况

- 文件：`0724-批注-response letter.docx`
- 文档页数：14 页
- 批注总数：13 条
- 批注作者：王颖
- 批注状态：13 条均未解决
- 批注时间：2026-07-24 15:43-17:54（北京时间）
- 批注分布：第 2、3、5、7-10、12、13 页

## 二、集中反馈与修改重点

### 1. Author action 普遍过于简略

多条批注指出，当前 `Author action` 多为“修改了某节”的概括，缺少足以让审稿人快速核验的细节。建议统一补充：

- 精确的章节号、页码；
- 图、表、公式编号；
- 关键修改后的原文或简短摘录；
- 新增实验的规模、设置和主要定量结果；
- 修改目的及修改后解决了什么问题。

该问题首先在第 5 页明确提出，并以“同前，下同”延伸至后续各条回复，应视为全文通用修改要求。

### 2. 需要强化算法创新性和专门化设计

不能只罗列 DAREED 包含 initialization、crossover、mutation、repair 等步骤，还应说明：

- 每个机制如何针对 DNN DAG、动态资源状态或可靠性问题设计；
- 相比 generic NSGA-II 的实质区别；
- 这些专门化设计带来的效果，最好由消融实验或定量结果支撑；
- 在基线介绍中用定语从句等方式明确其他算法的性质，从而凸显 DAREED 的 tailored design。

### 3. 实验回应需要定量化

涉及基线、规模、复杂度和可扩展性的回复，应补充实际数据而非只说“已增加实验”：

- 大、中、小拓扑的具体节点数量；
- 运行时间、复杂度结论及随规模增长的变化；
- DAREED 相对基线的定量差距；
- 均值、标准差、随机种子数量；
- 新机制的目的、效果和消融结果。

### 4. 检查是否真正回应审稿意见

两处批注直接要求确认“是否回应”：

- Reviewer #1 Concern #12：新增基线后，是否同时解释了定量差距以及原基线的局限性；
- Reviewer #3 Concern #5：改为对 unique nodes/links 求可靠性，是否充分回应同一节点上多个层发生相关故障的问题。

### 5. Related Work 尚有未完成内容

Reviewer #2 Concern #1 处仍保留 `【tabular 待解决】`。需要：

- 添加与近期工作的对比表；
- 重组 Related Work 的论述逻辑；
- 精简重复文字；
- 用表格突出 DNN topology、dynamic reliability、link reliability、energy consumption、preference-based multi-objective deployment 等差异。

## 三、逐条批注清单

### 1. 第 2 页｜致副主编的总体回复：DAREED 算法重设计

- **锚定内容：** `Second, the algorithm has been redesigned and renamed as DAREED...`
- **原批注：** 强调专门化的设计和创新。
- **整理建议：** 不只列举初始化、交叉、变异、修复、局部搜索等模块；逐项说明这些模块如何利用 DNN DAG 信息和动态资源状态，以及相对 generic NSGA-II 的新增机制与效果。

### 2. 第 3 页｜Reviewer #1, Concern #1：四个优化目标

- **锚定内容：** 将 `accuracy reliability` 改为 `inference accuracy`，并统一为 service reliability、inference accuracy、inference delay、total energy consumption 四个目标。
- **原批注：** 再斟酌一下论文题目，当前仅体现其中两个目标。
- **整理建议：** 检查论文题目是否需要同时体现 accuracy 和 delay，或在正文中解释题目为何只突出 reliable 与 energy-efficient，确保标题与四目标定位一致。

### 3. 第 5 页｜Reviewer #1, Concern #5：动态可靠性、随机种子与统计结果

- **锚定内容：** `We revised Section 4 to define dynamic node reliability and dynamic inference accuracy...`
- **原批注：** 总体来说，action 部分均略显简略；这里可以描述得更详细，或直接引用原文部分文字、图表，或提供更细节的章节号、页码、图、表、公式编号。
- **整理建议：** 将此视为所有 `Author action` 的通用要求。此处至少列出动态可靠性/准确率公式编号、实验小节、随机种子数量、重复次数，以及均值和标准差出现在哪些表中。

### 4. 第 5 页｜Reviewer #1, Concern #6：可靠性和准确率公式重构

- **锚定内容：** `We revised the mathematical formulation of service reliability and inference accuracy in Section 4 and updated the notation table accordingly.`
- **原批注：** 同前，下同。
- **整理建议：** 补充新旧公式的关键区别，并给出公式号、符号表编号和页码；明确 non-selected nodes 不再参与计算的数学实现。

### 5. 第 7 页｜Reviewer #1, Concern #12：基线公平性和改进幅度

- **锚定内容：** 审稿人指出 LB 和 LPD 是单策略启发式方法，要求更合理地解释百分比提升及这些基线的局限性。
- **原批注：** 确认这一点是否呼应了，定量的差距报告。
- **整理建议：** 当前回复说明了新增基线和评价拆分，但还应明确：正文是否报告 DAREED 相对各类基线的定量差距；是否区分简单启发式参考与真正多目标竞争算法；是否讨论 LB、LPD 的适用边界。

### 6. 第 8 页｜Reviewer #2, Concern #1：近期工作与对比表

- **锚定内容：** `【tabular 待解决】`
- **原批注：** 添加表格、优化 Related Work 的逻辑、精简文字。
- **整理建议：** 完成文献对比表，并以表格为主线重组该段；删除与表格重复的逐篇描述，突出本文相对近期工作的独特组合与缺口。

### 7. 第 9 页｜Reviewer #2, Concern #4：大规模网络实验

- **锚定内容：** `The revised experimental design includes small, medium, and large network scenarios.`
- **原批注：** 需要引用描述一下具体的规模。
- **整理建议：** 直接写明各拓扑规模。至少补入大规模场景的 `1 cloud node、60 edge nodes、300 end devices`，并给出中、小规模设置及对应表/图编号；删除句末多余的句点。

### 8. 第 9 页｜Reviewer #2, Concern #5：复杂度和可扩展性

- **锚定内容：** DAREED 增加复杂度分析、可扩展机制和运行时间报告。
- **原批注：** 需要进一步提供并说明一下当前的复杂度分析结论和运行时间实验结果。
- **整理建议：** 在回复中直接给出时间复杂度表达式、主导项、关键参数，以及大规模场景下的实际运行时间和增长趋势；说明这些结果是否支持算法在在线部署中的可用性。

### 9. 第 10 页｜Reviewer #2, Concern #7：能耗与成本

- **锚定内容：** 将 total energy consumption 纳入优化目标，并说明本轮不把货币成本作为主要目标。
- **原批注：** 可以补充说明：能耗也是成本模型的重要组成部分，和资源成本相关。
- **整理建议：** 在解释未显式建模货币成本时补充：能耗与基础设施运行成本、资源使用成本密切相关，因此能源目标在一定程度上反映成本效率，但不能完全替代平台定价模型。

### 10. 第 12 页｜Reviewer #3, Concern #3：Pareto 搜索与偏好选择的两阶段框架

- **锚定内容：** `We revised Section 5 to clarify the two-stage framework of evolutionary candidate search and preference-based deployment selection...`
- **原批注：** 解释目的和效果是什么。
- **整理建议：** 明确第一阶段的目的在于保留多样的非支配候选，第二阶段的目的在于针对不同请求偏好作出单一在线部署决策；补充其效果，例如避免预设权重限制搜索、提高偏好适配能力，并引用偏好实验结果。

### 11. 第 12 页｜Reviewer #3, Concern #5：同节点多层的相关故障

- **锚定内容：** 审稿人质疑多个层部署在同一节点时，故障并非独立；回复改为只对 unique used nodes 和 traversed links 计算一次可靠性。
- **原批注：** 这一点是否回应了？
- **整理建议：** 明确说明“同一节点上的多个层共享节点故障事件，因此该节点只计一次”如何消除原公式的重复独立性假设；同时检查是否还需要讨论不同节点/链路之间的独立性假设和局限性。

### 12. 第 13 页｜Reviewer #3, Concern #6：增强基线集合

- **锚定内容：** 新增 generic NSGA-II、simulated annealing、random feasible deployment、maximum-resource deployment、local-first deployment 和 RTBL。
- **原批注：** 用定语从句强调一下算法的创新性和 tailored design。
- **整理建议：** 在介绍 generic NSGA-II 等基线时增加限定性说明，指出它们缺少 DNN-DAG-aware、dynamic-reliability-aware 或 preference-based tailored mechanisms，从而自然突出 DAREED 的专门化设计；避免仅靠评价性语言声称创新。

### 13. 第 13 页｜Reviewer #3, Concern #7：规模与运行时间

- **锚定内容：** 大规模拓扑为 `1 cloud node, 60 edge nodes, and 300 end devices`，并增加运行时间报告。
- **原批注：** 结论？
- **整理建议：** 补充实验结论：运行时间随拓扑和请求负载如何增长、最大规模下是否仍可接受、与基线相比如何，以及由此能否支持 scalability/practicality 的论断。

## 四、推荐处理顺序

1. 先补完第 8 页的 Related Work 对比表和未完成占位符。
2. 批量增强全部 `Author action`：章节、页码、公式、表格和实验编号。
3. 将规模、复杂度、运行时间和基线差距写成定量回应。
4. 强化 DAREED 相对 generic NSGA-II 的专门化机制与证据。
5. 检查两处“是否回应”问题，必要时重写 Author response。
6. 最后统一题目、四目标表述、能耗与成本关系，以及英文措辞。

> 注：页码依据当前 DOCX 渲染后的 14 页版本；后续排版变化可能导致页码移动。
