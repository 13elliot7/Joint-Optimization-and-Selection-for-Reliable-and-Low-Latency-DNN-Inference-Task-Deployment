# Code Optimization 文档索引

## 当前有效文档

- `experimental_environment_organization.md`：两层实验环境的职责、可比较范围、公平性规范、结果目录和统一泊松基线迁移顺序；涉及实验环境选择时以此为总入口。
- `current_system_model_and_algorithm_design.md`：周期规划主路径、旧即时基线、信息边界、时间语义和模块对应关系。
- `experiments.md`：区分周期主路径与旧即时 CLI 的实验口径、指标、执行方式和正式实验矩阵；周期入口支持 `periodic_experiment_runner.py --suite all`。
- `model_submission_readiness_issues.md`：已完成代码基线、仍成立的模型边界以及投稿证据与复现任务；不再把缺正式实验列为代码阶段 P0。

## 已完成的统一实施方案

- `plans/pending/unified_periodic_planning_model_closure_implementation_plan.md`：P0-01～P0-03 模型闭环、泊松批量到达、异步周期规划、方案池、在线复验和联合预算控制的代码阶段已经完成；正式多 seed 论文实验另行执行。路径暂保留以避免设计来源文档中的追溯链接失效。
- `plans/pending/remove_ifs_execution_plan.md`：OSS—时延—能耗三目标迁移已经完成；文件路径暂保留以维持追溯链接，文档状态已标记为已实施。

## 待实施方案

- `plans/pending/customized_generalizable_plan_pool_enhancement_plan.md`：在保持 OSS—时延—能耗三目标不变的前提下，通过因果多快照鲁棒门控、资源裕量、节点/物理链路多样性、在线反馈和可选 deadline 分桶提升 Customized 方案池的跨状态复用能力。

## 设计来源（不独立实施）

- `plans/reference/p0_model_closure_code_modification_plan.md`：P0 模型闭环的原始详细方案。
- `plans/reference/poisson_arrival_periodic_planning_code_modification_plan.md`：泊松到达与周期规划的原始详细方案。
- `plans/reference/periodic_planning_model_consistency_remediation_plan.md`：周期架构下模型一致性修复的原始详细方案。
- `plans/reference/adaptive_joint_planning_budget_control_implementation_plan.md`：规划周期、分画像候选池规模和在线复验时间预算联合控制的专项实施细化。

上述文档仅用于追溯设计过程；若与统一实施方案或当前有效文档冲突，以当前代码、`semantics.py` 和当前有效文档为准。

## 归档

- `archive/implemented plans/`：已经实施或被后续实现替代的方案。
- `archive/audits/`：阶段性代码清理和弃用审计记录。
- `archive/references/`：早期方案 PDF 与调研参考材料。

归档文档只用于追溯设计演进，不应作为当前代码接口、字段名称或实验 Schema 的实现依据。
