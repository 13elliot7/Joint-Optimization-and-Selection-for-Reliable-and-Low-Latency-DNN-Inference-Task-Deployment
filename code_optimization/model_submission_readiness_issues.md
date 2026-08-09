# 模型投稿准备问题清单

> 更新时间：2026-08-09  
> 活动目标：OSS、时延满意度、能耗满意度  
> P0 模型闭环代码问题已按周期规划方案处理；当前处于实验重跑与证据补齐阶段。

## 已关闭的模型问题

- [x] 候选 delay、offered rate、链路负载、有效带宽和 OSS 来自同一接纳后固定点快照。
- [x] 节点当前离线作为接纳硬约束；服务窗口连续可用性进入 OSS。
- [x] 规划运行时间和在线复验时间与推理估计 delay 分开建模。
- [x] 后台规划复用原 Customized 遗传搜索生成候选池。
- [x] 请求偏好、最终选解、在线排序和 Goodput 统一为 stability/delay/energy 三权重公式。
- [x] 删除缺少真实测量支撑的合成输出质量目标，Pareto/HV 改为三维。
- [x] 旧四目标结果与新三目标 Schema 通过语义版本严格隔离。
- [x] 九类算法可进入统一泊松到达的周期事件循环进行系统比较。

## P1：投稿前必须补齐的证据

### P1-01 OSS 参数来源与校准

当前节点和链路先验、负载/热度衰减系数及 Beta 窗口均属于受控仿真参数。论文必须给出参数表、范围和来源，并通过 weak/medium/strong 与 availability suite 报告敏感性。若没有真实故障轨迹，只能表述为合成可用性场景下的相对结果。

### P1-02 OSS 的区分效度

报告 OSS、节点连续可用性、链路稳定性、delay 和 energy 的 Pearson/Spearman 相关、散点图和三目标冲突率。需要证明 OSS 不是负载或 delay 的简单重复代理；否则应简化目标或将其改为约束。

### P1-03 三目标搜索有效性

在固定单 DNN 快照实例库上比较 Customized、普通演化、Random 和 SA：

- 三维 Hypervolume 和 Pareto 点数；
- 可行率、约束拒绝率、固定点不收敛率；
- 相同评价预算下的质量—墙钟时间；
- DAG 初始化、块交叉、偏斜变异和精英搜索消融。

### P1-04 周期规划的系统收益

在相同 profile catalog、泊松到达、故障轨迹和 paired seed 下比较：

- 周期 Customized 方案池；
- 逐请求 Customized；
- Random、SA、Local First、Max Resource；
- availability-only RTBL adaptation。

同时报告 Goodput、完成/失败/deadline miss、平均 delay/OSS/energy、规划时间、在线 P95、直接命中/repair/fallback 和请求守恒。

### P1-05 联合 T/K/B 控制有效性

需要固定 T/K/B、单变量自适应和联合控制消融；报告 action 分布、预测误差、fallback、预算违约和不同负载区间的收益。控制器训练/拟合数据不得与最终评价 seed 重叠。

### P1-06 规划耗时真实性

主结果至少提供一组 measured planning runtime，而不只使用注入的固定毫秒数。需要展示规划任务跨时隙完成、旧仓库继续服务和新仓库原子发布。

### P1-07 统计与可复现性

- 至少 20 个 paired seed；
- 均值、标准差或 95% CI；
- 配对显著性检验与效应量；
- 明确随机种子、软硬件、Python/依赖版本和完整命令；
- 正式结果目录与 smoke 目录隔离。

## P2：表述与外部有效性

- OSS 是稳定性评分，不写成严格端到端成功概率。
- delay 为仿真模型的估计完成时延，单位毫秒，不包含后台规划墙钟时间。
- 能耗是一阶部署/传输模型，除非校准，否则不写成真实设备绝对功耗。
- RTBL 需标记为 availability-delay adaptation，不能声称与含输出质量收益的原方法完全等价。
- 结论限定在给定拓扑、到达、故障和资源参数范围内。

## 正式实验顺序

1. A 层 sensitivity；
2. A 层 pareto 与 ablation；
3. B 层 search_budget 与 generator_ablation；
4. B 层 load；
5. B 层 algorithm_baseline；
6. control、availability、planning_runtime、scale；
7. 统计检验和论文产图。

P0-4“缺少实验”不再作为代码改造阻塞项；上述实验均属于投稿证据阶段。
