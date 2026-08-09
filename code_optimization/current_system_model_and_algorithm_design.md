# 当前系统模型与算法设计

> 当前活动目标语义：`stability_delay_energy_v4`  
> 候选预测语义：`unified_fixed_point_v3`  
> 周期结果结构：`periodic_three_objective_v3`

## 1. 问题定义

系统把 DNN 推理表示为有向无环任务图，并将子任务部署到云、边缘和用户节点。算法在资源、层级、拓扑、节点在线和 deadline 约束下最大化：

```text
F(x) = [OSS(x), delay_satisfaction(x), energy_satisfaction(x)]
```

项目不掌握真实模型精度、量化误差或数据集测量值，因此不再构造合成的输出质量目标。若未来获得 measured accuracy，应以新的数据来源、版本和实验协议独立引入。

## 2. 信息边界

算法可读取显式 `ObservationSnapshot`：

- 当前时隙、拓扑和环境状态版本；
- 节点在线状态、可用 CPU、负载和热度；
- 节点可用性历史与 availability mode；
- 方向 offered rate、链路负载、物理链路热度和有效带宽。

算法不可直接读取仿真器未来故障、未来请求、随机数内部状态或执行后的真实结果。节点当前离线是硬约束；历史数据只能用于估计未来服务窗口内的连续可用性。

## 3. 泊松请求模型

系统启动前构造可复用的 DNN profile catalog。运行时，第 t 个时隙的请求数满足：

```text
N_t ~ Poisson(lambda)
```

到达请求只保存 profile、发起节点、deadline、priority 和三个偏好权重：

```text
preference_stability
preference_delay
preference_energy
```

三个权重均在 `[0,1]`，且总和必须大于零。同一时隙可到达多个请求，调度器逐个原子提交资源，因此后续请求会看到前序请求造成的最新状态。

## 4. 节点可用性与 OSS

节点维护时隙级 up/down 状态及有限窗口历史。随机节点使用 Beta 平滑经验估计，always-on 节点的连续可用性固定为 1。给定候选估计时延，其服务暴露长度为：

```text
horizon_slots = max(1, ceil(delay_ms / slot_length_ms))
```

候选的节点连续可用性按任务 FLOPs 权重在 log 域聚合；链路稳定性根据候选接纳后的共享物理链路负载和热度计算。OSS 再将节点域与链路域按配置权重聚合。OSS 是可解释的稳定性评分，不宣称为严格端到端成功概率。

## 5. offered rate 与时延固定点

`offered rate` 是运行中及待接纳 DNN 在单位时间内向某条方向链路施加的数据量。候选接纳后：

1. 由当前 delay 猜测得到请求服务率；
2. 汇总运行队列与候选的方向 offered rate；
3. 映射到共享物理链路负载和热度；
4. 更新方向有效带宽；
5. 重新计算 DAG 关键路径执行与传输时延；
6. 阻尼更新并迭代至相对残差小于阈值。

所有 delay 使用毫秒。规划算法运行时间不加进单请求网络/计算 delay；它在周期事件循环中单独建模为后台 planning runtime。在线复验时间也单独报告，并受在线预算 B 约束。

## 6. 能耗与满意度

总能耗包含节点计算能耗、DNN 中间张量传输、输入上传和结果回传。环境为每个 profile 生成稳定的参考上下界，并把原始能耗转换为 `[0,1]` 的能耗满意度；原始 `total_energy` 始终保留用于报告。

时延满意度为：

```text
max(0, min(1, (deadline - delay) / deadline))
```

## 7. Customized 三目标搜索

后台规划复用原 Customized 遗传搜索骨架：DAG 感知初始化、块交叉、偏斜变异、修复、精英局部搜索、约束支配和 archive。公开候选目标严格为三元组：

```text
(OSS, delay_satisfaction, energy_satisfaction)
```

只有固定点收敛且满足全部硬约束的候选才可进入方案池。Pareto 支配、archive 筛选和 Hypervolume 均使用三维口径；HV 版本为 `3d_oss_delay_energy_v1`。

## 8. 周期性规划

周期框架建立在 Customized 候选生成器之上，而不是替代原算法：

```text
系统快照 -> Customized 后台搜索 -> 多样化方案池 -> 原子发布
请求到达 -> key 匹配 -> 快速过滤 -> 预算内完整复验 -> 原子提交
```

方案池按 `(profile_id, origin_group)` 分组，因为 profile 决定 DAG/计算/传输需求，而 origin group 决定接入路径与层级可行域。方案只保存部署结构、资源需求和 OSS/时延/能耗参考值；真正接纳前必须基于最新在线快照复验。

触发条件包括固定周期、仓库过期、拓扑/在线状态变化、节点或链路负载漂移、画像到达分布漂移以及 fallback/rejection 反馈。

## 9. 在线评分与 Goodput

Customized 最终选解、方案参考排序、在线复验、逐请求 Customized 和完成请求的 Goodput 共用：

```text
U = (w_s*OSS + w_d*delay_sat + w_e*energy_sat) / (w_s+w_d+w_e)
```

deadline miss 或运行失败的请求效用为零。Goodput 是单位模拟时间内成功完成请求效用之和，不能替代各分项指标；结果同时报告平均 OSS、delay 和 energy。

## 10. 联合 T/K/B 控制

控制器周期性读取请求到达、直接命中、fallback、拒绝、规划开销和在线复验开销等统计量，从离散动作集合联合选择：

- T：下一次规划间隔；
- K：各 profile/origin key 的方案池总预算；
- B：每请求在线完整复验预算。

控制器预测每个动作的 Goodput—开销目标，并保留 decision/outcome 的因果关联。控制器只改变预算，不绕过统一候选预测或硬约束。

## 11. 基线与评价

- `periodic_customized`：周期方案池主算法；
- `per_request_customized`：每次到达运行相同 Customized 核心；
- Random、SA、Local First、Max Resource：统一环境下的部署基线；
- RTBL：删除无测量支撑的质量收益后，作为 availability-delay adaptation 报告。

核心输出包括请求守恒、接纳/完成/失败、deadline miss、Goodput、平均 delay/OSS/energy、规划与在线运行时间、方案池命中路径、搜索统计和联合控制误差。

## 12. 结论边界

当前模型可以评价“在给定合成拓扑、故障过程、请求到达和一阶能耗模型下”的相对调度表现。它不能直接证明真实设备上的准确率、绝对可靠性或部署收益。正式投稿仍需多 seed、置信区间、配对检验、负载拐点、真实规划耗时和参数敏感性结果。
