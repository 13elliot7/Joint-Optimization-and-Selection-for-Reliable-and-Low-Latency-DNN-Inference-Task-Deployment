## Related Work

### 新增 Reliability-aware edge computing 小节

*新增小节，强调现有研究通常把可靠性作为静态节点属性或约束，而本文考虑运行过程中负载和热度造成的动态退化。*

在可靠性感知边缘计算方面，确保任务部署与卸载决策的可靠性是维持服务质量（QoS）的关键。然而，现有的大多数研究通常将可靠性视为静态的节点属性或恒定的物理约束。**J. Zhou 等人 [2]** 研究了多用户边缘计算系统中的联合卸载与调度，其模型基于静态指数分布建立处理器的执行成功概率。**A. M. Rasouli 等人 [3]** 提出的 RASOUL 策略利用在线强化学习为物联网设备选择边缘服务器，但其服务器可用性是基于平均无故障时间（MTBF）构建的长期静态概率，未能考虑连续工作负载对硬件损伤的累积效应。在无人机协同网络中，**S. Yang 等人 [1]** 将链路瞬时失效建模为独立的泊松过程，通过 TD3 算法提升任务成功率，但依然视设备失效率为静态先验分布。**V. Deka 等人 [4]** 针对多服务器系统提出了一种可靠性感知任务映射方法（RATM），虽然利用自适应 DVFS 调节应对瞬态故障，但其对物理器件的硬失效风险仍设定为离线已知的固定状态。

与上述将可靠性视为静态指标的研究不同，边缘节点在处理 DNN 推理等计算密集型任务时，高强度负载引发的热累积会导致显著的动态退化（Physical Degradation）。集成电路物理学表明，负偏置温度不稳定性（NBTI）和电迁移（EM）等物理失效过程对运行温度展现出指数级的敏感性。**J. Zhou 等人 [5]** 率先针对多处理器系统提出了面向寿命、能量和完工时间共优化的调度框架，利用阻容（RC）等效热网络跟踪瞬态温升，并实时更新设备的累计失效损耗。针对物联网分布式系统，**K. Ergun 等人 [6]** 提出了动态可靠性管理（DRM）方案，通过感知网关由于历史活动累积的物理降级状态来动态调整卸载比率，主动避免高降级设备承担重载。在针对机器学习推理的专用研究中，**O. Shafi 等人 [7]** 开发了实时推理框架 IceEdge。由于紧凑型边缘服务器在部署深度学习模型时极易发生热积聚，该框架通过温度感知的闭环调控降低了 GPU 温升，在长效维护系统物理可靠性的同时提升了服务稳定性。

上述研究表明，忽略负载和热度造成的动态退化会导致部署方案在长期运行中失效。与现有研究相比，本文不仅针对 DNN 推理任务的层级依赖结构，还建立了由实时负载和热动态驱动的退化模型，实现了对推理延迟与动态可靠性的精准权衡。

[1] Hao H, Xu C, Zhang W, et al. Reliability-aware optimization of task offloading for uav-assisted edge computing[J]. IEEE Transactions on Computers, 2025.

[2] Zhou J, Hou X, Zeng Y, et al. Quality of experience and reliability-aware task offloading and scheduling for multi-user mobile-edge computing systems[J]. IEEE Transactions on Services Computing, 2025.

[3] Rasouli A M, Esnaashari M, Ansari M. RASOUL: A Reliability-Aware Task Allocation Strategy to Improve Success Rate and Energy Saving in Mobile-Edge Computing[J]. IEEE Internet of Things Journal, 2025.

[4] Zhou Z, Han J, Mo L, et al. Reliability-Aware Real-Time Task Mapping for Multi-Server Edge Computing Systems With DVFS[C]//2025 40th Youth Academic Annual Conference of Chinese Association of Automation (YAC). IEEE, 2025: 1994-2000.

[5] Zhou J, Cao K, Sun J, et al. A framework to solve the energy, makespan and lifetime problems in reliability-driven task scheduling[C]//2019 International Conference on Internet of Things (iThings) and IEEE Green Computing and Communications (GreenCom) and IEEE Cyber, Physical and Social Computing (CPSCom) and IEEE Smart Data (SmartData). IEEE, 2019: 608-614.

[6] Ergun K, Ayoub R, Mercati P, et al. Dynamic reliability management of multigateway IoT edge computing systems[J]. IEEE Internet of Things Journal, 2022, 10(5): 3864-3889.

[7] Shafi O, Pandit M K, Gujarati A, et al. IceEdge: Thermal-Aware Machine Learning Inference Serving for Emerging Edge Applications[J]. ACM Transactions on Sensor Networks, 2026, 22(4): 1-26.



### Multi-Objective optimization 小节修改

*说明 NSGA-II、MOPSO、强化学习等方法在多目标问题中的应用。然后指出：通用多目标算法需要结合具体任务结构，否则搜索空间大、不可行解多、收敛不稳定。*

现实世界中的许多问题都具有多目标并存的特征。在多样化约束条件下权衡这些不同的目标，需要考虑并优先处理多个相互竞争或互补的目标。

多目标优化通常涉及复杂的决策过程，需要考虑众多可能的权衡以确定最优解，从而更好地满足多目标的要求。多目标优化在许多领域受到越来越多的关注。目前已开发出诸多多目标优化算法，包括遗传算法、NSGA-II [28]、多目标粒子群优化算法 [29] 以及多目标强化学习算法等，并在多个领域得到应用。例如，XujXuj 等人 [30] 提出了一种多目标进化学习算法来发现帕累托前沿，以指导连续机器人的控制过程。F. Song 等人 [31] 使用改进的进化多目标强化学习算法来确定最优的无人机轨迹控制和任务卸载策略。Ma W 等人 [32] 设计了一种基于多目标优化原理的边缘计算优化算法。M. Mounesan 等人 [33] 提出了一个优势执行者-评论者强化学习框架，以平衡端到端延迟、推理精度和设备能耗。H. Huang 等人 [34] 通过基于异构智能体强化学习的算法，对部署位置、DNN 模型选择和应用配置进行联合优化，以同时优化延迟、精度和资源成本。H. Hao 等人 [35] 提出了一种基于潜在空间的新型深度强化学习算法，以最大化由能耗、任务延迟和优先级组成的多无人机协作边缘计算系统的长期平均系统增益。

然而，尽管这些通用多目标算法在理论上具有强大的全局搜索能力，但直接将其应用于 DNN 推理任务部署时面临严峻挑战。**通用算法通常将优化空间视为黑盒，若不结合具体的 DNN 任务层级结构（如计算图依赖）与异构资源约束，会导致搜索空间呈爆炸式增长。** 这种盲目搜索往往会产生大量违反逻辑依赖或显存限制的**不可行解（Infeasible solutions）**，进而导致算法**收敛不稳定**且计算效率低下。

此外，现有关于 DNN 推理部署的研究 [9], [11], [15] 大多通过加权法将多目标转化为单目标，这忽略了动态网络场景中权重分配的复杂性。虽然 **N. Mu 等人 [36]** 指出 NSGA-II 预计算的最优解集可以作为“静态蓝图”以应对偏好变化，但如何从帕累托前沿中高效选择满足特定延迟与可靠性要求的解仍是难题。为此，本文在利用 NSGA-II 进行多目标探索时，**深度融合了 DNN 推理任务的链式结构特性**，以缩小有效搜索空间并剔除无效解。随后，通过量身定制的选择启发式算法（DSSA），在不依赖固定权重的情况下，实现延迟与可靠性的最优部署。

