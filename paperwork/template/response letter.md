**Original Manuscript ID:** TSC-2025-01-0005

**Original Article Title: “**Multi-Objective DNN Inference Task Deployment for Reliability and Delay”

 

**To:** IEEE TSC Editor

**Re:** Response to reviewers

 

Dear Editor,

 

We thank you and the reviewers for the constructive feedback and the opportunity to revise and resubmit our manuscript titled “Multi-Objective DNN Inference Task Deployment for Reliability and Delay.”

We have carefully studied all the comments and have conducted extensive additional experiments and revisions to address the concerns raised. The manuscript has been significantly improved as a result. 

The comments are reproduced in green, followed by our point-by-point responses.

We are uploading (a) our point-by-point response to the comments (below) (response to reviewers) as Summary of Changes file, (b) an updated manuscript as the PDF main document.

 

Best regards,

Ying Wang, Huang Lianze, Wei Junjie et al.



Reviewer#1, Concern#1: Is Eq.3 incorrect? Is the transmission time equal to the transmission rate divided by the amount of data?

***\*Author response:\**** 

Thanks for the reviewer’s correction. We must apologize for the carelessness that had been shown when editing the equation. Transmission time is equal to the amount of data divided by the transmission rate, which is consistent with our implementation in the experimental section.

***\*Author action:\**** 

We made corresponding correction in Eq.3.

***\*Before:\****



***\*After:\****

 

![img](file:////Users/hlz/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-hlz/ksohtml//wps3.jpg) 

Reviewer#1, Concern#2: “terminal devices” may be "end device" in edge computing.

***\*Author response:\**** 

Thanks for the reviewer’s suggestion. We have reexamined the whole paper and adopted the “end device” expression, which is more frequently used in the field of edge computing.

***\*Author action:\**** 

We have replaced all the expressions of “terminal devices” with “end devices”.

![img](file:////Users/hlz/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-hlz/ksohtml//wps4.jpg) 

Reviewer#1, Concern#3: Does the term 'layer' in the article refer to the layers of a neural network? If so, the author did not consider the order relationship between layers when designing the model. For example, a task is distributed in different layers, but layer 2 must be executed followed by layer 1. How to consider the waiting time of the model when calculating the delay?

***\*Author response:\**** 

Thanks for the reviewer’s suggestion. To clarify, in our paper, we didn’t specifically consider the deep neural network inference task to be multi-layered. But instead, a DNN inference task consisted of certain subtasks has topological characteristics, in which the execution of each subtask must come after the execution of its predecessor if there is any. Regarding the waiting time of the model, we have formulated our deployment problem in the context of a single time slot where we first collect all DNN tasks and then generate a feasible deployment schedule. In addition, our edge nodes are modeled as multi-core servers, allowing parallel execution of multiple subtasks within the node's resource constraint. Therefore, the common queuing delay due to dynamic task arrival is not considered. The total task delay is primarily calculated based on the sum of execution times and transmission time, thereby excluding explicit waiting time components in the final delay calculation.

***\*Author action:\**** 

We have added explanation on the topological characteristics of our task model in “4.1.2 DNN Inference Task Model”.

***\*Before: None.\****

***\*After:\****

“A DNN inference task request is initiated by an end device. Let **Task** denote the set of DNN inference tasks. An individual DNN inference task **task****i** within this set has a specific inference delay requirement denoted by ![img](file:////Users/hlz/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-hlz/ksohtml//wps5.jpg). Each DNN inference task can be modeled as a directed acyclic graph **G****i** **= (V****i****, E****i****)**. This graph has topological characteristics, that is, the execution of each subtask depends on the implementation of the previous subtask, and the correct execution of the entire DNN inference task depends on the completion of all subtasks.”

![img](file:////Users/hlz/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-hlz/ksohtml//wps6.jpg) 

Reviewer#1, Concern#4: The experiment was not conducted specifically for a particular neural network (e.g. VGG, Resnet, etc), Is the experimental setup in line with the actual situation? The authors need to explain this. There is no previous method for experimental comparison, and the methods mentioned in related work need to be compared.

***\*Author response:\**** 

Thanks to the reviewer for raising this important question. We acknowledge that our experimental setup does not specifically target particular neural network architectures such as VGG or ResNet. However, we believe this design choice is justified and aligns with the actual research objectives. Our primary research objective is to validate the effectiveness of multi-objective optimization scheduling algorithms rather than optimizing for specific DNN architectures. The randomly generated DNN structures serve to test algorithm robustness across diverse network topologies, avoid overfitting to specific architectural patterns and also provide comprehensive performance evaluation across varied and more general scenarios. In edge computing and cloud computing research, using randomly generated DAGs for workflow scheduling is a well-established practice. Many seminal studies employ randomly generated DAGs to evaluate scheduling algorithms. This approach avoids bias toward specific application scenarios and it provides generalizable algorithm performance assessment.
To address the second concern regarding lack of comparison between our proposed algorithm and related work, we conducted supplemental experimental comparison, taking into account the algorithm RTBL(reliability-aware tasks scheduling scheme with bandit learning) from reference [9], which incorporates online learning and a SOTA(state-of-the-art approximation) algorithm to generate solutions.

***\*Author action:\**** 

We have added corresponding introduction of the RTBL algorithm and the following experimental results in “6 Performance Analysis”.

***\*6.2 Metrics and Comparison Method\****

***\*Before:\****

![img](file:////Users/hlz/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-hlz/ksohtml//wps7.jpg) 

***\*After:\****

![img](file:////Users/hlz/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-hlz/ksohtml//wps8.jpg) 

***\*6.3.3 Average Service Reliability\****

***\*Before:\**** 

![img](file:////Users/hlz/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-hlz/ksohtml//wps9.jpg) 

Fig. 6. Service reliability under different task quantities

Figure 6 illustrates the variation in average service reliability for successfully deployed DNN inference tasks as the number of tasks increases. Among the four strategies, the average service reliability of RDODA+DSSA surpasses that of the RD, LB, and LPD algorithms by 19.1%, 3.4%, and 19.6%, respectively. This indicates that our proposed algorithm achieves superior average service reliability compared to the comparative algorithms. Particularly noteworthy is the minimal difference in results between our proposed algorithm and the LB method. This can be attributed to the LB algorithm’s emphasis on deploying DNN inference tasks to central cloud computing nodes, resulting in higher reliability but potentially introducing significant transmission delays.

***\*After:\****

![img](file:////Users/hlz/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-hlz/ksohtml//wps10.jpg) 

Fig. 6. Service reliability under different task quantities

Figure 6 illustrates the variation in average service reliability for successfully deployed DNN inference tasks as the number of tasks increases. Among the five strategies, the average service reliability of RDODA+DSSA surpasses that of the RTBL, RD, LB, and LPD algorithms by 11.6%, 19.1%, 3.4%, and 19.6%, respectively. This indicates that our proposed algorithm achieves superior average service reliability compared to the comparative algorithms. The service reliability of the RTBL algorithm increases as the number of DNNs goes up. When the number of DNNs was relatively low and the system was underloaded, the RTBL algorithm chose edge nodes over cloud nodes when weighing between reliability and delay. With the increase of the number of DNNs, the RTBL would deploy more tasks on cloud nodes, leading to growth in reliability.[删减] Particularly noteworthy is the minimal difference in results between our proposed algorithm and the LB method. This can be attributed to the LB algorithm’s emphasis on deploying DNN inference tasks to central cloud computing nodes, resulting in higher reliability but potentially introducing significant transmission delays.

***\*6.3.4 Average Accuracy Reliability\**** 

***\*Before:\****

![img](file:////Users/hlz/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-hlz/ksohtml//wps11.jpg) 

Fig. 7. Accuracy and reliability under different task quantities

Figure 7 illustrates the variation of the average accuracy reliability for successfully deployed DNN inference tasks as the number of tasks increases. Among these four schemes, our proposed algorithm showcases average accuracy reliability surpassing that of the RD, LB, and LPD algorithms by 25.1%, 5.8%, and 27.4%, respectively. Similarly to the finding regarding service reliability, the difference between the results obtained by our algorithm and the LB method is very small.

***\*After:\****

![img](file:////Users/hlz/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-hlz/ksohtml//wps12.jpg) 

Fig. 7. Accuracy and reliability under different task quantities

Figure 7 illustrates the variation of the average accuracy reliability for successfully deployed DNN inference tasks as the number of tasks increases. Among these five schemes, our proposed algorithm showcases average accuracy reliability surpassing that of the RTBL, RD, LB, and LPD algorithms by 18.1%, 25.1%, 5.8%, and 27.4%, respectively. Similarly to the finding regarding service reliability, the difference between the results obtained by our algorithm and the LB method is very small.

***\*6.3.5 Average Inference Delay\**** 

***\*Before:\****

![img](file:////Users/hlz/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-hlz/ksohtml//wps13.jpg) 

Fig. 8. Comparison of inference delay

Figure 8 illustrates the variation of the average inference delay for successfully deployed DNN inference tasks as the number of DNN inference tasks increases. Among these five schemes, our algorithm demonstrates an average inference delay that is 6.5% higher than the LPD algorithm, 5.8% lower than the RD algorithm, and 13.7% lower than the LB algorithm. The local first algorithm exhibits the lowest delay due to its prioritization of deploying DNN inference tasks locally or on the nearest computing node, effectively mitigating the impact of transmission delays.

***\*After:\****

![img](file:////Users/hlz/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-hlz/ksohtml//wps14.jpg) 

Fig. 8. Comparison of inference delay

Figure 8 illustrates the variation of the average inference delay for successfully deployed DNN inference tasks as the number of DNN inference tasks increases. Among these five schemes, our algorithm demonstrates an average inference delay that is 6.5% higher than the LPD algorithm, 0.8% higher than the RTBL algorithm, 5.8% lower than the RD algorithm, and 13.7% lower than the LB algorithm. Within the range of error, the RTBL algorithm can achieve similar optimization on the delay objective as our proposed algorithm. The local first algorithm exhibits the lowest delay due to its prioritization of deploying DNN inference tasks locally or on the nearest computing node, effectively mitigating the impact of transmission delays.

***\*6.3.6 The number of failed deployments\****

***\*Before:\****

![img](file:////Users/hlz/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-hlz/ksohtml//wps15.jpg) 

***\*After:\****

![img](file:////Users/hlz/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-hlz/ksohtml//wps16.jpg) 

***\*6.3.7 Average Running Time\****

***\*Before:\****

![img](file:////Users/hlz/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-hlz/ksohtml//wps17.jpg) 

***\*After:\****

![img](file:////Users/hlz/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-hlz/ksohtml//wps18.jpg) 

![img](file:////Users/hlz/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-hlz/ksohtml//wps19.jpg)Reviewer#2, Concern#1: The paper needs a more detailed clarification about what makes RDODA stand out compared to current NSGA-II versions.

***\*Author response:\**** 

Thanks for the reviewer’s suggestion. In our paper, we adopted the multi-objective optimization algorithm ‘NSGA-II’ with improvements in fitness processing and offspring selection to diversify population. 

***\*Author action:\**** 

We have added clarification on the highlights of our algorithm in “5.1 Reliability and delay-oriented DNN inference task deployment algorithm based on NSGA-II”.

***\*Before:\**** None.

***\*After:\****

The NSGA-II algorithm achieves its objectives by maintaining a set of non-dominated solutions. The algorithm generates new solutions for each generation through a combination of genetic operations. It sorts the population through non-dominated properties, selecting the best solution set for the next iteration. In this paper, we design the specific population generation and genetic operations, and perform special processing on the fitness function and offspring selection to optimize the diversity of population selection. The specific steps for finding the Pareto solution set for DNN inference task deployment strategies are as follows:

![img](file:////Users/hlz/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-hlz/ksohtml//wps20.jpg) 

Reviewer#2, Concern#2: The usefulness of DSSA depends largely on additional testing of weight parameters.

***\*Author response:\**** 

To address the reviewer’s concern regarding weight parameters, we would clarify that we do not require extensive prior testing. Instead, our approach utilizes the upper and lower bounds of the optimized objectives to normalize each objective's contribution. This method allows the optimization to focus on setting the preference ratio between the objectives, rather than determining absolute weight values. Crucially, our DSSA is robust against minor variations in these preference weights, as the normalization stabilizes the objective space. The required bounds are easily determined either empirically or through brief preliminary tests during system setup.

***\*Author action:\**** 

We have added a brief analysis on the setting of weight parameters in “5.2 Deployment scheme selection algorithm based on expectation” to demonstrate robustness.

***\*Before:\**** None.

***\*After:\****

Algorithm 1 yields a Pareto solution set, offering a range of solutions without specifying a particular deployment plan. Therefore, to derive a deployment plan tailored for a given DNN inference task, we need to select the most suitable deployment from the Pareto solution set. To accomplish this, we introduce three expectation conditions: the expected service reliability ![img](file:////Users/hlz/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-hlz/ksohtml//wps21.jpg), the expected accuracy reliability ![img](file:////Users/hlz/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-hlz/ksohtml//wps22.jpg), and the expected delay of DNN inference **task****i**. In order to select an optimal solution suitable for a specific DNN inference task among various solutions, we quantify the weights of the task across three objective functions. We utilize the normalization formula of formula (16) to calculate the weights of service reliability and accuracy reliability, denoted as Rw and Aw, respectively. Here, **I****w** represents **R****w** or **A****w**. When **I****w** represents **R****w**, ![img](file:////Users/hlz/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-hlz/ksohtml//wps23.jpg), **I****max** represents the maximum expected service reliability, and **I****min** represents the minimum service reliability. When **I****w** represents **A****w**, ![img](file:////Users/hlz/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-hlz/ksohtml//wps24.jpg) , **I****max** represents the maximum accuracy reliability, and **I****min** represents the minimum accuracy reliability. formula (17) is used to calculate the weight of delay, where expected delay T as a reference value is set as the delay constraint of the task, **T****max** represents the maximum delay requirement among all the tasks, while **T****min** represents the minimum delay requirement among all the tasks. In practical network scenarios, **T**, ![img](file:////Users/hlz/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-hlz/ksohtml//wps25.jpg), and![img](file:////Users/hlz/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-hlz/ksohtml//wps26.jpg) can be obtained through the expected values of DNN inference tasks corresponding to their features, rather than through subjective settings, which helps to improve the flexibility of the method in different scenarios.

![img](file:////Users/hlz/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-hlz/ksohtml//wps27.jpg) 

Reviewer#2, Concern#3: The experimental design needs additional scalability tests using numerous tasks and nodes in order to validate feasibility.

***\*Author response:\**** 

Thanks for the reviewer's valuable suggestion regarding the scalability test. We acknowledge that testing with a massive number of nodes and tasks is important for fully validating feasibility. To justify our current experimental scope, we clarify the rationale behind using a 10-node topology. In this paper, we focus more the regional edge computing deployments. In a regional edge network, the operational topology often consists of a limited number of powerful edge servers connected to a local gateway or a single cloud point. The 10-node scale is highly representative of this typical local cluster size in resource-constrained environments. Additionally, our primary contribution is the RDODA-DSSA algorithm for optimizing the multi-objective task deployment, which centers on complex task dependencies and resource constraints at the individual node level. The current scale is sufficient to fully stress-test the core mechanisms of our algorithm, including dependency graph management, resource allocation complexity, and trade-off decision-making between delay and reliability.

***\*Author action:\**** 

We have added clarification on the scalability of the proposed algorithm in “6.1 Simulation Environment Setup”.

***\*Before:\**** None.

***\*After:\****

All simulations are conducted using a network comprising 10 edge nodes[9] and 1 central cloud. The number of end devices served by each edge computing node is randomly selected within the range of [2,5]. To evaluate the scalability and robustness of the proposed method, we specifically designed the comparative experiments to reflect its effectiveness under various resource loads and task complexity levels. Rather than solely relying on an extensive increase in the number of nodes, we consider the deployment performance of DNN inference tasks of different scales in the comparative sections. This approach focuses on challenging the core scheduling and resource allocation mechanisms of our method under high load, which is essential for validating its feasibility in real-world, dynamic environments.[删减]

![img](file:////Users/hlz/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-hlz/ksohtml//wps28.jpg) 

Reviewer#2, Concern#4: System productivity would improve when authors conduct extensive language edits and simplify the sentences.

***\*Author response and action:\****

Thanks for the reviewer’s suggestion. We have thoroughly revised our article to improve readability.

![img](file:////Users/hlz/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-hlz/ksohtml//wps29.jpg) 

Reviewer#2, Concern#5: An evaluation strengthening mechanism could be achieved by incorporating learning-based deployment baselines and their corresponding discussions.

***\*Author response and action:\**** 

Thanks for the reviewer’s suggestion. We would like to direct you to our detailed response under Reviewer#1, Concern#4, where we added a learning-based deployment baseline and conducted corresponding experiments.

![img](file:////Users/hlz/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-hlz/ksohtml//wps30.jpg)
Reviewer#3, Concern#1: The paper's literature review is inadequate, lacking recent citations on multi - objective optimization algorithms for DNN inference task deployment. The references are too old to back up the research motivation and method innovation. The authors should list recent advances in the field from the past two years, especially those on multi - objective optimization strategies and DNN inference task deployment. This will help readers understand the research background, highlight this paper's contribution, and strengthen its academic rigor.

***\*Author response:\**** 

Thanks for the reviewer’s suggestion. We have reinvestigated more recent work on multi-objective optimization strategies and DNN inference task deployment.

***\*Author action:\**** 

We have added the following references to back up our research and corresponding analysis in “2 Related Work”.

**The supplemented literature is as follows:**

[22] M. Deutel, G. Kontes, at al., “Multi-Objective Bayesian Optimization with Reinforcement Learning for Edge Deployment of DNNs on Microcontrollers,” in Proceedings of the Genetic and Evolutionary Computation Conference Companion (GECCO ’25 Companion). Association for Computing Machinery, New York, NY, USA, 19–20.

[27] M. Mounesan, X. Zhang and S. Debroy, “Infer-EDGE: Dynamic DNN Inference Optimization in Just-in-Time Edge-AI Implementations,” in NOMS 2025-2025 IEEE Network Operations and Management Symposium, Honolulu, HI, USA, 2025, pp. 1-9, DOI:10.1109/NOMS57970.2025.11073623.

[28] H. Huang, J. Liang and G. Min, “Joint DNN Model Deployment, Selection, and Configuration for Heterogeneous Inference Services Toward Edge Intelligence,” in IEEE Transactions on Mobile Computing, vol. 24, no. 11, pp. 12726-12741, Nov. 2025, doi: 10.1109/TMC.2025.3586793.

[29] N. Mu, Y. Luan and Q. -S. Jia, “Preference-Based Multi-Objective Reinforcement Learning,” in IEEE Transactions on Automation Science and Engineering, vol. 22, pp. 18737-18749, 2025, DOI:10.1109/TASE.2025.3589271.

***\*2.1 DNN inference task deployment\****

***\*Before:\**** 

To sum up, most of the current studies center on optimizing the delay or resource efficiency of DNN inference tasks, with only a limited focus on reliability. As advancements in DNN and related fields continue, there is a growing recognition of the significance of reliability. Within this domain of research, reliability modeling is often treated as a constraint or a singular optimization goal in the problem formulation, which may overlook the crucial optimization direction of minimizing DNN inference task delays. Consequently, this paper studies the DNN deployment, aiming to address multiple optimization objectives encompassing both reliability and delay.

***\*After:\****

To sum up, most of the current studies center on optimizing the delay or resource efficiency of DNN inference tasks, with only a limited focus on reliability. As advancements in DNN and related fields continue, there is a growing recognition of the significance of reliability. Within this domain of research, reliability modeling is often treated as a constraint or a singular optimization goal in the problem formulation, which may overlook the crucial optimization direction of minimizing DNN inference task delays. The challenge of finding an optimal deployment for DNN inference is pervasive across the compute continuum. M. Deutel et al. [22] employ highly sophisticated methods like Multi-Objective Bayesian Optimization combined with Reinforcement Learning to balance conflicting goals such as accuracy, memory, and computational complexity on severely resource-constrained microcontroller Units. The necessity of such advanced MOO techniques, even at the lowest layer of the edge hierarchy, underscores the fundamental inadequacy of traditional single-objective optimization for the broader DNN deployment problem. Motivated by this systemic conflict, this paper studies the DNN deployment, aiming to address multiple optimization objectives encompassing both reliability and delay in a Cloud-Edge-Terminal deployment scenario.

***\*2.2 Multi-Objective optimization\****

***\*Before:\**** 

Multi-objective optimization usually involves a complex decision-making process that involves considering numerous possible trade-offs to identify the optimal solution, thus better fulfilling the requirements of multiple objectives. Multi-objective optimization has attracted increasing attention in many fields. Numerous multi-objective optimization algorithms have been developed, including genetic algorithm, NSGA-II [22], multi-objective particle swarm optimization algorithm [23], and multi-objective reinforcement learning algorithm, and have found applications in diverse fields. For instance, Xu j et al. [24] proposed a multi-objective evolutionary learning algorithm to discover Pareto front to guide the control process of continuous robots. Ma W et al. [25] devised an edge computing optimization algorithm rooted in multi-objective optimization principles. 

The exploration and utilization of the aforementioned multi-objective optimization algorithms provide valuable insights for this paper. As previously discussed, existing research on DNN inference task deployment mostly considers only one major optimization objective or combines multiple objectives through weighting to derive an optimized objective function, subsequently used for solution computation [9], [11], [15]. Nonetheless, these studies overlook the complexity of assigning weights to different optimization objectives in practical network scenarios and the challenges associated with seamlessly switching optimization algorithms across varying contexts to achieve flexible optimal objectives. Furthermore, solutions derived from multi-objective optimization are non- dominated, complicating the selection of an ideal deterministic solution that comprehensively guides specific resource allocation or task deployment.

***\*After:\****

Multi-objective optimization usually involves a complex decision-making process that involves considering numerous possible trade-offs to identify the optimal solution, thus better fulfilling the requirements of multiple objectives. Multi-objective optimization has attracted increasing attention in many fields. Numerous multi-objective optimization algorithms have been developed, including genetic algorithm, NSGA-II [23], multi-objective particle swarm optimization algorithm [24], and multi-objective reinforcement learning algorithm, and have found applications in diverse fields. For instance, Xu j et al. [25] proposed a multi-objective evolutionary learning algorithm to discover Pareto front to guide the control process of continuous robots. Ma W et al. [26] devised an edge computing optimization algorithm rooted in multi-objective optimization principles. M. Mounesan et al. [27] propose an Advantage Actor-Critic Reinforcement Learning framework to dynamically select DNN model versions and partition cut points on ”just-in-time” edge devices to balance end-to-end latency, inference accuracy, and device energy consumption. H. Huang et al. [28] adopt joint optimization of deployment location, DNN model selection, and application configuration to simultaneously optimize delay, accuracy, and resource cost through a Heterogeneous Agent Reinforcement Learning (HARL) based algorithm.

The exploration and utilization of the aforementioned multi-objective optimization algorithms provide valuable insights for this paper. As previously discussed, existing research on DNN inference task deployment mostly considers only one major optimization objective or combines multiple objectives through weighting to derive an optimized objective function, subsequently used for solution computation [9], [11], [15]. Nonetheless, these studies overlook the complexity of assigning weights to different optimization objectives in practical network scenarios and the challenges associated with seamlessly switching optimization algorithms across varying contexts to achieve flexible optimal objectives. Furthermore, solutions derived from multi-objective optimization are non- dominated, complicating the selection of an ideal deterministic solution that comprehensively guides specific resource allocation or task deployment. N. Mu et al. [29] show that relying on fixed, hand-designed reward weights is outdated. And by using NSGA-II to pre-compute the complete optimal set of deployment options, it gives system designers the necessary static blueprint, enabling future, advanced preference-driven systems to select the perfect deployment instantly, without rerunning complex optimization every time a user’s preference changes.

![img](file:////Users/hlz/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-hlz/ksohtml//wps31.jpg) 

Reviewer#3, Concern#2: The descriptions of the formulas in the paper are unclear. The $v^i_j$‘s deployment on computing nodes requires the utilization of CPU resources, denoted as ${C'}_{cpu}^{i,j}$, however, when calculating the computation delay of $v^i_j$,  the parameter ${C'}_{cpu}^{i,j}$ is not utilized. The authors should provide an explanation for this discrepancy.

***\*Author response:\**** 

We apologize for the typographical error shown during the formula editing process. The actual formulation used in our implementation is correct. The exact computing power assigned to a deployed task equals the computing power of unit computing resources ![img](file:////Users/hlz/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-hlz/ksohtml//wps32.jpg) times the CPU resources required for the task ![img](file:////Users/hlz/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-hlz/ksohtml//wps33.jpg). This definition is consistent with our model's underlying logic and the reported results.

***\*Author action:\**** 

We have reedited formula (1) to align with our actual implementation.

***\*Before:\****

![img](file:////Users/hlz/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-hlz/ksohtml//wps34.jpg) 

***\*After:\****

![img](file:////Users/hlz/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-hlz/ksohtml//wps35.jpg) 

![img](file:////Users/hlz/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-hlz/ksohtml//wps36.jpg) 

Reviewer#3, Concern#3: The paper contains serious theoretical errors in calculating communication delay between inference layers. The correct formula should be communication data volume ${F’}_{e_i^j}^{i}$ divided by transmission rate ${B_l}$, but formula (3) incorrectly uses the reverse. This undermines the theoretical model and may impact experimental results. The authors need to correct this and re-run experiments if necessary.

***\*Author response and action:\**** 

Thanks for the reviewer’s suggestion. We would like to direct you to our detailed response under Reviewer#1, Concern#1, where we thoroughly addressed a similar concern.

![img](file:////Users/hlz/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-hlz/ksohtml//wps37.jpg) 

Reviewer#3, Concern#4: The paper reasonably models the DNN inference task as a DAG. However, it is flawed in calculating the inference delay after determining the deployment strategy. The actual DNN inference task delay cannot be obtained by simply summing the execution delay of each layer. Considering that the inference layers can be computed in parallel with each other, the critical path based on DAG may be a better choice to measure the inference delay of tasks.

***\*Author response:\**** 

We apologize for the vagueness in our paper where we didn’t explicitly state the calculation of the delay of a DNN inference task. During our implementation, we refer to the path with the largest delay as the critical path. The actual delay of a DNN task equals the delay of its critical path instead of a simple summation of the execution delay of each layer and the transmission delay between layers. 

***\*Author action:\**** 

We have added clarification on the actual calculation of DNN task delay in “4.2 Delay model”.

***\*Before:\**** None.

***\*After:\****

The delay of DNN inference tasks generally comprises processing delay, queuing delay, propagation delay, and transmission delay. Among these, propagation delay and queuing delay are peripheral to our focus, typically small and thus negligible. Therefore, this paper assumes that the total delay of DNN inference tasks equals the sum of processing delay and transmission delay. Each DNN inference task ![img](file:////Users/hlz/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-hlz/ksohtml//wps38.jpg) has multiple paths, and we refer to the path with the longest delay as the critical path. The set of inference layers on the critical path is denoted as ![img](file:////Users/hlz/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-hlz/ksohtml//wps39.jpg), and the set of dependency layers is denoted as ![img](file:////Users/hlz/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-hlz/ksohtml//wps40.jpg). The delay of the critical path is the total inference delay of the DNN inference task.

![img](file:////Users/hlz/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-hlz/ksohtml//wps41.jpg)
Reviewer#3, Concern#5: The definition of $T_min$ is ambiguous and problematic. Although defined as a “minimum delay constraint,” execution delay below $T_min$ are more desirable in DNN inference tasks. The paper lacks a clear explanation of its necessity and practical significance, as well as its specific role in the overall model and experiments.

***\*Author response:\**** 

We apologize for the ambiguity in our previous expression concerning the delay constraints. To clarify our modeling approach: **T****max** and **T****min** are used primarily for the normalization of the delay objective in our multi-objective function. It represents the theoretical maximum and minimum delay achievable in the ideal scenario which can be obtained either empirically or by prior experimental tests, providing a necessary reference point to scale the delay objective to the [0,1] range.

***\*Author action:\**** 

We have changed the definitions for **T****max** and **T****min** into more accurate ones in “5.2 Deployment scheme selection algorithm based on expectation”.

***\*Before:\****

formula (17) is used to calculate the weight of delay, where expected delay T as a reference value is set as the delay constraint of the task, **T****max** represents the maximum delay constraint, while **T****min** represents the minimum delay constraint.

***\*After:\****

formula (17) is used to calculate the weight of delay, where expected delay T as a reference value is set as the delay constraint of the task, **T****max** represents the maximum delay requirement among all the tasks, while **T****min** represents the minimum delay requirement among all the tasks. 

![img](file:////Users/hlz/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-hlz/ksohtml//wps42.jpg)
Reviewer#3, Concern#6: The research in this paper focuses on multi-objective optimization of DNN inference task deployment, but the model fails to adequately highlight the uniqueness of DNN inference tasks. Considering the differences between DNN inference tasks and common tasks, it is recommended that the authors delve deeper into the unique characteristics of DNN inference tasks and incorporate them into the modelling and optimization process. This will enhance the relevance and expertise of the research.

***\*Author response and action:\****

Thanks for the reviewer’s suggestion. We would like to direct you to our detailed response under Reviewer#1, Concern#4, where we addressed a similar concern.

![img](file:////Users/hlz/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-hlz/ksohtml//wps43.jpg)
Reviewer#3, Concern#7: The thesis suffers from certain typographical problems, in particular the lack of clarity of Figures I and II and the need to adjust the font size. The authors are advised to revisit the typography of the paper to improve the clarity and readability of the illustrations.

***\*Author response\**** ***\*and action:\****

Thanks for the reviewer’s suggestion. To address the concerns regarding typographical problems and clarity of figures, we have regenerated Figures I and II using higher resolution settings and adjusted the graphical elements to significantly improve their clarity and sharpness. Additionally, we have conducted a systematic review of the entire manuscript's typography, adjusting font sizes within the figures, tables, and the main text to improve readability. 

![img](file:////Users/hlz/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-hlz/ksohtml//wps44.jpg)
Reviewer#3, Concern#8: The overall writing quality and logical coherence of the thesis could be improved. In terms of formula presentation, there is the problem of not adding appropriate punctuation after the formula, which affects the reading fluency and professionalism of the paper. It is recommended that the authors carefully proofread the whole paper, and optimize the overall writing logic in order to improve the clarity of expression and academic rigor of the paper.

***\*Author response\**** ***\*and action:\****

We sincerely appreciate the reviewer's valuable feedback regarding the overall writing quality, logical coherence, and formula presentation of the manuscript. The entire manuscript has undergone a thorough round of professional proofreading to eliminate grammatical errors and improve the overall readability. We have systematically proofread every displayed equation in the manuscript and added the appropriate punctuation after each formula to align with standard academic publishing conventions and improve reading fluency. We also confirmed the consistency of all mathematical notation.

 