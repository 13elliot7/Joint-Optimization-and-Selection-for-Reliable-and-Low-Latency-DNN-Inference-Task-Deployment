该文件为算法和模型优化之前的审稿意见，现在需要根据template/response letter.md为模版，针对审稿意见写一份response letter进行回复

# 审稿意见

Associate Editor
Comments to the Author:
The manuscript addresses the deployment of DNN inference tasks in a cloud-edge-end computing continuum by jointly optimizing reliability and inference delay, using an NSGA-II-based algorithm (RDODA) and an expectation-based selection heuristic (DSSA). While the topic is relevant and the structure is logical, all three reviewers agree that the manuscript requires major revision before it can be considered for publication. The key concerns are: insufficient algorithmic novelty, as RDODA is largely a direct application of standard **NSGA-II without DNN-topology-aware customization**; **unrealistic modeling assumptions,** including constant node reliability, independent failure correlations, and neglect of communication link reliability; unclear and inconsistent mathematical formulations in the objective functions and constraints; **weak experimental evaluation** with an overly small simulation setup, lack of statistical reporting, and comparison only against simple heuristics rather than state-of-the-art methods; and writing quality issues including grammatical errors, inconsistent terminology, and promotional language. The authors are also encouraged to consider additional optimization objectives such as **energy consumption and cost**, and to test the framework **under larger and more dynamic network conditions**.



## Reviewer: 1

Recommendation: **Author Should Prepare A Major Revision For A Second Review**

Comments:
This manuscript investigates the deployment of DNN inference tasks in a cloud-edge-end computing continuum by jointly optimizing reliability and inference delay.    The authors formulate a multi-objective optimization problem and propose an NSGA-II-based algorithm (RDODA) to derive a Pareto solution set, followed by an expectation-based selection heuristic (DSSA) to choose a concrete deployment scheme.  The two-stage design is intended to decouple global trade-off exploration from task-specific decision making, thereby accommodating heterogeneous delay and reliability requirements without relying on fixed scalar weights.
Overall, the motivation is relevant, and the manuscript is structured logically, but **the current manuscript contains substantial weaknesses in modeling, algorithm specification, and experimental rigor**; these issues limit reproducibility and weaken the strength of the technical claims, and the manuscript requires major revision before it can be considered for publication.

1. In the Abstract, the manuscript claims to use NSGA-II “to optimize both reliability and delay”, but it then explicitly lists service reliability, accuracy reliability, and delay as optimization objectives, which is effectively a three-objective formulation. Without clarifying that “reliability” is decomposed into service and accuracy aspects, this wording may appear internally inconsistent and weaken the clarity of the main contribution. (Abstract, pp. 1) 

2. In Related Work, the manuscript uses strongly judgmental phrasing such as “relying on fixed, hand-designed reward weights is outdated” and suggests that preference-driven systems can “select the perfect deployment strategy instantly” This tone reads as subjective and promotional, and is not clearly supported by evidence in the surrounding discussion. A more neutral, evidence-driven phrasing. (Section II-B, p. 3)

3. In Equation (6), task service reliability is defined as a product over all deployed layers and nodes. This implicitly assumes independent failures and effectively multiplies reliability for each layer execution. For models with many layers, the value can become extremely small even if each node is highly reliable. The manuscript would benefit from a brief justification of this multiplicative assumption or a discussion of alternative formulations. (Section IV-C, pp. 5)

4. The manuscript contains several copy-editing and grammar issues that affect readability. For example, the caption “Fig. 2. THE Overall Framework of the Algorithm” uses unusual capitalization; Section IV-C includes the phrase “the potential inability or inaccurate of DNN inference tasks,” which is grammatically incorrect; and the performance discussion includes “the algorithm is significantly outperforms …,” which is also ungrammatical. A careful proofreading pass would improve clarity and professionalism. (Section IV-C, p. 5; Section V, p. 6, Fig. 2; Section VI-C.2, p. 11, Table IV discussion)

5. The reliability model assumes that node reliability parameters remain constant during task execution. However, in the experiments, reliability values are randomly drawn from intervals. It is unclear whether these values are fixed per node across all tasks or re-sampled across runs. This ambiguity affects reproducibility and interpretation of averaged reliability results. (Section IV-C and VI-A, pp. 5 and 9)

6. In Equations (11) and (12), service reliability and accuracy reliability are formulated as products over all nodes and layers using the binary variable . Since each layer is deployed on exactly one node (Equation (8)), the current product form implicitly includes reliability terms of non-selected nodes, which is mathematically unclear. The authors should clarify the intended semantics or reformulate the objective. (Section IV-D, pp. 5-6))

7. The RDODA algorithm retains delay-violating solutions by assigning negative fitness values in the delay direction, instead of removing them. While this can preserve diversity, the manuscript does not explain how negative values affect non-dominated sorting and whether this may distort Pareto relations. (Section V-A, pp. 6-7)

8. Constraints such as Equation (9) sum over all tasks and layers, suggesting a joint deployment across multiple tasks. Meanwhile, the objective functions (11)-(13) are written with an -indexed single-task form, which can confuse whether the optimization is per-task or multi-task. The manuscript should clarify whether RDODA solves a batch optimization for multiple tasks simultaneously, or runs per task and only shares resource constraints globally. (Section IV-D, pp. 6-7)

9. The simulation assumes the central cloud has unlimited resources and sets its service reliability to 0.99, while edge nodes have lower reliabilities. The manuscript also observes that LB is close to the proposed method in reliability because LB “prioritizes deploying … on central cloud computing nodes.” Under this setup, reliability gains can be partly driven by the cloud’s idealized advantage. Adding stress tests with constrained cloud compute, reduced cloud reliability, or tighter cloud-edge bandwidth would better demonstrate the reliability-delay trade-off and the benefit of the proposed framework beyond cloud-favoring settings. (Section VI-A, pp. 9; Section VI-C.3-4, p. 11)

10. In the discussion of Table IV, the manuscript claims that “The RDODA algorithm presented in this paper exhibits the highest hypervolume value and achieves the optimal Pareto front.” However, since the fitness values are transformed negated prior to hypervolume computation, the manuscript does not explicitly clarify whether the reported hypervolume values are directly comparable across algorithms under identical normalization and reference point settings. This omission may lead to ambiguity in interpreting the magnitude of the reported improvements. (Section VI-C.2, pp. 10, Table IV)

11. In the simulation environment setup, many key parameters are described as being randomly selected within ranges, but the manuscript does not specify how many random instances were generated, whether results are averaged over multiple seeds, or what variability measure is used. This becomes more noticeable because the results discussion later states “Within the range of error…”, yet no error bars or confidence intervals or standard deviations are reported, leaving the statistical robustness of the comparisons unclear. (Section VI-A, p. 9, Table II; Section VI-C.5, p. 11, Fig. 8 discussion).

12. Baselines LB and LPD are single-policy heuristics rather than multi-objective methods. While including them is fine, the manuscript could better contextualize percentage improvements, especially since Pareto-based methods optimize trade-offs and DSSA introduces preference-based selection. A discussion about the limitations of these baselines would strengthen result interpretation. (Section VI-C, pp. 10-12)



Please rate the manuscript. Please explain under Public Comments below.: Good

## Reviewer: 2

Recommendation: **Author Should Prepare A Major Revision For A Second Review**

Comments:

1. The authors can include the contributions of recent papers and provide a tabular comparative analysis between their work and previous research, highlighting the unique contribution of their paper.
2. The authors assume over simplistic assumption of constant service reliability of computing nodes during the execution. However, in real world this static assumption is impractical.
3. The authors only consider reliability of computing nodes while ignoring the reliability of communication links which may lead to incomplete reliability assessment.
4. The simulation environment is quite small and limited to only 10 edge nodes and 1 central cloud. It is suggested to test the model with larger network topologies.
5. The time complexity is quite high that will degrade at larger scaler is also a major issue that should be addressed.
6. The proposed framework assumes a static network environment. However, in practice, the network conditions are dynamic in nature.
7. The paper only considers reliability and delay as optimization objectives, but did not consider energy consumption, cost in their model.


Please rate the manuscript. Please explain under Public Comments below.: Good



## Reviewer: 3

Recommendation: Reject

Comments:
This paper investigates the multi-objective deployment of deep neural network (DNN) inference tasks in edge computing by jointly considering service reliability, accuracy, and latency. The authors propose a deployment approach utilizing the NSGA-II algorithm to find the Pareto solution set, followed by an expectation-based selection algorithm to determine the final scheme. After careful review, I find that the manuscript lacks sufficient algorithmic novelty, realistic system modeling, and rigorous experimental evaluation. The detailed comments are as follows:

1. The proposed RDODA is essentially a direct application of the classic NSGA-II algorithm using standard genetic operations like two-point crossover and random mutation. It lacks customized algorithmic designs tailored to the topological dependencies of DNN graphs or the unique characteristics of edge networks. The authors must clearly articulate and justify their core algorithmic contributions beyond applying an existing framework to a new scenario.
2. The strategy of assigning negative fitness values to individuals violating the delay constraint is overly simplistic. This brute-force penalty can easily destroy population diversity during the early genetic search stages, leading to a suboptimal Pareto front.
3. The proposed selection algorithm uses a simple linear weighted sum to determine the final deployment scheme, which fundamentally degrades the multi-objective optimization back into a single-objective problem. If the user's expectation weights are already known, the authors need to justify why it is still necessary to compute the computationally expensive Pareto front instead of using a weighted single-objective optimization from the start.
4. Assuming that the service reliability of computing nodes remains constant during task execution is highly unrealistic for mobile edge environments. In practice, node availability is highly dynamic due to factors like battery constraints, mobility, and bursty workloads, which severely limit the practical applicability of the current model.
5. The reliability calculation simply multiplies the reliabilities of the assigned nodes, implying that subtask failures are independent events. However, if multiple inference layers of the same DNN are deployed on a single node, a failure of that node would cause all those subtasks to fail simultaneously, indicating a critical logical flaw in how the model handles correlated failures.
6. Comparing the proposed multi-objective optimization approach against extremely simple heuristics like random deployment or load balancing creates an unfair baseline. To demonstrate true effectiveness, the proposed algorithm must be evaluated against **recent state-of-the-art multi-objective optimization algorithms or modern learning-based deployment strategies.**
7. The experimental setup, featuring only ten edge nodes and one cloud node, is too small to represent realistic, large-scale edge computing scenarios. Given the high theoretical time complexity of O(n * m^2 * v^2 * s), it is highly questionable whether the algorithm can scale and converge within a reasonable timeframe when applied to massive networks and complex DNNs.
8. The simulation relies heavily on static parameter ranges, such as fixed transmission rates between edge nodes. The experiments fail to incorporate dynamic network factors like bandwidth fluctuations, node mobility, or background traffic variations, making the evaluation results unconvincing for mobile computing contexts.


Please rate the manuscript. Please explain under Public Comments below.: Poor

